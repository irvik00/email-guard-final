"""Validated py7zr 1.1.3 streaming factory: one solid decompression, bounded disk spools."""
import hashlib
import re
import stat
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath

import py7zr
from py7zr.io import Py7zIO, WriterFactory
from normalization_adapters import InputRow, project_path

MESSAGE = re.compile(r"^\d{9,11}\.[A-Za-z0-9_.-]+$")


def validate_member(name, *, directory=False, symlink=False, junction=False, socket=False, mode=None):
    if not name or "\\" in name or ":" in name or "\x00" in name or name.startswith("/") or PureWindowsPath(name).drive:
        raise ValueError("Unsafe archive member path")
    parts = name.split("/")
    if any(p in {"", ".", ".."} or p.endswith((" ", ".")) for p in parts):
        raise ValueError("Unsafe archive member components")
    if any(re.fullmatch(r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", p, re.I) for p in parts):
        raise ValueError("Windows reserved archive path")
    if symlink or junction or socket or (mode is not None and stat.S_IFMT(mode) not in {0, stat.S_IFREG, stat.S_IFDIR}):
        raise ValueError("Links/special archive files refused")
    if mode is not None and stat.S_IFMT(mode) == stat.S_IFDIR and not directory:
        raise ValueError("Inconsistent archive member type")
    return PurePosixPath(name)


def inspect_archive(path, archive_config):
    with py7zr.SevenZipFile(path, "r") as archive:
        if archive.needs_password():
            raise ValueError("Password-protected archive")
        public = archive.list()
        if len(public) > archive_config["max_members"]:
            raise ValueError("Archive member count exceeds limit")
        members, seen, total = [], set(), 0
        for position, (info, raw) in enumerate(zip(public, archive.files), 1):
            props = raw.file_properties()
            validate_member(info.filename, directory=info.is_directory, symlink=raw.is_symlink,
                            junction=raw.is_junction, socket=raw.is_socket, mode=props.get("posix_mode"))
            if info.filename.casefold() in seen:
                raise ValueError("Duplicate/case-colliding archive names")
            seen.add(info.filename.casefold())
            if not info.is_directory and not info.is_file:
                raise ValueError("Non-regular archive member")
            size = info.uncompressed or 0
            if size < 0:
                raise ValueError("Negative archive member size")
            total += size
            if total > archive_config["max_uncompressed_bytes"]:
                raise ValueError("Archive uncompressed volume exceeds limit")
            members.append({"name": info.filename, "position": position, "size_bytes": size,
                            "type": "directory" if info.is_directory else "regular_file",
                            "email_filename_candidate": not info.is_directory and bool(MESSAGE.fullmatch(PurePosixPath(info.filename).name))})
        metadata = archive.archiveinfo()
        return {"adapter": "py7zr WriterFactory/Py7zIO.close single-pass bounded spools", "py7zr_version": py7zr.__version__,
                "solid": metadata.solid, "blocks": metadata.blocks, "methods": metadata.method_names,
                "member_count": len(members), "uncompressed_bytes": total, "members": members,
                "source_url": "https://untroubled.org/spam/", "source_notes": "publisher describes spam-trap source, forged headers and Unix timestamp filenames; archive not downloaded again"}


def process_archive(path, root, work_dir, config, on_row, trial_limit=None):
    inventory = inspect_archive(path, config["archive"])
    lookup = {m["name"]: m for m in inventory["members"] if m["type"] == "regular_file"}
    max_bytes = config["normalization"]["max_record_bytes"]
    completed, accepted, written_bytes = set(), 0, 0
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    extraction = work_dir / "virtual_archive_root"
    extraction.mkdir()

    class Spool(Py7zIO):
        def __init__(self, member):
            self.member = member
            self.handle = tempfile.TemporaryFile(dir=work_dir, mode="w+b")
            self.length = 0
            self.finished = False

        def write(self, data):
            nonlocal written_bytes
            if self.length + len(data) > self.member["size_bytes"]:
                raise ValueError("Archive member exceeded declared size")
            written_bytes += len(data)
            if written_bytes > config["archive"]["max_uncompressed_bytes"]:
                raise ValueError("Actual decompression volume exceeded limit")
            self.length += len(data)
            # Oversized/service files are counted without materializing their payload.
            if self.member["size_bytes"] <= max_bytes and self.member["email_filename_candidate"]:
                return self.handle.write(data)
            return len(data)

        def read(self, size=None):
            return self.handle.read(-1 if size is None else size)

        def seek(self, offset, whence=0):
            return self.handle.seek(offset, whence)

        def flush(self):
            self.handle.flush()

        def size(self):
            return self.length

        def close(self):
            nonlocal accepted
            if self.finished:
                return
            self.finished = True
            try:
                if self.length != self.member["size_bytes"]:
                    raise ValueError("Truncated decompressed member")
                completed.add(self.member["name"])
                row = InputRow("untroubled", path, f"{project_path(path, root)}::{self.member['name']}",
                               self.member["position"], self.member["name"].split("/")[0], "raw_email",
                               original_label="spam", size_bytes=self.length)
                if not self.member["email_filename_candidate"]:
                    row.excluded_reason = "service_file"
                elif self.length > max_bytes:
                    row.excluded_reason = "raw_record_too_large"
                elif trial_limit is not None and accepted >= trial_limit:
                    return
                else:
                    accepted += 1
                    self.handle.seek(0)
                    row.raw = self.handle.read(max_bytes + 1)
                    if len(row.raw) != self.length:
                        raise ValueError("Incomplete spool")
                on_row(row, self.member["name"])
            finally:
                self.handle.close()

    class Factory(WriterFactory):
        def create(self, filename):
            # Library supplies a sanitized virtual pathname; no member path is written.
            member_name = Path(filename).resolve().relative_to(extraction.resolve()).as_posix()
            return Spool(lookup[member_name])

    with path.open("rb") as archive_input:
        with py7zr.SevenZipFile(archive_input, "r", max_extract_size=config["archive"]["max_uncompressed_bytes"]) as archive:
            archive.extractall(path=extraction, factory=Factory())
    if completed != set(lookup):
        raise ValueError("Not all regular archive members decompressed")
    inventory["actual_decompressed_bytes"] = written_bytes
    inventory["normalized_candidates_this_run"] = accepted
    return inventory
