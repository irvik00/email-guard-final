"""Explicit reproduction: one NEW bounded day-5 run; old artifacts stay unchanged."""
import subprocess
import sys
from day5_common import ROOT,read_json


def run(python,script):
    completed=subprocess.run([str(ROOT/python),'-u','-B',str(ROOT/script)],cwd=ROOT)
    if completed.returncode:
        raise SystemExit(f'{script}: real exit code {completed.returncode}; do not claim completion')


def main():
    # Explicitly invoking this driver authorizes reproducing a new run, not reopening old candidates.
    for python,script in [
        ('.venv/Scripts/python.exe','src/prepare_day5.py'),
        ('.venv-decision/Scripts/python.exe','src/train_laya_day5.py'),
        ('.venv-decision/Scripts/python.exe','src/evaluate_laya_day5.py'),
        ('.venv/Scripts/python.exe','src/evaluate_frozen_lr_day5.py'),
        ('.venv/Scripts/python.exe','src/personal_email_review.py'),
        ('.venv/Scripts/python.exe','src/check_day5.py'),
        ('.venv-decision/Scripts/python.exe','src/reload_day5.py'),
        ('.venv-decision/Scripts/python.exe','src/verify_day5.py'),
        ('.venv/Scripts/python.exe','src/report_day5.py'),
    ]:run(python,script)


if __name__=='__main__':main()
