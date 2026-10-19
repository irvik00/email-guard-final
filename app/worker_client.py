"""One persistent CUDA subprocess, one in-flight neural inference, bounded waits."""
import json
import os
import queue
import subprocess
import threading
from .contract import ROOT, unavailable


class LayaWorker:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.process = None
        self.responses = queue.Queue(maxsize=4)
        self.lock = threading.Lock()
        self.closed = threading.Event()
        self.state = 'unavailable' if not enabled else 'initializing'
        self.reason = 'disabled_by_operator' if not enabled else None

    def start(self):
        if self.enabled:
            threading.Thread(target=self._start, daemon=True, name='laya-startup').start()

    def _start(self):
        try:
            if self.closed.is_set():
                return
            configured = os.environ.get('EMAIL_GUARD_LAYA_PYTHON')
            default = '.venv-decision/Scripts/python.exe' if os.name == 'nt' else '.venv-decision/bin/python'
            executable = (ROOT / (configured or default)).resolve()
            if not executable.is_file():
                raise FileNotFoundError()
            env = os.environ | {'PYTHONUTF8': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                                'HF_HUB_DISABLE_TELEMETRY': '1', 'DO_NOT_TRACK': '1', 'TOKENIZERS_PARALLELISM': 'false'}
            self.process = subprocess.Popen([str(executable), '-u', '-m', 'app.laya_worker'], cwd=ROOT,
                env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                encoding='utf-8', bufsize=1, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            if self.closed.is_set():
                self._stop()
                return
            threading.Thread(target=self._read, daemon=True, name='laya-pipe-reader').start()
            hello = self.responses.get(timeout=180)
            if hello.get('event') != 'ready':
                self.reason = hello.get('reason', 'startup_error')
                self.state = 'unavailable'
                self._stop()
                return
            self.state = 'ready'
        except Exception as error:
            self.state, self.reason = 'unavailable', type(error).__name__
            self._stop()

    def _read(self):
        try:
            for line in self.process.stdout:
                item = json.loads(line)
                self.responses.put(item, timeout=1)
        except Exception:
            pass
        self.state, self.reason = 'unavailable', 'worker_exited'
        try:
            self.responses.put_nowait({'status': 'unavailable', 'reason': 'worker_exited'})
        except queue.Full:
            pass

    def readiness(self):
        return {'status': self.state, 'reason': self.reason, 'concurrency': 1, 'device': 'cuda'}

    def predict(self, text):
        if self.state != 'ready':
            return unavailable('Laya_day5', self.reason or self.state, self.state)
        if not self.lock.acquire(timeout=1):
            return unavailable('Laya_day5', 'neural_busy', 'busy')
        try:
            self.process.stdin.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
            self.process.stdin.flush()
            response = self.responses.get(timeout=60)
            if response.get('model_id') != 'Laya_day5':
                raise RuntimeError('invalid_worker_response')
            return response
        except Exception:
            self.state, self.reason = 'unavailable', 'worker_failed_or_timeout'
            self._stop()
            return unavailable('Laya_day5', self.reason)
        finally:
            self.lock.release()

    def _stop(self):
        p = self.process
        if p and p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=5)

    def close(self):
        self.enabled = False
        self.closed.set()
        self._stop()
