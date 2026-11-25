"""Bounded local worker lifecycle; microphone audio is temporary, never logged."""
import atexit
import base64
import json
from pathlib import Path
import selectors
import subprocess
import tempfile
import threading
import time

from .audio import inspect_wav

ROOT=Path(__file__).resolve().parent.parent


class VoiceService:
    def __init__(self):
        self.lock=threading.Lock()
        self.process=None
        self.selector=None
        self.log=None
        self.last_error=None
        self.session=None
        self.ready=False
        atexit.register(self.close)

    def status(self):
        alive=self.process is not None and self.process.poll() is None
        return {'ready':bool(alive and self.ready),'busy':self.lock.locked(),
                'installed':(ROOT/'.venv-sts/bin/python').exists(),
                'assets_ready':(ROOT/'.runtime/lfm-4bit/model.safetensors').exists(),
                'model':'LFM2.5-Audio-1.5B / MLX 4-bit','mode':'native_sts_turn_based',
                'full_duplex':False,'error':self.last_error}

    def _read(self, timeout):
        if not self.selector.select(timeout):
            self.close()
            raise TimeoutError('Native speech worker exceeded its time budget; reload the model')
        line=self.process.stdout.readline()
        if not line:
            self.close()
            raise RuntimeError('Native speech worker exited; see local worker log')
        return json.loads(line)

    def _load(self):
        if self.status()['ready']:
            return {'ready':True,'already_loaded':True}
        if not self.status()['installed'] or not self.status()['assets_ready']:
            raise RuntimeError('Install the native STS runtime and download the pinned model first')
        self.close()
        logs=ROOT/'.runtime';logs.mkdir(exist_ok=True)
        self.log=(logs/'voice-worker.log').open('a')
        self.process=subprocess.Popen([str(ROOT/'.venv-sts/bin/python'),str(ROOT/'scripts/voice_worker.py')],
                                      stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1,cwd=ROOT)
        self.selector=selectors.DefaultSelector();self.selector.register(self.process.stdout,selectors.EVENT_READ)
        result=self._read(90)
        self.ready=bool(result.get('ready'));self.last_error=None
        return result

    def load(self):
        if not self.lock.acquire(False):raise BlockingIOError('Speech worker is busy')
        try:return self._load()
        finally:self.lock.release()

    def reset(self,session):
        if not self.lock.acquire(False):raise BlockingIOError('Speech worker is busy')
        try:
            if not self.status()['ready']:return {'reset':True}
            self.process.stdin.write(json.dumps({'command':'reset','session':session})+'\n');self.process.stdin.flush()
            result=self._read(15)
            if not result.get('ok'):raise RuntimeError(result.get('error','Reset failed'))
            self.session=None
            return result['result']
        finally:self.lock.release()

    def turn(self,payload,session,reset=False):
        if not self.lock.acquire(False):raise BlockingIOError('Speech worker is busy')
        try:
            (ROOT/'.runtime').mkdir(exist_ok=True)
            # A new browser may take ownership only by explicitly starting anew.
            if self.session and self.session!=session and not reset:
                raise BlockingIOError('Another browser owns this conversation; start a new conversation')
            with tempfile.TemporaryDirectory(prefix='sts-voice-',dir=ROOT/'.runtime') as directory:
                source=Path(directory)/'input.wav';output=Path(directory)/'reply.wav';source.write_bytes(payload)
                metadata=inspect_wav(source)
                if metadata['duration_seconds']>20:raise ValueError('Voice turns must be at most 20 seconds')
                if metadata['rms']<.0005:raise ValueError('No audible speech detected; record again')
                self._load()
                self.session=session
                self.process.stdin.write(json.dumps({'command':'turn','session':session,'input':str(source),'output':str(output),'reset':reset})+'\n');self.process.stdin.flush()
                result=self._read(120)
                if not result.get('ok'):raise ValueError(result.get('error','Native inference failed'))
                self.last_error=None
                return result['result'] | {'audio_base64':base64.b64encode(output.read_bytes()).decode()}
        except Exception as error:
            self.last_error=str(error)
            raise
        finally:self.lock.release()

    def close(self):
        self.ready=False
        self.session=None
        if self.process:
            if self.process.poll() is None:
                self.process.terminate()
                try:self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
            if self.process.stdin:self.process.stdin.close()
            if self.process.stdout:self.process.stdout.close()
            self.process=None
        if self.selector:self.selector.close();self.selector=None
        if self.log:self.log.close();self.log=None


VOICE=VoiceService()
