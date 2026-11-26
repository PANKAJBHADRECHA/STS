"""Continuous Core Audio capture/playback with an exact render reference."""

from collections import deque
import queue, threading, time
import numpy as np
import sounddevice as sd


class NativeAudio:
    def __init__(self, engine, notify):
        self.engine = engine
        self.notify = notify
        self.lock = threading.RLock()
        self.pending = deque()
        self.epoch = 0
        self.gain = 1.0
        self.done = False
        self.frames = 0
        self.output_frames = 0
        self.overlap_frames = 0
        self.underflows = 0
        self.starvation_frames = 0
        self.has_audio = False
        self.suppress_pending = None
        self.actions = queue.Queue()
        self.closed = threading.Event()

    def start(self):
        # PortAudio caches CoreAudio device IDs at import. Refresh between
        # sessions so unplug/replug does not reuse an invalid AudioDeviceID.
        # No other stream is open: the server permits only one active session.
        sd._terminate()
        sd._initialize()
        self.worker = threading.Thread(target=self._events, daemon=True)
        self.worker.start()
        capture = sd.default.device[0]
        self.stream = sd.Stream(
            device=(capture, sd.default.device[1]),
            samplerate=16000,
            blocksize=320,
            channels=(1, 1),
            dtype="float32",
            latency="low",
            callback=self._callback,
        )
        self.stream.start()
        self.devices = {
            "input": sd.query_devices(self.stream.device[0])["name"],
            "output": sd.query_devices(self.stream.device[1])["name"],
        }

    def accept(self, event):
        kind = event["type"]
        with self.lock:
            if kind == "cancel":
                self.epoch = event["next_epoch"]
                self.pending.clear()
                self.has_audio = False
                self.done = False
            elif kind == "reply_start":
                if event["epoch"] < self.epoch:
                    return
                self.epoch = event["epoch"]
                self.done = False
                self.gain = 1.0
                self.has_audio = False
            elif kind == "speech_start":
                self.gain = 0.25
                if self.pending:
                    self.suppress_pending = self.epoch
            elif kind == "duck_restore":
                self.gain = 1.0
            elif kind == "audio":
                if event["epoch"] != self.epoch:
                    return
                samples = np.frombuffer(event["samples"], dtype="<f4")
                audio = np.interp(
                    np.arange(0, len(samples), event["rate"] / 16000),
                    np.arange(len(samples)),
                    samples,
                ).astype(np.float32)
                if (
                    sum(len(p["audio"]) - p["at"] for p in self.pending) + len(audio)
                    > 16000 * 60
                ):
                    raise RuntimeError(
                        "Speaker queue exceeded 60 seconds; restart the conversation"
                    )
                self.pending.append(
                    {
                        "audio": audio,
                        "at": 0,
                        "chunk": event["chunk"],
                        "epoch": event["epoch"],
                        "suppressed": False,
                    }
                )
            elif kind == "reply_done":
                if event["epoch"] != self.epoch:
                    return
                self.done = True
                if not self.pending:
                    self.has_audio = False
                    self.actions.put(("played", self.epoch))

    def _callback(self, indata, outdata, frames, timing, status):
        output = np.zeros(frames, dtype=np.float32)
        finished = []
        started = None
        completed_epoch = None
        with self.lock:
            if self.suppress_pending is not None:
                self.actions.put(
                    (
                        "event",
                        {
                            "type": "playback_suppressed",
                            "epoch": self.suppress_pending,
                            "at": time.monotonic(),
                        },
                    )
                )
                self.suppress_pending = None
            position = 0
            while self.pending and position < frames:
                self.has_audio = True
                item = self.pending[0]
                size = min(frames - position, len(item["audio"]) - item["at"])
                if item["at"] == 0:
                    started = {
                        "type": "playback_start",
                        "epoch": item["epoch"],
                        "chunk": item["chunk"],
                        "at": time.monotonic(),
                    }
                output[position : position + size] = (
                    item["audio"][item["at"] : item["at"] + size] * self.gain
                )
                if self.gain == 0.0:
                    item["suppressed"] = True
                item["at"] += size
                position += size
                if item["at"] == len(item["audio"]):
                    if not item.get("suppressed", False):
                        finished.append((item["epoch"], item["chunk"]))
                    self.pending.popleft()
            if (
                self.has_audio
                and not position
                and not self.pending
                and not self.done
                and self.gain == 1.0
            ):
                self.starvation_frames += 1
            if position and not self.pending and self.done:
                completed_epoch = self.epoch
                self.done = False
                self.has_audio = False
        outdata[:, 0] = output
        try:
            self.engine.feed(
                np.concatenate([indata[:, 0], output]).astype("<f4").tobytes()
            )
        except (queue.Full, ValueError) as error:
            self.actions.put(
                (
                    "error",
                    "Audio input stopped ("
                    + type(error).__name__
                    + "); restart the conversation",
                )
            )
            raise sd.CallbackAbort
        self.frames += 1
        if np.max(np.abs(output)) > 0.001:
            self.output_frames += 1
            if np.sqrt(np.mean(indata[:, 0] ** 2)) > 0.004:
                self.overlap_frames += 1
        if started:
            self.actions.put(("event", started))
        for epoch, chunk in finished:
            self.actions.put(("ack", epoch, chunk))
        if completed_epoch is not None:
            self.actions.put(("played", completed_epoch))
        if self.frames % 50 == 0:
            self.actions.put(
                (
                    "event",
                    {
                        "type": "capture",
                        "frames": self.frames,
                        "output_frames": self.output_frames,
                        "overlap_frames": self.overlap_frames,
                        "device_status_events": self.underflows,
                        "synthesis_starvation_frames": self.starvation_frames,
                    },
                )
            )
        if status:
            self.underflows += 1

    def _events(self):
        while not self.closed.is_set():
            try:
                item = self.actions.get(timeout=0.2)
            except queue.Empty:
                continue
            if item[0] == "ack":
                self.engine.ack(item[1], item[2])
            elif item[0] == "played":
                self.engine.finish_playback(item[1])
                self.notify({"type": "playback_end", "epoch": item[1]})
            elif item[0] == "event":
                self.notify(item[1])
            elif item[0] == "error":
                self.engine.interrupt("audio error")
                self.notify({"type": "error", "message": item[1]})

    def close(self):
        self.closed.set()
        if hasattr(self, "stream"):
            self.stream.stop()
            self.stream.close()
        if hasattr(self, "worker"):
            self.worker.join(timeout=2)
