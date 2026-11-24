"""Audio validation, bounded queues and explicit native-runtime capabilities."""
from __future__ import annotations

import array
import math
import sys
import wave
from collections import deque
from pathlib import Path


class AudioBacklog(RuntimeError):
    pass


class FrameQueue:
    """80 ms frames; a slow consumer stops rather than silently dropping audio."""
    def __init__(self, max_frames=6):
        if type(max_frames) is not int or max_frames < 1:
            raise ValueError("max_frames must be a positive integer")
        self.frames = deque()
        self.max_frames = max_frames
        self.closed = False

    def put(self, frame: bytes):
        if self.closed:
            raise RuntimeError("queue is closed")
        if not isinstance(frame, bytes) or len(frame) != 3840:
            raise ValueError("expected 1920 mono PCM16 samples")
        if len(self.frames) >= self.max_frames:
            self.close()
            raise AudioBacklog("audio backlog exceeded 480 ms budget; session stopped")
        self.frames.append(frame)

    def get(self):
        return self.frames.popleft() if self.frames else None

    def close(self):
        self.closed = True
        self.frames.clear()


def inspect_wav(path):
    path = Path(path)
    if path.stat().st_size > 30 * 1024 * 1024:
        raise ValueError("WAV exceeds 30 MiB limit")
    try:
        with wave.open(str(path), "rb") as stream:
            channels, width, rate, count = stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getnframes()
            if channels not in (1, 2) or width != 2 or rate not in (16000, 24000, 44100, 48000):
                raise ValueError("expected PCM16 mono/stereo at 16, 24, 44.1 or 48 kHz")
            if not count or count/rate > 60:
                raise ValueError("audio duration must be greater than zero and at most 60 seconds")
            raw = stream.readframes(count)
            if len(raw) != count*channels*width:
                raise ValueError("truncated WAV payload")
            values = array.array("h", raw)
            if sys.byteorder != "little":
                values.byteswap()
            rms = math.sqrt(sum(x*x for x in values)/len(values))/32768
            return {"sample_rate": rate, "channels": channels, "samples_per_channel": count,
                    "duration_seconds": count/rate, "rms": rms,
                    "clipped_fraction": sum(abs(x) >= 32760 for x in values)/len(values)}
    except (wave.Error, EOFError) as error:
        raise ValueError("invalid PCM WAV") from error


def normalize_wav(source, destination):
    """Linear resampling for fixture preparation; not a production DSP claim."""
    metadata = inspect_wav(source)
    with wave.open(str(source), "rb") as stream:
        values = array.array("h", stream.readframes(stream.getnframes()))
        if sys.byteorder != "little":
            values.byteswap()
    channels = metadata["channels"]
    mono = [sum(values[i:i+channels])/channels for i in range(0, len(values), channels)]
    count = max(1, round(len(mono)*24000/metadata["sample_rate"]))
    output = array.array("h")
    for i in range(count):
        position = min(len(mono)-1, i*metadata["sample_rate"]/24000)
        left = int(position)
        weight = position-left
        output.append(round(mono[left]*(1-weight)+mono[min(left+1, len(mono)-1)]*weight))
    with wave.open(str(destination), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(24000)
        if sys.byteorder != "little":
            output.byteswap()
        stream.writeframes(output.tobytes())
    return inspect_wav(destination)
