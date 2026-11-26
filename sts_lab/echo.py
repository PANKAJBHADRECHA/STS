"""SpeexDSP acoustic echo cancellation using the client's rendered reference."""

import ctypes as c
import ctypes.util
from pathlib import Path
import numpy as np


class EchoCanceller:
    def __init__(self):
        library = ctypes.util.find_library("speexdsp")
        if library is None:
            library = next(
                (
                    str(p)
                    for p in [
                        Path("/opt/homebrew/lib/libspeexdsp.dylib"),
                        Path("/usr/local/lib/libspeexdsp.dylib"),
                    ]
                    if p.exists()
                ),
                None,
            )
        if library is None:
            raise RuntimeError("Install SpeexDSP with brew install speexdsp")
        self.lib = c.CDLL(library)
        p = c.c_void_p
        i = c.c_int
        short = c.POINTER(c.c_int16)
        for name, args, result in [
            ("speex_echo_state_init", [i, i], p),
            ("speex_echo_ctl", [p, i, p], i),
            ("speex_echo_cancellation", [p, short, short, short], None),
            ("speex_echo_state_destroy", [p], None),
            ("speex_preprocess_state_init", [i, i], p),
            ("speex_preprocess_ctl", [p, i, p], i),
            ("speex_preprocess_run", [p, short], i),
            ("speex_preprocess_state_destroy", [p], None),
        ]:
            fn = getattr(self.lib, name)
            fn.argtypes = args
            fn.restype = result
        self.state = self.lib.speex_echo_state_init(320, 4096)
        rate = i(16000)
        self.lib.speex_echo_ctl(self.state, 24, c.byref(rate))
        self.pre = self.lib.speex_preprocess_state_init(320, 16000)
        self.tail = 0
        self.lib.speex_preprocess_ctl(self.pre, 24, self.state)
        for control, value in [(0, 1), (18, -25), (20, -45), (22, -15)]:
            setting = i(value)
            self.lib.speex_preprocess_ctl(self.pre, control, c.byref(setting))

    def process(self, microphone, reference):
        pointer = c.POINTER(c.c_int16)
        mic = (np.clip(microphone, -1, 1) * 32767).astype(np.int16)
        ref = (np.clip(reference, -1, 1) * 32767).astype(np.int16)
        out = np.empty(320, dtype=np.int16)
        self.lib.speex_echo_cancellation(
            self.state,
            mic.ctypes.data_as(pointer),
            ref.ctypes.data_as(pointer),
            out.ctypes.data_as(pointer),
        )
        self.lib.speex_preprocess_run(self.pre, out.ctypes.data_as(pointer))
        if np.max(np.abs(reference)) > 0.002:
            self.tail = 15
        elif self.tail:
            self.tail -= 1
        # Do not denoise quiet user speech when there is no playback echo.
        return out.astype(np.float32) / 32768 if self.tail else microphone.copy()

    def close(self):
        if self.pre:
            self.lib.speex_preprocess_state_destroy(self.pre)
            self.pre = None
        if self.state:
            self.lib.speex_echo_state_destroy(self.state)
            self.state = None
