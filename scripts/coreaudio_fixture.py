"""Read and temporarily enable only the acoustic fixture's built-in speaker.

Never changes the default output device or its USB/headphone volume. Restores
that speaker's exact mute/volume state even when the acoustic test fails.
"""

import ctypes as c
from contextlib import contextmanager


class Address(c.Structure):
    _fields_ = [
        ("selector", c.c_uint32),
        ("scope", c.c_uint32),
        ("element", c.c_uint32),
    ]


def four(value):
    return int.from_bytes(value.encode("ascii"), "big")


class CoreAudio:
    def __init__(self):
        self.api = c.CDLL("/System/Library/Frameworks/CoreAudio.framework/CoreAudio")
        self.cf = c.CDLL(
            "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
        )
        self.api.AudioObjectGetPropertyData.argtypes = [
            c.c_uint32,
            c.POINTER(Address),
            c.c_uint32,
            c.c_void_p,
            c.POINTER(c.c_uint32),
            c.c_void_p,
        ]
        self.api.AudioObjectGetPropertyDataSize.argtypes = [
            c.c_uint32,
            c.POINTER(Address),
            c.c_uint32,
            c.c_void_p,
            c.POINTER(c.c_uint32),
        ]
        self.api.AudioObjectSetPropertyData.argtypes = [
            c.c_uint32,
            c.POINTER(Address),
            c.c_uint32,
            c.c_void_p,
            c.c_uint32,
            c.c_void_p,
        ]
        self.cf.CFStringGetCString.argtypes = [
            c.c_void_p,
            c.c_void_p,
            c.c_long,
            c.c_uint32,
        ]
        self.cf.CFStringGetCString.restype = c.c_bool
        self.cf.CFRelease.argtypes = [c.c_void_p]

    def read(self, device, key, kind, scope="outp"):
        a = Address(four(key), four(scope), 0)
        size = c.c_uint32(c.sizeof(kind))
        value = kind()
        result = self.api.AudioObjectGetPropertyData(
            device, c.byref(a), 0, None, c.byref(size), c.byref(value)
        )
        if result:
            raise OSError(f"CoreAudio property {key}: {result}")
        return value.value

    def write(self, device, key, value, kind):
        a = Address(four(key), four("outp"), 0)
        v = kind(value)
        result = self.api.AudioObjectSetPropertyData(
            device, c.byref(a), 0, None, c.sizeof(v), c.byref(v)
        )
        if result:
            raise OSError(f"CoreAudio set {key}: {result}")

    def speaker(self):
        a = Address(four("dev#"), four("glob"), 0)
        size = c.c_uint32()
        if self.api.AudioObjectGetPropertyDataSize(
            1, c.byref(a), 0, None, c.byref(size)
        ):
            raise OSError("Cannot list audio devices")
        devices = (c.c_uint32 * (size.value // 4))()
        if self.api.AudioObjectGetPropertyData(
            1, c.byref(a), 0, None, c.byref(size), devices
        ):
            raise OSError("Cannot read audio devices")
        for device in devices:
            name = self.read(device, "lnam", c.c_void_p, "glob")
            buf = c.create_string_buffer(1024)
            self.cf.CFStringGetCString(name, buf, len(buf), 0x08000100)
            self.cf.CFRelease(name)
            if buf.value.decode() == "MacBook Air Speakers":
                return device
        raise RuntimeError("Built-in acoustic fixture speaker not found")


@contextmanager
def fixture_speaker(enable=False):
    api = CoreAudio()
    device = api.speaker()
    state = {
        "muted": api.read(device, "mute", c.c_uint32),
        "volume": api.read(device, "volm", c.c_float),
    }
    if state["muted"] and not enable:
        raise RuntimeError(
            "Fixture speaker is muted; use --enable-fixture-speaker for a temporary audible test"
        )
    info = {
        "original_state": state,
        "temporary_fixture_volume": 0.55 if enable else None,
        "default_output_unchanged": True,
    }
    try:
        if enable:
            api.write(device, "volm", 0.55, c.c_float)
            api.write(device, "mute", 0, c.c_uint32)
        yield info
    finally:
        if enable:
            api.write(device, "mute", state["muted"], c.c_uint32)
            api.write(device, "volm", state["volume"], c.c_float)
        restored = {
            "muted": api.read(device, "mute", c.c_uint32),
            "volume": api.read(device, "volm", c.c_float),
        }
        info["restored_state"] = restored
        info["restoration_verified"] = (
            restored["muted"] == state["muted"]
            and abs(restored["volume"] - state["volume"]) < 1e-6
        )
        if not info["restoration_verified"]:
            raise RuntimeError("Fixture speaker state was not restored")
