"""Actual built-in speaker -> air -> built-in microphone check.

Raw captured audio stays ignored under .runtime; only metrics/test transcript
from the synthetic spoken phrase are written to reports.
"""

from pathlib import Path
import json, time
import numpy as np
import sounddevice as sd

ROOT = Path(__file__).resolve().parent.parent


def main():
    import sys

    sys.path.insert(0, str(ROOT))
    from sts_lab.live_paths import ASSETS, LIVE_HOME
    from moonshine_voice import Transcriber, ModelArch

    path, arch = (
        ASSETS
        / "moonshine/download.moonshine.ai/model/small-streaming-en/quantized_26_11_02",
        ModelArch.SMALL_STREAMING,
    )
    asr = Transcriber(path, arch)
    from live_inference_test import speech

    audio = speech(
        "hardware-correction",
        "I want to meet on Tuesday. Sorry, I mean Friday. Which day did I choose?",
    )
    rate = 16000
    received = []

    def record(indata, frames, timing, status):
        received.append(indata[:, 0].copy())

    with sd.InputStream(
        samplerate=16000,
        channels=1,
        dtype="float32",
        device="MacBook Air Microphone",
        callback=record,
    ):
        time.sleep(0.4)
        with sd.OutputStream(
            samplerate=rate, channels=1, dtype="float32", device="MacBook Air Speakers"
        ) as speaker:
            speaker.write((audio * 0.65)[:, None])
        time.sleep(0.8)
    pcm = np.concatenate(received)
    s = time.perf_counter()
    text = asr.transcribe_without_streaming(pcm.tolist(), sample_rate=16000)
    wall = time.perf_counter() - s
    transcript = " ".join(l.text for l in text.lines)
    result = {
        "actual_hardware": True,
        "input": "MacBook Air Microphone",
        "output": "MacBook Air Speakers",
        "capture_seconds": len(pcm) / 16000,
        "rms": float(np.sqrt(np.mean(pcm**2))),
        "transcript": transcript if "friday" in transcript.lower() else None,
        "transcript_saved": "friday" in transcript.lower(),
        "asr_wall_s": wall,
        "recognized_friday": "friday" in transcript.lower(),
    }
    (LIVE_HOME / "diagnostics").mkdir(parents=True, exist_ok=True)
    np.save(LIVE_HOME / "diagnostics/physical-probe.npy", pcm)
    (ROOT / "reports/live-hardware-probe.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))
    asr.close()


if __name__ == "__main__":
    main()
