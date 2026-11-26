"""Actual neural streaming regression with clean recorded input, not hardware.

Playback is a timed software renderer; microphone hardware is tested separately.
"""

import argparse, json, queue, sys, threading, time, wave, subprocess
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sts_lab.live_engine import LiveEngine


def speech(name, text, rate=175):
    d = Path.home() / "Library/Application Support/STS/test-fixtures/inference"
    d.mkdir(parents=True, exist_ok=True)
    a = d / (name + ".aiff")
    w = d / (name + ".wav")
    subprocess.run(
        ["say", "-v", "Samantha", "-r", str(rate), "-o", str(a), text], check=True
    )
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", str(a), str(w)], check=True
    )
    with wave.open(str(w)) as f:
        return (
            np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(np.float32)
            / 32768
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen-2b")
    parser.add_argument("--voice", default="piper_en_US-lessac-medium")
    parser.add_argument("--asr", default="MEDIUM_STREAMING")
    parser.add_argument("--report", default="live-inference.json")
    args = parser.parse_args()
    events = queue.Queue()
    engine = LiveEngine(events.put, model=args.model, voice=args.voice, arch=args.asr)
    engine.load()
    stop = threading.Event()
    input_audio = queue.Queue()
    rows = []
    frames = 0

    def capture():
        nonlocal frames
        frame = np.zeros(320, dtype=np.float32)
        while not stop.is_set():
            try:
                frame = input_audio.get_nowait()
            except queue.Empty:
                frame = np.zeros(320, dtype=np.float32)
            try:
                engine.feed(frame.astype("<f4").tobytes())
            except queue.Full:
                stop.set()
                raise
            frames += 1
            time.sleep(0.02)

    def inject(a):
        for i in range(0, len(a), 320):
            input_audio.put(
                np.pad(a[i : i + 320], (0, max(0, 320 - len(a[i : i + 320]))))
            )

    cases = [
        ("normal", "What is two plus two?", ["four", "4"]),
        (
            "memory",
            "My name is Alex and I live in London. Remember that.",
            ["alex", "remember", "understood", "got"],
        ),
        ("recall", "What is my name and where do I live?", ["alex", "london"]),
        (
            "correction",
            "My meeting is Tuesday. Sorry, I mean Friday. Which day is my meeting?",
            ["friday"],
        ),
        (
            "changed_question",
            "What is five plus five? Actually, tell me what two plus three is.",
            ["five", "5"],
        ),
        ("background_noise", "What is four plus four?", ["eight", "8"]),
        ("fast_speech", "What is the capital of France?", ["paris"]),
        ("slow_speech", "How many days are there in one week?", ["seven", "7"]),
        ("retained_correction", "Which day is my meeting now?", ["friday"]),
    ]
    sources = {
        n: speech(
            n, t, rate=245 if n == "fast_speech" else 110 if n == "slow_speech" else 175
        )
        for n, t, _ in cases
    }
    sources["background_noise"] = np.clip(
        sources["background_noise"]
        + np.random.default_rng(3)
        .normal(0, 0.004, len(sources["background_noise"]))
        .astype(np.float32),
        -1,
        1,
    )
    sources["long"] = speech("long", "Explain how rain forms, in six sentences.")
    sources["interrupt"] = speech(
        "interrupt", "Wait. Stop. Instead, tell me what three plus three is."
    )
    thread = threading.Thread(target=capture, daemon=True)
    thread.start()
    render_end = 0.0
    timers = []

    def handle(e):
        nonlocal render_end
        if e["type"] == "audio":
            audio = np.frombuffer(e["samples"], dtype="<f4")
            duration = len(audio) / e["rate"]
            render_end = max(time.monotonic(), render_end) + duration
            timer = threading.Timer(
                max(0, render_end - time.monotonic()),
                lambda: engine.ack(e["epoch"], e["chunk"]),
            )
            timer.daemon = True
            timer.start()
            timers.append(timer)
            assert np.isfinite(audio).all()
        elif e["type"] == "cancel":
            render_end = time.monotonic()

    for name, text, expected in cases:
        while not events.empty():
            handle(events.get())
        inject(sources[name])
        deadline = time.monotonic() + 35
        result = None
        user = ""
        while time.monotonic() < deadline:
            try:
                e = events.get(timeout=0.2)
            except queue.Empty:
                continue
            handle(e)
            if e["type"] == "user":
                user = e["text"]
            if e["type"] == "reply_done":
                result = e
                break
            if e["type"] == "error":
                raise RuntimeError(e)
        passed = bool(
            result
            and (
                all(x in result["text"].lower() for x in expected)
                if name == "recall"
                else any(x in result["text"].lower() for x in expected)
            )
            and result["audio_s"] > 0.1
        )
        row = {"case": name, "recognized": user, "passed": passed, "result": result}
        rows.append(row)
        print(row, flush=True)
        while time.monotonic() < render_end + 0.1:
            time.sleep(0.02)
        if result:
            engine.finish_playback(result["epoch"])
        if not passed and not stop.is_set():
            engine.interrupt("test recovery")
            while not input_audio.empty():
                input_audio.get_nowait()
        if stop.is_set():
            for pending, _, _ in cases[len(rows) :]:
                rows.append(
                    {
                        "case": pending,
                        "passed": False,
                        "status": "not_executed_after_capture_failure",
                    }
                )
            break
    if all(r["passed"] for r in rows):
        inject(sources["long"])
        deadline = time.monotonic() + 35
        interrupted = False
        cancel_at = None
        first_audio = None
        result = None
        while time.monotonic() < deadline:
            try:
                e = events.get(timeout=0.2)
            except queue.Empty:
                continue
            handle(e)
            if e["type"] == "audio" and not interrupted:
                first_audio = time.monotonic()
                inject(sources["interrupt"])
                interrupted = True
            if (
                interrupted
                and e["type"] == "cancel"
                and e.get("reason") == "recognized speech"
            ):
                cancel_at = time.monotonic()
            if (
                interrupted
                and e["type"] == "reply_done"
                and any(x in e["text"].lower() for x in ["six", "6"])
            ):
                result = e
                break
        rows.append(
            {
                "case": "spoken_interruption",
                "passed": result is not None,
                "result": result,
                "time_to_cancellation_s": cancel_at - first_audio
                if cancel_at and first_audio
                else None,
            }
        )
    stop.set()
    thread.join(timeout=2)
    for timer in timers:
        timer.cancel()
    engine.close()
    report = {
        "mode": "actual_neural_inference_recorded_input",
        "configuration": {"model": args.model, "voice": args.voice, "asr": args.asr},
        "hardware": False,
        "inference_mocked": False,
        "renderer": "timed software acknowledgements",
        "frames_sent": frames,
        "engine_metrics": engine.metrics,
        "cases": rows,
        "status": "passed"
        if len(rows) == 10 and all(x["passed"] for x in rows)
        else "failed",
    }
    (ROOT / "reports" / args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
