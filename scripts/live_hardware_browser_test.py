"""Five acoustic voice checks using real speakers and microphone, no human.

OS-generated fixtures travel through air; individual failures are recorded and
remaining cases are still attempted. No microphone audio or ambient transcript
is published. The optional fixture-speaker flag temporarily changes only that speaker and restores it.
"""

import argparse, json, subprocess, time, wave
from pathlib import Path
import numpy as np
import sounddevice as sd
from voice_checks import check, answer_number

ROOT = Path(__file__).resolve().parent.parent
PROGRESS = []
FIXTURE_STATE = {}
OUTPUT = ROOT / "reports/live-hardware-final.json"


def speak(text, name, rate=165):
    folder = Path.home() / "Library/Application Support/STS/test-fixtures/hardware"
    folder.mkdir(parents=True, exist_ok=True)
    source = folder / (name + ".aiff")
    target = folder / (name + ".wav")
    subprocess.run(
        ["say", "-v", "Samantha", "-r", str(rate), "-o", str(source), text],
        check=True,
        timeout=30,
    )
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", str(source), str(target)],
        check=True,
        timeout=30,
    )
    with wave.open(str(target)) as f:
        a = (
            np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(np.float32)
            / 32768
        )
    return a


def play(a):
    with sd.OutputStream(
        samplerate=16000, channels=1, dtype="float32", device="MacBook Air Speakers"
    ) as stream:
        stream.write((a * 0.8)[:, None])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--extended", action="store_true")
    p.add_argument(
        "--hold-seconds",
        type=int,
        default=0,
        help="Keep real microphone capture running for at least this duration",
    )
    p.add_argument("--case", help="Run one named acoustic case")
    p.add_argument(
        "--enable-fixture-speaker",
        action="store_true",
        help="Temporarily enable fixture speaker; restore its exact mute/volume afterward",
    )
    return p.parse_args()


def main(args):
    from playwright.sync_api import sync_playwright

    fixtures = [
        (
            "normal",
            "Please tell me what two plus two is. Give a short answer.",
            r"\b(four|4)\b",
        ),
        (
            "memory",
            "Remember my name is Alex and I live in London.",
            r"(remember|got|understood|alex|london|okay)",
        ),
        (
            "recall",
            "What is my name, and which city do I live in?",
            r"(?=.*alex)(?=.*london)",
        ),
        (
            "correction",
            "My meeting is Tuesday. Sorry, I mean Friday. Which day is my meeting?",
            r"friday",
        ),
    ]
    if args.quick:
        fixtures = fixtures[:1]
    if args.extended:
        fixtures += [
            (
                "changed_question",
                "What is five plus five? Actually, tell me what two plus three is.",
                r"\b(five|5)\b",
            ),
            ("fast_speech", "What is the capital of France?", r"paris"),
            ("slow_speech", "How many days are there in one week?", r"\b(seven|7)\b"),
            ("background_noise", "What is four plus four?", r"\b(eight|8)\b"),
        ]
        fixtures += [
            ("rapid_" + str(i), text, pattern)
            for i, text, pattern in [
                (1, "What is two plus three?", r"\b(five|5)\b"),
                (2, "What is three plus four?", r"\b(seven|7)\b"),
                (3, "What is four plus five?", r"\b(nine|9)\b"),
            ]
        ]
    if args.case:
        fixtures = [f for f in fixtures if f[0] == args.case]
        if not fixtures and args.case != "spoken_interruption":
            raise ValueError("Unknown acoustic case")
    sounds = {
        n: speak(
            t, n, 235 if n == "fast_speech" else 110 if n == "slow_speech" else 165
        )
        for n, t, _ in fixtures
    }
    if args.extended and "background_noise" in sounds:
        a = sounds["background_noise"]
        sounds["background_noise"] = np.clip(
            a + np.random.default_rng(3).normal(0, 0.004, len(a)).astype(np.float32),
            -1,
            1,
        )
    interrupt = speak(
        "Wait. Stop. Instead, tell me what three plus three is.", "interrupt", 190
    )
    errors = []
    rows = PROGRESS
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=False, ignore_default_args=["--mute-audio"]
        )
        page = browser.new_page(
            permissions=["microphone"],
            bypass_csp=True,
            viewport={"width": 1280, "height": 980},
        )
        page.on("pageerror", lambda e: errors.append(type(e).__name__))
        page.goto(f"http://127.0.0.1:{args.port}")
        page.locator("#start").click()
        page.wait_for_function(
            "!document.getElementById('text').disabled || ws===null", timeout=120000
        )
        if page.evaluate("ws===null"):
            raise RuntimeError(
                "Audio startup failed: " + page.locator("#status").inner_text()
            )
        session_start = time.monotonic()
        for name, text, expected in fixtures:
            previous = page.evaluate("liveMetrics.replies.length")
            start = time.monotonic()
            row = {"case": name, "passed": False}
            try:
                play(sounds[name])
                end = time.monotonic()
                speech_end_js = page.evaluate("performance.now()")
                page.wait_for_function(
                    f"liveMetrics.replies.length>{previous}", timeout=20000
                )
                reply = page.evaluate("liveMetrics.replies.at(-1)")
                row["answer_number"] = answer_number(reply["text"])
                reply_observed = time.monotonic()
                page.wait_for_function(
                    f"liveMetrics.audioStarts.some(x=>x.epoch==={reply['epoch']})",
                    timeout=20000,
                )
                starts = page.evaluate("liveMetrics.audioStarts")
                first = next(
                    (x["at_ms"] for x in starts if x["epoch"] == reply["epoch"]), None
                )
                if first is not None:
                    row["speech_end_to_device_start_s"] = (first - speech_end_js) / 1000
                row.update(
                    {
                        "passed": check(name, reply["text"], expected)
                        and reply["audio_s"] > 0.1,
                        "reply_metrics": {
                            k: v for k, v in reply.items() if k != "text"
                        },
                        "speech_end_to_reply_done_s": reply_observed - end,
                    }
                )
                page.wait_for_function(
                    f"liveMetrics.events.some(x=>x.type==='playback_end' && x.epoch==={reply['epoch']})",
                    timeout=20000,
                )
            except Exception as e:
                row["passed"] = False
                row["ui_status"] = page.locator("#status").inner_text()
                row["error_type"] = type(e).__name__
                if page.locator("#interrupt").is_enabled():
                    page.locator("#interrupt").click()
            row["duration_s"] = time.monotonic() - start
            rows.append(row)
            print(row, flush=True)
            if name not in ("memory",):
                page.locator("#reset").click()
            page.wait_for_timeout(500)
        if not args.quick and (not args.case or args.case == "spoken_interruption"):
            row = {"case": "spoken_interruption", "passed": False}
            try:
                # Typed priming isolates spoken barge-in from failures to hear
                # the first question. The interruption itself is acoustic.
                page.locator("#text").fill("Explain how rain forms in six sentences.")
                page.locator("#send").click()
                page.wait_for_function("playing", timeout=30000)
                count = page.evaluate("liveMetrics.cancellations.length")
                previous = page.evaluate("liveMetrics.replies.length")
                primed_epoch = page.evaluate("activeEpoch")
                onset_js = page.evaluate("performance.now()")
                play(interrupt)
                page.wait_for_function(
                    f"liveMetrics.cancellations.length>{count}", timeout=10000
                )
                page.wait_for_function(
                    f"liveMetrics.replies.some(r=>r.epoch>{primed_epoch})",
                    timeout=30000,
                )
                reply = page.evaluate("liveMetrics.replies.at(-1)")
                page.wait_for_function(
                    f"liveMetrics.events.some(x=>x.type==='playback_end' && x.epoch==={reply['epoch']})",
                    timeout=20000,
                )
                row["answer_number"] = answer_number(reply["text"])
                suppressed = page.evaluate(
                    "liveMetrics.events.filter(x=>x.type==='playback_suppressed')"
                )
                after = [e for e in suppressed if e["at_ms"] >= onset_js]
                if after:
                    row["onset_to_render_suppression_s"] = (
                        after[0]["at_ms"] - onset_js
                    ) / 1000
                row.update(
                    {
                        "passed": check("spoken_interruption", reply["text"])
                        and reply["audio_s"] > 0.1,
                        "cancellation_s": (
                            page.evaluate(f"liveMetrics.cancellations[{count}].at_ms")
                            - onset_js
                        )
                        / 1000,
                        "reply_metrics": {
                            k: v for k, v in reply.items() if k != "text"
                        },
                    }
                )
            except Exception as e:
                row["passed"] = False
                row["ui_status"] = page.locator("#status").inner_text()
                row["error_type"] = type(e).__name__
                if page.locator("#interrupt").is_enabled():
                    page.locator("#interrupt").click()
            rows.append(row)
            print(row, flush=True)
        while time.monotonic() - session_start < args.hold_seconds:
            page.wait_for_timeout(1000)
            if page.evaluate("ws===null"):
                errors.append(
                    page.evaluate(
                        "liveMetrics.lastError || 'Session ended during continuous capture'"
                    )
                )
                break
        session_duration = time.monotonic() - session_start
        metrics = page.evaluate(
            "({capture_settings:liveMetrics.captureSettings,audio_context_rate:liveMetrics.audioContextRate,last_error:liveMetrics.lastError,engine_metrics:liveMetrics.engineMetrics,configuration:liveMetrics.configuration,frames:liveMetrics.captureFrames,output_frames:liveMetrics.outputFrames,devices:liveMetrics.devices,speakers_muted:liveMetrics.speakersMuted,device_status_events:liveMetrics.deviceStatusEvents,synthesis_starvation_frames:liveMetrics.synthesisStarvationFrames})"
        )
        if page.locator("#stop").is_enabled():
            page.locator("#stop").click()
        page.wait_for_function(
            "ws===null && liveMetrics.microphoneReleased===true", timeout=30000
        )
        metrics["engine_metrics"] = page.evaluate("liveMetrics.engineMetrics")
        browser.close()
    if not rows:
        raise RuntimeError("No acoustic cases executed")
    report = {
        "power_source": subprocess.check_output(
            ["pmset", "-g", "batt"], text=True
        ).splitlines()[0],
        "session_duration_s": session_duration,
        "continuous_capture_completed": session_duration >= args.hold_seconds
        and not errors,
        "actual_hardware": True,
        "fixture_speaker": FIXTURE_STATE,
        "human_participant": False,
        "fake_audio_capture": False,
        "source": "OS-generated speech through built-in speakers, air and built-in microphone",
        "cases": rows,
        "metrics": metrics,
        "microphone_released": True,
        "console_errors": errors,
        "status": "passed"
        if all(x["passed"] for x in rows) and not errors
        else "failed",
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], flush=True)


if __name__ == "__main__":
    try:
        from coreaudio_fixture import fixture_speaker

        args = parse_args()
        with fixture_speaker(args.enable_fixture_speaker) as state:
            FIXTURE_STATE.update(state)
            main(args)
        report = json.loads(OUTPUT.read_text())
        report["fixture_speaker"] = state
        OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    except Exception as error:
        OUTPUT.write_text(
            json.dumps(
                {
                    "actual_hardware": True,
                    "human_participant": False,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "completed_cases": PROGRESS,
                },
                indent=2,
            )
            + "\n"
        )
        raise
