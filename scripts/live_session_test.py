"""Five-minute real CoreAudio lifecycle test, typed prompts, not a voice qualification.

Keeps actual microphone capture open during neural synthesis/device rendering.
Never unmutes speakers, changes input gain, saves audio, or records ambient text.
"""

import argparse, asyncio, json, sys, time
from pathlib import Path
import psutil
from websockets.asyncio.client import connect

ROOT = Path(__file__).resolve().parent.parent
PROGRESS = {}


async def run(port, seconds):
    result = {
        "mode": "native_device_lifecycle_typed_prompts",
        "human_participant": False,
        "voice_recognition_qualified": False,
        "speakers_unmuted_by_test": False,
        "cases": [],
        "errors": [],
    }
    PROGRESS.clear()
    PROGRESS.update(result)
    result = PROGRESS
    start = time.monotonic()
    capture = {}
    replies = []
    cpu = []
    rss = []
    asr_events = 0
    server_pid = next(
        (
            p.pid
            for p in psutil.process_iter(["cmdline"])
            if p.info["cmdline"]
            and "scripts/live_server.py" in p.info["cmdline"]
            and str(port) in p.info["cmdline"]
        ),
        None,
    )
    process = psutil.Process(server_pid) if server_pid else None
    if process:
        process.cpu_percent()
    async with connect(
        f"ws://127.0.0.1:{port + 1}", origin=f"http://127.0.0.1:{port}", max_size=200000
    ) as ws:

        async def event(timeout=120):
            nonlocal capture, asr_events
            e = json.loads(await asyncio.wait_for(ws.recv(), timeout))
            if e["type"] == "capture":
                capture = e
                result["capture"] = e
                result["duration_s"] = time.monotonic() - start
                if process:
                    cpu.append(process.cpu_percent())
                    rss.append(process.memory_info().rss)
                    result["cpu_percent_max"] = max(cpu)
                    result["rss_peak_bytes"] = max(rss)
            if e["type"] == "reply_done":
                replies.append(e)
                print("Reply", len(replies), e["first_audio_s"], e["text"], flush=True)
            if e["type"] == "error":
                result["errors"].append(e["message"])
                raise RuntimeError(e["message"])
            if e["type"] == "user":
                asr_events += 1
            return e

        while (await event())["type"] != "ready":
            pass
        result["devices_ready"] = True
        start = time.monotonic()
        print("Native devices ready", flush=True)

        async def ask(text, expected):
            old = len(replies)
            await ws.send(json.dumps({"type": "text", "text": text}))
            deadline = time.monotonic() + 45
            reply = None
            while time.monotonic() < deadline:
                e = await event(50)
                if e["type"] == "reply_done" and len(replies) > old:
                    reply = e
                if (
                    reply
                    and e["type"] == "playback_end"
                    and e["epoch"] == reply["epoch"]
                ):
                    break
            if reply is None:
                raise TimeoutError("No completed neural reply")
            answer = reply["text"].lower()
            for name, num in [
                ("one", "1"),
                ("two", "2"),
                ("three", "3"),
                ("four", "4"),
                ("five", "5"),
                ("six", "6"),
                ("seven", "7"),
                ("eight", "8"),
                ("nine", "9"),
                ("ten", "10"),
            ]:
                answer = answer.replace(name, num)
            passed = all(word in answer for word in expected)
            return {"passed": passed, "reply": reply}

        result["cases"].append(
            {"case": "typed_neural_audio", **await ask("What is two plus two?", ["4"])}
        )
        result["cases"].append(
            {
                "case": "context_seed",
                **await ask("Remember my name is Alex and my meeting is Friday.", [""]),
            }
        )
        for i in range(16):
            r = await ask(f"Say only the number {i + 1}.", [str(i + 1)])
            result["cases"].append({"case": "context_filler_" + str(i + 1), **r})
        result["cases"].append(
            {
                "case": "long_context_recall",
                **await ask(
                    "What is my name and which day is my meeting?", ["alex", "friday"]
                ),
            }
        )
        # Actual output device can be muted; these are callback/render checks.
        await ws.send(
            json.dumps({"type": "text", "text": "Explain rain in six sentences."})
        )
        while (await event())["type"] != "playback_start":
            pass
        before = time.monotonic()
        await ws.send(json.dumps({"type": "interrupt"}))
        while True:
            e = await event()
            if e["type"] == "cancel" and e.get("reason") == "manual":
                break
        result["cases"].append(
            {
                "case": "manual_cancel",
                "passed": True,
                "control_roundtrip_s": time.monotonic() - before,
            }
        )
        result["cases"].append(
            {
                "case": "post_cancel_recovery",
                **await ask("What is three plus three?", ["6"]),
            }
        )
        while time.monotonic() - start < seconds:
            await event(5)
        result["capture"] = capture
        result["duration_s"] = time.monotonic() - start
        await ws.send(json.dumps({"type": "stop"}))
        while True:
            e = await event()
            if e["type"] == "stopped":
                result["microphone_released"] = e["microphone_released"]
                break
    # Ensure bad controls close one session and a later connection can recover.
    await asyncio.sleep(2)
    async with connect(
        f"ws://127.0.0.1:{port + 1}", origin=f"http://127.0.0.1:{port}"
    ) as ws:
        while json.loads(await ws.recv())["type"] != "ready":
            pass
        await ws.send("[]")
        try:
            while True:
                await ws.recv()
        except Exception as e:
            result["invalid_control_closed"] = getattr(e, "code", None) == 1008
    await asyncio.sleep(2)
    async with connect(
        f"ws://127.0.0.1:{port + 1}", origin=f"http://127.0.0.1:{port}"
    ) as ws:
        while json.loads(await ws.recv())["type"] != "ready":
            pass
        result["restart_ready"] = True
        await ws.send(json.dumps({"type": "stop"}))
        while True:
            e = json.loads(await ws.recv())
            if e["type"] == "stopped":
                result["restart_microphone_released"] = e["microphone_released"]
                break
    result["cpu_percent_max"] = max(cpu, default=0)
    result["rss_peak_bytes"] = max(rss, default=0)
    result["capture_during_device_output"] = capture.get(
        "output_frames", 0
    ) > 0 and capture.get("frames", 0) > capture.get("output_frames", 0)
    result["status"] = (
        "passed"
        if all(x["passed"] for x in result["cases"])
        and result.get("microphone_released")
        and result.get("invalid_control_closed")
        and result.get("restart_ready")
        and not result["errors"]
        else "failed"
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--seconds", type=int, default=300)
    args = p.parse_args()
    try:
        report = asyncio.run(run(args.port, args.seconds))
    except Exception as e:
        report = PROGRESS
        report.update({"status": "failed", "error": type(e).__name__ + ": " + str(e)})
    (ROOT / "reports/live-session.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    sys.exit(0 if report["status"] == "passed" else 1)
