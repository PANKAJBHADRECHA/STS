"""Actual-device UI smoke, with typed input; no fake audio devices."""

import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(port):
    from playwright.sync_api import sync_playwright

    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False, ignore_default_args=["--mute-audio"]
        )
        page = browser.new_page(
            permissions=["microphone"],
            bypass_csp=True,
            viewport={"width": 1280, "height": 980},
        )
        page.on("pageerror", lambda e: errors.append(type(e).__name__))
        response = page.goto(f"http://127.0.0.1:{port}")
        assert response.status == 200
        assert "default-src 'self'" in response.headers["content-security-policy"]
        assert page.locator("#listening").inner_text() == "● Microphone off"
        page.locator("#start").click()
        page.wait_for_function(
            "!document.getElementById('text').disabled", timeout=120000
        )
        page.locator("#text").fill("What is two plus two?")
        page.locator("#send").click()
        page.wait_for_function("liveMetrics.replies.length>=1", timeout=45000)
        reply = page.evaluate("liveMetrics.replies[0]")
        assert "4" in reply["text"] or "four" in reply["text"].lower()
        page.wait_for_function("liveMetrics.outputFrames>0 && !playing", timeout=30000)
        page.locator("#text").fill("Explain why the sky is blue in one short sentence.")
        page.locator("#send").click()
        page.wait_for_function(
            "liveMetrics.replies.length>=2 && !playing", timeout=45000
        )
        page.screenshot(path=str(ROOT / "reports/live-desktop.png"), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth<=innerWidth")
        page.screenshot(path=str(ROOT / "reports/live-mobile.png"), full_page=True)
        page.locator("#reset").click()
        page.wait_for_function("!document.querySelector('.message')")
        metrics = page.evaluate("liveMetrics")
        page.locator("#stop").click()
        page.wait_for_function(
            "ws===null && liveMetrics.microphoneReleased===true", timeout=30000
        )
        assert page.locator("#start").is_enabled()
        page.locator("#start").click()
        page.wait_for_function(
            "!document.getElementById('text').disabled", timeout=120000
        )
        page.locator("#stop").click()
        page.wait_for_function(
            "ws===null && liveMetrics.microphoneReleased===true", timeout=30000
        )
        browser.close()
    return {
        "status": "passed" if not errors else "failed",
        "mode": "real_audio_ui_typed_prompt",
        "human_participant": False,
        "fake_capture": False,
        "capture_frames": metrics["captureFrames"],
        "output_frames": metrics["outputFrames"],
        "reply": reply,
        "microphone_released": True,
        "immediate_restart_passed": True,
        "devices": metrics.get("devices"),
        "configuration": metrics.get("configuration"),
        "capture_settings": metrics.get("captureSettings"),
        "device_status_events": metrics.get("deviceStatusEvents"),
        "synthesis_starvation_frames": metrics.get("synthesisStarvationFrames"),
        "console_errors": errors,
        "mobile_overflow": False,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    try:
        report = run(args.port)
    except Exception as e:
        import traceback

        traceback.print_exc()
        report = {
            "status": "failed",
            "error_type": type(e).__name__,
            "mode": "real_audio_ui_typed_prompt",
        }
    (ROOT / "reports/live-ui.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    sys.exit(0 if report["status"] == "passed" else 1)
