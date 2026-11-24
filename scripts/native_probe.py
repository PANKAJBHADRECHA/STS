"""Download a pinned public q4 checkpoint and run a resource-bounded audio smoke test.

Not an end-to-end agent evaluation. No user recording or microphone access.
"""
import argparse
import array
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import wave

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sts_lab.evaluation import atomic_json
from sts_lab.runtime import probe_model_access

CHECKPOINT = "18e4df760a34d5977a34517d7d1580e07acbb2f1"
REPO = "kyutai/moshiko-mlx-q4"
FILES = ("model.q4.safetensors", "tokenizer-e351c8d8-checkpoint125.safetensors", "tokenizer_spm_32k_3.model")


def swap_mb():
    text = subprocess.check_output(["sysctl", "vm.swapusage"], text=True)
    match = re.search(r"used = ([0-9.]+)M", text)
    return float(match[1]) if match else 0


def fixtures(directory):
    import random
    paths = []
    for name, text in (("greeting", "Hello. How are you?"), ("correction", "Tuesday. Sorry, I mean Friday."),
                       ("interruption", "Wait. Please stop. Let me finish."), ("noise", "Can you hear me?"), ("silence", "")):
        path = directory / f"{name}.wav"
        if text:
            source = directory / f"{name}.aiff"
            subprocess.run(["say", "-v", "Samantha", "-r", "180", "-o", str(source), text], check=True)
            subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@24000", str(source), str(path)], check=True)
            with wave.open(str(path), "rb") as stream:
                samples = array.array("h", stream.readframes(stream.getnframes()))
        else:
            samples = array.array("h", [0]*24000*2)
        if name == "noise":
            rng = random.Random(7)
            samples = array.array("h", [max(-32768, min(32767, x+rng.randint(-1500,1500))) for x in samples])
        samples.extend([0]*24000*2)
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(24000);stream.writeframes(samples.tobytes())
        paths.append(str(path))
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    directory = ROOT / ".runtime/native"
    directory.mkdir(parents=True, exist_ok=True)
    output = ROOT / "reports/native-audio.json"
    report = {"mode": "native_audio_smoke_test", "checkpoint": REPO, "checkpoint_revision": CHECKPOINT,
              "runtime": "moshi_mlx==0.3.0", "status": "not_run", "cases": [], "live_audio_qualified": False,
              "agentic_integration_tested": False, "limitations": ["Synthetic OS-generated fixture voices, not human conversations.",
                  "No closed-loop interruption recovery, tool integration, voice-quality judge or wall-clock long-session test."]}
    try:
        if sys.platform != "darwin":
            raise RuntimeError("Apple Silicon macOS required")
        import mlx.core as mx
        report["metal_available"] = mx.metal.is_available()
        if not report["metal_available"]:
            raise RuntimeError("Metal unavailable")
        needed = 15*1024**3 + sum((4_805_545_317, 384_644_900, 552_778))
        if shutil.disk_usage(ROOT).free < needed:
            raise RuntimeError("insufficient disk headroom for download plus 15 GiB reserve")
        from huggingface_hub import hf_hub_download
        models = directory / "models"
        models.mkdir(exist_ok=True)
        assets = []
        for filename in FILES:
            local = models / filename
            if not local.exists():
                if not args.download:
                    raise RuntimeError("pinned local model assets missing; use --download")
                print(f"Fetching pinned {filename}", flush=True)
                local = Path(hf_hub_download(REPO, filename, revision=CHECKPOINT, local_dir=models))
            assets.append(str(local))
        inputs = fixtures(directory)
        job = directory / "job.json"
        worker_output = directory / "worker-report.json"
        worker_output.unlink(missing_ok=True)
        atomic_json(job, {"assets": assets, "inputs": inputs, "output": str(worker_output)})
        start, initial_swap, peak_rss = time.monotonic(), swap_mb(), 0
        with (directory / "worker.log").open("w") as log:
            process = subprocess.Popen([sys.executable, str(ROOT/"scripts/native_worker.py"), str(job)], stdout=log, stderr=log, cwd=ROOT)
            stop = None
            while process.poll() is None:
                try:
                    rss = int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(process.pid)], text=True).strip() or 0)*1024
                except subprocess.CalledProcessError:
                    rss = 0
                peak_rss = max(peak_rss, rss)
                if rss > 6*1024**3:
                    stop = "6 GiB RSS budget exceeded"
                elif swap_mb()-initial_swap > 768:
                    stop = "swap grew by more than 768 MiB"
                elif time.monotonic()-start > 180:
                    stop = "180-second worker budget exceeded"
                if stop:
                    process.terminate()
                    try: process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill();process.wait()
                    break
                time.sleep(.5)
        report.update({"worker_exit_code": process.returncode, "peak_worker_rss_bytes": peak_rss,
                       "worker_wall_seconds": time.monotonic()-start, "swap_growth_mb": swap_mb()-initial_swap})
        if worker_output.exists():
            data=json.loads(worker_output.read_text());report.update(data)
        report["status"] = "resource_stopped" if stop else ("completed" if process.returncode == 0 else "runtime_failed")
        if stop: report["reason"] = stop
        if process.returncode and not stop:
            report["reason"] = "native runtime failed; see .runtime/native/worker.log locally"
    except Exception as error:
        report["status"] = "blocked"
        # No network exception bodies or authenticated URLs enter public reports.
        report["reason"] = str(error) if isinstance(error, RuntimeError) else type(error).__name__
    atomic_json(output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
