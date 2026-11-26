"""Process CPU/RSS and explicitly system-wide GPU samples during a live test."""

import argparse, json, re, subprocess, time
from pathlib import Path
import psutil

ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--seconds", type=int, default=45)
    p.add_argument(
        "--scope",
        default="native UI smoke plus continuous-capture tail; GPU includes all applications",
    )
    args = p.parse_args()
    process = psutil.Process(args.pid)
    process.cpu_percent()
    samples = []
    start = time.monotonic()
    while time.monotonic() - start < args.seconds:
        time.sleep(1)
        raw = subprocess.check_output(
            ["/usr/sbin/ioreg", "-r", "-c", "AGXAccelerator", "-l"], text=True
        )
        gpu = re.search(r'"Device Utilization %"=(\d+)', raw)
        samples.append(
            {
                "elapsed_s": time.monotonic() - start,
                "process_cpu_percent": process.cpu_percent(),
                "process_rss_bytes": process.memory_info().rss,
                "system_gpu_utilization_percent": int(gpu[1]) if gpu else None,
            }
        )
    report = {
        "scope": args.scope,
        "gpu_is_model_specific": False,
        "cpu_100_percent_means_one_core": True,
        "duration_s": time.monotonic() - start,
        "samples": samples,
        "process_cpu_percent_peak": max(x["process_cpu_percent"] for x in samples),
        "process_rss_peak_bytes": max(x["process_rss_bytes"] for x in samples),
        "system_gpu_utilization_percent_peak": max(
            x["system_gpu_utilization_percent"] or 0 for x in samples
        ),
    }
    (ROOT / "reports/live-resources.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
