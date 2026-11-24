from __future__ import annotations

import hashlib
import gzip
import json
import math
import statistics
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .core import POLICIES, evaluate
from .scenarios import FAMILIES, GENERATOR_VERSION, generate


def wilson(passes, n):
    if not n:
        return [0, 0]
    z = 1.96
    p = passes / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    margin = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1+z*z/n)
    return [round(max(0, center-margin), 4), round(min(1, center+margin), 4)]


def run_suite(split="held_out", seeds=(0, 1, 2), output=None):
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and unique")
    started = time.perf_counter()
    cases = [case for seed in seeds for case in generate(split, seed)]
    manifest = [asdict(case) for case in cases]
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if output:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        atomic_gzip_json(output / f"{split}-manifest.json.gz", manifest)
    rows, traces, summaries = [], [], []
    for policy in POLICIES.values():
        timings = []
        for case in cases:
            start = time.perf_counter()
            result = evaluate(case, policy)
            timings.append((time.perf_counter()-start)*1000)
            trace = result.pop("trace")
            if not result["passed"]:
                traces.append({"scenario": case.id, "variant": policy.name, "trace": trace})
            rows.append(result)
        subset = [r for r in rows if r["variant"] == policy.name]
        passes = sum(r["passed"] for r in subset)
        attempts = sum(r["argument_attempts"] for r in subset)
        summaries.append({
            "variant": policy.name, "passed": passes, "total": len(cases),
            "completion_rate": passes/len(cases), "wilson_95_descriptive": wilson(passes, len(cases)),
            "argument_accuracy": sum(r["arguments_correct"] for r in subset)/attempts if attempts else None,
            "duplicate_actions": sum(r["duplicate_actions"] for r in subset),
            "unsupported_claims": sum(r["unsupported_claims"] for r in subset),
            "recovery_tasks": sum(r["passed"] and r["recoveries"] > 0 for r in subset),
            "injected_failure_tasks": sum(case.tool_fault != "none" or bool(case.blocked_slots) for case in cases),
            "controller_runtime_p50_ms": statistics.median(timings),
            "controller_runtime_p95_ms": sorted(timings)[math.ceil(.95*len(timings))-1],
            "by_family": {f: {"passed": sum(r["passed"] for r in subset if r["family"] == f),
                              "total": sum(r["family"] == f for r in subset)} for f in FAMILIES},
        })
    eligible = [s for s in summaries if s["duplicate_actions"] == 0 and s["unsupported_claims"] == 0]
    ranked = sorted(eligible, key=lambda s: (-s["completion_rate"], -s["recovery_tasks"], s["controller_runtime_p95_ms"]))
    best = ranked[0] if ranked else None
    baseline = summaries[0]["completion_rate"]
    gain = best["completion_rate"]-baseline if best else 0
    result = {
        "schema_version": 1, "generator_version": GENERATOR_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "controller_simulation", "split": split, "seeds": list(seeds),
        "scenario_manifest_sha256": digest, "scenario_count": len(cases), "run_count": len(rows),
        "wall_seconds": time.perf_counter()-started,
        "selection": {"winner": best["variant"] if best else None, "gain_over_baseline": gain,
                      "target_met": bool(best and best["completion_rate"] >= .6 and gain >= .1),
                      "rule": "zero duplicate actions and unsupported claims; then completion, recovery, controller p95",
                      "live_audio_qualified": False},
        "limitations": [
            "Synthetic structured events; no speech model, ASR, TTS or real user evaluated.",
            "Thirty-minute virtual timelines are not wall-clock audio soak tests.",
            "Seed variants share templates; Wilson intervals are descriptive, not independent-population confidence.",
            "Handwritten baseline and fault mix limit generalization; these are not official tau-Voice scores.",
            "Controller timing excludes speech inference, audio buffering and network latency.",
        ], "summaries": summaries, "rows": rows,
    }
    if output:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / f"{split}.json", result)
        atomic_gzip_json(output / f"{split}-failures.json.gz", traces)
        if split == "held_out":
            atomic_json(output / "selection.json", result["selection"] | {"manifest_sha256": digest})
    return result


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def atomic_gzip_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    # mtime=0 keeps compressed artifacts reproducible across repeated runs.
    with tmp.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as stream:
            stream.write(json.dumps(value, separators=(",", ":")).encode())
    tmp.replace(path)
