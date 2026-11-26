"""Multi-turn text reasoning comparison; explicitly not an audio benchmark."""

import argparse, json, re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sts_lab.live_paths import ASSETS

SYSTEM = "You are Sol, a helpful AI voice assistant. The user is a different person. Address the user as you; never speak as if you were the user. Answer the latest question directly in one or two short sentences, under 45 words. Use plain spoken English, no markdown. Remember what the user tells you, using the latest correction. User facts belong to the user: say your name or your meeting, not my name or my meeting. Do not add filler like I can confirm that. If interrupted, follow the new request. Do not invent facts or pretend to have performed actions. An interrupted assistant message contains only speech rendered before cancellation."
SIMPLE = "You are a helpful voice assistant. Give brief, direct answers. User facts belong to the user. Follow the latest correction."
QUESTIONS = [
    ("arithmetic", "What is two plus two?", r"four|\b4\b"),
    ("remember", "My name is Maya and I live in Pune. Please remember that.", r"."),
    ("recall", "What is my name and where do I live?", r"maya.*pune|pune.*maya"),
    (
        "correction",
        "My meeting is Tuesday. Sorry, I mean Friday. Which day is my meeting?",
        r"friday",
    ),
    (
        "update",
        "Actually my meeting has moved to Monday. What day is it now?",
        r"monday",
    ),
    ("new_question", "What is three plus three?", r"six|\b6\b"),
    (
        "retained_state",
        "What is my name and what day is my meeting?",
        r"maya.*monday|monday.*maya",
    ),
]


def main():
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx

    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["qwen-08b", "qwen-2b", "lfm-text"])
    args = p.parse_args()
    rows = []
    for name in args.models:
        if not (ASSETS / name / "model.safetensors").exists():
            continue
        model, tok = load(str(ASSETS / name))
        mx.set_cache_limit(64 * 1024**2)
        for label, system in [("simple", SIMPLE), ("detailed", SYSTEM)]:
            history = [{"role": "system", "content": system}]
            results = []
            for case, text, pattern in QUESTIONS:
                history.append({"role": "user", "content": text})
                prompt = tok.apply_chat_template(
                    history,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                start = time.perf_counter()
                out = ""
                for r in stream_generate(
                    model,
                    tok,
                    prompt=prompt,
                    max_tokens=90,
                    sampler=make_sampler(temp=0),
                ):
                    out += r.text
                history.append({"role": "assistant", "content": out})
                passed = bool(re.search(pattern, out, re.I))
                if re.search(
                    r"\b(i live|i am in pune|my name is maya|my meeting)\b", out, re.I
                ):
                    passed = False
                result = {
                    "case": case,
                    "text": out,
                    "passed": passed,
                    "wall_s": time.perf_counter() - start,
                }
                results.append(result)
                print(name, label, result, flush=True)
            rows.append(
                {
                    "model": name,
                    "prompt": label,
                    "system_text": system,
                    "passed": sum(x["passed"] for x in results),
                    "total": len(results),
                    "cases": results,
                }
            )
        del model, tok
        mx.clear_cache()
    (ROOT / "reports/live-dialogue-models.json").write_text(
        json.dumps(
            {
                "mode": "text_only_multi_turn_regression",
                "sampling": "greedy",
                "profiles": rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
