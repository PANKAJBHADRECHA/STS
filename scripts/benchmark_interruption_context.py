import json, time
from pathlib import Path
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

root = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(root))
from sts_lab.live_paths import ASSETS

profiles = json.loads((root / "reports/live-dialogue-models.json").read_text())[
    "profiles"
]
simple = next(
    p["system_text"]
    for p in profiles
    if p["model"] == "qwen-2b" and p["prompt"] == "simple"
)
detailed = next(
    p["system_text"]
    for p in profiles
    if p["model"] == "qwen-2b" and p["prompt"] == "detailed"
)
model, tok = load(str(ASSETS / "qwen-2b"))
rows = []
for name, system, turns, thinking in [
    (
        "fresh",
        simple,
        [
            {
                "role": "user",
                "content": "What? Stop. Instead, tell me what three plus three is.",
            }
        ],
        False,
    ),
    (
        "consecutive_users",
        simple,
        [
            {"role": "user", "content": "Explain how rain forms in six sentences."},
            {
                "role": "user",
                "content": "What? Stop. Instead, tell me what three plus three is.",
            },
        ],
        False,
    ),
    (
        "cancel_marker",
        simple,
        [
            {"role": "user", "content": "Explain how rain forms in six sentences."},
            {"role": "assistant", "content": "[Response cancelled before completion.]"},
            {
                "role": "user",
                "content": "What? Stop. Instead, tell me what three plus three is.",
            },
        ],
        False,
    ),
    (
        "detailed",
        detailed,
        [
            {"role": "user", "content": "Explain how rain forms in six sentences."},
            {
                "role": "user",
                "content": "What? Stop. Instead, tell me what three plus three is.",
            },
        ],
        False,
    ),
    (
        "thinking",
        simple,
        [
            {"role": "user", "content": "Explain how rain forms in six sentences."},
            {
                "role": "user",
                "content": "What? Stop. Instead, tell me what three plus three is.",
            },
        ],
        True,
    ),
]:
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": system}] + turns,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )
    start = time.perf_counter()
    text = generate(
        model,
        tok,
        prompt=prompt,
        max_tokens=128,
        sampler=make_sampler(temp=0),
        verbose=False,
    )
    row = {"case": name, "text": text, "wall_s": time.perf_counter() - start}
    rows.append(row)
    print(row, flush=True)
(root / "reports/live-interruption-prompts.json").write_text(
    json.dumps(
        {"mode": "actual_text_inference_only", "model": "qwen-2b", "cases": rows},
        indent=2,
    )
    + "\n"
)
