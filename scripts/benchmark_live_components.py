"""Comparable local ASR/TTS/model probes. Uses recorded speech, not hardware."""

import json
from pathlib import Path
import sys
import time
import wave
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sts_lab.live_paths import ASSETS


def pcm(path):
    with wave.open(str(path)) as f:
        a = (
            np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(np.float32)
            / 32768
        )
        rate = f.getframerate()
    return np.interp(np.arange(0, len(a), rate / 16000), np.arange(len(a)), a).astype(
        np.float32
    )


def main():
    from moonshine_voice import (
        get_model_for_language,
        ModelArch,
        Transcriber,
        TextToSpeech,
    )

    root = ASSETS / "moonshine"
    results = {"asr": [], "tts": [], "llm": [], "hardware_test": False}
    from live_inference_test import speech

    fixtures = {
        name: speech(name, text)
        for name, text in [
            ("greeting", "Hello, how are you today?"),
            ("arithmetic", "What is two plus three? Give a short answer."),
            (
                "correction",
                "I want to meet on Tuesday. Sorry, I mean Friday. Which day did I choose?",
            ),
        ]
    }
    for arch in [ModelArch.TINY_STREAMING, ModelArch.SMALL_STREAMING]:
        path, arch = get_model_for_language("en", arch, cache_root=root)
        model = Transcriber(path, arch, update_interval=0.16)
        for name in ["greeting", "arithmetic", "correction"]:
            audio = fixtures[name]
            s = model.create_stream(update_interval=0.16)
            s.start()
            before = time.perf_counter()
            for i in range(0, len(audio), 320):
                s.add_audio(audio[i : i + 320].tolist(), 16000)
            transcript = s.stop()
            wall = time.perf_counter() - before
            text = " ".join(l.text for l in transcript.lines)
            row = {
                "model": arch.name,
                "case": name,
                "text": text,
                "wall_s": wall,
                "audio_s": len(audio) / 16000,
                "rtf": wall / (len(audio) / 16000),
            }
            results["asr"].append(row)
            print(row, flush=True)
            s.close()
        model.close()
    for voice in ["piper_en_US-lessac-medium", "kokoro_af_heart"]:
        t = TextToSpeech().language("en-us").voice(voice).models_from(root).load()
        for i in range(3):
            started = time.perf_counter()
            first = None
            samples = 0
            for chunk in t.stream(["Your meeting is on Friday."]):
                if first is None:
                    first = time.perf_counter() - started
                samples += len(chunk.samples)
                rate = chunk.sample_rate
            wall = time.perf_counter() - started
            row = {
                "voice": voice,
                "trial": i,
                "first_audio_s": first,
                "wall_s": wall,
                "audio_s": samples / rate,
                "rtf": wall / (samples / rate),
            }
            results["tts"].append(row)
            print(row, flush=True)
        t.close()
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx

    for name in ["qwen-08b", "qwen-2b"]:
        if not (ASSETS / name / "model.safetensors").exists():
            continue
        m, t = load(str(ASSETS / name))
        mx.set_cache_limit(64 * 1024**2)
        questions = [
            "Hello. How are you?",
            "What is two plus two?",
            "My meeting is Tuesday. Sorry, I mean Friday. Which day is my meeting?",
            "Remember: my name is Maya and I live in Pune. What is my name and where do I live?",
            "Explain why the sky looks blue in one short sentence.",
        ]
        for q in questions:
            p = t.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": "You are a friendly voice assistant. Answer directly in one short sentence. No markdown.",
                    },
                    {"role": "user", "content": q},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            started = time.perf_counter()
            first = None
            text = ""
            for r in stream_generate(
                m, t, prompt=p, max_tokens=100, sampler=make_sampler(temp=0)
            ):
                if first is None:
                    first = time.perf_counter() - started
                text += r.text
            row = {
                "model": name,
                "question": q,
                "text": text,
                "first_token_s": first,
                "wall_s": time.perf_counter() - started,
            }
            results["llm"].append(row)
            print(row, flush=True)
        del m, t
        mx.clear_cache()
    (ROOT / "reports/live-components.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
