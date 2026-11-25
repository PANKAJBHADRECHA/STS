# Native duplex work on the 8 GB M2 Mac

Local validation: 56 unit/API/checkpoint tests passed; the native browser streaming check and three transport rejection/overload checks passed.

Hugging Face setup is complete. A read credential is stored by the local Hugging Face login utility, and the authenticated PersonaPlex weight probe returned HTTP 200. No credential, email address, model weight or microphone recording is committed. See [access evidence](../reports/hugging-face-access.json).

## Selection

The selected general voice application remains LFM2.5-Audio 1.5B, 4-bit, brief-response recipe. It generates actual native speech from input audio, but is turn-based. **No tested candidate has passed sustained real-time full-duplex or conversation-quality qualification on this Mac.**

A separate F-Actor experiment now performs real native duplex inference. Its model is a Llama 3.2 derivative with two Mimi audio streams and a CSM depth decoder. The local MLX port loads the complete public checkpoint without executing remote model code. Packed weights occupy 1,270,727,950 bytes. Peak MLX allocation in the five-case runs was approximately 1.50 GB; that number excludes codec, browser, and operating-system memory.

## What was tested

Each F-Actor recipe processes five 12.8-second recorded-speech timelines: greeting, spoken correction, interruption, noisy speech, and silence. Incoming audio is fed on every frame during output generation. Pipeline runs place the interruption after a greeting at four seconds; they record frames with nontrivial speech energy in both directions. These are genuine neural audio responses to OS-generated voice fixtures, not controller simulations. They are not live human or physical-microphone conversations.

Four recipes completed all five cases: **20 actual native duplex cases**, each 12.8 seconds. The [selection record](../reports/duplex-selection.json) preserves every outcome. The completed comparisons are stored separately:

- [Serial codec and waiting for the user](../reports/factor-duplex.json): mean processing/audio ratio approximately 2.35.
- [Pipelined codec and model-initiated conversation](../reports/factor-duplex-pipeline.json): approximately 1.94, the best of the first three runs.
- [Pipelined codec with fewer depth-decoder synchronization points](../reports/factor-duplex-fused.json): approximately 2.44, rejected as slower in this run.
- [Casual narrative matching the experimental microphone demo](../reports/factor-duplex-casual.json): approximately 2.16. The prompt is more suited to casual dialogue, but grounding and timing remain unqualified.

A ratio of 1.94 means 1.94 seconds of processing for one second of audio. A live stream needs a sustained ratio at or below 1, with acceptable tail latency and speech quality. These short runs are practical regressions with other Mac applications open, not controlled hardware benchmarks. They do not measure long-session reliability or external benchmark scores. Changing both pipeline scheduling and dialogue initiation prevents attributing that comparison to scheduling alone.

[Independent Whisper checks](../reports/factor-audio-transcripts.json) confirm intelligible generated speech but expose unrelated reports, marketing and budget dialogue. They transcribe the single-synchronization recipe's outputs. A [separate ASR report](../reports/factor-casual-audio-transcripts.json) records the final casual-prompt outputs, which still fail to ground the day correction. Prompt accuracy and correction grounding are not established. ASR is used only for evaluation, never in the reply-generation path.

## Actual browser capture and playback

[The browser check](../reports/duplex-browser.json) uses Chromium's WAV-backed microphone and actual MLX inference. It received 13 model audio frames while capture continued, observed 11 capture callbacks while output was scheduled, checked mobile layout, and verified that Stop releases capture and its audio context. This is a short streaming-path check, not proof of sustained real-time conversation or acoustic echo cancellation. The desktop and mobile screenshots are in `reports/duplex-desktop.png` and `reports/duplex-mobile.png`.

[Transport tests](../reports/duplex-transport.json) also exercise foreign-origin rejection, malformed-frame rejection and explicit queue-overload closure.

The first browser attempt stopped from startup overload. The corrected engine warms its codec and decoder before enabling capture. Its input queue holds at most 16 frames (1.28 seconds), with a separate bounded WebSocket receive queue. Overload closes the session explicitly; input is not silently dropped. Sessions are limited to 60 seconds, with one active connection and loopback-only origin checks. The page states the unresolved speed and quality limitations.

## Larger model failures

[Packed Moshi q4](../reports/duplex-packed-smoke.json) avoided the old loader's dense random-weight quantization, but its first frame took about 88 seconds under memory pressure. [Temporal 2-bit Moshi](../reports/duplex-temporal-q2.json) reduced model allocation to about 3.16 GB, but processed two 80 ms frames in about 62 seconds. Both were stopped and rejected. They did not complete the five conversation cases. The loader and conversion integrity tests remain reproducible.

The [Mimi compatibility check](../reports/mimi-compatibility.json) compared 2,097,152 bytes of the semantic codebook with the official Hugging Face Mimi checkpoint and found an exact match. This is a component check, not full encoder equivalence or a quality benchmark.

Haru 4B was investigated from its official runtime and community model cards, but was not downloaded or benchmarked. Its published conversions were substantially larger than the successful F-Actor footprint; no performance claim on this Mac is made. PersonaPlex access is available, but its full-precision weights were not downloaded as a hidden fallback.

## Reproduce

Use the existing `.venv-sts` environment, or install `requirements-duplex.txt` in a Python 3.12 virtual environment. Optional evaluation additionally needs the existing Whisper dependencies.

```bash
.venv-sts/bin/python -m pip install -r requirements-duplex.txt
.venv-sts/bin/python scripts/setup_duplex.py
VECLIB_MAXIMUM_THREADS=1 .venv-sts/bin/python scripts/factor_probe.py
VECLIB_MAXIMUM_THREADS=1 .venv-sts/bin/python scripts/factor_probe.py --pipeline --starts-first --report reports/factor-duplex-pipeline.json
VECLIB_MAXIMUM_THREADS=1 .venv-sts/bin/python scripts/factor_probe.py --pipeline --starts-first --fused-depth --report reports/factor-duplex-fused.json
VECLIB_MAXIMUM_THREADS=1 .venv-sts/bin/python scripts/factor_probe.py --pipeline --starts-first --casual --report reports/factor-duplex-casual.json
.venv-sts/bin/python scripts/judge_duplex_audio.py
.venv/bin/python scripts/duplex_browser_test.py
.venv-sts/bin/python scripts/duplex_transport_test.py
.venv-sts/bin/python -m unittest discover -s tests -v
```

Double-click `Start Duplex Experiment.command` and open http://127.0.0.1:8767 to inspect the experimental microphone mode. The selected turn-based app remains at http://127.0.0.1:8765 via `Start Voice.command`. Use headphones for the duplex experiment.

For the optional older Moshi probe, install `moshi_mlx==0.3.0` with `--no-deps` in an isolated environment containing MLX 0.32.4, NumPy 2.2.6, safetensors 0.5.3, sentencepiece 0.2.0, rustymimi 0.4.1 and sphn 0.2.1. Its declared older MLX dependency conflicts with the newer experimental environment; do not alter the working voice environment to resolve it.

## Sources and licenses

- [F-Actor repository](https://github.com/MaikeZuefle/f-actor), [pinned model](https://huggingface.co/JSALT2026-Conv-AI-Simulator/f-actor-mimi-behavior-sd/tree/7e45bf192ab1c7a3db5e7a26715119f5b6f4f1f1), and [paper](https://arxiv.org/abs/2601.11329).
- [MLX Audio](https://github.com/Blaizzy/mlx-audio) supplies the Llama implementation used by the port.
- [Kyutai Moshi/Mimi](https://github.com/kyutai-labs/moshi) supplies codec runtime and Moshi inference components.
- [Official Haru model](https://huggingface.co/higurashi-ai/Haru-4B), [8-bit conversion](https://huggingface.co/nano16/Haru-4B-mlx-8bit), and [selective quantization](https://huggingface.co/groxaxo/Haru-4B-MLX-oQ5-Selective).

Model licenses are separate from this repository's code license. F-Actor weights are Llama 3.2 derivatives; see its model card and the [Llama 3.2 license](https://www.llama.com/llama3_2/license/). The example speaker vectors are the upstream demonstration voices, not a clone of the user's voice. No fine-tuning or commercial benchmark result is claimed.
