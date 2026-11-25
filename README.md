# Local native speech-to-speech on Mac

**Actual microphone audio in, model-generated speech out**, running locally on Apple Silicon. The selected model is **LFM2.5-Audio-1.5B, MLX 4-bit**, with a tested brief-response prompt. Its audio encoder takes the waveform directly; interleaved model audio tokens are decoded into a spoken answer. No separate speech-to-text/chat/text-to-speech pipeline is used for replies.

This version is **turn-based English voice chat**, not PersonaPlex-style simultaneous listening/speaking. Record, stop and send, then listen to the generated reply. Model replies can take several seconds on an 8 GB M2 and may contain factual or pronoun errors.

## Run on this Mac

Double-click **Start Voice.command**, or run:

```bash
./.venv-sts/bin/python -m sts_lab serve
```

Open **http://127.0.0.1:8765**, click **Load voice model**, then **Record voice**. Allow microphone access in the browser when it requests it. Speak for up to 20 seconds, click **Stop and send**, and the model will reply with playable audio. WAV upload is also supported. Start a new conversation after six turns.

The native runtime and pinned 4-bit model are already installed in this checkout. They are ignored by Git; source and reproducible tests are pushed to the private repository.

## Fresh installation

macOS 26 on Apple Silicon and Python 3.12 match the tested environment.

```bash
python3.12 -m venv .venv-sts
.venv-sts/bin/python -m pip install -r requirements-voice.txt
.venv-sts/bin/python scripts/setup_voice.py
.venv-sts/bin/python -m sts_lab serve
```

The model download is approximately 1.8 GB. `setup_voice.py` pins the checkpoint revision and never substitutes a larger model. The full tested environment, including optional audio-evaluation tools, is in `requirements-voice-macos.lock.txt`.

## Actual audio verification

Four real-inference recipes were compared: 4-bit/default, 4-bit/greedy audio, 6-bit/default and 4-bit/brief prompt. Each ran six spoken turns covering five scenarios: greeting, arithmetic, two-turn fact recall, a spoken correction and an upstream recorded question. These are actual native model responses, not controller simulations.

The selected brief-prompt recipe generated non-silent, untruncated speech for all six turns and passed the five disclosed factual spot checks. Its measured median response time and peak MLX memory are in [voice-selection.json](reports/voice-selection.json). The prompt was tuned using these examples, so this is a small regression set, not a blind reasoning benchmark.

[Voice browser verification](reports/voice-browser.json) exercises actual model loading, WAV upload, browser microphone capture, response decoding/playback and mobile layout. Automated capture uses Chromium's recorded-audio microphone device; **the user's physical microphone and a live human conversation were not tested unattended**. The same capture code is used when you click Record voice.

[Independent ASR checks](reports/lfm-audio-transcripts.json) transcribe generated WAVs with Whisper tiny.en to check that output is speech. Whisper is used only for evaluation and is absent from the serving path. It can make transcription mistakes and is not a human quality rating.

```bash
.venv-sts/bin/python scripts/lfm_probe.py --brief
.venv-sts/bin/python scripts/select_voice.py
python3.12 -m unittest discover -s tests -v
```

Browser tests use Playwright in the separate `.venv` environment:

```bash
.venv/bin/python scripts/voice_browser_test.py
.venv/bin/python scripts/browser_test.py
```

See [native voice results](reports/VOICE_RESULTS.md), [voice-engine implementation](sts_lab/voice_engine.py) and [model documentation](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B).

## Hugging Face and PersonaPlex access

The working LFM voice model is public and requires no account. For PersonaPlex, accept its model terms under your Hugging Face account, create a read token, then run the installed interactive login:

```bash
./.venv-sts/bin/hf auth login
```

Keep the token in the local login utility, not chat or repository files. Full steps and official links: [Hugging Face access](docs/HUGGING_FACE.md). Authentication does not remove PersonaPlex's memory requirements.

## Privacy and limitations

The server binds only to 127.0.0.1 and rejects cross-origin requests. Microphone capture starts only after your click. Audio stays on the Mac; server-side input/output files are temporary and removed after a reply. The browser holds playable replies until you reset or close it. No external actions or model fine-tuning occur.

Response generation is capped and context is bounded to six turns. The selected model sometimes phrases user facts in the first person, and accuracy beyond this small test set is not established. Stop playback stops the speaker; it does not implement full-duplex barge-in.

## Experimental native duplex mode

A smaller **F-Actor/Mimi MLX 4-bit** port now supports simultaneous microphone capture and generated audio playback. The actual browser streaming path was tested with recorded speech. It remains experimental: the best initial five-case run processed audio about **1.94× slower than real time**, and independent transcripts exposed off-topic replies. It is not the selected voice app.

Double-click **Start Duplex Experiment.command** and open **http://127.0.0.1:8767** to inspect it. Use headphones. Startup warms the decoder before capture; bounded queues stop a session if inference falls behind. The user's physical microphone and live human conversation remain untested.

See [native duplex results and reproduction](docs/DUPLEX_RESULTS.md). Hugging Face authentication and PersonaPlex access have been completed locally; [the access probe](reports/hugging-face-access.json) returned HTTP 200. Credentials and weights are excluded from Git.

## Earlier controller research

The same repository retains the five-policy reliability experiment under **Controller experiments** and **Scenario replay**. Its 1,500 synthetic evaluations measure task-state logic, not speech-model performance. All five variants remain reproducible; the combined controller won that separate suite. See [controller outcomes](reports/RESULTS.md), [evaluation definitions](docs/evaluation.md), [research dossier](docs/research.md) and [build memo](docs/build-memo.md).

Full controller manifests and traces are compressed JSON artifacts. The failed larger Moshi probe remains documented; it does not describe the smaller working LFM path.

Code is MIT; model licenses are separate. LFM model use is subject to [Liquid's model license](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B/blob/main/LICENSE). The CI configuration remains a template in `ci/github-actions.yml` because the available GitHub token lacks workflow-write scope. Recorded test results are from local execution.
