# Local duplex voice on Mac

A local voice application for Apple Silicon: **Moonshine Medium streaming English ASR → Qwen3.5 2B MLX 4-bit → Piper neural speech**. Microphone capture and recognition continue during generated speech. Speaking ducks playback; confirmed new speech cancels the old reply and updates the conversation. No cloud inference or paid API is required.

**Experimental: repeated acoustic tests still expose false interruptions and occasional overload. The requested consistently natural phone-call experience is not achieved.** The previous working turn-based app remains available below.

This is a streaming cascade, rather than an audio-native PersonaPlex model. Actual acoustic microphone/speaker interruption has been demonstrated automatically, including the corrected spoken answer. The [qualification results](docs/LIVE_RESULTS.md) distinguish hardware tests, recorded-input tests and remaining failures. No human naturalness rating has been performed, and sub-500 ms end-to-end voice latency is not established.

## Launch on this Mac

Double-click **Start Conversation.command**, or open [Local Conversation](http://127.0.0.1:8765) when the server is running. Click **Start conversation**, then speak in English. The app shows the selected microphone/output and listening/speaking state. **Interrupt** clears queued speech; **End** releases both devices. **New conversation** clears context and pending recognition. Spoken “stop,” “wait” and “cancel” also stop speech without starting another generated answer.

The default audio path uses the browser’s real microphone with WebRTC echo cancellation, noise suppression and gain control. Generated speech plays through a local AudioWorklet. Model inference remains entirely on this Mac. The browser uses the current default devices and the app preserves their volume settings. Unmute the selected output to hear replies; startup reports mute state. Restart the conversation after changing devices. Allow microphone access in the browser when requested. macOS may also request permission. Speak when the status changes to Listening. Chrome/Chromium is the tested browser; use headphones if another browser lacks reliable echo cancellation.

## Installation

Tested on **MacBook Air M2, 8 GB unified memory, macOS 26, Python 3.12**. Allow 5 GiB free disk. Setup also installs SpeexDSP through Homebrew for the optional native adapter. Setup needs internet; conversations use local files only.

```bash
./"Setup Conversation.command"
./"Start Conversation.command"
```

The dedicated environment, models, installed app and Python caches live under `~/Library/Application Support/STS`, outside cloud-managed Documents. The source repository stays here. Setup creates `.venv-live` as a convenience symlink on a fresh installation. Set `STS_LIVE_HOME` to override the installation directory.

[Setup](scripts/setup_conversation.py) pins Qwen's exact revision and checks the speech asset revision. Dependencies are in [requirements-live.txt](requirements-live.txt); the [tested lockfile](requirements-live-macos.lock.txt) includes optional test tools. The selected public models require no Hugging Face login. Tokens and weights are excluded from Git.

To develop directly from the checkout:

```bash
.venv-live/bin/python scripts/live_server.py --port 8765
```

Add `--audio native` to use the earlier CoreAudio adapter, preferably with headphones. The browser path is selected for speaker echo handling.

Only loopback ports 8765/8766 are used. One session is permitted per server. Avoid running another local model benchmark during a conversation on this 8 GB Mac: competing model loads can starve audio processing. Battery/Low Power Mode can increase latency; the final hardware report records its power condition.

## Implementation

A browser AudioWorklet continuously transfers 20 ms microphone frames to an independent ASR worker. WebRTC handles acoustic echo. A lightweight local energy gate ducks playback promptly; recognized new speech invalidates the reply epoch. The optional native adapter uses a sample-synchronous output reference and SpeexDSP. Late audio, completion events and acknowledgements cannot revive cancelled speech. Local streaming synthesis overlaps MLX token generation.

Context contains user messages and completed rendered assistant phrases. An explicit, non-spoken cancellation marker separates turns when no phrase finished, preventing the small model from merging abandoned and replacement requests. Missing question punctuation is restored without changing recognized words. Older context is summarized locally; recent turns remain verbatim. Input, output and context are bounded. Each reply uses MLX-LM's normal generation cache; manual cross-turn cache reuse is disabled.

[Architecture and alternatives](docs/LIVE_AUDIT.md) cover Moshi/PersonaPlex, F-Actor, Dessa, compact language models and streaming speech components. [Measured results](docs/LIVE_RESULTS.md) explain profile selection, correctness checks, latency definitions and limits. Test answers are generated by the models; no canned conversation replies or calculator substitute is used in serving.

## Reproduce tests

Install optional tools first:

```bash
.venv-live/bin/python -m pip install -r requirements-live-test.txt
.venv-live/bin/python -m playwright install chromium
.venv-live/bin/python -m unittest discover -s tests -v
```

With a server running, the acoustic suite plays OS-generated test questions through the built-in speakers and recaptures them through the real microphone. The explicit fixture flag temporarily enables **only the fixture speaker**, then restores its exact mute/volume state. It does not change the default output selection.

```bash
.venv-live/bin/python scripts/live_hardware_browser_test.py --port 8765 --enable-fixture-speaker --extended --hold-seconds 300
.venv-live/bin/python scripts/live_ui_test.py --port 8765
```

Separate checks:

```bash
# Neural inference with recorded input and software rendering, not hardware:
.venv-live/bin/python scripts/live_inference_test.py
# The earlier native-only typed-context probe requires a native-mode server:
# .venv-live/bin/python scripts/live_server.py --audio native --port 8770
.venv-live/bin/python scripts/live_session_test.py --port 8770 --seconds 300
# Reproduce the interrupted-turn context investigation:
.venv-live/bin/python scripts/benchmark_interruption_context.py
```

Run these sequentially. Acoustic tests use real devices and air transmission, with an automated speech source; they are not live-human conversations. Arithmetic checks inspect the answer, rather than accepting a matching operand. The scripts retain individual failures rather than silently skipping them.

## Limits and privacy

ASR errors, echoes, long pauses, accents and factual mistakes remain possible. The optional native mode can mistake exact short repetitions for echo; speaker echo handling is stronger in the browser mode. Recognition segments are capped at 25 seconds and generated replies at 96 tokens; long speech can be segmented or truncated. This is not a hard-real-time or universally error-free system.

Normal serving saves neither microphone recordings nor chat transcripts. Conversation state stays in process/page memory until reset or close. The optional `--diagnostics` mode saves a bounded private test trace and error details in Application Support; it is off in the normal launcher. Test fixtures, diagnostics, environments, credentials and weights are outside Git. Device playback acknowledgements establish delivery, not that a person heard or understood every word.

## Preserved earlier work

**Start Voice.command** opens the previous LFM2.5-Audio turn-based app on port **8769**. Its separate setup and historical measurements remain in [legacy voice instructions](docs/LEGACY_VOICE.md). [F-Actor results](docs/DUPLEX_RESULTS.md) document its slower-than-real-time and off-topic behavior; it is not the selected app. The [controller lab](reports/RESULTS.md) remains preserved; its synthetic scores are not voice-performance evidence.

Code is MIT; model/library licenses are separate. See [Qwen](https://huggingface.co/mlx-community/Qwen3.5-2B-4bit), [Moonshine](https://github.com/moonshine-ai/moonshine) and [SpeexDSP](https://www.speex.org/). Existing Hugging Face/PersonaPlex access remains configured locally, but PersonaPlex is not the selected model. The existing CI template is in `ci/github-actions.yml`; reported checks were executed locally.
