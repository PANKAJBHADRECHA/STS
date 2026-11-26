# Contributing

Contributions should improve a measured behavior without obscuring the difference between native audio inference, a streaming cascade and controller simulation. Include the trigger, the resulting behavior, and the validation relevant to your change.

## Development setup

Use Python 3.12. The controller and HTTP/API tests do not need downloaded model weights:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m unittest discover -s tests -v
```

Optional MLX/audio tests skip when their runtime is unavailable. For the recorded full Apple Silicon test environment, install the streaming runtime and optional test tools:

```bash
./"Setup Conversation.command"
.venv-live/bin/python -m pip install -r requirements-live-test.txt
.venv-live/bin/python -m playwright install chromium
.venv-live/bin/python -m unittest discover -s tests -v
node scripts/test_live_worklet.cjs
```

Node.js is needed only for the standalone AudioWorklet checks. A model download is not needed for those checks.

## Test the application you change

For native turn-based work, follow [native setup](docs/GETTING_STARTED.md#native-voice-chat), then run the corresponding inference/browser checks:

```bash
.venv-sts/bin/python scripts/lfm_probe.py --brief
.venv-sts/bin/python scripts/select_voice.py
.venv/bin/python scripts/voice_browser_test.py
```

Install Chromium with `.venv/bin/python -m playwright install chromium` before browser tests. The browser suite uses recorded microphone input; it is not a physical microphone qualification.

For streaming work, start one local server and run tests sequentially:

```bash
.venv-live/bin/python scripts/live_server.py --port 8765
```

In another terminal:

```bash
.venv-live/bin/python scripts/live_ui_test.py --port 8765
.venv-live/bin/python scripts/live_hardware_browser_test.py \
  --port 8765 --enable-fixture-speaker --extended --hold-seconds 300
```

The acoustic test opens headed Chromium, uses actual microphone/speaker hardware and plays audible synthetic test speech. Its fixture flag temporarily enables the built-in speaker and restores its original mute/volume afterward. It requires the built-in devices used by the reference Mac; adapt device selection explicitly for different hardware. It is not a live-human evaluation.

Run inference comparisons independently of hardware stability tests. Competing local-model processes can starve audio on an 8 GB Mac.

## Evidence to include

- Model identifier and revision, runtime, hardware and power condition.
- Whether input is synthetic, recorded, acoustically recaptured or live human speech.
- The timing boundary: component time, recognized-text-to-audio, or acoustic end-to-end.
- All relevant failed cases, not just successful examples.
- The exact command used and any conditions required to reproduce the result.

Preserve raw test definitions when comparing configurations. Report changes to scoring rules; a revised scorer does not make an earlier run pass retroactively. Never use a mock, transcript keyword match or controller score as proof of speech-model quality.

## Scope and conventions

Keep the native turn-based app available when working on the streaming experiment. Keep audio queues and model context bounded, reject stale output after cancellation, and release capture on stop/disconnect. Changes to the serving code must also be installed with setup before testing the installed launcher.

Do not commit environments, model weights, tokens, captured microphone audio, private transcripts, physical-microphone UI screenshots or runtime databases. Use synthetic fixtures and the ignored local runtime directories. Review model licenses separately from the MIT code license.

The CI file in `ci/github-actions.yml` is a template, not an active workflow. Do not describe local validation as a hosted CI pass.
