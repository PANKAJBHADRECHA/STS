# Installation and usage

Both applications run locally on Apple Silicon. They use different models and environments; install the mode you want to try first. Reference measurements were made on an M2 MacBook Air with 8 GB unified memory, macOS 26 and Python 3.12.

## Prerequisites

- Apple Silicon Mac and Python 3.12 available as `python3.12`.
- Internet access during dependency/model setup.
- Microphone permission for the browser. Chrome/Chromium is the tested streaming browser.
- Homebrew for the streaming setup's SpeexDSP installation.
- At least 17 GiB free disk for the native setup's headroom check, or 5 GiB for a fresh streaming installation.

Clone the project and enter it:

```bash
git clone https://github.com/PANKAJBHADRECHA/STS.git
cd STS
```

## Native voice chat

This application uses LFM2.5-Audio-1.5B, MLX 4-bit. It is turn-based: record, submit, then listen.

```bash
python3.12 -m venv .venv-sts
.venv-sts/bin/python -m pip install -r requirements-voice.txt
.venv-sts/bin/python scripts/setup_voice.py
./"Start Voice.command"
```

Open **http://127.0.0.1:8769**. Click **Load voice model**, allow microphone access, and record a short English question. Click **Stop and send** to generate a reply. WAV upload is also supported. Use a new conversation after six turns.

The selected checkpoint download is approximately 1.8 GB. It is pinned by revision in `scripts/setup_voice.py`. The environment is `.venv-sts`; weights and temporary runtime assets live under `.runtime`. Neither is committed. The public selected checkpoint requires no Hugging Face account.

Manual launch:

```bash
.venv-sts/bin/python -m sts_lab serve --port 8769
```

Model responses can take several seconds on the reference Mac and can contain factual/pronoun errors. This mode does not implement listening during playback or voice barge-in. [Recorded native results](../reports/VOICE_RESULTS.md).

## Streaming conversation

This experimental mode combines Moonshine Medium streaming recognition, Qwen3.5 2B MLX 4-bit and Piper speech generation.

```bash
./"Setup Conversation.command"
./"Start Conversation.command"
```

Open **http://127.0.0.1:8765**, click **Start conversation**, and grant microphone access. The interface displays the active input/output devices. Speak when it shows Listening. **Interrupt** stops a reply, **New conversation** clears context, and **End** releases the microphone. Spoken stop/wait/cancel controls are supported.

The default browser path uses WebRTC echo cancellation, noise suppression and gain control. Audio processing runs in an AudioWorklet while all model inference runs in the local Python process. Typed input is available for isolating recognition errors.

Setup stores the environment, models, installed serving code and Python caches under:

```text
~/Library/Application Support/STS/
```

This avoids observed cloud-offloaded files inside Documents. Set `STS_LIVE_HOME` before setup and launch to choose another local directory. On a fresh installation, setup creates `.venv-live` as a convenience symlink. Rerun setup after editing serving files to update the installed app copy.

To run the current checkout directly:

```bash
.venv-live/bin/python scripts/live_server.py --port 8765
```

Optional native CoreAudio/SpeexDSP path:

```bash
.venv-live/bin/python scripts/live_server.py --audio native --port 8770
```

Open **http://127.0.0.1:8770** for that server. Headphones can help with echo. This adapter is retained for comparison; its built-in-speaker tests exposed self-echo problems.

The streaming app is a research preview. Physical interruption has been demonstrated, but repeated acoustic runs still fail and overload recovery can end a session. It is not a production voice assistant. [Qualification and limitations](LIVE_RESULTS.md).

## Troubleshooting

**No audio:** check the selected output device and its mute setting. Device mute is independent per output; a USB device and built-in speakers may have different states. Restart the conversation after changing devices.

**Microphone denied:** grant access in browser settings and macOS Privacy & Security → Microphone, then reload the page. The native adapter may require permission for the launching terminal/process as well.

**Model/assets missing:** rerun the setup for the mode you are using. Models are downloaded separately and excluded from the repository. Do not substitute a larger model automatically on an 8 GB system.

**Audio processing falls behind:** end the session, stop other model experiments, and start again. Persistent overload is a known streaming limitation; use the turn-based app as a fallback.

**Port already occupied:** close the previous server terminal, or choose another `--port`. Streaming uses the specified HTTP port and the next port for WebSocket audio/control.

**Imports stall in a cloud-managed checkout:** use the streaming installed launcher and local Application Support storage. Native-mode setup still uses checkout-local paths; a directory excluded from cloud offloading is preferable for that environment and its weights.

## Test audio and diagnostics

Normal sessions do not save audio/transcripts. Streaming `--diagnostics` is an explicit debugging option and writes private traces locally. Test fixtures and captured diagnostics must remain outside Git.

The acoustic test's `--enable-fixture-speaker` flag makes test speech audible through the built-in speakers, then restores that speaker's original mute/volume. Run it in a suitable environment. Test commands and their scope are in [CONTRIBUTING.md](../CONTRIBUTING.md).
