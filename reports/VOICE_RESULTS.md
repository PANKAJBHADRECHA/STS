# Working local native STS: measured results

The current application performs **actual audio-to-audio inference on this M2 MacBook Air with 8 GB memory**. It uses LFM2.5-Audio-1.5B through MLX Audio 0.6.8. User PCM audio enters the model's audio encoder; the model generates interleaved text/audio tokens and its detokenizer produces speech. There is no separate ASR/chat/TTS response pipeline.

## Best tested recipe

Selected `lfm-brief-upstream.json`: 4-bit weights with the brief-response system prompt in `sts_lab/voice_engine.py`, greedy text and upstream audio sampling (temperature 1.0, top-k 4). Checkpoint revision: `55d4de65fa10fa935561aaaa995271d8b8ff2ee0`.

Four actual-inference recipes ran six spoken turns each (24 turns total), covering five scenarios. The memory scenario uses two turns. The selected recipe:

- Generated valid non-silent, untruncated speech in **6/6 turns**.
- Passed **5/5 disclosed keyword-based factual spot checks**, including “five,” recalled Maya/coffee, and “Friday” after a correction.
- Had median full-response time **8.65 seconds** on this host.
- Reached peak **MLX allocation 2.14 GiB**. This excludes browser/process/OS memory and is not total system RAM usage.

The default 4-bit and 6-bit recipes each passed 3/5 factual checks; greedy-audio decoding also had a truncated response. All measurements, failed cases, exact output text and limitations remain in `lfm-*.json` and `voice-selection.json`. This is the best among these tested configurations, not a model leaderboard claim.

## Actual browser voice path

`voice-browser.json` reports **passed**, with model inference mocked: **False**.

- WAV upload produced: “2 plus 3 equals 5.” in 3.19 seconds.
- Browser microphone capture produced: “I'm doing well, thanks for asking. How can I help you today?” in 9.21 seconds.
- Generated WAV audio decoded in the browser's audio player; desktop and mobile layouts passed without JavaScript console errors.

The automated microphone is Chromium's recorded-audio capture device. It exercises getUserMedia, PCM capture, browser resampling, the real HTTP endpoint, real native inference and playable output. **No unattended physical microphone recording or live human conversation was performed.** The Record voice button uses the same capture path for the user. Screenshots: `voice-desktop.png` and `voice-mobile.png`.

Independent Whisper tiny.en transcribes generated WAVs in `lfm-audio-transcripts.json`. It is only an evaluation tool; it is never used to prepare the model's response. Its transcripts sometimes miss names or negation, so they are evidence of intelligible output, not a human speech-quality certification.

## Limits and honesty

This is turn-based English STS. It does not continuously listen while talking, implement native full-duplex barge-in or execute business tools. Stop playback stops the audio player. The prompt was tuned after observing this tiny known test set; 5/5 is regression coverage, not an unbiased accuracy estimate. User/assistant pronouns can still be confused. Timing comparisons were not isolated from all host load.

The historical failed Moshi resource test and PersonaPlex access gate are still documented; neither describes this smaller working LFM path. PersonaPlex authentication is optional for the current app; see `docs/HUGGING_FACE.md` for local login steps.

## Reproduce

Run the current README setup, `scripts/lfm_probe.py --brief`, `scripts/select_voice.py` and `scripts/voice_browser_test.py`. Unit/API/audio validation is in `tests-voice.txt`. The separate controller/browser regression suite is retained without conflating it with these native voice results.
