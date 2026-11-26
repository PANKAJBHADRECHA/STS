# Local voice implementation and qualification

Final delivery review: 27 November 2026. Development reports retain their original run data.

The delivered application uses **Moonshine Medium streaming English recognition, Qwen3.5 2B MLX 4-bit, and Piper neural speech**. The default browser audio path uses a real microphone with WebRTC echo cancellation, noise suppression and automatic gain control. An AudioWorklet captures audio continuously while rendering generated speech. All model inference runs on the Mac. An optional native CoreAudio/SpeexDSP path remains available.

**Physical simultaneous capture/playback and spoken interruption have been demonstrated, but repeatability and extended-session reliability still have failures.** This is an experimental local duplex application. It is not qualified as a universally natural, sub-500 ms phone-call replacement. No human conversation or naturalness rating was performed.

## What was actually tested

The hardware is an M2 MacBook Air with 8 GB unified memory. Automated acoustic tests generate speech with macOS `say`, play it through the built-in speakers, and recapture it through the real microphone. Headed Chromium uses actual audio devices; no fake microphone or fake audio input is supplied. Track settings, device names, power source, capture/output callback counts and every case result are in [the latest hardware report](../reports/live-hardware-final.json).

The fixture checks the built-in speaker's individual mute state, independent of the default USB output. Its explicit audible-test flag temporarily sets only the fixture speaker to 55%, then restores its original mute and volume. Default device selection is preserved. Some early silent-fixture failures were invalid recognition comparisons, and later default-device changes also altered acoustic conditions.

The stricter final answer checks reject a matching operand in a wrong calculation: `6 + 6 = 12` does not pass the request for three plus three. Markdown around a number is stripped before scoring. Early reports with permissive checks are explicitly development records; they must not be used as final qualification.

## Executed scenarios and limits

- Normal arithmetic, user memory/recall, a spoken correction, a changed question, fast and slow speech, low stationary background noise, three successive exchanges and spoken interruption are implemented in the acoustic suite. The suite records individual failures and continues where possible. See the latest JSON for exact passes, failures and duration.
- During a passing targeted interruption, real audio input continued while speech rendered. New speech ducked output; confirmed recognized words cancelled its epoch; the model generated the updated answer of six. A strict targeted native run measured ducking at about 308 ms and confirmed cancellation at 1.64 seconds. Browser runs demonstrated the same behavior with different timing. These are automated acoustic measurements, not human perception ratings.
- One earlier browser run passed all 12 acoustic answer checks but failed during the subsequent five-minute hold after an input queue overflow. Later runs exposed echo-related false interruptions and playback timeouts. These failures remain relevant; a list of correct answers does not establish stable continuous conversation.
- An earlier **native-device typed-input** session passed 300.33 seconds, 20 generated replies, retained Alex/Friday after 16 intervening turns/local summarization, manual cancellation recovery, invalid protocol closure, restart and microphone release. It reported one CoreAudio status event. This is a separate mode and input source, not proof of five minutes of spoken browser conversation. [Native lifecycle data](../reports/live-session.json).
- The UI checks desktop/mobile layout, real-device rendering, reset, stop, microphone release and immediate restart. [Latest UI data](../reports/live-ui.json). Its mode/configuration determines which audio path it covers.
- Unit/API tests cover state cancellation, stale acknowledgements, cancelled-turn context, reset flushing, echo matching, question punctuation, PCM validation, queue bounds, real DSP attenuation, native render ordering and strict scoring. The standalone JavaScript DSP test covers worklet capture/resampling, ordering, stale output, queue bounds and local ducking. [Python result](../reports/live-unit-tests.json), [worklet result](../reports/live-worklet-tests.json).

Long natural pauses, arbitrary room noise/music, different accents, live human overlap and device unplug/replug remain incompletely qualified. Raw captured/ambient text is not published. Missing or failed results are never treated as passes.

## Selection and optimization evidence

Six text-only dialogue profiles compared Qwen 0.8B/2B and LFM 1.2B with simple/detailed prompts. Qwen 2B passed seven fact checks with both prompts; the smaller Qwen confused user ownership and failed repeatedly with the longer prompt. LFM later denied user facts in actual streaming inference, so it was not retained. [Dialogue comparisons](../reports/live-dialogue-models.json), [LFM streaming failure](../reports/live-inference-lfm.json).

Tiny and Small ASR decoded three clean recorded clips correctly; isolated processing ratios were approximately 0.059–0.071 and 0.110–0.150 respectively. Piper's isolated first audio was about 36–41 ms and RTF 0.040–0.044; Kokoro was about 193–202 ms and RTF 0.273–0.287. These component results exclude input pacing, acoustic difficulty, recognition endpointing and model-response latency. Medium recognition was selected after it decoded a noisy acoustic case the smaller recognizers missed. [Component data](../reports/live-components.json), [Medium noise probe](../reports/live-hardware-medium-noise.json).

Manual cross-turn model-cache reuse was removed. The language model uses its normal per-response generation cache. A targeted inference experiment found that two consecutive user turns after unspoken cancellation could produce a wrong answer; an explicit, non-spoken cancellation marker preserves the turn boundary. Another experiment found omitted question punctuation changed an arithmetic answer; recognized questions now receive their missing question mark without changing words. [Interruption context probe](../reports/live-interruption-prompts.json), [punctuation probe](../reports/live-tokenization-probe.json).

The browser path avoids native speaker self-echo limitations. Cancellation no longer waits on synthesis from the ASR thread. Bounded transport backpressure waits briefly for capture-queue capacity rather than terminating on a transient incoming burst; persistent overload still ends the session visibly. Browser gain is left to WebRTC AGC rather than adding a second gain boost. Recent generated-text echo rejection remains active while browser output plays. These improve resource handling without replacing model answers with fixtures.

## Latency, resources and interpretation

Read per-case `speech_end_to_device_start_s` for the measured acoustic end-of-fixture to playback-start delay. `first_audio_s` starts later, at submission of the recognized text, and excludes recognition/endpointing. Callback notification timestamps approximate device rendering; they are not sound-pressure measurements at a listener's ear. Sub-500 ms end-to-end speech latency was not established. Individual short generated replies can have a wall/audio ratio above 1.0 even when synthesis itself is faster than real time.

MLX allocator peaks around 1.2 GiB exclude CPU/ONNX models, browser memory and other applications. CPU/RSS probes and system-wide GPU readings are available in [resource data](../reports/live-resources.json); their recorded scope matters. GPU utilization includes every application and cannot be attributed exclusively to inference. CPU uses one core = 100%.

Early comparisons ran on battery with Low Power Mode enabled; later hardware work used AC and device conditions changed. No power policy or unrelated process was changed. Performance differences cannot be attributed solely to one model or code edit.

## Delivery

Setup installs the runtime, models, Python caches and serving code under `~/Library/Application Support/STS`, avoiding observed cloud-file offloading stalls in Documents. The repository is the source of truth; rerun setup after editing serving files. The default launcher opens the local browser interface. Normal serving stores no microphone recording or transcript. Optional private diagnostics are bounded and off by default.

The earlier LFM native turn-based app and F-Actor experiment remain preserved with their historical reports. The selected model requires neither paid cloud APIs nor PersonaPlex credentials. Code and separate model licenses are documented in README.

## Final review fixes

The last review found that browser capture used WebRTC automatic gain control plus a second ASR gain boost. Browser-mode ASR now uses unity additional gain. Recent-output echo matching is also explicitly enabled for browser playback, where there is no native reference channel to update its activity timer. Native-mode reference processing is unchanged. Browser input can now await bounded queue capacity briefly when callbacks arrive in a burst; the client also rejects more than two seconds of buffered socket audio. These changes do not imply that earlier failed runs passed retroactively.

## Resumed validation on 27 November

The browser input transport change completed a **300.05-second actual-microphone hold** and released capture cleanly, but multiple acoustic reply/playback checks in that run failed. This is a passing capture-lifecycle result and an overall failed voice run. The report is preserved in [resumed stability run](../reports/live-hardware-resumed-before-echo-fix.json). The browser echo-activity and gain correction was applied afterward; the latest five-case report records that build's outcome separately. No human qualification is inferred.

The last resumed acoustic retry hit input overload on the first case despite the passing unit/DSP checks. The best local duplex candidate is therefore delivered **as experimental**, not a fully qualified real-time conversation product. The five-minute capture lifecycle passed in the preceding run, but its overall voice test failed. The preserved turn-based application remains the dependable fallback. More unattended retries do not erase these limits.

The final browser UI regression passed two typed neural replies, real capture/output, desktop/mobile layout, reset, microphone release and immediate restart. Effective WebRTC settings were echo cancellation, noise suppression and automatic gain control enabled, with 48 kHz mono capture. This validates the launch and lifecycle, not the failed natural-voice qualification. The final Python suite passed 79 tests; six standalone worklet checks and dependency validation also passed.
