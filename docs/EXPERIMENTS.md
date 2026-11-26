# Experiments and evidence

Use this index to distinguish the application modes from development probes. Reports are historical observations under their recorded conditions, not a general capability guarantee.

## Native turn-based speech

The LFM2.5-Audio application is retained as the primary reproducible native-audio example. Four configurations were compared across six spoken turns each; the 4-bit brief-prompt recipe was retained. Its reported median generation wall time was approximately **8.65 seconds**, with an MLX allocator peak of approximately **2.14 GiB**. These figures describe that small known regression set and exclude some CPU/runtime memory.

- [Native results and methodology](../reports/VOICE_RESULTS.md)
- [Configuration selection](../reports/voice-selection.json)
- [Browser/API verification](../reports/voice-browser.json)
- [Independent output transcription](../reports/lfm-audio-transcripts.json)
- [Earlier native documentation](LEGACY_VOICE.md)

The factual checks used during selection are spot checks, and the prompt was tuned on the development examples. They do not establish general reasoning or live-human speech quality.

## Streaming local conversation

The selected cascade uses Moonshine Medium streaming recognition, Qwen3.5 2B MLX 4-bit and Piper synthesis. Recognition, generation and rendering overlap, with cancellation epochs guarding against stale output. Targeted hardware runs demonstrate interruption; repeated runs still expose recognition/echo errors and overload.

- [Architecture and decisions](LIVE_AUDIT.md)
- [Qualification, failures and metric definitions](LIVE_RESULTS.md)
- [Latest acoustic test](../reports/live-hardware-final.json)
- [Five-minute resumed capture run](../reports/live-hardware-resumed-before-echo-fix.json)
- [UI lifecycle result](../reports/live-ui.json)
- [Unit/API checks](../reports/live-unit-tests.json) and [worklet checks](../reports/live-worklet-tests.json)
- [Component comparisons](../reports/live-components.json) and [text dialogue profiles](../reports/live-dialogue-models.json)

Other `live-*` reports are intermediate development records. Some early arithmetic scorers were permissive; those records are annotated and should not be promoted to final qualification. Timed software playback, muted/headless browser activity, acoustic recapture and a human conversation are different tests.

## Native duplex model research

Moshi/PersonaPlex and F-Actor/Mimi were investigated as audio-native alternatives. The smaller F-Actor port achieved a feasible model footprint but its measured processing ratio remained around **1.94**, and output transcripts exposed off-topic responses. It remains an optional experiment rather than the selected application.

- [Native duplex results](DUPLEX_RESULTS.md)
- [Model/runtime research](research.md)
- [Initial feasibility report](../reports/feasibility.json)

The `Start Duplex Experiment.command` launcher is retained for this track. Separate dependencies and checkpoints are required; see the track's documentation.

## Controller reliability lab

The original five-policy study evaluates interruption/task-state logic using synthetic scenarios. Its 1,500 evaluations measure controller behavior, not neural speech quality, acoustic latency or real-world task completion.

- [Controller outcomes](../reports/RESULTS.md)
- [Evaluation definitions](evaluation.md)
- [Build rationale](build-memo.md)

Reproduce the study without model downloads:

```bash
python3.12 -m sts_lab evaluate --output reports
```

This regenerates reports. Use a separate output directory if you want to preserve the committed snapshot while comparing changes.
