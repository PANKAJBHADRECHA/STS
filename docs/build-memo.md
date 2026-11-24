# STS reliability: build memo

## 1. Differentiation thesis and target numbers

Choose **agentic reliability under interruption**. The gap between selecting an action and executing it correctly is measurable in the published VoiceChat evaluation; see the sourced figures and version caveats in the research dossier. [VoiceChat](https://arxiv.org/html/2609.21967v2)

**Hypothesis, not established fact:** tracking current intent, authoritative execution receipts and realized speech will improve complete-task success without degrading conversational dynamics.

Immediate experiment: at least 60% synthetic held-out task completion and at least 10 percentage points above the shared baseline, with zero duplicate actions or unsupported execution claims. This is an internal engineering criterion, not a τ-Voice score. The exact adjudicator is `python -m sts_lab evaluate`, with its recorded manifest hash.

Research target: exceed 60% pass@1 on a pinned official τ-Voice evaluation with at least a 10-point gain over the same unmodified checkpoint/runtime. The published τ-Voice results are historical baselines, not proof that 60% now beats every commercial model. [τ-Voice](https://arxiv.org/abs/2603.13686)

The critical regression metric is unauthorized/duplicate committed actions: any occurrence disqualifies a controller in this fault suite. In the later audio experiment, a greater than 10-point absolute loss on paired Full-Duplex-Bench interaction success, or greater than 100 ms additional p95 interruption-to-silence delay, falsifies the proposed “preserves duplex interaction” claim.

## 2. Architectural bet and justification

Use a frozen native speech frontend and a small reliability controller before considering model changes. Moshi's official MLX runtime is the public local smoke-test baseline; PersonaPlex is the preferred future persona-controlled frontend once model access and quantized assets are available. Moshi/PersonaPlex do not supply native tool use merely because they produce text. [Moshi MLX](https://github.com/kyutai-labs/moshi/tree/main/moshi_mlx), [PersonaPlex](https://github.com/NVIDIA/personaplex)

The controller receives structured observations and proposals in the immediate experiment. It stores confirmed fields, intent revisions, pending operation IDs and tool receipts. Before execution, it checks the latest revision and grounds each argument. A timeout retries the same operation key; a stale result cannot overwrite a newer intent. The SQLite tool service validates schemas/identity for every policy and records mutations and receipts transactionally.

Actual speech integration needs an exposed native action channel or an explicitly trained delegation protocol. VoiceChat is a better research starting point for that interface than pretending PersonaPlex's inner text is a user transcript. The delegation paper establishes a possible mechanism but uses a backend text model; this must remain an explicitly different experimental architecture. [VoiceChat architecture](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B/blob/main/overview.md), [Delegation paper](https://arxiv.org/abs/2609.19334)

The five implemented policies are baseline, task memory, validated execution, playback recovery and their combination. All use the same scenario observations, tool service and injected faults. The baseline has an eight-observation context, no persistent task ledger, ordinary retries and generation-based communication assumptions. These choices are transparent engineering baselines, not a commercial-model replica.

Playback recovery distinguishes generated/queued/interrupted/rendered confirmations. An acknowledgement is not proof of human comprehension. True neural self-listening requires a model accepting its already-played speech on a dedicated channel, as in LoopSpeech; it is not implemented by looping speaker audio into the user microphone. [Self-listening](https://arxiv.org/abs/2609.05592)

## 3. Training data plan

The immediate prototype needs no weight updates. Generate synthetic appointment tasks with known state, proposed arguments, revisions, tool outcomes and playback events. Keep goals out of the controller. Hold dates/timing seeds and case IDs separate between development and evaluation; acknowledge the shared scenario templates.

For research, construct 10,000 training trajectories, 1,000 development trajectories and at least 1,000 held-out trajectories split by scenario templates and entities before synthesis. Include correction before dispatch, correction after commit, commit-with-lost-receipt, unavailable alternatives, overlapping tool responses, interrupted confirmations and long-range references. Synthesize user audio and annotate exact event/sample positions. Human-check at least 100 held-out conversations before drawing speech-quality conclusions.

Possible speech sources are those used in the cited research, but open weights do not grant training rights to Fisher or vendor-internal data. Audit each source's terms before use. Start from our entirely synthetic tool environment and consented or appropriately licensed speech. No external synthetic-generation API is necessary for the local prototype.

For an action-capable training model, supervise complete structured calls and response incorporation; mask injected tool results from next-token loss. Add hard negative no-call frames and stale/cancelled-intent cases. This follows the distinction between generated actions and authoritative returned context, rather than teaching a model to hallucinate execution. [Function-channel supervision](https://arxiv.org/html/2609.21967v2)

## 4. Evaluation protocol

Development: 30 cases per seed. Held out: 100 cases per seed, with three seeds and all five policies, producing 1,500 evaluations. The five families are argument corrections, tool failures, interruptions, delayed feedback and 30-minute virtual timelines. Freeze and hash the manifest before evaluation; record generator version and complete failures.

Primary metric is final-state correctness AND complete accurate confirmation, with duplicate mutations and unsupported claims disqualifying success. Argument accuracy compares dispatched arguments to current observed intent; policy constraints and final state are checked separately. Report attempts/denominators and recoveries after explicitly injected service failures. No-call avoidance must not count as successful completion of a required task.

Use deterministic assertions for transport and controller correctness, 100 randomized correction-ordering regression trials, recorded audio fixtures for codec/model smoke tests, and browser checks covering every policy/family combination. Keep time-derived seeds and shared-template dependence explicit. The reported Wilson bands are descriptive only.

The future controlled self-listening ablation uses matched checkpoint, data, seeds and codec settings, changing only access to the played-speech channel. Measure anchor error, post-interruption task recovery and latency. The implemented playback-ledger ablation is useful system evidence but cannot answer whether neural self-listening is necessary.

The future 30-minute wall-clock audio evaluation measures persona/instruction violations, factual contradictions, task-state errors, p95 turn/interrupt latency, queue growth, RSS and swap at five-minute intervals. The current virtual-time test measures controller state only; it cannot establish model persona retention, thermal stability or vocal naturalness.

## 5. Minimal viable research experiment: 8–12 weeks

- **Weeks 1–2:** reproduce the five-controller experiment, inspect failures and verify one native audio path on actual hardware. Freeze baseline prompts, runtime versions and task schema.
- **Weeks 3–4:** integrate an action-capable speech checkpoint with real execution feedback. Reproduce its published tool protocol before adding the controller; qualify accuracy on unmodified tasks.
- **Weeks 5–6:** build the disjoint trajectory corpus and a no-training system baseline. If it already closes the targeted gap, keep that simpler result.
- **Weeks 7–8:** only if needed, train rank-8 LoRA on the selected action-capable backbone's attention query/value projections and function output adapter. Freeze perception and speech decoder. Verify actual module names in the pinned implementation before allocating training hardware; this is a proposed intervention, not an implemented training result.
- **Weeks 9–10:** run paired τ-Voice/FDB evaluations, long audio sessions and the played-channel ablation; examine failures blind to variant identity.
- **Weeks 11–12:** replicate with new seeds, document uncertainty and publish reproducible artifacts if the target survives.

The local 8 GB M2 supports controller research and potentially bounded quantized inference, not this training run. A future 80 GB NVIDIA GPU is a conservative starting allocation for profiling adapter training; required memory/throughput must be measured on the chosen architecture and sequence length. This is a planning envelope, not a verified minimum or permission to rent hardware. No GPU provisioning is included.

## 6. Expected failures and early warnings

- **Stale intent:** a call commits after a correction; detect at the first revision mismatch in the execution audit. Never suppress a real post-commit change as though nothing happened.
- **Unknown commit:** timeout causes a second mutation; detect repeated state transitions from one intent. Retry a stable operation key and verify receipts.
- **Wrong arguments:** action selection succeeds but customer/date fields fail; track field accuracy and complete-call exact match separately.
- **Unheard confirmation:** generated text is treated as delivered; detect a missing/full-play acknowledgement after interruption and replay without rerunning the tool.
- **State loss:** facts disappear after eight observations or a long timeline; detect final-state failures versus an unchanged intent and bounded-memory assertions.
- **Delay-induced hallucination:** stale receipts overwrite current state, or success is spoken before a receipt; count every unsupported claim, stratified by 0/300/1,000/3,000 ms injected delay.
- **Repeated observations:** duplicate user events cancel pending work; reproduce this found bug with a regression test and keep revisions stable for unchanged fields.
- **Backchannel over-generation/deafness/topic drift:** not measurable from structured traces; in the audio study require separate playback/input tracks and human-rated recovery. A post-interruption score below 4/5 is an early warning, not a current measured result.
- **Resource pressure:** model loading, codec allocation or growing queues overwhelm the Mac; stop the native worker above 6 GiB RSS, 768 MiB additional swap or its 180-second wall budget. A stopped probe is a failed feasibility gate, never a live-qualified model.

Select the best measured controller, retain all alternatives for reproduction, and report native-audio availability independently. No threshold achieved in a scripted simulator proves commercial-model superiority.
