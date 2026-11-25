# Evaluation definitions and limits

The report is an engineering controller simulation. It contains 100 held-out tasks × three seeds × five policies = 1,500 evaluations. Development is a separate 30-task split, also run with three seeds. Both share disclosed templates; held-out here means disjoint task instances, not unseen task families or human conversations.

## Task success

A case passes only if the final appointment slot/status matches the evaluator's goal, the complete correct result has been acknowledged as played when communication is required, there are no duplicate mutations, and there are no unsupported success claims. The controller never receives `expected_slot` or `expected_status`. Its observations are user events, draft actions, execution receipts and playback events.

The appointment backend is an independent SQLite database with transactional receipts and a mutation audit. Every variant uses the same schema/identity checks. Guarded policies additionally validate arguments against observations and use stable intent keys for retries. Duplicate actions are repeated committed slot/status transitions within a case; legitimate changes to different states are not counted as duplicate retries. This definition is specific to the one-appointment experiment, not a general business-action equivalence test.

## Metrics

- Completion rate: passing cases divided by all cases for that policy.
- Argument accuracy: dispatched calls exactly matching the current observed canonical intent, divided by all dispatched calls. A schema-valid but ungrounded field can still be wrong. No-call avoidance does not raise task completion.
- Recovery tasks: cases that pass after at least one explicit retry/fallback recovery. The injected-failure denominator includes transient service failure and unavailable requested slots; unknown-commit timeouts are analyzed separately in traces.
- Unsupported claims: complete played confirmations whose slot/status disagrees with authoritative state at playback time. This metric does not inspect arbitrary natural-language claims because no speech model is attached to the controller.
- Controller p50/p95: measured execution time of the whole simulated case on this host. It is not audio response latency, interruption latency or model frame throughput.
- Wilson bands: descriptive intervals over generated case instances. Reused templates and seeded variants are correlated, so these are not defensible population-level confidence intervals or significance claims.

## Selection rule

Disqualify a policy with any duplicate action or unsupported claim. Rank eligible policies by completion, successful recovery and p95 controller runtime. No model is live-qualified by controller success. All policy implementations and failure traces remain available after selection.

The combined policy is enabled by default in the replay view only after loading a matching held-out selection report. The other policies remain selectable; deleting losing policies would prevent reproducing the comparison.

The suite was revised from generator 1.0.0 to 1.1.0 after an additional queued-output regression exposed that acknowledging only one message could hide stale speech. Version 1.1.0 uses FIFO playback and drains another message in delayed-feedback cases. Old queued output is invalidated when the playback-aware controller receives a changed intent. Because we inspected failures and reran the suite during development, these held-out instance scores are regression evidence, not an untouched final research test set or an unbiased estimate of real-world performance.

## Five scenario families

Argument corrections include a changed slot, corrupted identity, missing identity followed by clarification, and unrelated remarks. Tool failures include transient unavailability, explicitly authorized alternatives, a lost/late receipt after commit, and malformed proposed arguments. Interruptions include a correction before dispatch, an interrupted confirmation, cancellation before an old proposal executes, and a non-invalidating backchannel. Delayed feedback exercises 0/300/1,000/3,000 ms receipt delays, stale results, retries and playback recovery. Long sessions use thirty minutes of virtual time with many unrelated observations and a final state query/action.

These cases deliberately stress the mechanisms under test. The baseline is a handwritten short-context controller, not a commercial speech model. A perfect score means the selected controller handles these templates; it does not establish broad conversational intelligence or state-of-the-art STS performance.

## Recursive verification

Run development cases, inspect observed failures, add targeted regression cases, fix the implementation, then run the full five-policy suite again. A discovered duplicate-observation bug showed why the fixed suite is insufficient: replaying identical facts initially invalidated a pending action. A regression test now verifies stable intent revision when the values have not changed.

The automated tests also cover stale proposals and receipts, unknown commits, bounded retries, identity/schema rejection, SQL-as-data, playback truncation, 100 randomized correction orderings, bounded context, bad events, corrupt/truncated/empty WAVs, channel/rate normalization, bounded audio queues, request-size limits, cross-origin requests, DNS rebinding and path traversal.

Browser QA visits every policy/family combination with all four scenario subtypes (100 replays), reruns evaluation, reads the three documents, checks result export, inspects a mobile viewport and captures desktop/mobile screenshots. Console errors fail the browser test.

## Native audio gate

The separate Moshi smoke test uses the public q4 checkpoint at a pinned Hugging Face revision and `moshi_mlx==0.3.0`. It attempts five synthesized fixtures: greeting, correction, interruption phrase, noisy speech and silence. These are audio-in/audio-out model probes, not closed-loop task conversations. Waveform validity and real-time factor do not certify intelligibility, reasoning, speech quality or interruption recovery.

The child worker has explicit RSS, swap-growth and wall-clock budgets. If it fails or is stopped, the report preserves the outcome and no live-mode claim is made. PersonaPlex access is checked separately; an HTTP 401 is a blocked access gate, not a failed model-quality evaluation.

No actual microphone session, native tool-calling model integration, neural self-listening ablation, official τ-Voice run or thirty-minute wall-clock speech soak is represented by the controller report.

## Working native voice addition

The current Voice chat tab uses LFM2.5-Audio directly for actual waveform input and generated speech. Its own recipe comparison, microphone-path browser tests and limitations are in `reports/VOICE_RESULTS.md`. Those new results are separate from the controller report described above. A fixture-backed browser microphone is tested without mocking model responses; no physical microphone/human conversation or full-duplex capability is claimed.
