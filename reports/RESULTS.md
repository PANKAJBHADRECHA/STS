# Measured outcomes

Generator 1.1.0; all measurements from this checkout on the local M2 / 8 GB Mac.

## Controller selection

Selected **combined** by the predeclared safety/completion rule. The strengthened suite ran 1,500 evaluations: 100 synthetic tasks × 3 seeds × 5 policies.

- **baseline**: 75/300 completed (25%); 45 duplicate actions; 45 unsupported confirmations.
- **task_memory**: 120/300 completed (40%); 45 duplicate actions; 45 unsupported confirmations.
- **validated_execution**: 225/300 completed (75%); 0 duplicate actions; 0 unsupported confirmations.
- **playback_recovery**: 90/300 completed (30%); 45 duplicate actions; 45 unsupported confirmations.
- **combined**: 300/300 completed (100%); 0 duplicate actions; 0 unsupported confirmations.

The selected policy gains 75 percentage points over this handwritten baseline. It is the default replay policy. All alternatives remain selectable and reproducible.

Manifest SHA-256: `7c19015e4923316615bddf3c9c9e1ab333d916bb0f31f88c3423f18b726f4dd9`.

The cases use structured observations and fault injection. The suite was inspected and rerun while fixing defects, and template variants are correlated. These results establish regression coverage for the controller, not an untouched research holdout, official τ-Voice performance or commercial-model superiority.

## Recursive test rounds

1. Initial development evaluation: five policies × 30 tasks × three seeds (450 runs). Combined completed 90/90.
2. Regression expansion found and fixed duplicate observation delivery incorrectly invalidating a pending operation. A 39-test pass and 1,500-run comparison followed.
3. An adversarial queued-output probe found that stale confirmations could play after an intent change. Added invalidation, FIFO playback and second-message draining; bumped the generator to 1.1.0.
4. Reran development, the full 1,500-run matrix, randomized correction-ordering tests, API/audio tests and browser QA after the fixes. Final unit/integration log: `tests-final.txt`.

## Browser verification

Chromium / Playwright: 100 replays across 25 policy/family combinations; desktop 1440×1050 and mobile 390×844. Evaluation rerun, all three document views and JSON download passed, with no console errors. See `browser.json`, `desktop.png` and `mobile.png`.

## Native speech feasibility

PersonaPlex access probe returned HTTP 401; no credentials were configured, and no PersonaPlex weights were downloaded. The inspected community loader can fall back to full precision when q4 is absent; this path was not used.

The pinned public Moshi q4 assets were downloaded (approximately 5.2 GB). The native worker was **resource_stopped** during loading, before any of its five audio fixtures completed. System swap rose by 2670.6 MiB during the attempt, exceeding the 768 MiB growth budget; the watchdog terminated it after 23.1 seconds.

System-wide swap growth is not an isolated measurement of model memory; other host activity may contribute. The observed failure means the local resource gate did not pass, not a universal proof that every optimized runtime cannot run on an M2. No audio quality, model RTF, tool integration or long wall-clock conversation result was obtained. Live STS stays unqualified. See `native-audio.json` and `feasibility.json`.

## Reproduce

Run the commands in the README. The controller suite has no third-party dependencies; browser QA and optional native runtime dependencies are pinned separately. The included `ci/github-actions.yml` template reproduces controller/browser verification without downloading model weights. It is not active on GitHub: the available token lacks workflow-write scope, so GitHub rejected that portion of the original push. All recorded checks were run locally.
