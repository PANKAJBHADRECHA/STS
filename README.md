# STS Reliability Lab

A local experiment for speech-agent **system reliability**: implement five controllers, inject five classes of conversational failures, evaluate every combination, and select the best measured controller.

**Current scope:** structured-event controller experiments, an interactive replay interface, a source-grounded research dossier and a separately instrumented Moshi audio probe. This is not a trained new STS model or a live tool-using voice assistant. No controller result is presented as speech-model performance.

## Start locally

Python 3.12 or later is sufficient for the controller lab; it has no third-party runtime dependencies. Native Moshi testing uses Python 3.12 on Apple Silicon.

```bash
python3.12 -m venv .venv
.venv/bin/python -m sts_lab evaluate
.venv/bin/python -m sts_lab serve
```

Open **http://127.0.0.1:8765**. The server binds only to loopback. In the UI, rerun all experiments, select any variant/failure family, inspect event traces and read the research/memo. The chosen winner becomes the default replay policy; all five remain available.

## Five variants, five failure families

The variants are baseline, task memory, validated execution, playback recovery and combined. Ablations share one backend, proposal-fault generator, observation stream and evaluator. The independent SQLite backend checks schemas/identity and records authoritative receipts for every variant.

The families are argument corrections, tool failures, interruptions, delayed feedback and long sessions. Four scenario subtypes per family cover changed facts, missing identity, unavailable slots, unknown commits, cancellations, backchannels, stale results, partially played confirmations and forgotten task facts. The long-session test uses thirty minutes of **virtual time**, not a wall-clock speech-model soak.

The frozen held-out evaluation contains 100 tasks × 3 seeds × 5 variants = **1,500 controller runs**. Seed variants share templates; they are not independent human conversations. See [evaluation definitions](docs/evaluation.md) and [complete results](reports/held_out.json).

Full manifests and failure traces are losslessly compressed as `*-manifest.json.gz` and `*-failures.json.gz` to keep generated artifacts manageable. Read them with Python's `gzip.open()` or any gzip utility; scenario replay is available directly in the UI.

The selected controller must have zero duplicate actions and zero unsupported execution claims. Eligible variants rank by task completion, recovery and controller p95 runtime. Task completion requires correct database state **and** complete accurate confirmation. Exact measurements and the selected default are in [selection.json](reports/selection.json); the simulator target is not an official τ-Voice result.

## Tests and reproduction

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m sts_lab evaluate --split development --output reports
.venv/bin/python -m sts_lab evaluate --split held_out --output reports
.venv/bin/python -m sts_lab doctor --network --output reports/feasibility.json
```

The controller tests include 100 randomized correction-ordering trials and regression tests for discovered bugs. To reproduce browser QA:

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m playwright install chromium
.venv/bin/python scripts/browser_test.py
```

The browser script starts an isolated ephemeral server, checks all 25 policy/family combinations with four subtypes each (100 replays), reruns evaluation, checks documents/export and mobile layout, then closes its browser/server. Results and screenshots are in `reports/`.

## Native audio probe

The public Moshi checkpoint is pinned to `kyutai/moshiko-mlx-q4@18e4df760a34d5977a34517d7d1580e07acbb2f1`. The tested runtime is `moshi_mlx==0.3.0`; this is different from the newer source revision reviewed in the dossier. Model artifacts consume approximately 5.2 GB and are ignored by Git.

```bash
.venv/bin/python -m pip install -r requirements-native-macos.lock.txt
.venv/bin/python scripts/native_probe.py --download
```

This creates synthetic fixture speech with macOS `say`, uses Mimi/Moshi directly for audio input/output, and records frame timings/real-time factors if inference succeeds. There is no STT→LLM→TTS replacement inference path; OS TTS only creates test inputs. The watchdog stops the worker if RSS exceeds 6 GiB, swap grows by 768 MiB or wall time exceeds 180 seconds. Native smoke tests do not test task execution or true interactive interruption recovery. See [native probe outcome](reports/native-audio.json).

PersonaPlex is a separate access gate. The official model requires Hugging Face license acceptance. The inspected community runtime can fall back from a missing q4 file to its 16.7 GB full-precision model; that path is not enabled on this 8 GB Mac. Tokens belong in the environment, never this repository.

## Research and limitations

- [Technical dossier: ten models, training, latency, benchmarks, research threads and source map](docs/research.md)
- [Six-section differentiation/build memo and 8–12-week research experiment](docs/build-memo.md)
- [Evaluation methodology and untested capabilities](docs/evaluation.md)
- [Final measured outcomes](reports/RESULTS.md)

Not established by this prototype: conversational reasoning quality, persona consistency, neural self-listening, speech-to-tool argument grounding, live voice-agent performance, official τ-Voice/FDB improvements or commercial-model superiority. The speech model is not wired to the simulated controller. The local interface does not capture a microphone or record user audio.

Reports contain synthetic data only. Checkpoint weights, audio fixtures, virtual environments, caches and credentials are excluded from Git. Code is MIT; model licenses remain separate. See [third-party notices](THIRD_PARTY_NOTICES.md).

## CI template

`ci/github-actions.yml` reproduces controller and browser verification on Ubuntu. It is stored as a template because the available GitHub OAuth token lacks the `workflow` scope; GitHub rejected installing it under `.github/workflows/`. Local verification completed successfully. No credential permissions were changed.
