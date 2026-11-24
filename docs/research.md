# Native STS: technical evidence dossier

Reviewed 25 November 2026. This document separates published architectural evidence, vendor measurements, hypotheses and local tests. The numerical frontier in the pasted prompts is not treated as a verified current leaderboard. Missing measurements remain missing.

## Architecture

“Native audio input/output,” “single jointly trained network,” “no internal text,” and “full duplex” are different properties. A streaming speech system may have several trained modules and still accept audio directly. An audio-capable model does not establish concurrent listening and speaking. A separate transcript can be an auxiliary output rather than the model's reasoning input. The distinctions below follow the actual model descriptions, rather than the STS label alone. [Moshi](https://arxiv.org/abs/2410.00037), [VoiceChat](https://arxiv.org/html/2609.21967v2), [Qwen3-Omni](https://arxiv.org/abs/2509.17765)

### Moshi 7B

Helium supplies the text backbone. Mimi supplies semantic/acoustic discrete codes; a temporal transformer models time and a smaller depth transformer models codebook dependencies within each step. User and assistant audio have separate streams. Time-aligned assistant text precedes audio as an inner monologue. Padding represents silence/absence of text; duplex behavior is learned on a common timeline rather than requiring a turn-ending VAD state machine. Mimi operates at 24 kHz and 12.5 frames/s; the 80 ms frame is an acoustic latency floor. Codebook and acoustic-delay settings belong to the checkpoint configuration, not a universal STS specification. [Paper §§3–4](https://arxiv.org/html/2410.00037v2)

The official MLX implementation offers q4/q8 and uses the Rust Mimi implementation. Its barebones clients explicitly lack echo cancellation and automatic compensation for growing lag. Consequently, sustained frame throughput and bounded playback delay must be tested independently of whether a model loads. [Official Mac runtime](https://github.com/kyutai-labs/moshi/tree/main/moshi_mlx)

### PersonaPlex 7B

PersonaPlex retains Moshi's audio/text machinery and adds role and voice conditioning. The hybrid prompt puts role tokens on the agent-text stream while agent audio is silent, and voice examples on the agent-audio stream while text is padded. User audio is replaced by a 440 Hz conditioning signal during this prefix. Delimiters separate prefix and dialogue. This is structured stream conditioning, not a hidden STT→chat→TTS chain. [Paper §3.1](https://arxiv.org/html/2602.06053v1)

The current NVIDIA implementation pins the prompt in a 2,048-frame sliding context; retained conversational history is shorter than the whole context because the prefix consumes frames. Prompt persistence therefore does not imply indefinite factual memory. The repository's latest behavior must not be assumed in an older MLX conversion. [NVIDIA implementation](https://github.com/NVIDIA/personaplex)

The card specifies English input/output, 24 kHz audio, Linux and NVIDIA A100/H100 targets. Code and weights have different licenses: MIT and NVIDIA Open Model License respectively. Access to weights is gated. [Model card](https://huggingface.co/nvidia/personaplex-7b-v1)

### NemotronLabs VoiceChat

The repository slug says 11B while the current overview calls the system 12B. Record both identifiers rather than silently normalizing the discrepancy. The card identifies a Nano v2 9B backbone, FastConformer perception and a streaming speech decoder; it specifies 16 kHz input and 22.05 kHz output. [Overview](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B/blob/main/overview.md)

The paper describes 80 ms encoder steps, parallel agent-text/function channels and an auxiliary RNN-T transcription branch. Tool boundaries delimit call and response spans; injected responses supply context with masked loss. Crucially, the RNN-T is trained afterward with the encoder/backbone frozen, and the TTS decoder is trained separately. The pasted assumption of joint RNN-T training is incorrect for this version. SFT uses 0.5 times agent-text plus function losses, plus 0.5 text-only loss; direct audio loss is zero. This is a unified streaming system, not proof of one end-to-end audio-token training objective. [Paper §§2–3](https://arxiv.org/html/2609.21967v2)

### Qwen3-Omni 30B-A3B

A MoE Thinker receives multimodal representations; a separate Talker predicts speech codes, using lightweight multi-token prediction and a causal convolutional waveform decoder. Hidden-state conditioning makes this different from passing an ASR transcript through three unrelated APIs. A 30B-A3B designation refers to total versus active backbone parameters; it does not imply that only 3B weights need storage. The report gives a 3B-A0.3B Talker, 80M MTP and 200M code-to-wave module. It reports streaming perception/generation, but this alone does not establish an application-level interruption protocol matching Moshi's dual-stream conversations. [Report §§2–3](https://arxiv.org/html/2509.17765v1)

Fine codec IDs, codebook configuration and model sub-configurations are available in the released configuration and must be pinned with the checkpoint before implementation; Mimi settings cannot be imported from Moshi. [Released config](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct/blob/main/config.json)

### MiniCPM-o 4.5, 9B

The official stack combines SigLip2, Whisper-medium, Qwen3-8B and CosyVoice2 through hidden-state connections. Its Omni-Flow processing serializes concurrent modality streams with time-division multiplexing. Speech generation interleaves text and speech tokens to receive new input promptly. Proactive speaking decisions are made at 1 Hz; this is distinct from the codec's frame rate. Voice references are part of the system conditioning. English/Chinese spoken interaction is documented. [Official technical documentation](https://github.com/OpenBMB/MiniCPM-V/blob/main/docs/minicpm_o4dot5_en.md)

The report separately covers time alignment, stream serialization, speech pretraining, joint multimodal training and RL. Its joint architecture is neither audio-only learning nor a claim that a generic offline Whisper endpoint supplies its runtime transcript. Exact deployable precision and duplex support must be checked in the selected llama.cpp-omni or PyTorch build; a text-only quantization is insufficient. [Report](https://arxiv.org/html/2604.27393v1)

### Step-Audio 2 mini

The release is an end-to-end audio understanding/conversation model with tool calling and multimodal RAG, under Apache 2.0. The documented installation targets PyTorch/CUDA and includes speech-token generation components. Its public conversation functionality should not automatically be classified as continuous full duplex: the inspected model card does not specify a concurrent user/agent timeline or a native interruption contract. [Model card](https://huggingface.co/stepfun-ai/Step-Audio-2-mini)

The report describes latent audio encoding, supervised training and reasoning-oriented reinforcement learning. It is relevant for stronger content reasoning, but the exact mini-checkpoint codec layout, training-mixture accounting and interruption behavior require code-level verification before an adapter is marked supported. Those quantities are not filled with Moshi defaults. [Report](https://arxiv.org/html/2507.16632v1)

### LongCat-Flash-Omni

This 560B-total/27B-active MoE adds audio perception and reconstruction to the LongCat architecture. The technical report separates audio tokenization/encoding/decoding, streaming audiovisual interaction, multimodal continued pretraining, audio alignment, SFT and RL. Shared active compute is not shared storage: inactive experts still occupy memory. This is excluded from the 8 GB Mac implementation. [Report](https://arxiv.org/html/2511.00279v1)

The official repository specifies at least 8×H20-141G for FP8 weights or 16×H800-80G for BF16. These are deployment examples, not the hardware for the local lab. A 27B-active headline cannot be interpreted as laptop deployability. [Repository](https://github.com/meituan-longcat/LongCat-Flash-Omni)

### Hibiki-Zero 3B

Hibiki-Zero is simultaneous translation, not a general customer-service agent. Its repository lists French, Spanish, Portuguese and German to English, with voice transfer. The official server targets NVIDIA GPUs: 8 GB VRAM may work and 12 GB is described as safe. [Repository](https://github.com/kyutai-labs/hibiki-zero)

The key training distinction is sentence-level aligned speech translation followed by GRPO to reduce lag while preserving quality, avoiding word-level alignment heuristics. Translation delay must be scored with quality/lag metrics, not conversational turn-taking scores. The inspected abstract does not specify all codec hyperparameters, reward coefficients or exact corpus hours; obtain them from the pinned implementation/report before replication. [Paper](https://arxiv.org/abs/2602.11072)

### Hertz-dev

This is a genuine released full-duplex conversational-audio base model, not merely a product mention. The official repository supplies one-/two-channel completion and experimental live client/server paths. Known-working setup is Python 3.10 and CUDA 12.1, with Ubuntu server and Mac client testing. The README alone does not establish exact codec bitrate, codebook count, loss weights, training corpus composition or 8 GB Apple inference. Those details remain unverified here; no compatibility claim is inferred from its Mac client. [Repository](https://github.com/Standard-Intelligence/hertz-dev)

### Dessa

Block states it trained from human speech without text or pretrained models and supports concurrent listening/speaking. It specifies an M4 Max MacBook Pro or better and early researcher access. The public page does not disclose enough tokenizer, training-objective or checkpoint information to reconstruct the token stream or run a controlled audio-only/hybrid ablation. “No text” describes supervision; it does not reveal whether the internal acoustic representation is discrete, continuous or hierarchical. No numerical reasoning/nuance advantage over PersonaPlex is inferred from demos. [Official page](https://block.xyz/dessa)

## Training

Moshi's card separates 7 million hours of unsupervised audio, roughly 2,000 hours of Fisher telephone conversations, 170 hours of collected multi-speaker material for the synthesis system, and 20,000 synthetic dialogue hours. These are stages/components, not a license to treat all sources as redistributable training data. The model uses aligned text/audio next-token objectives; codec reconstruction/adversarial learning is a separate training problem. [Checkpoint card](https://huggingface.co/kyutai/moshiko-mlx-q4), [Model/training sections](https://arxiv.org/html/2410.00037v2)

PersonaPlex's paper lists 1,840 synthetic service hours over 105,410 dialogues and 410 QA hours over 39,322 dialogues; the release adds 1,217 hours from 7,303 Fisher conversations. Consequently the model-card phrase “less than 10,000 hours” is not a precise standalone dataset description. These reported components sum to 3,467 hours, but this arithmetic is not an independently audited unique-hours total and excludes inherited Moshi training. The paper reports separate temporal/depth learning rates of 2e-6/4e-6, 24,576 updates and batch 32. [Paper §4 and release appendix](https://arxiv.org/html/2602.06053v1)

VoiceChat's current card reports approximately 550k hours spanning real speech and TTS-rendered text corpora, including Fisher, LibriTTS and internal data. This is an author account, not a fully reproducible open corpus. The RNN-T attaches to the shared encoder but has separate optimization; it is not a transcript bottleneck on the response path. [Overview](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B/blob/main/overview.md), [Training recipe](https://arxiv.org/html/2609.21967v2)

For the remaining models, the linked reports provide their own alignment/pretraining/SFT/RL descriptions. This project does not assert an audited common training-hours table: stage totals, unique recordings, synthetic renderings and token counts are not interchangeable. Exact unrecovered quantities are research gaps rather than zeros. No training corpus has been copied into this repository.

## Latency

Instrument capture framing → encoder → backbone → depth/MTP decoding → waveform decoder → playback queue → client render acknowledgement. Record useful-answer latency separately from first non-silent audio. Also distinguish interruption-to-silence from interruption-end-to-recovered-response. These are the lab's measurement definitions.

Moshi's paper reports 160 ms theoretical and approximately 200 ms practical latency. Its 80 ms codec frame is not the whole response time. PersonaPlex's card gives 170 ms smooth-turn response and 240 ms interruption latency for its reported evaluation; these are author benchmark results on their setup, not guaranteed browser latency. Earlier paper tables and released-checkpoint/card results differ, so retain the source/version alongside each number. [Moshi](https://arxiv.org/abs/2410.00037), [PersonaPlex card](https://huggingface.co/nvidia/personaplex-7b-v1)

Qwen3-Omni's theoretical single-concurrency audio first-packet budget is 72 ms tail preprocessing + 88 ms Thinker + 57 ms Talker + 14 ms MTP + 3 ms decoder = 234 ms. Its reported four-concurrency budget is 728 ms. Neither includes an arbitrary browser/network stack, nor proves sub-250 ms task completion. [Report Table 2](https://arxiv.org/html/2509.17765v1)

No corresponding encoder/backbone/decoder/playback breakdown has been locally measured for the other models. The native probe reports frame timing and real-time factor only if it actually runs; the controller's millisecond timings exclude all speech computation.

## Benchmarks and disagreements

Full-Duplex-Bench evaluates pause handling, backchannels, turn taking and interruption behavior. Takeover rate has different desired directions by task: taking the floor is desirable after a yielded turn and undesirable during a hesitation. It is not a single universal accuracy. [Benchmark paper](https://arxiv.org/abs/2503.04721)

VoiceChat's paper Table 4 reports FDB3 tool-selection F1 82.5%, argument accuracy 42.2% and strict pass@1 33.0%; Gemini Live 3.1 is listed at 81.7%, 58.8%, 54.0%. These are that paper's evaluation, not our reproduction. Strict pass@1 requires all expected calls and arguments, explaining why routing F1 does not demonstrate task reliability. The card has separately revised figures; preserve both provenance and snapshot. [Paper Table 4](https://arxiv.org/html/2609.21967v2)

The τ-Voice paper describes 278 tasks and reports historical voice-agent pass rates of 31–51% clean and 26–38% realistic, versus 85% for its text baseline. These describe its evaluated systems and conditions, not a perpetual market ceiling. [τ-Voice](https://arxiv.org/abs/2603.13686)

Artificial Analysis currently defines its composite as reasoning, agentic performance, arena preference and task success, equally weighted. Its methodology uses 1,000 Big Bench Audio questions across four reasoning categories and 23 synthetic voices. Its separately charted conversational dynamics is not the same thing as the current composite. The pasted 12.6/19.1 reasoning figures, older composite leaders and “sub-53%” ceiling cannot be combined into a current Pareto table without matching dates/checkpoints. [Methodology](https://artificialanalysis.ai/methodology/speech-to-speech-benchmarking), [Leaderboard](https://artificialanalysis.ai/speech-to-speech)

Our report supplies five-policy numerical comparisons with explicit denominators, seed/template limitations and full traces. It must not be plotted on the same frontier as official voice benchmarks.

## Tool and agentic mechanisms

Parallel function heads, delegation and a shared action stream make different engineering commitments. VoiceChat supplies an explicit function channel; correctness still requires executing results to affect subsequent actions. The frontend/backend paper learns a delegation token, sends incremental transcripts to a backend LLM and reinjects results using prefill-and-repeat. That approach is modular but differs from our strict single native-audio inference experiment; its reported single-turn recall is not multi-step task completion. [Delegation paper](https://arxiv.org/abs/2609.19334)

DuplexSLA instead describes continuous user audio, discrete assistant audio and rate-limited textual actions on a shared 160 ms timeline. The action bandwidth and timing synchronization are part of the architectural bet. A paper/demo/benchmark release is not evidence of available production-serving weights. [DuplexSLA](https://arxiv.org/abs/2605.20755)

Our selected system adds grounded intent revisions, idempotent receipts and rendered-output tracking around a controller. It does not add a native action head to PersonaPlex. Real speech-to-action integration remains a separate capability gate.

## Research threads

- **Backchannels:** a lightweight hidden-state head predicts onset probability; a threshold triggers force decoding. This exposes a timing control, rather than merely hoping a role prompt alters frequency. [Paper](https://arxiv.org/abs/2609.29418)
- **Self-listening:** user speech, realized model speech and model text occupy three channels. The played waveform anchors recovery to what was actually rendered. Our playback ledger is an engineering ablation, not this neural architecture. [Paper](https://arxiv.org/html/2609.05592v1), [Code](https://github.com/FreedomIntelligence/LoopSpeech)
- **Turn-taking instructions:** Instruct-FD tests explicit behavioral policies; its abstract reports a best adherence of 64.4%. Natural conversational timing does not establish compliance with instructions such as waiting through a pause. [Paper](https://arxiv.org/abs/2607.20460)
- **HiPLEX:** factorizes the text policy into pad/epad/content control and conditional lexical content, assigning timing and semantic advantages differently. It is RL credit assignment, not an independent VAD bolted onto playback. [Paper](https://arxiv.org/abs/2610.07727)
- **Asynchronous retrieval:** MoshiRAG retrieves while the model continues speaking; knowledge must arrive before the factual portion rather than before any speech. This motivates measuring useful-content latency as well as first audio. [Kyutai explanation](https://kyutai.org/blog/2026-04-30-moshi-rag/), [Implementation](https://github.com/kyutai-labs/moshi-rag)
- **Interruption safety:** DuplexJail tests spoken prompts inserted at fixed delays or refusal cues during generation. Safety evaluation must inspect the resulting whole response, not just its opening refusal. [Paper](https://arxiv.org/abs/2609.09420)

## Source map and implementation decision

Most useful implementation sources: [NVIDIA PersonaPlex](https://github.com/NVIDIA/personaplex) for conditioning and session context; [Kyutai Moshi](https://github.com/kyutai-labs/moshi) for actual frame/token streaming and official MLX; [MiniCPM-o documentation](https://github.com/OpenBMB/MiniCPM-V/blob/main/docs/minicpm_o4dot5_en.md) for synchronized multimodal streams. For actions, read the [delegation paper](https://arxiv.org/abs/2609.19334) and [DuplexSLA](https://arxiv.org/abs/2605.20755) alongside implementation availability, not as drop-in PersonaPlex features.

Most revealing papers: [Moshi](https://arxiv.org/abs/2410.00037), [PersonaPlex](https://arxiv.org/abs/2602.06053), [VoiceChat](https://arxiv.org/abs/2609.21967). For evaluation, pair [Full-Duplex-Bench](https://arxiv.org/abs/2503.04721) with [τ-Voice](https://arxiv.org/abs/2603.13686). The most useful explanatory resources are [NVIDIA's project page](https://research.nvidia.com/labs/adlr/personaplex/), [Kyutai's MoshiRAG explanation](https://kyutai.org/blog/2026-04-30-moshi-rag/) and [Artificial Analysis methodology](https://artificialanalysis.ai/methodology/speech-to-speech-benchmarking). Roxabi was requested but no verified primary engineering source from it was recovered; no architecture claim relies on it.

For this Mac: test public q4 Moshi through a bounded worker, keep PersonaPlex blocked until accessible prequantized assets are verified, and develop the reliability layer independently. Reports expose precisely which gates passed. No commercial superiority or new model-training result is claimed.
