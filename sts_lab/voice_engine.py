"""Actual local LFM audio-to-audio inference. No transcript replaces audio input."""
from __future__ import annotations

import time
from pathlib import Path

from .audio import inspect_wav

BRIEF_PROMPT = "Respond with interleaved text and audio. You are a helpful assistant. Answer the user's spoken question directly in one short sentence. Use the latest correction when a fact changes. Remember facts the user gives about themselves; do not confuse them with yourself."


class VoiceEngine:
    def __init__(self, model_path, max_tokens=256, sampling="upstream", system_prompt=None):
        import mlx.core as mx
        from mlx_audio.sts.models.lfm_audio import LFM2AudioModel, LFM2AudioProcessor
        self.mx = mx
        self.model_path = str(Path(model_path).resolve())
        self.max_tokens, self.sampling = max_tokens, sampling
        self.system_prompt = system_prompt or "Respond with interleaved text and audio."
        mx.set_cache_limit(128*1024**2)
        mx.set_memory_limit(4*1024**3)
        started = time.perf_counter()
        self.model = LFM2AudioModel.from_pretrained(self.model_path)
        self.processor = LFM2AudioProcessor.from_pretrained(self.model_path)
        self.load_seconds = time.perf_counter()-started
        self.reset()

    def reset(self):
        from mlx_audio.sts.models.lfm_audio import ChatState
        self.chat = ChatState(self.processor)
        self.chat.new_turn("system")
        self.chat.add_text(self.system_prompt)
        self.chat.end_turn()
        self.turns = 0

    def _add_audio(self, waveform, rate):
        """Keep each mel span aligned to the encoder's eightfold subsampling.

        The runtime concatenates previous mel features. Padding each span avoids
        ceil(sum(lengths)/8) differing from sum(ceil(length/8)) across turns.
        """
        from mlx_audio.sts.models.lfm_audio import LFMModality
        mx = self.mx
        features = self.processor.preprocess_audio(waveform, rate)
        remainder = (-features.shape[0]) % 8
        if remainder:
            features = mx.concatenate([features, mx.repeat(features[-1:], remainder, axis=0)], axis=0)
        self.chat.audio_features = features if self.chat.audio_features is None else mx.concatenate([self.chat.audio_features, features], axis=0)
        self.chat.modalities.extend([LFMModality.AUDIO_IN]*(features.shape[0]//8))

    def respond(self, source, destination, *, reset=False, on_text=None, cancelled=None):
        from mlx_audio.audio_io import read as audio_read, write as audio_write
        from mlx_audio.sts.models.lfm_audio import LFMModality
        from mlx_audio.sts.models.lfm_audio.model import AUDIO_EOS_TOKEN
        import numpy as np
        mx = self.mx
        meta = inspect_wav(source)
        if meta["duration_seconds"] > 20:
            raise ValueError("Voice turns must be at most 20 seconds")
        if meta["rms"] < .0005:
            raise ValueError("No audible speech detected; record again")
        if reset:
            self.reset()
        if self.turns >= 6 or len(self.chat.modalities) > 3500:
            raise ValueError("Conversation context limit reached; start a new conversation")
        started = time.perf_counter()
        audio, rate = audio_read(str(source))
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        self.chat.new_turn("user")
        self._add_audio(mx.array(audio.astype(np.float32)), rate)
        self.chat.end_turn()
        self.chat.new_turn("assistant")
        inputs = dict(self.chat)
        if self.chat.audio_out_codes:
            # The upstream ChatState mapping omits this field: replay actual
            # generated audio codes, not zero embeddings, on subsequent turns.
            inputs["audio_codes"] = mx.stack(self.chat.audio_out_codes, axis=0)[None, :]
        text, audio_frames, generated = [], [], []
        first_token = first_audio_token = None
        mx.random.seed(42 + self.turns)
        for token, modality in self.model.generate_interleaved(
            **inputs, max_new_tokens=self.max_tokens,
            audio_temperature=1.0 if self.sampling == "upstream" else None,
            audio_top_k=4 if self.sampling == "upstream" else 1,
        ):
            mx.eval(token)
            if cancelled and cancelled():
                self.reset()
                raise InterruptedError("Generation cancelled; conversation reset")
            elapsed = time.perf_counter()-started
            first_token = elapsed if first_token is None else first_token
            generated.append((token, modality))
            if modality == LFMModality.TEXT:
                text.append(token)
                if on_text:
                    on_text(self.processor.decode_text(token))
            elif token[0].item() != AUDIO_EOS_TOKEN:
                first_audio_token = elapsed if first_audio_token is None else first_audio_token
                audio_frames.append(token)
        if not audio_frames:
            self.reset()
            raise RuntimeError("Model did not generate audio; conversation reset")
        generation_seconds = time.perf_counter()-started
        codes = mx.stack(audio_frames, axis=1)[None, :]
        waveform = self.processor.decode_audio(codes, codec="detokenizer")
        mx.eval(waveform)
        audio_write(str(destination), waveform[0].tolist(), 24000)
        for token, modality in generated:
            self.chat.append(token, modality)
        self.chat.end_turn()
        self.turns += 1
        result = {
            "text": self.processor.decode_text(mx.concatenate(text)).replace("<|text_end|>", "").strip() if text else "",
            "input_seconds": meta["duration_seconds"], "output": inspect_wav(destination),
            "wall_seconds": time.perf_counter()-started, "generation_seconds": generation_seconds,
            "first_token_seconds": first_token, "first_audio_token_seconds": first_audio_token,
            "audio_frames": len(audio_frames), "generated_tokens": len(generated),
            "truncated": len(generated) >= self.max_tokens,
            "turn": self.turns, "model_peak_memory_bytes": mx.get_peak_memory(),
            "mode": "native_sts_turn_based", "sampling": self.sampling,
        }
        result["real_time_factor"] = result["wall_seconds"]/result["output"]["duration_seconds"]
        mx.clear_cache()
        return result
