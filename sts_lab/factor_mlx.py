"""Experimental native MLX inference port of F-Actor's public Mimi model.

Architecture: https://github.com/MaikeZuefle/f-actor
No PyTorch execution or remote model code is used. Qualification is separate.
"""
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
from mlx_audio.lm.models.llama import LlamaModel, ModelArgs
from mlx_audio.lm.models.cache import make_prompt_cache


class DepthModel(LlamaModel):
    def __init__(self):
        args = ModelArgs(model_type='llama', hidden_size=1024, num_hidden_layers=4,
                         intermediate_size=8192, num_attention_heads=8, num_key_value_heads=2,
                         head_dim=128, rms_norm_eps=1e-5, vocab_size=65632,
                         rope_theta=500000, max_position_embeddings=33)
        super().__init__(args)
        self.embed_tokens = nn.Embedding(65632, 2048)
        self.inputs_embeds_projector = nn.Linear(2048, 1024, bias=False)

    def run(self, embeds, cache):
        return super().__call__(None, cache=cache, input_embeddings=self.inputs_embeds_projector(embeds))


class CodebookHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = mx.zeros((31, 1024, 2051), dtype=mx.bfloat16)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = DepthModel()
        self.codebooks_head = CodebookHead()


class DepthHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.semantic_head = nn.Linear(2048, 2051)
        self.backbone_adapter = nn.Linear(2048, 2048, bias=False)
        self.acoustic_speaker_adapter = nn.Linear(192, 2048, bias=False)
        self.semantic_speaker_adapter = nn.Linear(192, 2048, bias=False)
        self.depth_decoder = Decoder()


class Factor(nn.Module):
    def __init__(self, config):
        super().__init__()
        args = ModelArgs.from_dict(config)
        self.model = LlamaModel(args)
        self.lm_head = nn.Linear(2048, config['vocab_size'], bias=False)
        self.audio_embeds = [nn.Embedding(2051, 2048) for _ in range(16)]
        self.speaker_embed_proj = nn.Linear(192, 2048)
        self.depth_decoder_head = DepthHead()
        self.fast_depth = False

    def reset(self, prompt, speaker):
        self.cache = make_prompt_cache(self.model)
        self.speaker = mx.array(speaker, dtype=mx.bfloat16)[None]
        prefix = self.speaker_embed_proj(self.speaker)[:, None]
        text = self.model.embed_tokens(mx.array(prompt)[None])
        hidden = self.model(None, cache=self.cache, input_embeddings=mx.concatenate([prefix, text], axis=1))
        mx.eval(hidden, [c.state for c in self.cache])
        self.hidden = hidden[:, -1]
        self.semantic_context = self.depth_decoder_head.semantic_speaker_adapter(self.speaker)
        self.acoustic_context = self.depth_decoder_head.acoustic_speaker_adapter(self.speaker)
        # Upstream prepends two audio-delay frames; text is still generated.
        delay = mx.full((1, 8), 2049, dtype=mx.int32)
        for _ in range(2):
            text = self.sample(self.lm_head(self.hidden))
            self.advance(delay, delay, text)

    def sample(self, logits, temperature=.8, top_k=50):
        # Sample in float32; identical top-k contract to upstream.
        logits = logits.astype(mx.float32)
        if temperature <= 0:
            return mx.argmax(logits, axis=-1)
        cutoff = (mx.partition(logits, logits.shape[-1]-top_k, axis=-1) if self.fast_depth else mx.sort(logits, axis=-1))[..., -top_k, None]
        logits = mx.where(logits >= cutoff, logits / temperature, -float('inf'))
        return mx.random.categorical(logits)

    def predict(self):
        head = self.depth_decoder_head
        semantic = self.sample(head.semantic_head(self.hidden + self.semantic_context))
        text = self.sample(self.lm_head(self.hidden))
        context = head.backbone_adapter(self.hidden) + self.acoustic_context
        decoder = head.depth_decoder.model
        cache = make_prompt_cache(decoder)
        # Position 0 seeds the depth cache; semantic id is position 1.
        seeded = decoder.run(context[:, None], cache)
        if not self.fast_depth: mx.eval(seeded)
        codes = [semantic]
        for codebook in range(7):
            embed = decoder.embed_tokens((codes[-1] + codebook * 2051)[:, None])
            hidden = decoder.run(embed, cache)
            logits = hidden[:, -1] @ head.depth_decoder.codebooks_head.weight[codebook]
            code = self.sample(logits)
            if not self.fast_depth: mx.eval(code)
            codes.append(code)
        audio = mx.stack(codes, axis=1)
        mx.eval(audio, text)
        return audio, text

    def advance(self, own_audio, user_audio, text):
        # Each timestep sums both audio streams plus the model's text stream.
        both = mx.concatenate([own_audio, user_audio], axis=1)
        embeds = sum(layer(both[:, i]) for i, layer in enumerate(self.audio_embeds))
        embeds = embeds + self.model.embed_tokens(text)
        self.hidden = self.model(None, cache=self.cache, input_embeddings=embeds[:, None])[:, -1]
        mx.eval(self.hidden, [c.state for c in self.cache])


def load_factor(directory):
    directory = Path(directory)
    config = json.loads((directory / 'config.json').read_text())
    model = Factor(config)
    model.set_dtype(mx.bfloat16)
    weights = mx.load(str(directory / 'model.q4.safetensors'))
    modules = dict(model.named_modules())
    replacements = []
    for key in weights:
        if not key.endswith('.scales'):
            continue
        prefix = key.removesuffix('.scales')
        from sts_lab.packed_moshi import packed_layer
        packed = packed_layer(modules[prefix], weights, prefix)
        replacements.append((prefix, packed))
    model.update_modules(tree_unflatten(replacements))
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model
