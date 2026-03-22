import math
from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.base import Seq2SeqModel

logger = getLogger()


def get_norm_layer(norm_type, dim):
    if norm_type == "layernorm":
        return nn.LayerNorm(dim)
    elif norm_type == "rmsnorm":
        return nn.RMSNorm(dim)
    elif norm_type == "rmsnorm_no_params":
        return nn.RMSNorm(dim, elementwise_affine=False)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


def get_activation_fn(name):
    if name == "relu":
        return F.relu
    elif name == "relu_squared":
        return lambda x: F.relu(x).square()
    elif name == "gelu":
        return F.gelu
    else:
        raise ValueError(f"Unknown activation function: {name}")


def _pad_mask_4d(lengths_or_left_pad, total_len, device, mode="right"):
    """Build a (batch, 1, 1, total_len) additive -inf mask for padded positions.

    mode="right": positions >= lengths are padding (encoder-style).
    mode="left":  positions < left_pad are padding (decoder left-pad style).
    """
    pos = torch.arange(total_len, device=device).unsqueeze(0)
    if mode == "right":
        padding = pos >= lengths_or_left_pad.unsqueeze(1)
    else:
        padding = pos < lengths_or_left_pad.unsqueeze(1)
    mask = torch.zeros(lengths_or_left_pad.size(0), 1, 1, total_len, device=device)
    mask.masked_fill_(padding.unsqueeze(1).unsqueeze(2), float("-inf"))
    return mask


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, max_len, dim):
        super().__init__()
        assert dim % 2 == 0
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x, start_pos=0):
        return self.pe[:, start_pos : start_pos + x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, dim, src_dim, dropout, is_cross_attn=False):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.is_cross_attn = is_cross_attn
        if not is_cross_attn:
            self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        else:
            self.q_proj = nn.Linear(dim, dim, bias=False)
            self.k_proj = nn.Linear(src_dim, dim, bias=False)
            self.v_proj = nn.Linear(src_dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def _reshape(self, x, batch_size):
        return x.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, context=None, mask=None, kv_cache=None, first_loop=True):
        batch_size, seq_len, _ = x.shape

        if not self.is_cross_attn:
            q_raw, k_raw, v_raw = self.qkv_proj(x).chunk(3, dim=-1)
            q = self._reshape(q_raw, batch_size)

            if kv_cache is not None and "pos" in kv_cache:
                k_new = self._reshape(k_raw, batch_size)
                v_new = self._reshape(v_raw, batch_size)

                pos = kv_cache["pos"]
                if not first_loop:
                    pos = pos - seq_len
                kv_cache["k"][:, :, pos : pos + seq_len] = k_new
                kv_cache["v"][:, :, pos : pos + seq_len] = v_new
                kv_cache["pos"] = pos + seq_len
                k = kv_cache["k"][:, :, : kv_cache["pos"]]
                v = kv_cache["v"][:, :, : kv_cache["pos"]]
            elif kv_cache is not None and "k" in kv_cache:
                k, v = kv_cache["k"], kv_cache["v"]
            else:
                k = self._reshape(k_raw, batch_size)
                v = self._reshape(v_raw, batch_size)

                if kv_cache is not None:
                    kv_cache["k"], kv_cache["v"] = k, v
        else:
            q = self._reshape(self.q_proj(x), batch_size)
            if kv_cache is not None and "k" in kv_cache:
                k, v = kv_cache["k"], kv_cache["v"]
            else:
                src = context
                k = self._reshape(self.k_proj(src), batch_size)
                v = self._reshape(self.v_proj(src), batch_size)
                if kv_cache is not None:
                    kv_cache["k"], kv_cache["v"] = k, v

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(out)


class MLP(nn.Module):
    def __init__(self, dim, dropout, activation, n_hidden_layers):
        super().__init__()
        hidden_dim = 4 * dim
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.mid = nn.ModuleList()
        for _ in range(1, n_hidden_layers):
            self.mid.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
        self.down = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.activation = get_activation_fn(activation)

    def forward(self, x):
        x = self.activation(self.up(x))
        if self.dropout is not None:
            x = self.dropout(x)
        for mid_layer in self.mid:
            x = self.activation(mid_layer(x))
            if self.dropout is not None:
                x = self.dropout(x)
        return self.down(x)


class Gate(nn.Module):
    def __init__(self, dim, scalar, gate_bias, dropout, activation):
        super().__init__()
        self.up = nn.Linear(dim, 4 * dim, bias=False)
        self.down = nn.Linear(4 * dim, 1 if scalar else dim, bias=True)
        self.down.bias.data.fill_(gate_bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.activation = get_activation_fn(activation)

    def forward(self, x):
        x = self.activation(self.up(x))
        if self.dropout is not None:
            x = self.dropout(x)
        return torch.sigmoid(self.down(x))


class Residual(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x, residual):
        if self.dropout is not None:
            x = self.dropout(x)
        return residual + x


class TransformerLayer(nn.Module):
    def __init__(self, params, is_encoder, gated=False):
        """
        Transformer model (encoder or decoder).
        """
        super().__init__()
        self.is_encoder = is_encoder
        self.dim = params.enc_emb_dim if is_encoder else params.dec_emb_dim
        self.src_dim = params.enc_emb_dim
        self.n_heads = params.n_enc_heads if is_encoder else params.n_dec_heads
        self.dropout = params.dropout
        self.attention_dropout = params.attention_dropout
        self.gated = gated

        self.self_attn = MultiHeadAttention(self.n_heads, self.dim, self.dim, self.attention_dropout)
        self.ln1 = get_norm_layer(params.norm, self.dim)
        self.res1 = Residual(self.dropout)

        self.has_cross_attn = not self.is_encoder and params.architecture == "encoder_decoder"
        if self.has_cross_attn:
            self.cross_attn = MultiHeadAttention(self.n_heads, self.dim, self.src_dim, self.attention_dropout, is_cross_attn=True)
            self.ln_cross = get_norm_layer(params.norm, self.dim)
            self.res_cross = Residual(self.dropout)

        n_hidden_layers = params.n_enc_hidden_layers if is_encoder else params.n_dec_hidden_layers
        self.ffn = MLP(self.dim, self.dropout, activation=params.activation, n_hidden_layers=n_hidden_layers)
        self.ln2 = get_norm_layer(params.norm, self.dim)
        self.res2 = Residual(self.dropout)

        if self.gated:
            self.gate = Gate(self.dim, params.scalar_gate, params.gate_bias, self.dropout, activation=params.activation)

    def forward(self, x, src_enc=None, src_mask=None, tgt_mask=None, kv_cache=None, loop_count=1):
        mask = src_mask if self.is_encoder else tgt_mask

        for i in range(loop_count):
            loop_input = x

            self_attn_cache = kv_cache["self_attn"] if kv_cache else None
            attn_out = self.self_attn(self.ln1(x), mask=mask, kv_cache=self_attn_cache, first_loop=i == 0)
            x = self.res1(attn_out, x)

            if self.has_cross_attn:
                cross_attn_cache = kv_cache["cross_attn"] if kv_cache else None
                cross_out = self.cross_attn(self.ln_cross(x), context=src_enc, mask=src_mask, kv_cache=cross_attn_cache, first_loop=i == 0)
                x = self.res_cross(cross_out, x)

            x = self.res2(self.ffn(self.ln2(x)), x)

            if self.gated:
                gate_value = self.gate(x)
                x = gate_value * x + (1 - gate_value) * loop_input

        return x


class TransformerModel(nn.Module):
    def __init__(self, params, is_encoder, with_output):
        super().__init__()
        self.is_encoder = is_encoder
        self.with_output = with_output
        self.dim = params.enc_emb_dim if is_encoder else params.dec_emb_dim
        self.n_heads = params.n_enc_heads if is_encoder else params.n_dec_heads
        self.head_dim = self.dim // self.n_heads
        self.n_layers = params.n_enc_layers if is_encoder else params.n_dec_layers
        self.pos_emb = params.enc_pos_emb if is_encoder else params.dec_pos_emb
        self.loops = params.enc_loops if is_encoder else params.dec_loops
        self.loop_idx = params.enc_loop_idx if is_encoder else params.dec_loop_idx
        self.dropout = params.dropout
        assert self.loop_idx == -2 or self.loop_idx < self.n_layers, "loop index must be -2 (all layers) or a valid layer index"

        self.n_words = params.n_words
        self.eos_index = params.eos_index
        self.pad_index = params.pad_index

        # Positional embeddings
        pos_max = params.max_len if is_encoder else (params.max_len + params.max_output_len + 4)
        if self.pos_emb == "abs_sinusoidal":
            self.position_embeddings = SinusoidalPositionalEmbedding(pos_max, self.dim)
        elif self.pos_emb == "abs_learned":
            self.position_embeddings = nn.Embedding(pos_max, self.dim)
        self.token_embeddings = nn.Embedding(self.n_words, self.dim, padding_idx=self.pad_index)
        self.ln_emb = get_norm_layer(params.norm, self.dim)

        self.layers = nn.ModuleList()
        for layer_id in range(self.n_layers):
            gated = (
                (params.enc_gated and is_encoder)
                or (params.dec_gated and not is_encoder)
                or (params.gated and (layer_id == self.loop_idx or self.loop_idx == -2))
            )
            self.layers.append(TransformerLayer(params, is_encoder, gated))

        self.ln_final = get_norm_layer(params.norm, self.dim)

        if self.with_output:
            self.proj = nn.Linear(self.dim, params.n_words, bias=False)
            if params.share_inout_emb:
                self.proj.weight = self.token_embeddings.weight

    def _compute_positions(self, seq_len, lengths, start_pos, kv_cache, device):
        """Compute position indices, using content-relative positions for left-padded decoder prefill."""
        is_prefill_left_padded = not self.is_encoder and kv_cache is not None and kv_cache[0]["self_attn"]["pos"] == 0 and (lengths < seq_len).any()
        if is_prefill_left_padded:
            left_pad = seq_len - lengths
            return (torch.arange(seq_len, device=device).unsqueeze(0) - left_pad.unsqueeze(1)).clamp(min=0)
        if isinstance(start_pos, torch.Tensor):
            if start_pos.dim() == 0:
                return start_pos + torch.arange(seq_len, device=device).unsqueeze(0)
            elif start_pos.dim() == 1:
                return start_pos.unsqueeze(1) + torch.arange(seq_len, device=device).unsqueeze(0)
            else:
                return start_pos
        return torch.arange(start_pos, start_pos + seq_len, device=device).unsqueeze(0)

    def forward(self, x, lengths, src_enc=None, src_mask=None, kv_cache=None, start_pos=0):
        _, seq_len = x.shape
        device = x.device

        hidden = self.token_embeddings(x)
        if self.pos_emb in ("abs_sinusoidal", "abs_learned"):
            positions = self._compute_positions(seq_len, lengths, start_pos, kv_cache, device)
            if self.pos_emb == "abs_sinusoidal":
                pos_emb = self.position_embeddings.pe.squeeze(0)[positions]
            else:
                pos_emb = self.position_embeddings(positions)
            is_pad = x == self.pad_index
            if is_pad.any():
                pos_emb = pos_emb.masked_fill(is_pad.unsqueeze(-1), 0.0)
            hidden = hidden + pos_emb
        hidden = self.ln_emb(hidden)
        if self.dropout > 0:
            hidden = F.dropout(hidden, p=self.dropout, training=self.training)

        if self.is_encoder:
            attn_mask = _pad_mask_4d(lengths, seq_len, device, mode="right")
            pad_mask = torch.arange(seq_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        else:
            pad_mask = None
            has_cache = kv_cache is not None and kv_cache[0]["self_attn"]["pos"] > 0
            if has_cache:
                left_pad_len = kv_cache[0]["self_attn"].get("left_pad_len")
                if left_pad_len is not None:
                    total_len = kv_cache[0]["self_attn"]["pos"] + seq_len
                    attn_mask = _pad_mask_4d(left_pad_len, total_len, device, mode="left")
                else:
                    attn_mask = None
            else:
                attn_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1).unsqueeze(0).unsqueeze(0)
                if kv_cache is not None:
                    left_pad_len = seq_len - lengths
                    if (left_pad_len > 0).any():
                        attn_mask = attn_mask + _pad_mask_4d(left_pad_len, seq_len, device, mode="left")
                        kv_cache[0]["self_attn"]["left_pad_len"] = left_pad_len

        for i, layer in enumerate(self.layers):
            loop_count = self.loops if (i == self.loop_idx or self.loop_idx == -2) else 1
            layer_cache = kv_cache[i] if kv_cache is not None else None

            if self.is_encoder:
                hidden = layer(hidden, src_mask=attn_mask, kv_cache=layer_cache, loop_count=loop_count)
                hidden = hidden * pad_mask.unsqueeze(-1)
            else:
                hidden = layer(hidden, src_enc=src_enc, src_mask=src_mask, tgt_mask=attn_mask, kv_cache=layer_cache, loop_count=loop_count)

        hidden = self.ln_final(hidden)
        if self.with_output:
            hidden = self.proj(hidden)
        return hidden


class TransformerSeq2Seq(Seq2SeqModel):
    def __init__(self, params):
        super().__init__(params)
        if self.architecture in ("encoder_only", "encoder_decoder"):
            self.encoder = TransformerModel(params, is_encoder=True, with_output=(self.architecture == "encoder_only"))
        if self.architecture in ("decoder_only", "encoder_decoder"):
            self.decoder = TransformerModel(params, is_encoder=False, with_output=True)

    def _get_decoder(self, task):
        return self.decoder

    def _init_kv_cache(self, decoder, batch_size, max_len):
        kv_cache = []
        for layer in decoder.layers:
            self_attn_cache = {
                "pos": 0,
                "k": torch.zeros(batch_size, decoder.n_heads, max_len, decoder.head_dim, device=self.device),
                "v": torch.zeros(batch_size, decoder.n_heads, max_len, decoder.head_dim, device=self.device),
            }
            layer_cache = {"self_attn": self_attn_cache}
            if layer.has_cross_attn:
                layer_cache["cross_attn"] = {}
            kv_cache.append(layer_cache)
        return kv_cache

    def _reorder_kv_cache(self, kv_cache, indices):
        for layer_cache in kv_cache:
            for attn_type in ("self_attn", "cross_attn"):
                if attn_type not in layer_cache:
                    continue
                cache = layer_cache[attn_type]
                for key, val in cache.items():
                    if isinstance(val, torch.Tensor):
                        cache[key] = val.index_select(0, indices)

    def _encode(self, src, src_len):
        enc_output = self.encoder(src, src_len)
        src_mask = self._make_src_mask(src_len, enc_output.size(1), src.device)
        return enc_output, src_mask

    def _decode_train(self, task, dec_input, dec_input_len, src_enc, src_mask):
        decoder = self._get_decoder(task)
        return decoder(dec_input, dec_input_len, src_enc=src_enc, src_mask=src_mask)

    def _prefill(self, task, gen_prefix, gen_prefix_len, max_new_tokens, src_enc, src_mask):
        decoder = self._get_decoder(task)
        batch_size = gen_prefix.size(0)
        total_len = gen_prefix.size(1) + max_new_tokens
        kv_cache = self._init_kv_cache(decoder, batch_size, total_len)
        logits = decoder(gen_prefix, gen_prefix_len, src_enc=src_enc, src_mask=src_mask, kv_cache=kv_cache)
        return logits, (kv_cache, gen_prefix_len, 0)

    def _generate_step(self, task, token, token_len, src_enc, src_mask, gen_state):
        decoder = self._get_decoder(task)
        kv_cache, start_steps, step = gen_state
        step += 1
        start_pos = start_steps + step - 1
        logits = decoder(token, token_len, src_enc=src_enc, src_mask=src_mask, kv_cache=kv_cache, start_pos=start_pos)
        return logits, (kv_cache, start_steps, step)

    def _expand_enc_out(self, src_enc, src_mask, beam_size):
        src_enc = src_enc.unsqueeze(1).expand(-1, beam_size, -1, -1).reshape(-1, src_enc.size(1), src_enc.size(2))
        src_mask = src_mask.unsqueeze(1).expand(-1, beam_size, -1, -1, -1).reshape(-1, src_mask.size(1), src_mask.size(2), src_mask.size(3))
        return src_enc, src_mask

    def _reorder_gen_state(self, gen_state, indices):
        kv_cache, start_steps, step = gen_state
        self._reorder_kv_cache(kv_cache, indices)
        if isinstance(start_steps, torch.Tensor):
            start_steps = start_steps.index_select(0, indices)
        return (kv_cache, start_steps, step)
