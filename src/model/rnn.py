import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.base import Seq2SeqModel


class RNNEncoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.dim = params.enc_emb_dim
        self.hidden_dim = params.enc_emb_dim
        self.n_layers = params.n_enc_layers
        self.dropout = params.dropout
        self.n_words = params.n_words
        self.pad_index = params.pad_index
        self.is_lstm = params.model_type == "lstm"

        self.token_embeddings = nn.Embedding(self.n_words, self.dim, padding_idx=self.pad_index)
        self.ln_emb = nn.LayerNorm(self.dim)

        rnn_cls = nn.LSTM if self.is_lstm else nn.GRU
        self.rnn = rnn_cls(self.dim, self.hidden_dim, self.n_layers, bias=False, dropout=self.dropout if self.n_layers > 1 else 0, batch_first=True)
        self.proj_out = nn.Linear(self.hidden_dim, self.dim, bias=False)

    def forward(self, x, lengths):
        bs, slen = x.shape

        hidden = self.token_embeddings(x)
        hidden = self.ln_emb(hidden)
        if self.dropout > 0:
            hidden = F.dropout(hidden, p=self.dropout, training=self.training)

        pad_mask = torch.arange(slen, dtype=torch.long, device=lengths.device) < lengths[:, None]
        hidden = hidden * pad_mask.unsqueeze(-1).to(hidden.dtype)

        output, rnn_hidden = self.rnn(hidden)
        output = self.proj_out(output)
        return output, rnn_hidden


class RNNDecoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.dim = params.dec_emb_dim
        self.hidden_dim = params.dec_emb_dim
        self.n_layers = params.n_dec_layers
        self.dropout = params.dropout
        self.n_words = params.n_words
        self.pad_index = params.pad_index
        self.is_lstm = params.model_type == "lstm"

        self.token_embeddings = nn.Embedding(self.n_words, self.dim, padding_idx=self.pad_index)
        self.ln_emb = nn.LayerNorm(self.dim)

        rnn_cls = nn.LSTM if self.is_lstm else nn.GRU
        self.rnn = rnn_cls(self.dim, self.hidden_dim, self.n_layers, bias=False, dropout=self.dropout if self.n_layers > 1 else 0, batch_first=True)
        self.proj_out = nn.Linear(self.hidden_dim, self.dim, bias=False)

        self.proj = nn.Linear(self.dim, self.n_words, bias=False)
        if params.share_inout_emb:
            self.proj.weight = self.token_embeddings.weight

    def forward(self, x, lengths, hidden=None):
        bs, slen = x.shape

        emb = self.token_embeddings(x)
        emb = self.ln_emb(emb)
        if self.dropout > 0:
            emb = F.dropout(emb, p=self.dropout, training=self.training)

        pad_mask = torch.arange(slen, dtype=torch.long, device=lengths.device) < lengths[:, None]
        emb = emb * pad_mask.unsqueeze(-1).to(emb.dtype)

        output, new_hidden = self.rnn(emb, hidden)
        output = self.proj_out(output)
        logits = self.proj(output)
        return logits, new_hidden


class RNNSeq2Seq(Seq2SeqModel):
    def __init__(self, params):
        super().__init__(params)
        self.encoder = RNNEncoder(params)
        self.decoder = RNNDecoder(params)
        self.is_lstm = params.model_type == "lstm"

    def _encode(self, src, src_len):
        _output, hidden = self.encoder(src, src_len)
        return hidden, None

    def _decode_train(self, task, dec_input, dec_input_len, src_enc, src_mask):
        logits, _hidden = self.decoder(dec_input, dec_input_len, hidden=src_enc)
        return logits

    def _prefill(self, task, gen_prefix, gen_prefix_len, max_new_tokens, src_enc, src_mask):
        logits, hidden = self.decoder(gen_prefix, gen_prefix_len, hidden=src_enc)
        return logits, hidden

    def _generate_step(self, task, token, token_len, src_enc, src_mask, gen_state):
        logits, hidden = self.decoder(token, token_len, hidden=gen_state)
        return logits, hidden

    def _expand_enc_out(self, src_enc, src_mask, beam_size):
        if self.is_lstm:
            h, c = src_enc
            h = h.unsqueeze(2).expand(-1, -1, beam_size, -1).reshape(h.size(0), -1, h.size(2))
            c = c.unsqueeze(2).expand(-1, -1, beam_size, -1).reshape(c.size(0), -1, c.size(2))
            return (h, c), None
        else:
            h = src_enc
            h = h.unsqueeze(2).expand(-1, -1, beam_size, -1).reshape(h.size(0), -1, h.size(2))
            return h, None

    def _reorder_gen_state(self, gen_state, indices):
        if self.is_lstm:
            h, c = gen_state
            return (h.index_select(1, indices), c.index_select(1, indices))
        else:
            return gen_state.index_select(1, indices)
