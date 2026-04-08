from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = getLogger()


class BaseModel(nn.Module):
    """
    Base class for all models (Transformer, LSTM, GRU).
    Shared logic: forward (training), decode (encoder-only), generate, beam_generate.
    """

    def __init__(self, params):
        super().__init__()
        self.architecture = params.architecture
        self.pad_index = params.pad_index
        self.eos_index = params.eos_index
        self.n_words = params.n_words
        self.max_src_len = params.max_src_len
        self.device = params.device

    def _make_src_mask(self, lengths, max_len, device):
        mask = torch.zeros(lengths.size(0), 1, 1, max_len, device=device)
        if self.max_src_len > 0:
            effective_lengths = torch.clamp(lengths, max=self.max_src_len)
        else:
            effective_lengths = lengths
        padding = torch.arange(max_len, device=device).unsqueeze(0) >= effective_lengths.unsqueeze(1)
        mask.masked_fill_(padding.unsqueeze(1).unsqueeze(2), float("-inf"))
        return mask

    def _filter_logits(self, logits, temperature, top_k, top_p):
        logits = logits / max(temperature, 1e-8)
        if top_k > 0:
            top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < top_values[:, -1:]] = float("-inf")
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)
            remove = (cumulative - probs) >= top_p
            sorted_logits[remove] = float("-inf")
            logits = torch.zeros_like(logits).scatter_(1, sorted_indices, sorted_logits)
        return logits

    def _sample(self, logits, temperature, top_k, top_p, n):
        logits = self._filter_logits(logits, temperature, top_k, top_p)
        return torch.multinomial(F.softmax(logits, dim=-1), n)

    def _encode(self, src, src_len):
        """Encode source. Returns (src_enc, src_mask). For encoder_only, src_enc is logits."""
        raise NotImplementedError

    def _decode_train(self, task, dec_input, dec_input_len, src_enc, src_mask):
        """Decoder forward during training. Returns logits (bs, slen, n_words)."""
        raise NotImplementedError

    def _prefill(self, task, gen_prefix, gen_prefix_len, max_new_tokens, src_enc, src_mask):
        """Prefill generation state with prefix. Returns (logits, gen_state)."""
        raise NotImplementedError

    def _generate_step(self, task, token, token_len, src_enc, src_mask, gen_state):
        """Single autoregressive decode step. Returns (logits, gen_state)."""
        raise NotImplementedError

    def _expand_enc_out(self, src_enc, src_mask, beam_size):
        """Expand encoder output for beam search. Returns (src_enc, src_mask)."""
        raise NotImplementedError

    def _reorder_gen_state(self, gen_state, indices):
        """Reorder generation state for beam search. Returns new gen_state."""
        raise NotImplementedError

    def forward(self, enc_problem, enc_problem_len, dec_tgt, dec_tgt_len, prefix_len, task):
        if self.architecture == "encoder_only":
            logits, _ = self._encode(enc_problem, enc_problem_len)
            targets = dec_tgt
            min_len = min(logits.size(1), targets.size(1))
            logits = logits[:, :min_len, :]
            targets = targets[:, :min_len]
        elif self.architecture == "encoder_decoder":
            src_enc, src_mask = self._encode(enc_problem, enc_problem_len)
            dec_input = dec_tgt[:, :-1]
            logits = self._decode_train(task, dec_input, dec_tgt_len - 1, src_enc, src_mask)
            targets = dec_tgt[:, 1:]
        else:
            dec_input = dec_tgt[:, :-1]
            logits = self._decode_train(task, dec_input, dec_tgt_len - 1, None, None)
            targets = dec_tgt[:, 1:]

        # Mask prefix tokens from loss
        if prefix_len is not None and (prefix_len > 1).any():
            arange = torch.arange(targets.size(1), device=targets.device).unsqueeze(0)
            input_mask = arange < (prefix_len - 1).unsqueeze(1)
            targets = targets.clone()
            targets[input_mask] = self.pad_index

        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=self.pad_index)
        return logits, loss

    @torch.no_grad()
    def decode(self, enc_problem, enc_problem_len, max_output_len):
        assert self.architecture == "encoder_only"
        logits, _ = self._encode(enc_problem, enc_problem_len)
        generated_tokens = logits.argmax(-1)[:, :max_output_len]
        actual_len = generated_tokens.size(1)

        eos_mask = generated_tokens == self.eos_index
        has_eos = eos_mask.any(dim=1)
        first_eos = eos_mask.float().argmax(dim=1)
        generated_lengths = torch.where(has_eos, first_eos, torch.tensor(actual_len, device=self.device))

        return generated_tokens, generated_lengths

    @torch.no_grad()
    def generate(self, enc_src, enc_src_len, gen_prefix, gen_prefix_len, max_new_tokens, temperature, top_k, top_p, task):
        assert self.architecture != "encoder_only"
        batch_size = gen_prefix.size(0)

        if self.architecture == "encoder_decoder":
            src_enc, src_mask = self._encode(enc_src, enc_src_len)
        else:
            src_enc, src_mask = None, None
        logits, gen_state = self._prefill(task, gen_prefix, gen_prefix_len, max_new_tokens, src_enc, src_mask)

        generated_tokens = torch.full((batch_size, max_new_tokens), self.pad_index, dtype=torch.long, device=self.device)
        generated_lengths = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        ones = torch.ones(batch_size, dtype=torch.long, device=self.device)

        for step in range(max_new_tokens):
            if finished.all():
                break
            if step == 0:
                current_token = self._sample(logits[:, -1, :], temperature, top_k, top_p, 1)
            else:
                logits, gen_state = self._generate_step(task, current_token, ones, src_enc, src_mask, gen_state)
                current_token = self._sample(logits[:, -1, :], temperature, top_k, top_p, 1)
            current_token[finished] = self.pad_index
            active = ~finished
            generated_tokens[active, step] = current_token.squeeze(-1)[active]
            just_finished = current_token.squeeze(-1) == self.eos_index
            finished |= just_finished
            still_active = active & ~just_finished
            generated_lengths[still_active] += 1

        return generated_tokens, generated_lengths

    @torch.no_grad()
    def beam_generate(
        self, enc_src, enc_src_len, gen_prefix, gen_prefix_len, max_new_tokens, beam_size, length_penalty, temperature, top_k, top_p, task
    ):
        assert self.architecture != "encoder_only"
        n_words = self.n_words
        use_sampling = temperature != 1.0 or top_k > 0 or top_p < 1.0
        batch_size = gen_prefix.size(0)
        total_beams = batch_size * beam_size

        if self.architecture == "encoder_decoder":
            src_enc, src_mask = self._encode(enc_src, enc_src_len)
            src_enc, src_mask = self._expand_enc_out(src_enc, src_mask, beam_size)
        else:
            src_enc, src_mask = None, None

        gen_prefix_beams = gen_prefix.unsqueeze(1).expand(-1, beam_size, -1).reshape(total_beams, -1)
        gen_prefix_len_beams = gen_prefix_len.unsqueeze(1).expand(-1, beam_size).reshape(total_beams)
        logits, gen_state = self._prefill(task, gen_prefix_beams, gen_prefix_len_beams, max_new_tokens, src_enc, src_mask)

        done = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        finished_scores = torch.full((batch_size, beam_size), -1e9, device=self.device)
        finished_tokens = torch.full((batch_size, beam_size, max_new_tokens), self.pad_index, dtype=torch.long, device=self.device)
        finished_lengths = torch.zeros(batch_size, beam_size, dtype=torch.long, device=self.device)

        beam_scores = torch.zeros(total_beams, device=self.device)
        beam_scores.view(batch_size, beam_size)[:, 1:] = -1e9

        generated_tokens = torch.full((total_beams, max_new_tokens), self.pad_index, dtype=torch.long, device=self.device)
        gen_len = torch.zeros(total_beams, dtype=torch.long, device=self.device)

        ones = torch.ones(total_beams, dtype=torch.long, device=self.device)
        sent_offsets = torch.arange(batch_size, device=self.device).unsqueeze(1) * beam_size
        orig_indices = torch.arange(total_beams, device=self.device).view(batch_size, beam_size)

        for step in range(max_new_tokens):
            if done.all():
                break

            if step == 0:
                step_logits = logits[:, -1, :]
            else:
                logits, gen_state = self._generate_step(task, current_tokens, ones, src_enc, src_mask, gen_state)
                step_logits = logits[:, -1, :]
            if use_sampling:
                step_logits = self._filter_logits(step_logits, temperature, top_k, top_p)
            log_probs = F.log_softmax(step_logits, dim=-1)

            next_scores = (log_probs + beam_scores[:, None]).view(batch_size, beam_size * n_words)
            topk_scores, topk_flat = torch.topk(next_scores, 2 * beam_size, dim=1)

            topk_beam_ids = topk_flat // n_words
            topk_word_ids = topk_flat % n_words
            topk_source_beams = sent_offsets + topk_beam_ids

            is_eos = topk_word_ids == self.eos_index
            is_last = step + 1 == max_new_tokens
            is_finished_candidate = is_eos | is_last

            flat_sources = topk_source_beams.reshape(-1)
            candidate_lengths = gen_len[flat_sources].reshape(batch_size, 2 * beam_size)

            hyp_len = candidate_lengths.float().clone()
            need_append = ~is_eos & is_finished_candidate
            hyp_len[need_append] += 1
            hyp_len.clamp_(min=1)

            lp_scores = topk_scores / hyp_len**length_penalty
            lp_scores[~is_finished_candidate] = -1e9
            if done.any():
                lp_scores[done.unsqueeze(1).expand_as(lp_scores)] = -1e9

            candidate_tokens = generated_tokens[flat_sources].reshape(batch_size, 2 * beam_size, max_new_tokens)
            scatter_pos = candidate_lengths.unsqueeze(-1).clamp(max=max_new_tokens - 1)
            candidate_tokens.scatter_(2, scatter_pos, topk_word_ids.unsqueeze(-1))

            candidate_finished_lengths = candidate_lengths.clone()
            candidate_finished_lengths[need_append] += 1

            all_f_scores = torch.cat([finished_scores, lp_scores], dim=1)
            all_f_tokens = torch.cat([finished_tokens, candidate_tokens], dim=1)
            all_f_lengths = torch.cat([finished_lengths, candidate_finished_lengths], dim=1)

            top_f_scores, top_f_idx = torch.topk(all_f_scores, beam_size, dim=1)
            finished_scores = top_f_scores
            finished_tokens = all_f_tokens.gather(1, top_f_idx.unsqueeze(-1).expand(-1, -1, max_new_tokens))
            finished_lengths = all_f_lengths.gather(1, top_f_idx)

            active_scores = topk_scores.clone()
            active_scores[is_finished_candidate] = -1e9

            _, active_idx = torch.topk(active_scores, beam_size, dim=1)

            next_beam_scores_2d = topk_scores.gather(1, active_idx)
            next_beam_words_2d = topk_word_ids.gather(1, active_idx)
            next_beam_sources_2d = topk_source_beams.gather(1, active_idx)

            if done.any():
                done_mask = done.unsqueeze(1).expand(-1, beam_size)
                next_beam_sources_2d[done_mask] = orig_indices[done_mask]
                next_beam_scores_2d[done_mask] = beam_scores.view(batch_size, beam_size)[done_mask]
                next_beam_words_2d[done_mask] = self.pad_index

            n_finished_per_sent = (finished_scores > -1e8).sum(dim=1)
            has_enough = n_finished_per_sent >= beam_size

            best_active = active_scores.max(dim=1).values
            worst_finished = finished_scores.min(dim=1).values

            best_possible_lp = best_active / max(step + 1, 1) ** length_penalty
            no_improvement_possible = worst_finished >= best_possible_lp
            done = done | (has_enough & no_improvement_possible)

            next_beam_indices = next_beam_sources_2d.reshape(-1)
            next_beam_words = next_beam_words_2d.reshape(-1)
            beam_scores = next_beam_scores_2d.reshape(-1)

            current_tokens = next_beam_words.unsqueeze(1)
            gen_state = self._reorder_gen_state(gen_state, next_beam_indices)

            generated_tokens = generated_tokens[next_beam_indices]
            gen_len = gen_len[next_beam_indices]
            active_mask = next_beam_words != self.pad_index
            generated_tokens[active_mask, gen_len[active_mask]] = next_beam_words[active_mask]
            gen_len[active_mask] += 1

        generated_indices = finished_scores.argsort(dim=1, descending=True)
        generated_scores = finished_scores.gather(1, generated_indices)
        generated_tokens = finished_tokens.gather(1, generated_indices.unsqueeze(-1).expand(-1, -1, max_new_tokens))
        generated_lengths = finished_lengths.gather(1, generated_indices)
        return generated_tokens, generated_scores, generated_lengths
