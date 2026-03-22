import os
import queue
import threading
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from itertools import repeat
from logging import getLogger

import torch

from src.envs.environment import create_test_iterator
from src.trainer import _unwrap_model

logger = getLogger()


def check_hypothesis(eq, env):
    eq["hyp_tokens"] = [env.id2word[wid] for wid in eq["hyp"]]
    eq["metrics"] = env.check_prediction(eq["problem_data"], eq["question_data"], eq["answer_data"], eq["hyp_tokens"], metrics=eq["metrics_to_eval"])
    return eq


def _process_batch(single_batch, env, params, is_beam, task, display_logs, metrics, metrics_convention, executor=None):
    valid = single_batch["greedy_valid"].long()
    gen_tokens, gen_lengths, gen_scores = single_batch["gen_tokens"], single_batch["gen_lengths"], single_batch["gen_scores"]
    bs = single_batch["bs"]

    n_perfect_match = valid.sum().item()
    batch_n_well_formed = 0
    batch_metrics = []

    beam_log = {}
    for i in range(bs):
        problem_infix = single_batch["problem_data_list"][i]
        question_infix = single_batch["question_data_list"][i]
        answer_infix = single_batch["answer_data_list"][i]

        if valid[i] and params.eval_verbose < 2:
            beam_log[i] = {"problem": problem_infix, "question": question_infix, "answer": answer_infix, "hyps": [(answer_infix, None, True)]}
            batch_metrics.append({})
            continue

        hyp_inputs = []
        if is_beam:
            for j in range(gen_tokens.size(1)):
                hyp_len = gen_lengths[i, j].item()
                hyp_inputs.append(
                    {
                        "score": gen_scores[i, j].item(),
                        "problem_data": single_batch["problem_data_list"][i],
                        "question_data": single_batch["question_data_list"][i],
                        "answer_data": single_batch["answer_data_list"][i],
                        "hyp": gen_tokens[i, j, :hyp_len].tolist(),
                        "task": task,
                        "metrics_to_eval": metrics,
                    }
                )
        else:
            hyp_len = gen_lengths[i].item()
            gen_i = gen_tokens[i].tolist()[:hyp_len]
            hyp_inputs.append(
                {
                    "problem_data": single_batch["problem_data_list"][i],
                    "question_data": single_batch["question_data_list"][i],
                    "answer_data": single_batch["answer_data_list"][i],
                    "hyp": gen_i,
                    "task": task,
                    "metrics_to_eval": metrics,
                    # Greedy decoding has no beam score
                    "score": float("-inf"),
                }
            )

        if executor is not None:
            gens = list(executor.map(check_hypothesis, hyp_inputs, repeat(env)))
        else:
            gens = [check_hypothesis(inp, env) for inp in hyp_inputs]
        if is_beam:
            gens.sort(key=lambda x: x["score"], reverse=True)

        beam_log[i] = {"problem": problem_infix, "question": question_infix, "answer": answer_infix, "hyps": []}
        best_well_formed, best_valid = 0, 0
        best_hyp_metrics = {}

        for gen in gens:
            iv = gen["metrics"]["is_valid"]
            beam_log[i]["hyps"].append((gen["hyp"], gen["score"], iv > 0))
            if not valid[i]:
                if iv >= 0:
                    best_well_formed = 1
                if iv > 0:
                    best_valid = 1
            for k, v in gen["metrics"].items():
                if k == "is_valid":
                    continue
                biggest = metrics_convention.get(k, True)
                if k not in best_hyp_metrics:
                    best_hyp_metrics[k] = v
                elif biggest:
                    best_hyp_metrics[k] = max(best_hyp_metrics[k], v)
                else:
                    best_hyp_metrics[k] = min(best_hyp_metrics[k], v)

        if not valid[i]:
            batch_n_well_formed += best_well_formed
            valid[i] = best_valid
        batch_metrics.append(best_hyp_metrics)

    if params.eval_verbose:
        assert len(beam_log) == bs
        display_logs(beam_log, offset=single_batch["total_so_far"] - bs)

    return n_perfect_match, batch_n_well_formed, valid, batch_metrics


class _CpuSink:
    def __init__(self, fn, decouple=False):
        self._fn = fn
        self._decouple = decouple
        self._queue = None
        self._thread = None
        self._error = None

    def start(self):
        if not self._decouple:
            return
        self._queue = queue.Queue()

        def consumer():
            try:
                while True:
                    item = self._queue.get()
                    if item is None:
                        break
                    self._fn(*item)
            except Exception as e:
                self._error = e

        self._thread = threading.Thread(target=consumer, daemon=True)
        self._thread.start()

    def submit(self, *args):
        if self._decouple:
            if self._error is not None:
                raise self._error
            self._queue.put(args)
        else:
            self._fn(*args)

    def join(self):
        if self._decouple:
            self._queue.put(None)
            self._thread.join()
            if self._error is not None:
                raise self._error


@contextmanager
def cpu_sink(fn, decouple=False):
    sink = _CpuSink(fn, decouple)
    sink.start()
    try:
        yield sink
    finally:
        sink.join()


class Evaluator:
    def __init__(self, trainer):
        self.trainer = trainer
        self.model = trainer.model
        self.params = trainer.params
        self.env = trainer.env
        self.ctx = trainer.ctx
        self.task = trainer.task

        # Parse metrics_eval with trainer.py convention:
        # Metrics prefixed with "_" are minimized, others are maximized.
        self.metrics = []
        self.metrics_convention = {}
        if self.params.metrics_eval:
            for m in self.params.metrics_eval.split(","):
                if m.startswith("_"):
                    name = m[1:]
                    self.metrics_convention[name] = False
                else:
                    name = m
                    self.metrics_convention[name] = True
                self.metrics.append(name)

    def run_all_evals(self):
        scores = OrderedDict({"epoch": self.trainer.epoch})

        data_type_list = ["valid"]
        if self.params.eval_data != "":
            for i in range(1, len(self.params.eval_data.split(","))):
                data_type_list.append(f"test{i}")

        for data_type in data_type_list:
            self.enc_dec_step(data_type, self.task, scores)
        return scores

    def enc_dec_step(self, data_type, task, scores):
        params = self.params
        env = self.env
        decoder_only = params.architecture == "decoder_only"
        encoder_only = params.architecture == "encoder_only"

        is_beam = params.beam_eval and params.beam_size > 1 and not encoder_only
        # +2 to account for BOS and EOS tokens
        max_beam_length = params.max_output_len + 2

        model = _unwrap_model(self.model)
        model.eval()

        decouple = params.decouple_cpu_gpu and params.device != "cpu"

        # evaluation details
        if params.eval_verbose:
            beam_str = "beam." if params.beam_eval else ""
            eval_path = os.path.join(params.dump_path, f"eval.{beam_str}{data_type}.{task}.{scores['epoch']}")
            f_export = open(eval_path, "w")
            logger.info(f"Writing evaluation results in {eval_path} ...")

        def display_logs(logs, offset):
            if params.eval_verbose == 0:
                return
            for i, res in sorted(logs.items()):
                n_valid = sum([int(v) for _, _, v in res["hyps"]])
                s = f"Equation {offset + i} ({n_valid}/{len(res['hyps'])})\n"
                s += f"problem={res['problem']}\n"
                if res["question"] is not None:
                    s += f"question={res['question']}\n"
                s += f"answer={res['answer']}\n"
                for hyp, score, valid in res["hyps"]:
                    if score is None:
                        s += f"{int(valid)} {hyp}\n"
                    else:
                        s += f"{int(valid)} {score :.3e} {hyp}\n"
                f_export.write(s + "\n")
                f_export.flush()

        # stats
        stats = {
            "xe_loss": 0,
            "n_tokens": 0,
            "n_perfect_match": 0,
            "n_well_formed": 0,
            "n_valid": OrderedDict(),
            "n_total": OrderedDict(),
            "metrics_sum": {},
            "metrics_count": {},
        }
        iterator = create_test_iterator(
            env, task, data_type, data_path=params.eval_data.split(",") if params.eval_data != "" else None, params=params
        )
        eval_size = len(iterator.dataset)

        hyp_executor = None
        if params.process_pool:
            hyp_executor = ProcessPoolExecutor(max_workers=min(params.num_workers, 8))

        def process_and_accumulate(single_batch):
            for cid in single_batch["class_id"]:
                stats["n_total"][cid] = stats["n_total"].get(cid, 0) + 1
            single_batch["total_so_far"] = sum(stats["n_total"].values())
            bs = single_batch["bs"]

            greedy_valid = single_batch["greedy_valid"].sum().item()
            logger.info(f"({single_batch['total_so_far']}/{eval_size}) Found {greedy_valid}/{bs} valid top-1 predictions.")

            n_perfect_match, batch_n_well_formed, valid, batch_metrics = _process_batch(
                single_batch,
                env=env,
                params=params,
                is_beam=is_beam,
                task=task,
                display_logs=display_logs,
                metrics=self.metrics,
                metrics_convention=self.metrics_convention,
                executor=hyp_executor,
            )

            stats["xe_loss"] += single_batch["loss_val"] * single_batch["n_batch_tokens"]
            stats["n_tokens"] += single_batch["n_batch_tokens"]
            stats["n_perfect_match"] += n_perfect_match
            stats["n_well_formed"] += batch_n_well_formed
            for cid, v in zip(single_batch["class_id"], valid.tolist()):
                stats["n_valid"][cid] = stats["n_valid"].get(cid, 0) + v

            for m_dict in batch_metrics:
                for k, v in m_dict.items():
                    stats["metrics_sum"][k] = stats["metrics_sum"].get(k, 0.0) + v
                    stats["metrics_count"][k] = stats["metrics_count"].get(k, 0) + 1

            logger.info(f"Found {valid.sum().item()}/{bs} solutions in hypotheses.")

        with cpu_sink(process_and_accumulate, decouple=decouple) as sink:
            device = params.device
            for batch in iterator:
                (
                    (enc_problem, enc_problem_len),
                    (dec_tgt, dec_tgt_len),
                    prefix_len,
                    (gen_prefix, gen_prefix_len),
                    (ref_answer, ref_answer_len),
                    class_id,
                    problem_data_list,
                    question_data_list,
                    answer_data_list,
                ) = batch

                if enc_problem is not None:
                    enc_problem = enc_problem.to(device)
                if enc_problem_len is not None:
                    enc_problem_len = enc_problem_len.to(device)
                dec_tgt, dec_tgt_len = dec_tgt.to(device), dec_tgt_len.to(device)
                prefix_len = prefix_len.to(device)
                ref_answer, ref_answer_len = ref_answer.to(device), ref_answer_len.to(device)
                gen_prefix, gen_prefix_len = gen_prefix.to(device), gen_prefix_len.to(device)
                bs = dec_tgt.size(0)

                gpu_out = self._gpu_forward_and_generate(
                    model,
                    enc_problem,
                    enc_problem_len,
                    dec_tgt,
                    dec_tgt_len,
                    prefix_len,
                    ref_answer,
                    ref_answer_len,
                    gen_prefix,
                    gen_prefix_len,
                    encoder_only,
                    decoder_only,
                    is_beam,
                    max_beam_length,
                    task,
                )

                single_batch = {
                    "greedy_valid": gpu_out["greedy_valid"].cpu(),
                    "gen_tokens": gpu_out["gen_tokens"].cpu(),
                    "gen_scores": gpu_out["gen_scores"].cpu() if gpu_out["gen_scores"] is not None else None,
                    "gen_lengths": gpu_out["gen_lengths"].cpu(),
                    "loss_val": gpu_out["loss"].item(),
                    "n_batch_tokens": gpu_out["n_batch_tokens"],
                    "bs": bs,
                    "class_id": class_id,
                    "problem_data_list": problem_data_list,
                    "question_data_list": question_data_list,
                    "answer_data_list": answer_data_list,
                }
                sink.submit(single_batch)

        if hyp_executor is not None:
            hyp_executor.shutdown(wait=True)

        if params.eval_verbose:
            f_export.close()
            logger.info(f"Evaluation results written in {eval_path}")

        # scores
        _n_valid = sum(stats["n_valid"].values())
        _n_total = sum(stats["n_total"].values())
        assert _n_total == eval_size
        logger.info(f"{_n_valid}/{_n_total} ({100. * _n_valid / _n_total}%) equations were evaluated correctly.")

        scores[f"{data_type}_{task.upper()}_xe_loss"] = stats["xe_loss"] / stats["n_tokens"]
        scores[f"{data_type}_{task.upper()}_greedy_acc"] = 100.0 * stats["n_perfect_match"] / _n_total
        scores[f"{data_type}_{task.upper()}_acc"] = 100.0 * _n_valid / _n_total
        scores[f"{data_type}_{task.upper()}_well_formed"] = 100.0 * (stats["n_perfect_match"] + stats["n_well_formed"]) / _n_total

        for cid, total in stats["n_total"].items():
            valid_i = stats["n_valid"].get(cid, 0)
            scores[f"{data_type}_{task.upper()}_acc_{cid}"] = 100.0 * valid_i / total
            logger.info(f"{cid}: {valid_i} / {total} ({100. * valid_i / total:.2f}%)")

        for metric_name, metric_sum in stats["metrics_sum"].items():
            if metric_name == "is_valid":
                continue
            count = stats["metrics_count"][metric_name]
            if count > 0:
                avg = metric_sum / count
                scores[f"{data_type}_{task.upper()}_avg_{metric_name}"] = avg
                logger.info(f"{metric_name}: {avg:.6f} (over {count} samples)")

    @torch.inference_mode()
    def _gpu_forward_and_generate(
        self,
        model,
        enc_problem,
        enc_problem_len,
        dec_tgt,
        dec_tgt_len,
        prefix_len,
        ref_answer,
        ref_answer_len,
        gen_prefix,
        gen_prefix_len,
        encoder_only,
        decoder_only,
        is_beam,
        max_beam_length,
        task,
    ):
        params = self.params
        device = params.device

        with self.ctx:
            logits, loss = model(enc_problem, enc_problem_len, dec_tgt, dec_tgt_len, prefix_len=prefix_len, task=task)
        preds = logits.argmax(-1)

        # Greedy
        n_answer = ref_answer.size(1)
        arange = torch.arange(n_answer, device=device).unsqueeze(0)
        answer_mask = arange < ref_answer_len.unsqueeze(1)

        gather_idx = (arange + (prefix_len - 1).unsqueeze(1)).clamp(max=preds.size(1) - 1)
        aligned_preds = preds.gather(1, gather_idx)

        correct = (aligned_preds == ref_answer) | ~answer_mask
        greedy_valid = correct.all(1)

        n_batch_tokens = answer_mask.sum().item()

        # Generation
        gen_scores = None
        with self.ctx:
            if encoder_only:
                gen_tokens, gen_lengths = model.decode(enc_problem, enc_problem_len, max_beam_length)
                # Strip the <ans> prefix prediction
                gen_tokens = gen_tokens[:, 1:]
                gen_lengths = (gen_lengths - 1).clamp(min=0)
            else:
                enc_src = None if decoder_only else enc_problem
                enc_src_len = None if decoder_only else enc_problem_len
                if is_beam:
                    gen_tokens, gen_scores, gen_lengths = model.beam_generate(
                        enc_src=enc_src,
                        enc_src_len=enc_src_len,
                        gen_prefix=gen_prefix,
                        gen_prefix_len=gen_prefix_len,
                        max_new_tokens=max_beam_length,
                        beam_size=params.beam_size,
                        length_penalty=params.beam_length_penalty,
                        temperature=params.temperature,
                        top_k=params.top_k,
                        top_p=params.top_p,
                        task=task,
                    )
                else:
                    gen_tokens, gen_lengths = model.generate(
                        enc_src=enc_src,
                        enc_src_len=enc_src_len,
                        gen_prefix=gen_prefix,
                        gen_prefix_len=gen_prefix_len,
                        max_new_tokens=max_beam_length,
                        temperature=params.temperature,
                        top_k=params.top_k,
                        top_p=params.top_p,
                        task=task,
                    )

        return {
            "loss": loss,
            "n_batch_tokens": n_batch_tokens,
            "greedy_valid": greedy_valid,
            "gen_tokens": gen_tokens,
            "gen_scores": gen_scores,
            "gen_lengths": gen_lengths,
        }
