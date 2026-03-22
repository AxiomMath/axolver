import contextlib
import os
import sys
import time
from logging import getLogger

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from src.envs.environment import create_train_iterator
from src.optim import get_optimizer

logger = getLogger()


def _unwrap_model(model):
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    if hasattr(model, "module"):
        model = model.module
    return model


def is_bf16_supported(device):
    try:
        torch.tensor([1.0], dtype=torch.bfloat16, device=device)
        return True
    except Exception:
        return False


def is_fp16_supported(device):
    try:
        torch.tensor([1.0], dtype=torch.float16, device=device)
        return True
    except Exception:
        return False


def default_dtype(device, amp):
    if amp:
        if is_bf16_supported(device):
            return torch.bfloat16
        elif is_fp16_supported(device):
            return torch.float16
    return torch.float32


def setup_amp(device, dtype):
    use_amp = dtype in (torch.float16, torch.bfloat16)
    if not use_amp:
        return contextlib.nullcontext()
    return torch.amp.autocast(device_type=device, dtype=dtype, enabled=True)


class Trainer:
    def __init__(self, model, env, params):
        self.model = model
        self.params = params
        self.env = env
        self.task = params.task

        assert params.report_loss_every > 0

        # distributed training
        if params.multi_gpu:
            logger.info("Using nn.parallel.DistributedDataParallel ...")
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[params.local_rank], output_device=params.local_rank, broadcast_buffers=True
            )

        self.optimizer, self.scheduler = get_optimizer(_unwrap_model(self.model).parameters(), params.optimizer)

        self.dtype = default_dtype(params.device, params.amp)
        self.ctx = setup_amp(params.device, self.dtype)
        self.scaler = torch.amp.GradScaler(device=params.device, enabled=(self.dtype == torch.float16))

        # stopping criterion used for early stopping
        if params.stopping_criterion != "":
            split = params.stopping_criterion.split(",")
            assert len(split) == 2 and split[1].isdigit()
            self.decrease_counts_max = int(split[1])
            self.decrease_counts = 0
            if split[0][0] == "_":
                self.stopping_criterion = (split[0][1:], False)
            else:
                self.stopping_criterion = (split[0], True)
            self.best_stopping_criterion = -1e12 if self.stopping_criterion[1] else 1e12
        else:
            self.stopping_criterion = None
            self.best_stopping_criterion = None

        # validation metrics
        self.metrics = []
        metrics = [m for m in params.validation_metrics.split(",") if m != ""]
        for m in metrics:
            # Metrics prefixed with "_" are minimized (e.g. "_loss"), others are maximized.
            if m.startswith("_"):
                self.metrics.append((m[1:], False))
            else:
                self.metrics.append((m, True))
        self.best_metrics = {metric: (-1e12 if biggest else 1e12) for (metric, biggest) in self.metrics}

        self.epoch_size = params.epoch_size

        # training statistics
        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.n_equations = 0
        self.stats = {"processed_sequences": 0, "processed_tokens": 0, "loss": []}
        self.last_time = time.time()

        # reload potential checkpoints
        self.reload_checkpoint()

        # reload exported data
        self.data_path = None
        if params.reload_data != "":
            task_name, path = params.reload_data.split(":")
            assert task_name == self.task, f"Task '{task_name}' in reload_data does not match task '{self.task}'"
            assert os.path.isfile(path), f"Data file not found: {path}"
            self.data_path = path

        # open export files if needed
        self.export_files = {}
        if params.export_data and params.is_master:
            export_path = os.path.join(params.dump_path, f"{self.task}.data.prefix")
            self.export_files[self.task] = open(export_path, "w")

        # create data loader
        if not params.eval_only:
            if params.env_base_seed < 0:
                params.env_base_seed = np.random.randint(1_000_000_000)
            self.dataloader = iter(create_train_iterator(self.env, self.task, self.data_path, params))

    def optimize(self, loss):
        if (loss != loss).data.any():
            logger.info("NaN detected")
            self.save_checkpoint("checkpoint_nan")
            exit()

        self.scaler.scale(loss).backward()

        if self.params.clip_grad_norm > 0:
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(_unwrap_model(self.model).parameters(), self.params.clip_grad_norm)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        if self.scheduler is not None:
            self.scheduler.step()

    def iter(self):
        self.n_iter += 1
        self.n_total_iter += 1
        if self.n_total_iter % self.params.report_loss_every == 0:
            self.print_stats()

    def print_stats(self):
        s_iter = f"{self.n_total_iter:7d} - "
        s_stat = " || ".join([f"{k.upper().replace('_', '-')}: {np.mean(v):7.4f}" for k, v in self.stats.items() if type(v) is list and len(v) > 0])
        for k in self.stats.keys():
            if type(self.stats[k]) is list:
                del self.stats[k][:]

        # learning rate
        s_lr = " - LR: " + " / ".join(f"{group['lr']:.4e}" for group in self.optimizer.param_groups)

        # processing speed
        new_time = time.time()
        diff = new_time - self.last_time
        s_speed = f"{self.stats['processed_sequences'] * 1.0 / diff:7.2f} examples/s - {self.stats['processed_tokens'] * 1.0 / diff:8.2f} words/s - "
        self.stats["processed_sequences"] = 0
        self.stats["processed_tokens"] = 0
        self.last_time = new_time

        # log speed + stats + learning rate
        logger.info(s_iter + s_speed + s_stat + s_lr)

    def reset_epoch_stats(self):
        self.n_equations = 0
        self.model.train()

    def save_checkpoint(self, name):
        if not self.params.is_master:
            return

        path = os.path.join(self.params.dump_path, f"{name}.pth")
        logger.info(f"Saving {name} to {path} ...")

        data = {
            "model": _unwrap_model(self.model).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "scaler": self.scaler.state_dict(),
            "epoch": self.epoch,
            "n_total_iter": self.n_total_iter,
            "best_metrics": self.best_metrics,
            "best_stopping_criterion": self.best_stopping_criterion,
            "decrease_counts": getattr(self, "decrease_counts", 0),
            "params": {k: v for k, v in self.params.__dict__.items()},
        }
        torch.save(data, path)

    def reload_checkpoint(self):
        auto_path = os.path.join(self.params.dump_path, "checkpoint.pth")
        if os.path.isfile(auto_path):
            checkpoint_path = auto_path
        elif self.params.reload_checkpoint != "":
            checkpoint_path = self.params.reload_checkpoint
            assert os.path.isfile(checkpoint_path)
        else:
            return

        logger.info(f"Reloading checkpoint from {checkpoint_path} ...")
        data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        _unwrap_model(self.model).load_state_dict(data["model"])

        self.optimizer.load_state_dict(data["optimizer"])
        if self.scheduler is not None and data.get("scheduler") is not None:
            self.scheduler.load_state_dict(data["scheduler"])
        if "scaler" in data:
            self.scaler.load_state_dict(data["scaler"])
        self.epoch = data["epoch"] + 1
        self.n_total_iter = data["n_total_iter"]
        self.best_metrics = data["best_metrics"]
        self.best_stopping_criterion = data["best_stopping_criterion"]
        if hasattr(self, "decrease_counts"):
            self.decrease_counts = data.get("decrease_counts", 0)
        logger.info(f"Checkpoint reloaded. Resuming at epoch {self.epoch} / iteration {self.n_total_iter} ...")

    def save_best_model(self, scores):
        if not self.params.is_master:
            return
        for metric, biggest in self.metrics:
            if metric not in scores:
                logger.info(f'Metric "{metric}" not found in scores!')
                continue
            factor = 1 if biggest else -1
            if factor * scores[metric] > factor * self.best_metrics[metric]:
                self.best_metrics[metric] = scores[metric]
                logger.info(f"New best score for {metric}: {scores[metric]:.6f}")
                self.save_checkpoint(f"best-{metric}")

    def end_epoch(self, scores):
        # stop if the stopping criterion has not improved after a certain number of epochs
        if self.stopping_criterion is not None and self.params.is_master:
            metric, biggest = self.stopping_criterion
            assert metric in scores, metric
            factor = 1 if biggest else -1
            if factor * scores[metric] > factor * self.best_stopping_criterion:
                self.best_stopping_criterion = scores[metric]
                logger.info(f"New best validation score: {self.best_stopping_criterion}")
                self.decrease_counts = 0
            else:
                logger.info(f"Not a better validation score ({self.decrease_counts} / {self.decrease_counts_max}).")
                self.decrease_counts += 1
            if self.decrease_counts >= self.decrease_counts_max:
                logger.info(
                    f"Stopping criterion has been below its best value for more " f"than {self.decrease_counts_max} epochs. Ending the experiment..."
                )
                if self.params.multi_gpu and "SLURM_JOB_ID" in os.environ:
                    os.system("scancel " + os.environ["SLURM_JOB_ID"])
                exit()
        self.save_checkpoint("checkpoint")
        self.epoch += 1

    def get_batch(self):
        try:
            batch = next(self.dataloader)
        except Exception as e:
            logger.error(
                f"An unknown exception of type {type(e).__name__} occurred in line {sys.exc_info()[-1].tb_lineno} when fetching batch. Arguments:{e.args!r}. Restarting ..."
            )
            if self.params.is_master and "SLURM_JOB_ID" in os.environ:
                logger.info(f"Requeuing job {os.environ['SLURM_JOB_ID']}")
                os.system(f"scontrol requeue {os.environ['SLURM_JOB_ID']}")
            raise

        return batch

    def export_data(self):
        problem_list, question_list, answer_list = self.get_batch()
        for problem_tok, question_tok, answer_tok in zip(problem_list, question_list, answer_list):
            if len(problem_tok) == 0 or len(answer_tok) == 0:
                continue
            f = self.export_files[self.task]
            if question_tok:
                f.write(" ".join(problem_tok) + "\t" + " ".join(question_tok) + "\t" + " ".join(answer_tok) + "\n")
            else:
                f.write(" ".join(problem_tok) + "\t" + " ".join(answer_tok) + "\n")
        self.n_equations += len(problem_list)

    def close_export_files(self):
        for f in self.export_files.values():
            f.close()
        self.export_files = {}

    def enc_dec_step(self):
        params = self.params
        device = params.device

        (enc_src, enc_src_len), (dec_tgt, dec_tgt_len), prefix_len = self.get_batch()

        bs = dec_tgt_len.size(0)
        if params.architecture == "decoder_only":
            n_tokens = (dec_tgt_len - 1).sum().item()
        elif params.architecture == "encoder_only":
            n_tokens = (enc_src_len - 1).sum().item()
        else:
            n_tokens = (enc_src_len + dec_tgt_len - 2).sum().item()

        non_blocking = device == "cuda"
        if enc_src is not None:
            enc_src = enc_src.to(device, non_blocking=non_blocking)
        if enc_src_len is not None:
            enc_src_len = enc_src_len.to(device, non_blocking=non_blocking)
        dec_tgt, dec_tgt_len = dec_tgt.to(device, non_blocking=non_blocking), dec_tgt_len.to(device, non_blocking=non_blocking)
        prefix_len = prefix_len.to(device, non_blocking=non_blocking)

        with self.ctx:
            _, loss = self.model(enc_src, enc_src_len, dec_tgt, dec_tgt_len, prefix_len=prefix_len, task=self.task)

        self.stats["loss"].append(loss.item())

        # optimize
        self.optimize(loss)

        # number of processed sequences / words
        self.n_equations += bs
        self.stats["processed_sequences"] += bs
        self.stats["processed_tokens"] += n_tokens
