import inspect
import re

from torch import optim
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingWarmRestarts, LambdaLR, LinearLR, SequentialLR


def get_optimizer(parameters, s):
    """
    Parse optimizer parameters and return (optimizer, scheduler).
    Input should be of the form:
        - "adam,lr=0.001"
        - "adam_warmup,lr=0.001,warmup_updates=4000"
        - "adam_cosine,lr=0.001,warmup_updates=4000,init_period=100000"
    """
    if "," in s:
        method = s[: s.find(",")]
        all_params = {}
        for x in s[s.find(",") + 1 :].split(","):
            split = x.split("=")
            assert len(split) == 2
            assert re.match(r"^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
            all_params[split[0]] = float(split[1])
    else:
        method = s
        all_params = {}

    scheduler_keys = {"warmup_updates", "warmup_init_lr", "exp_factor", "min_lr", "init_period", "period_mult", "lr_shrink", "lr_shrink_min"}
    sched_params = {k: all_params.pop(k) for k in list(all_params) if k in scheduler_keys}

    schedule = None
    if method.startswith("adam_warmup"):
        schedule = "warmup"
        method = "adam"
    elif method.startswith("adam_inverse_sqrt"):
        schedule = "inverse_sqrt"
        method = "adam"
    elif method.startswith("adam_cosine") or method.startswith("adam_smooth_cosine"):
        schedule = "cosine"
        method = "adam"

    OPTIMIZERS = {
        "adadelta": optim.Adadelta,
        "adagrad": optim.Adagrad,
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "adamax": optim.Adamax,
        "asgd": optim.ASGD,
        "rmsprop": optim.RMSprop,
        "rprop": optim.Rprop,
        "sgd": optim.SGD,
    }

    if method not in OPTIMIZERS:
        raise ValueError(f'Unknown optimization method: "{method}"')
    optim_fn = OPTIMIZERS[method]

    if "beta1" in all_params or "beta2" in all_params:
        all_params["betas"] = (all_params.pop("beta1", 0.9), all_params.pop("beta2", 0.999))

    if method in ("adam", "adamw"):
        all_params.setdefault("fused", True)

    sig = inspect.signature(optim_fn)
    expected_args = [p for p in sig.parameters if p != "self"]
    if not all(k in expected_args for k in all_params):
        raise ValueError(f"Unexpected parameters: expected {expected_args}, got {list(all_params)}")

    optimizer = optim_fn(parameters, **all_params)
    scheduler = build_scheduler(optimizer, schedule, sched_params, all_params.get("lr", 1e-3))

    return optimizer, scheduler


def build_scheduler(optimizer, schedule, sched_params, lr):
    """
    Build a learning rate scheduler.
    Returns None for constant LR.
    """
    if schedule is None:
        return None

    warmup_steps = int(sched_params.get("warmup_updates", 4000))
    warmup_init_lr = sched_params.get("warmup_init_lr", 1e-7)
    start_factor = warmup_init_lr / lr

    warmup = LinearLR(optimizer, start_factor=start_factor, total_iters=warmup_steps)

    if schedule == "warmup":
        constant = ConstantLR(optimizer, factor=1.0, total_iters=0)
        return SequentialLR(optimizer, [warmup, constant], milestones=[warmup_steps])

    elif schedule == "inverse_sqrt":
        exp_factor = sched_params.get("exp_factor", 0.5)
        decay_factor = lr * warmup_steps**exp_factor

        def inv_sqrt_lr(step):
            t = max(step + warmup_steps, 1)
            return (decay_factor * t**-exp_factor) / lr

        decay = LambdaLR(optimizer, lr_lambda=inv_sqrt_lr)
        return SequentialLR(optimizer, [warmup, decay], milestones=[warmup_steps])

    elif schedule == "cosine":
        init_period = int(sched_params.get("init_period", 1_000_000))
        period_mult = int(sched_params.get("period_mult", 1))
        min_lr = sched_params.get("min_lr", 1e-9)

        cosine = CosineAnnealingWarmRestarts(optimizer, T_0=init_period, T_mult=period_mult, eta_min=min_lr)
        return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])

    else:
        raise ValueError(f'Unknown schedule: "{schedule}"')
