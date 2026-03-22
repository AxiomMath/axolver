import os
from logging import getLogger

import torch
import torch.distributed as dist

logger = getLogger()


def init_distributed_mode(params):
    """
    Initialize distributed training via torchrun environment variables.
    Usage: torchrun --nproc_per_node=N script.py
    """
    if "RANK" in os.environ:
        params.local_rank = int(os.environ["LOCAL_RANK"])
        params.global_rank = int(os.environ["RANK"])
        params.world_size = int(os.environ["WORLD_SIZE"])
        params.n_gpu_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", params.world_size))
    else:
        params.local_rank = 0
        params.global_rank = 0
        params.world_size = 1
        params.n_gpu_per_node = 1

    params.multi_gpu = params.world_size > 1
    params.is_master = params.global_rank == 0

    if params.multi_gpu:
        torch.cuda.set_device(params.local_rank)
        dist.init_process_group(backend="nccl")

    logger.info(f"Rank {params.global_rank}/{params.world_size}, local_rank={params.local_rank}, is_master={params.is_master}")
