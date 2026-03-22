import argparse
import os
import pickle
import shlex
import signal
import sys
import time

from src.logger import create_logger


class TimeoutError(Exception):
    pass


def timeout(seconds):
    def decorator(func):
        def handler(signum, frame):
            raise TimeoutError()

        def wrapper(*args, **kwargs):
            old_handler = signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            return result

        return wrapper

    return decorator


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() == "false":
        return False
    elif s.lower() == "true":
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")


def initialize_exp(params):
    """
    Initialize the experience:
    - dump parameters
    - create a logger
    """
    # dump parameters
    get_dump_path(params)
    with open(os.path.join(params.dump_path, "params.pkl"), "wb") as f:
        pickle.dump(params, f)

    command = shlex.join(sys.argv)
    params.command = command + f" --exp_id {shlex.quote(params.exp_id)}"

    # check experiment name
    assert len(params.exp_name.strip()) > 0

    # create a logger
    logger = create_logger(os.path.join(params.dump_path, "train.log"), rank=getattr(params, "global_rank", 0))
    logger.info("============ Initialized logger ============")
    logger.info("\n".join(f"{k}: {v}" for k, v in sorted(dict(vars(params)).items())))
    logger.info(f"The experiment will be stored in {params.dump_path}\n")
    logger.info(f"Running command: {command}")
    logger.info("")
    return logger


def get_dump_path(params):
    """
    Create a directory to store the experiment.
    """
    assert len(params.exp_name) > 0

    # create the sweep path if it does not exist
    sweep_path = os.path.join(params.dump_path, params.exp_name)
    os.makedirs(sweep_path, exist_ok=True)

    # create an ID for the job if it is not given in the parameters.
    # if we run on Modal, the job id is the timestamp of the run.
    # if we run on the cluster, the job ID is the one of SLURM.
    # otherwise, the job id is the timestamp of the run.
    exp_id = os.environ.get("MODAL_EXP_ID")
    if exp_id is not None:
        params.exp_id = exp_id
    elif params.exp_id == "":
        exp_id = os.environ.get("SLURM_JOB_ID")
        if exp_id is None:
            exp_id = time.strftime("%Y_%m_%d_%H_%M_%S")
        params.exp_id = exp_id

    # create the dump folder / update parameters
    params.dump_path = os.path.join(sweep_path, params.exp_id)
    os.makedirs(params.dump_path, exist_ok=True)
