import argparse
import json
import os
import pickle

import numpy as np
import torch

from src.envs import REGISTRY, build_env
from src.evaluator import Evaluator
from src.model import build_model, check_model_params
from src.slurm import init_distributed_mode
from src.trainer import Trainer
from src.utils import bool_flag, initialize_exp

np.seterr(all="raise")


def get_parser():
    """
    Generate a parameters parser.
    """
    parser = argparse.ArgumentParser(description="Language transfer", conflict_handler="resolve")

    # main parameters
    parser.add_argument("--dump_path", type=str, default="", help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")

    parser.add_argument("--report_loss_every", type=int, default=200, help="Log train loss every n optimisation steps")
    parser.add_argument("--epoch_size", type=int, default=300000, help="Epoch size / evaluation frequency")
    parser.add_argument("--max_epoch", type=int, default=100000, help="Maximum epoch size")
    parser.add_argument(
        "--stopping_criterion", type=str, default="", help="Stopping criterion, and number of non-increase before stopping the experiment"
    )
    parser.add_argument("--validation_metrics", type=str, default="", help="Validation metrics")

    # model parameters
    parser.add_argument("--model_type", type=str, default="transformer", help="transformer, lstm or gru")
    parser.add_argument("--architecture", type=str, default="encoder_decoder", help="encoder_decoder, encoder_only or decoder_only")
    parser.add_argument("--enc_emb_dim", type=int, default=256, help="Encoder embedding layer size")
    parser.add_argument("--dec_emb_dim", type=int, default=256, help="Decoder embedding layer size")
    parser.add_argument("--n_enc_layers", type=int, default=4, help="Number of Transformer layers in the encoder")
    parser.add_argument("--n_dec_layers", type=int, default=4, help="Number of Transformer layers in the decoder")
    parser.add_argument("--n_enc_heads", type=int, default=8, help="Number of Transformer encoder heads")
    parser.add_argument("--n_dec_heads", type=int, default=8, help="Number of Transformer decoder heads")
    parser.add_argument("--dropout", type=float, default=0, help="Dropout")
    parser.add_argument("--attention_dropout", type=float, default=0, help="Dropout in the attention layer")
    parser.add_argument("--norm", type=str, default="layernorm", help="Normalization: layernorm, rmsnorm")
    parser.add_argument("--activation", type=str, default="gelu", help="Activation function: relu, relu_squared, gelu")
    parser.add_argument("--n_enc_hidden_layers", type=int, default=1, help="Number of hidden layers in encoder FFN blocks")
    parser.add_argument("--n_dec_hidden_layers", type=int, default=1, help="Number of hidden layers in decoder FFN blocks")
    parser.add_argument("--enc_pos_emb", type=str, default="abs_learned", help="Encoder positional embedding: abs_sinusoidal, abs_learned, none")
    parser.add_argument("--dec_pos_emb", type=str, default="abs_learned", help="Decoder positional embedding: abs_sinusoidal, abs_learned, none")
    parser.add_argument("--share_inout_emb", type=bool_flag, default=True, help="Tie input and output embeddings")

    # sequence length
    parser.add_argument("--max_len", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--max_output_len", type=int, default=512, help="max length of output, beam max size")
    parser.add_argument("--max_src_len", type=int, default=0, help="Maximum number of tokens to consider in encoder output (0 to disable)")

    # loop layers
    parser.add_argument(
        "--enc_loop_idx", type=int, default=-1, help="Index of the encoder shared weight layers (-1 for none, -2 for all, or a valid layer index)"
    )
    parser.add_argument(
        "--dec_loop_idx", type=int, default=-1, help="Index of the decoder shared weight layers (-1 for none, -2 for all, or a valid layer index)"
    )
    parser.add_argument("--enc_loops", type=int, default=1, help="Fixed/max nr of train passes through the encoder loop")
    parser.add_argument("--dec_loops", type=int, default=1, help="Fixed/max nr of train passes through the decoder loop")

    # gates
    parser.add_argument("--gated", type=bool_flag, default=False, help="Gated loop layers")
    parser.add_argument("--enc_gated", type=bool_flag, default=False, help="All encoder layers gated")
    parser.add_argument("--dec_gated", type=bool_flag, default=False, help="All decoder layers gated")
    parser.add_argument("--scalar_gate", type=bool_flag, default=False, help="Scalar gates")
    parser.add_argument("--gate_bias", type=float, default=0, help="Gate_bias")

    # technical parameters
    parser.add_argument("--amp", type=bool_flag, default=False, help="Use automatic mixed precision (AMP) - should be 2x faster than float32")
    parser.add_argument("--num_workers", type=int, default=-1, help="Number of CPU workers for DataLoader (-1 to use os.cpu_count())")
    parser.add_argument("--env_base_seed", type=int, default=-1, help="Base seed for environments (-1 to use timestamp seed)")

    # CPU / multi-gpu / torch.compile
    parser.add_argument("--cpu", type=bool_flag, default=False, help="Run on CPU")
    parser.add_argument("--local_rank", type=int, default=-1, help="Multi-GPU - Local rank for torch.distributed.launch")
    parser.add_argument("--use_torch_compile", type=bool_flag, default=False, help="Use torch.compile to compile the model")

    # training parameters
    parser.add_argument("--eval_size", type=int, default=10000, help="Size of valid and test samples")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of sentences per batch")
    parser.add_argument("--batch_size_eval", type=int, default=128, help="Number of sentences per batch during evaluation")
    parser.add_argument("--optimizer", type=str, default="adam,lr=0.0001", help="Optimizer (SGD / RMSprop / Adam, etc.)")
    parser.add_argument("--clip_grad_norm", type=float, default=5, help="Clip gradients norm (0 to disable)")

    # export data / reload it
    parser.add_argument("--export_data", type=bool_flag, default=False, help="Export data and disable training.")
    parser.add_argument("--reload_data", type=str, default="", help="Data path. Format: task:path. If empty, data is generated on the fly.")
    parser.add_argument("--reload_size", type=int, default=-1, help="Reloaded training set size (-1 for everything, >0 enables batch loading)")
    parser.add_argument(
        "--index_dataset", type=bool_flag, default=False, help="Index the dataset and access it when needed instead of loading all on the RAM"
    )
    parser.add_argument("--max_examples", type=int, default=-1, help="Max number of examples to use from the dataset file (-1 for all)")
    parser.add_argument("--two_classes", type=bool_flag, default=False, help="Use two-class sampling for training data")
    parser.add_argument("--first_class_prob", type=float, default=0.5, help="Probability of sampling from the first class")
    parser.add_argument("--first_class_size", type=int, default=0, help="Size of the first class")

    # environment parameters
    parser.add_argument("--task", type=str, required=True, help="Task name")
    REGISTRY[parser.parse_known_args()[0].task]["register_args"](parser)

    # generation parameters
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for generation sampling")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling (0 to disable)")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling (1.0 to disable)")

    # beam search configuration
    parser.add_argument("--beam_eval", type=bool_flag, default=False, help="Evaluate with beam search decoding.")
    parser.add_argument("--beam_size", type=int, default=1, help="Beam size, default = 1 (greedy decoding)")
    parser.add_argument(
        "--beam_length_penalty",
        type=float,
        default=1,
        help="Length penalty, values < 1.0 favor shorter sentences, while values > 1.0 favor longer ones.",
    )

    # reload checkpoint
    parser.add_argument("--reload_checkpoint", type=str, default="", help="Reload from a checkpoint")

    # evaluation
    parser.add_argument("--metrics_eval", type=str, default="", help="Metrics to compute during evaluation. Format: metric1,metric2.")
    parser.add_argument("--eval_only", type=bool_flag, default=False, help="Only run evaluations")
    parser.add_argument("--eval_from_exp", type=str, default="", help="Path of experiment to use")
    parser.add_argument("--eval_data", type=str, default="", help="Path of data to eval")
    parser.add_argument("--eval_verbose", type=int, default=0, help="Export evaluation details")
    parser.add_argument(
        "--decouple_cpu_gpu", type=bool_flag, default=False, help="Overlap GPU generation with CPU hypothesis checking during evaluation"
    )
    parser.add_argument("--process_pool", type=bool_flag, default=False, help="Use a process pool for check_hypothesis during evaluation")

    return parser


def main(params):
    # CPU / CUDA / MPS
    params.device = "cpu" if params.cpu else ("mps" if torch.backends.mps.is_available() else "cuda")
    if params.device == "cuda":
        assert torch.cuda.is_available(), "CUDA requested but not available"
    elif params.device == "mps":
        assert torch.backends.mps.is_available(), "MPS requested but not available"

    # num_workers
    if params.num_workers == -1:
        params.num_workers = os.cpu_count()

    # initialize the multi-GPU / multi-node training
    init_distributed_mode(params)
    logger = initialize_exp(params)

    # build environment / model / trainer / evaluator
    env = build_env(params)
    use_torch_compile = params.use_torch_compile and not params.cpu
    model = build_model(params)
    # NOTE: torch.compile seems less efficient than no compilation
    if use_torch_compile:
        model = torch.compile(model, dynamic=True)
    trainer = Trainer(model, env, params)
    evaluator = Evaluator(trainer)

    # evaluation
    if params.eval_only:
        scores = evaluator.run_all_evals()
        for k, v in scores.items():
            logger.info(f"{k} -> {v:.6f}")
        logger.info(f"__log__:{json.dumps(scores)}")
        exit()

    # training
    for _ in range(params.max_epoch):
        logger.info(f"============ Starting epoch {trainer.epoch} ... ============")
        trainer.reset_epoch_stats()
        while trainer.n_equations < trainer.epoch_size:
            if params.export_data:
                trainer.export_data()
            else:
                trainer.enc_dec_step()
            trainer.iter()
        if params.export_data:
            trainer.close_export_files()
            logger.info("Exported data successfully")
            exit()

        if params.device == "cuda":
            logger.info(
                f"Memory allocated: {torch.cuda.memory_allocated(0)/(1024*1024):.2f}MB, reserved: {torch.cuda.memory_reserved(0)/(1024*1024):.2f}MB"
            )
        elif params.device == "mps":
            logger.info(
                f"Memory allocated: {torch.mps.current_allocated_memory()/(1024*1024):.2f}MB, reserved: {torch.mps.driver_allocated_memory()/(1024*1024):.2f}MB"
            )

        logger.info(f"============ End of epoch {trainer.epoch} ============")

        # evaluate perplexity
        scores = evaluator.run_all_evals()
        if params.device == "cuda":
            logger.info(
                f"Memory allocated: {torch.cuda.memory_allocated(0)/(1024*1024):.2f}MB, reserved: {torch.cuda.memory_reserved(0)/(1024*1024):.2f}MB"
            )
        elif params.device == "mps":
            logger.info(
                f"Memory allocated: {torch.mps.current_allocated_memory()/(1024*1024):.2f}MB, reserved: {torch.mps.driver_allocated_memory()/(1024*1024):.2f}MB"
            )

        if params.is_master:
            logger.info(f"__log__:{json.dumps(scores)}")

        # end of epoch
        trainer.save_best_model(scores)
        trainer.end_epoch(scores)


if __name__ == "__main__":

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()
    if params.eval_only and params.eval_from_exp != "":
        # read params from pickle
        pickle_file = params.eval_from_exp + "/params.pkl"
        assert os.path.isfile(pickle_file)
        with open(pickle_file, "rb") as f:
            pk = pickle.load(f)
        pickled_args = pk.__dict__
        del pickled_args["exp_id"]
        for k, v in pickled_args.items():
            if getattr(params, k, None) is None:
                setattr(params, k, v)

    # check parameters
    check_model_params(params)

    # run experiment
    main(params)
