[![](logo.svg)](https://axiommath.ai/)

# Axolver

## A modular framework for mathematics with transformers

Axolver is a framework for training sequence-to-sequence neural networks on mathematical problems. It ships with **26 ready-to-use tasks** spanning arithmetic, linear algebra, graph theory, symbolic calculus, and synthetic sequence manipulation. Data is generated on the fly, no datasets required.

Axolver is a complete rewrite of the [Int2Int](https://github.com/f-charton/Int2Int) codebase (arXiv: [2502.17513](https://arxiv.org/abs/2502.17513)).

## Setup

Axolver requires Python >= 3.10, PyTorch >= 2.0, and NumPy. Optional: SymPy (for integration tasks), Numba (for JIT acceleration in some generators).

```bash
git clone https://github.com/AxiomMath/axolver
cd axolver
pip install torch numpy sympy numba
```

**Hardware:** Axolver auto-detects NVIDIA GPUs (CUDA), Apple Silicon (MPS), or falls back to CPU. No configuration needed.

## My first experiment

Run this out of the box:

```bash
python train.py --task gcd --dump_path ./exp --exp_name my_first_gcd --exp_id 1 --base 10 --maxint 10000 --max_gcd 100
```

This trains a small transformer (7.6M parameters) to compute the greatest common divisor of two integers, with operands up to 10,000 and GCD values up to 100. Data is generated on the fly. Results are saved to `./exp/my_first_gcd/1/`.

If you don't have a GPU, add `--cpu true`. On a Mac with Apple Silicon, Axolver will automatically use the MPS backend.

### What you'll see

After printing the hyperparameters and vocabulary, the log reports training progress:

```
INFO - 03/16/26 10:02:30 - 0:00:07 -     200 -  975.98 examples/s - LOSS:  5.7409 - LR: 1.0000e-04
INFO - 03/16/26 10:02:35 - 0:00:12 -     400 - 1316.87 examples/s - LOSS:  1.1069 - LR: 1.0000e-04
INFO - 03/16/26 10:02:40 - 0:00:16 -     600 - 1330.12 examples/s - LOSS:  1.0175 - LR: 1.0000e-04
INFO - 03/16/26 10:02:45 - 0:00:21 -     800 - 1367.47 examples/s - LOSS:  0.9910 - LR: 1.0000e-04
INFO - 03/16/26 10:02:50 - 0:00:26 -    1000 - 1297.62 examples/s - LOSS:  0.9592 - LR: 1.0000e-04
```

Each line shows: date, elapsed time, optimisation steps (multiples of `--report_loss_every`, 200 by default), throughput, training loss, and learning rate. The loss drops from 5.74 to 0.96.

At the end of each epoch, the model is evaluated on 10,000 test examples with per-class breakdown:

```
INFO - 1655/10000 (16.55%) equations were evaluated correctly.
INFO - 1: 4 / 87 (4.60%)
INFO - 2: 4 / 96 (4.17%)
...
INFO - 20: 104 / 105 (99.05%)
INFO - 25: 51 / 72 (70.83%)
INFO - 50: 109 / 109 (100.00%)
INFO - 100: 105 / 105 (100.00%)
```

After one epoch: **16.55% overall accuracy**. Small GCD values (1, 2) are the hardest. Large GCD values that are multiples of 50 or 100 reach 100% immediately. Accuracy rises to **89.74%** after ten epochs.

## More examples

### Matrix operations

Train a model to transpose 4x4 integer matrices:

```bash
python train.py --task matrix_transpose --dump_path ./exp --exp_name transpose --exp_id 1 --dim1 4 --dim2 4 --maxint 5
```

For matrix eigenvalues, the output is a vector of floats, which requires specifying the float encoding parameters:

```bash
python train.py --task matrix_eigenvalues --dump_path ./exp --exp_name eigenvalues --exp_id 1 --dim1 3 --dim2 3 --maxint 5 --float_precision 3 --max_exponent 10
```

### Graph tasks

Graph tasks use a specialised graph tokenizer. To train on finding shortest paths in graphs with up to 8 nodes:

```bash
python train.py --task find_shortest_path --dump_path ./exp --exp_name shortest_path --exp_id 1 --max_nodes 8 --max_edges 15
```

The `find_shortest_path` task is an example of a task with a *query*: the graph is the problem, the source and target nodes are the query, and the shortest path is the answer.

### Symbolic integration

Symbolic integration uses expression trees, encoded with a dedicated symbolic sequence tokenizer:

```bash
python train.py --task integration --dump_path ./exp --exp_name integration --exp_id 1 --max_ops 15 --max_int 5 --n_variables 1
```

## Available tasks

Axolver ships with 26 tasks across six categories. All tasks generate data on the fly and require no external datasets.

### Arithmetic

These tasks operate on integers encoded in a configurable base (`--base`, default 10). Common parameters: `--minint` (default 1), `--maxint` (default 100).

| Task | Description |
|------|-------------|
| `gcd` | Greatest common divisor of two integers | 
| `fraction_simplify` | Simplify ka/kb to a/b |
| `fraction_round` | Compute floor(a/b) |
| `fraction_add` | Add a/b + c/d, return in lowest terms |
| `fraction_product` | Multiply a/b * c/d, return in lowest terms |
| `fraction_determinant` | Compute ad - bc from four integers |
| `fraction_compare` | Return 1 if a/b > c/d, 0 otherwise |
| `modular_add` | Compute (a + b) mod m |
| `modular_product` | Compute (a * b) mod m |

### Matrix

These tasks operate on integer matrices of dimension `--dim1` x `--dim2` with entries in [-maxint, maxint].

| Task | Description |
|------|-------------|
| `matrix_transpose` | Transpose a matrix |
| `matrix_sum` | Sum two matrices |
| `matrix_vector` | Multiply a matrix by a vector |
| `matrix_determinant` | Compute the determinant |
| `matrix_rank` | Compute the rank |
| `matrix_eigenvalues` | Eigenvalues of a symmetric matrix |
| `matrix_inverse` | Inverse of a symmetric matrix |

### Graph

These tasks operate on random graphs with at most `--max_nodes` nodes and `--max_edges` edges. Set `--weighted true` for weighted edges.

| Task | Description |
|------|-------------|
| `find_shortest_path` | Given a graph and a (source, target) query, return the shortest path as a node list |
| `laplacian_eigenvalues` | Eigenvalues of the graph Laplacian (output as floats; requires `--float_precision`, `--max_exponent`, `--base`) |
| `max_clique` | Find a maximum clique in the graph |

### Polynomial

| Task | Description |
|------|-------------|
| `polynomial_roots` | Compute roots of a polynomial from its coefficients |

### Symbolic

| Task | Description |
|------|-------------|
| `integration` | Symbolic integration of random expressions |

### Synthetic

Simple sequence manipulation tasks on symbolic tokens (integers 0 to n-1). Useful for testing and debugging. Parameters follow the pattern `--<task>_max_len` and `--<task>_n_tokens` (e.g. `--copy_max_len`, `--reverse_n_tokens`). 

| Task | Description |
|------|-------------|
| `copy` | Copy the input sequence |
| `reverse` | Reverse the input sequence |
| `sort` | Sort the input sequence |
| `parity` | Return the parity (sum mod 2) of the input |
| `deduplicate` | Remove duplicate tokens |
| `bracket_match` | Given a bracket sequence (0s and 1s), return matching indices |

## Adding a new task

Adding a new task takes three steps and does not require modifying any core code.

### Step 1: Write a generator

Create a class inheriting from `Generator` (in `src/envs/generators/base.py`). It must implement:

- `generate(rng, is_train)`: Returns a triple `(problem, question, answer)` where `question` can be `None`. Currently it's used in the find_shortest_path task where the problem is the graph, question is the tuple `(start, end)` and the answer is the shortest path between `start` and `end`. The `rng` argument is a NumPy random generator; use it for all randomness to ensure reproducibility.
- `evaluate(problem, question, answer, hyp, metrics)`: Returns a dict where `"is_valid"` is always present: 1 (correct), 0 (incorrect), or -1 (decoding error).
- Optionally, `encode_class_id(problem, question, answer)`: Returns an integer class ID for stratified evaluation metrics.

```python
from src.envs.generators.base import Generator

class AdditionGenerator(Generator):
    def __init__(self, params):
        self.maxint = params.maxint

    def generate(self, rng, is_train):
        a = int(rng.integers(1, self.maxint + 1))
        b = int(rng.integers(1, self.maxint + 1))
        return [a, b], None, a + b  # (problem, query, answer)

    def evaluate(self, problem, question, answer, hyp, metrics):
        if hyp is None:
            return {"is_valid": -1}
        if hyp == answer:
            return {"is_valid": 1}
        return {"is_valid": 0}
```

### Step 2: Choose tokenizers

Pick tokenizers for the problem and answer from the built-in library (`src/envs/tokenizers/`):

```python
from src.envs.tokenizers import NumberArrayTokenizer, IntegerTokenizer

problem_tokenizer = NumberArrayTokenizer(2, "V", 1, IntegerTokenizer(params.base))
answer_tokenizer = IntegerTokenizer(params.base)
```

Available tokenizers: `IntegerTokenizer`, `FloatTokenizer`, `ComplexTokenizer`, `SymbolicIntTokenizer`, `NumberArrayTokenizer`, `GraphTokenizer`, `GraphNodeListTokenizer`, `SymbolicSequenceTokenizer`.

### Step 3: Register the task

Create a file in `src/envs/ops/` with a build function, an argument registration function, and an `OPERATIONS` dictionary:

```python
from src.envs.generators.addition import AdditionGenerator
from src.envs.tokenizers import NumberArrayTokenizer, IntegerTokenizer

def build_addition(params):
    return {
        "problem_tokenizer": NumberArrayTokenizer(2, "V", 1, IntegerTokenizer(params.base)),
        "answer_tokenizer": IntegerTokenizer(params.base),
        "generator": AdditionGenerator(params),
    }

def register_args(parser):
    parser.add_argument("--maxint", type=int, default=100)
    parser.add_argument("--base", type=int, default=10)

OPERATIONS = {
    "addition": {"build": build_addition, "register_args": register_args},
}
```

Then import in `src/envs/__init__.py`:

```python
from src.envs.ops.addition import OPERATIONS as _addition_ops
REGISTRY.update(_addition_ops)
```

Your task is now available as `--task addition`.

## Training from a data file

Axolver can train from pre-computed data files instead of generating on the fly:

```bash
python train.py --task gcd --reload_data gcd:/path/to/train.data --eval_data /path/to/valid.data --base 10 --maxint 100 --dump_path ./exp --exp_name from_file --exp_id 1
```

The `--reload_data` flag takes the format `task:path`. The `--eval_data` flag specifies evaluation files; multiple files can be provided separated by commas (the first is used as the validation set, subsequent ones as test sets).

**File format:** plain text, one example per line. Tokens separated by spaces, columns separated by tabs. Two columns (problem, answer) or three columns (problem, query, answer).

Example (GCD in base 10):
```
V2 INT+ 1 2 INT+ 1 8	INT+ 6
```

For large files, `--reload_size N` enables batch loading (only N examples loaded at a time). `--index_dataset true` builds a file index and reads examples on demand, avoiding loading the entire file into memory. `--max_examples N` limits the number of examples used from the file (-1 for all).

The two-class sampling strategy is supported: `--two_classes true` with `--first_class_size` and `--first_class_prob` control data repetition.

### Generating large datasets

```bash
python train.py --task gcd --export_data true --epoch_size 1000000 --cpu true --num_workers 20 --dump_path ./exp --exp_name data_gen --exp_id 1 --base 10 --maxint 100
```

This creates a `gcd.data.prefix` file. To build a large train/valid/test split, run several instances with different `exp_id` and `--env_base_seed -1`, then:

```bash
cat data_gen/*/gcd.data.prefix > data.raw
shuf data.raw > data.shuf
head -n 10000 data.shuf > valid.data
tail -n 10000 data.shuf > test.data
tail -n +10001 data.shuf | head -n -10000 > train.data
```

## Command-line parameters

All parameters have default values. Boolean parameters must be set as `--param true` or `--param false` (not as flags).

### Base parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--task` | (required) | Task name (e.g. `gcd`, `reverse`, `matrix_rank`) |
| `--dump_path` | `""` | Where to save experiments |
| `--exp_name` | `"debug"` | Experiment name |
| `--exp_id` | (auto) | Experiment ID (auto-generated if empty) |
| `--epoch_size` | 300000 | Training examples per epoch |
| `--batch_size` | 32 | Mini-batch size |
| `--max_len` | 256 | Maximum input sequence length (longer examples are discarded) |
| `--max_output_len` | 512 | Maximum output length during generation |
| `--max_epoch` | 100000 | Maximum number of epochs |

### Technical parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--cpu` | `false` | Force CPU training |
| `--amp` | `false` | Automatic mixed precision (tries bf16, falls back to fp16) |
| `--num_workers` | (all cores) | Parallel data-generation workers. Set to 0 for single-threaded (useful for debugging). |
| `--env_base_seed` | -1 | Reproducibility seed. Positive value = reproducible; -1 = random. |

### Evaluation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--eval_size` | 10000 | Validation set size |
| `--batch_size_eval` | 128 | Evaluation batch size |
| `--beam_eval` | `false` | Use beam search during evaluation |
| `--beam_size` | 1 | Beam width |
| `--temperature` | 1.0 | Sampling temperature |
| `--eval_verbose` | 0 | Save per-example predictions (1 = errors only, 2 = all + beam hypotheses) |
| `--eval_only` | `false` | Evaluate only, no training (use with `--eval_from_exp`) |
| `--eval_from_exp` | `""` | Path to experiment directory for evaluation-only mode |
| `--metrics_eval` | `""` | Additional metrics (comma-separated; prefix `_` for lower-is-better) |
| `--decouple_cpu_gpu` | `false` | Overlap GPU generation with CPU hypothesis checking |
| `--process_pool` | `false` | Use a process pool for parallel hypothesis checking |

### Model architecture

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_type` | `transformer` | `transformer`, `lstm`, or `gru` |
| `--architecture` | `encoder_decoder` | `encoder_decoder`, `encoder_only` (BERT-like), or `decoder_only` (GPT-like) |
| `--n_enc_layers` | 4 | Encoder layers |
| `--n_dec_layers` | 4 | Decoder layers |
| `--enc_emb_dim` | 256 | Encoder embedding dimension |
| `--dec_emb_dim` | 256 | Decoder embedding dimension |
| `--n_enc_heads` | 8 | Encoder attention heads |
| `--n_dec_heads` | 8 | Decoder attention heads |
| `--n_enc_hidden_layers` | 1 | FFN depth within each encoder layer |
| `--n_dec_hidden_layers` | 1 | FFN depth within each decoder layer |
| `--norm` | `layernorm` | `layernorm` or `rmsnorm` |
| `--activation` | `gelu` | `gelu`, `relu`, or `relu_squared` |
| `--enc_pos_emb` | `abs_learned` | Positional embedding: `abs_learned`, `abs_sinusoidal`, or `none` |
| `--dec_pos_emb` | `abs_learned` | Positional embedding: `abs_learned`, `abs_sinusoidal`, or `none` |
| `--share_inout_emb` | `true` | Tie decoder input embedding and output projection |
| `--dropout` | 0 | FFN dropout |
| `--attention_dropout` | 0 | Attention weight dropout |

### Optimizer

Specified as `--optimizer name,key=value,...`. Supported optimizers: `adam`, `adamw`, `sgd`, `adadelta`, `adagrad`, `adamax`, `asgd`, `rmsprop`, `rprop`.

Learning rate scheduling is activated by appending a suffix:
- `adam_warmup`: linear warmup then constant LR
- `adam_inverse_sqrt`: linear warmup then inverse square root decay
- `adam_cosine` (or `adam_smooth_cosine`): linear warmup then cosine annealing with warm restarts

Examples:
- `adam,lr=0.0001` (default)
- `adam_warmup,lr=0.001,warmup_updates=4000`
- `adam_cosine,lr=0.001,warmup_updates=4000,init_period=10000`

Additional scheduler parameters: `warmup_init_lr`, `exp_factor`, `min_lr`, `init_period`, `period_mult`.

`--clip_grad_norm` (default 5.0) clips the gradient norm.

## Evaluation metrics

During evaluation, each prediction is checked in three ways:

- **Greedy accuracy** (`greedy_acc`): Whether the greedy (argmax) output matches the target token by token.
- **Valid** (`acc`): Whether the decoded prediction is mathematically correct, as determined by the generator's `evaluate()` method.
- **Well-formed** (`well_formed`): Whether the predicted tokens decode into a valid mathematical object.

Per-class accuracy is reported using the class IDs from `encode_class_id()`. For GCD, this is the GCD value itself; for matrix rank, the rank. Custom metrics (specified via `--metrics_eval`) are computed by the generator's `evaluate()` method and reported as averages.

## Resuming experiments

Re-running the same command resumes from the last checkpoint:

```bash
python train.py --task gcd --dump_path ./exp --exp_name my_first_gcd --exp_id 1 --base 10 --maxint 10000 --max_gcd 100
```

To initialise a new experiment from a pre-trained model:

```bash
python train.py --task gcd --dump_path ./exp --exp_name my_first_gcd --exp_id 2 --reload_checkpoint ./exp/my_first_gcd/1/checkpoint.pth --base 10 --maxint 10000 --max_gcd 100
```

## Multi-GPU training

Axolver supports multi-GPU training via PyTorch's DistributedDataParallel. Launch with `torchrun`:

```bash
torchrun --nproc_per_node=4 train.py --task gcd --dump_path ./exp --exp_name multi_gpu --exp_id 1 --base 10 --maxint 10000 --max_gcd 100
```

## Visualizing results

Axolver provides a Jupyter notebook (`tools/ReadXP.ipynb`) that reads experiment logs and produces learning curves and result tables.

To use the notebook, set the configuration cell:

```python
dump_path = "../exp"
exp_names = ["my_experiment"]
task_name = "GCD"
```

where `dump_path` is the root experiment directory, `exp_names` is a list of experiment name directories to compare, and `task_name` is the task name as it appears in the log metrics (e.g. `GCD`, `REVERSE`, `MATRIX_TRANSPOSE`).

The notebook produces three plots (training loss, validation accuracy, validation cross-entropy loss) and a summary table comparing all experiments. Multiple experiments can be overlaid on the same plot for easy comparison.

## Code structure

```
axolver/
  train.py                     # Main entry point
  src/
    utils.py                   # Experiment initialisation
    logger.py                  # Logging
    slurm.py                   # Distributed training
    optim.py                   # Optimizers and LR schedulers
    dataset.py                 # Data loading and batching
    trainer.py                 # Training loop
    evaluator.py               # Evaluation and beam search
    model/
      base.py                  # BaseModel (abstract base)
      transformer.py           # Transformer architecture
      rnn.py                   # LSTM / GRU
    envs/
      environment.py           # Vocabulary and data generation
      tokenizers/              # Integer, float, complex, array,
                               #   graph, symbolic expression
      generators/              # Arithmetic, matrix, graph,
                               #   polynomial, integration, synthetic
      ops/                     # Task registrations
```

## Results replication

We replicated results of several papers in the AI4math community. 

### Summary Table

| Task | Paper | Paper Accuracy | Our Accuracy | 
|------|-------|-----------|----------|
| Integration | Lample & Charton, "Deep Learning for Symbolic Mathematics", ICLR 2020. [arXiv:1912.01412](https://arxiv.org/abs/1912.01412). | 98.4% (BWD) | 97.4% |
| Matrix Transpose | Charton, "Linear algebra with transformers", TMLR 2022. [arXiv:2112.01898](https://arxiv.org/abs/2112.01898). | 99.8% (5x5, P10) | 100% |
| Matrix Eigenvalues | Charton, "Linear algebra with transformers", TMLR 2022. [arXiv:2112.01898](https://arxiv.org/abs/2112.01898). | 100% (5x5, FP15) | 99.3% |
| GCD | Charton, "Can transformers learn the greatest common divisor?", ICLR 2024. [arXiv:2308.15594](https://arxiv.org/abs/2308.15594). | 99.1% | 99.1% |

### Paper Replication Commands

#### 1. Integration - target 98.4% with beam size 10

```
python train.py --task integration --exp_name integration_bwd --max_ops 15 --max_int 5 --operators add:10,sub:3,mul:10,div:5,sqrt:4,pow2:4,pow3:2,pow4:1,pow5:1,ln:4,exp:4,sin:4,cos:4,tan:4,asin:1,acos:1,atan:1,sinh:1,cosh:1,tanh:1,asinh:1,acosh:1,atanh:1 --n_enc_layers 6 --n_dec_layers 6 --enc_emb_dim 512 --dec_emb_dim 512 --max_epoch 500 --amp true --max_len 512 --beam_eval true --beam_size 10 --batch_size_eval 32 --eval_size 1000
```

#### 2. Matrix Transpose — target 99.8%

```
python train.py --task matrix_transpose --exp_name matrix_transpose_5x5 --dim1 5 --dim2 5 --maxint 10 --n_enc_layers 1 --n_dec_layers 1 --optimizer adam_cosine,lr=0.0001,warmup_updates=10000 --batch_size 64 --max_epoch 200 --amp true
```

#### 3. Matrix Eigenvalues — target 100% (with 5% tol)

```
python train.py --task matrix_eigenvalues --exp_name matrix_eigenvalues_5x5_fp15 --dim1 5 --dim2 5 --maxint 10 --fp15_encoding true --rtol 0.05 --n_enc_layers 6 --n_dec_layers 6 --enc_emb_dim 512 --dec_emb_dim 512 --optimizer adam_cosine,lr=0.0001,warmup_updates=10000 --batch_size 64 --max_epoch 300 --amp true --float_precision 2 --max_exponent 16
```

#### 4. GCD — target 99.1%

```
python train.py --task gcd --exp_name gcd_base2401 --base 2401 --maxint 1000000 --max_gcd 1000 --outcome_distribution natural --operand_distribution log_uniform --optimizer adam,lr=0.00001 --batch_size 256 --max_epoch 1000 --amp true
```

## License

This repository uses the Apache-2.0 License. See [LICENSE](LICENSE) for details.

## Acknowledgements

The original code of Int2Int was written by François Charton, and can be found [here](https://github.com/f-charton/Int2Int). 
