from src.envs.generators import IntegrationGenerator
from src.envs.tokenizers import SymbolicSequenceTokenizer
from src.utils import bool_flag


def build_integration(params):
    tokenizer = SymbolicSequenceTokenizer(params.int_base, params.max_int)
    return {"problem_tokenizer": tokenizer, "answer_tokenizer": tokenizer, "generator": IntegrationGenerator(params, tokenizer.int_tokenizer)}


def register_args(parser):
    parser.add_argument("--int_base", type=int, default=10)
    parser.add_argument("--max_int", type=int, default=5)
    parser.add_argument("--max_ops", type=int, default=5)
    parser.add_argument("--positive", type=bool_flag, default=True)
    parser.add_argument("--n_variables", type=int, default=1)
    parser.add_argument("--n_coefficients", type=int, default=0)
    parser.add_argument("--operators", type=str, default="add:10,mul:10,pow:5,sin:4,cos:4,exp:3,ln:3")
    parser.add_argument("--leaf_probs", type=str, default="0.75,0,0.25,0")
    parser.add_argument("--rewrite_functions", type=str, default="")
    parser.add_argument("--fwd_prob", type=float, default=0.0)


OPERATIONS = {"integration": {"build": build_integration, "register_args": register_args}}
