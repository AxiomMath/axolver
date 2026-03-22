from src.envs.generators import PolynomialRootsGenerator
from src.envs.tokenizers import ComplexTokenizer, IntegerTokenizer, NumberArrayTokenizer


def build_polynomial_roots(params):
    max_degree = params.poly_max_degree
    inp_tok = NumberArrayTokenizer(max_degree + 1, "V", 1, IntegerTokenizer(params.base), encode_dim=True)
    out_tok = NumberArrayTokenizer(max_degree, "V", 1, ComplexTokenizer(params.base, params.float_precision, params.max_exponent), encode_dim=True)
    return {"problem_tokenizer": inp_tok, "answer_tokenizer": out_tok, "generator": PolynomialRootsGenerator(params)}


def register_args(parser):
    parser.add_argument("--base", type=int, default=10)
    parser.add_argument("--float_precision", type=int, default=3)
    parser.add_argument("--max_exponent", type=int, default=10)
    parser.add_argument("--poly_min_degree", type=int, default=3)
    parser.add_argument("--poly_max_degree", type=int, default=6)
    parser.add_argument("--poly_complex_prob", type=float, default=0.5)
    parser.add_argument("--poly_root_range", type=int, default=10)


OPERATIONS = {"polynomial_roots": {"build": build_polynomial_roots, "register_args": register_args}}
