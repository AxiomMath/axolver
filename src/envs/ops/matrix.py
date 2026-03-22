from src.envs.generators import (
    MatrixDeterminantGenerator,
    MatrixEigenvaluesGenerator,
    MatrixInverseGenerator,
    MatrixRankGenerator,
    MatrixSumGenerator,
    MatrixTransposeGenerator,
    MatrixVectorGenerator,
)
from src.envs.tokenizers import FloatTokenizer, FP15Tokenizer, IntegerTokenizer, NumberArrayTokenizer, SymbolicIntTokenizer
from src.utils import bool_flag


def _int_matrix_tok(params):
    max_dim = max(params.dim1, params.dim2)
    return NumberArrayTokenizer(max_dim * max_dim, "V", 2, IntegerTokenizer(params.base))


def _float_value_tokenizer(params):
    if params.fp15_encoding:
        return FP15Tokenizer(precision=params.float_precision, max_exponent=params.max_exponent)
    return FloatTokenizer(params.base, params.float_precision, params.max_exponent)


def _float_array_tok(params, max_len):
    return NumberArrayTokenizer(max_len, "V", 1, _float_value_tokenizer(params), encode_dim=True)


def _float_matrix_tok(params, max_size):
    return NumberArrayTokenizer(max_size, "V", 2, _float_value_tokenizer(params))


def build_transpose(params):
    max_dim = max(params.dim1, params.dim2)
    tok = NumberArrayTokenizer(max_dim * max_dim, "V", 2, IntegerTokenizer(params.base))
    return {"problem_tokenizer": tok, "answer_tokenizer": tok, "generator": MatrixTransposeGenerator(params)}


def build_matrix_sum(params):
    max_dim = max(params.dim1, params.dim2)
    inp_tok = NumberArrayTokenizer(2 * max_dim * max_dim, "V", 2, IntegerTokenizer(params.base))
    out_tok = NumberArrayTokenizer(max_dim * max_dim, "V", 2, IntegerTokenizer(params.base))
    return {"problem_tokenizer": inp_tok, "answer_tokenizer": out_tok, "generator": MatrixSumGenerator(params)}


def build_matrix_vector(params):
    max_dim = max(params.dim1, params.dim2)
    inp_tok = NumberArrayTokenizer(max_dim * (max_dim + 1), "V", 2, IntegerTokenizer(params.base))
    out_tok = NumberArrayTokenizer(max_dim, "V", 1, IntegerTokenizer(params.base), encode_dim=True)
    return {"problem_tokenizer": inp_tok, "answer_tokenizer": out_tok, "generator": MatrixVectorGenerator(params)}


def build_matrix_determinant(params):
    return {
        "problem_tokenizer": _int_matrix_tok(params),
        "answer_tokenizer": IntegerTokenizer(params.base),
        "generator": MatrixDeterminantGenerator(params),
    }


def build_eigenvalues(params):
    inp_tok = _int_matrix_tok(params)
    out_tok = _float_array_tok(params, params.dim1)
    return {"problem_tokenizer": inp_tok, "answer_tokenizer": out_tok, "generator": MatrixEigenvaluesGenerator(params)}


def build_invert_matrix(params):
    inp_tok = _int_matrix_tok(params)
    out_tok = _float_matrix_tok(params, params.dim1 * params.dim1)
    return {"problem_tokenizer": inp_tok, "answer_tokenizer": out_tok, "generator": MatrixInverseGenerator(params)}


def build_matrix_rank(params):
    max_dim = max(params.dim1, params.dim2)
    return {
        "problem_tokenizer": NumberArrayTokenizer(max_dim * max_dim, "V", 2, IntegerTokenizer(params.base)),
        "answer_tokenizer": SymbolicIntTokenizer(1, max_dim),
        "generator": MatrixRankGenerator(params),
    }


def register_int_args(parser):
    parser.add_argument("--dim1", type=int, default=4)
    parser.add_argument("--dim2", type=int, default=4)
    parser.add_argument("--maxint", type=int, default=5)
    parser.add_argument("--base", type=int, default=10)


def register_float_args(parser):
    register_int_args(parser)
    parser.add_argument("--fp15_encoding", type=bool_flag, default=False, help="Encoding numbers using FP15 Tokenizer")
    parser.add_argument("--float_precision", type=int, default=3)
    parser.add_argument("--max_exponent", type=int, default=10)
    parser.add_argument("--rtol", type=float, default=0.0, help="Relative tolerance for evaluation")


OPERATIONS = {
    "matrix_transpose": {"build": build_transpose, "register_args": register_int_args},
    "matrix_sum": {"build": build_matrix_sum, "register_args": register_int_args},
    "matrix_vector": {"build": build_matrix_vector, "register_args": register_int_args},
    "matrix_determinant": {"build": build_matrix_determinant, "register_args": register_int_args},
    "matrix_eigenvalues": {"build": build_eigenvalues, "register_args": register_float_args},
    "matrix_inverse": {"build": build_invert_matrix, "register_args": register_float_args},
    "matrix_rank": {"build": build_matrix_rank, "register_args": register_int_args},
}
