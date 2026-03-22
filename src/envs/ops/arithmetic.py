from src.envs.generators import (
    FractionAddGenerator,
    FractionCompareGenerator,
    FractionDeterminantGenerator,
    FractionProductGenerator,
    FractionRoundGenerator,
    FractionSimplifyGenerator,
    GCDGenerator,
    ModularAddGenerator,
    ModularMulGenerator,
)
from src.envs.tokenizers import IntegerTokenizer, NumberArrayTokenizer, SymbolicIntTokenizer


def build_gcd(params):
    return {
        "problem_tokenizer": NumberArrayTokenizer(2, "V", 1, IntegerTokenizer(params.base)),
        "answer_tokenizer": IntegerTokenizer(params.base),
        "generator": GCDGenerator(params),
    }


def build_fraction_simplify(params):
    return {
        "problem_tokenizer": NumberArrayTokenizer(2, "V", 1, IntegerTokenizer(params.base)),
        "answer_tokenizer": NumberArrayTokenizer(2, "V", 1, IntegerTokenizer(params.base)),
        "generator": FractionSimplifyGenerator(params),
    }


def build_fraction_round(params):
    return {
        "problem_tokenizer": NumberArrayTokenizer(2, "V", 1, IntegerTokenizer(params.base)),
        "answer_tokenizer": IntegerTokenizer(params.base),
        "generator": FractionRoundGenerator(params),
    }


def build_fraction_add(params):
    return {
        "problem_tokenizer": NumberArrayTokenizer(4, "V", 1, IntegerTokenizer(params.base)),
        "answer_tokenizer": NumberArrayTokenizer(2, "V", 1, IntegerTokenizer(params.base)),
        "generator": FractionAddGenerator(params),
    }


def build_fraction_product(params):
    return {
        "problem_tokenizer": NumberArrayTokenizer(4, "V", 1, IntegerTokenizer(params.base)),
        "answer_tokenizer": NumberArrayTokenizer(2, "V", 1, IntegerTokenizer(params.base)),
        "generator": FractionProductGenerator(params),
    }


def build_fraction_determinant(params):
    return {
        "problem_tokenizer": NumberArrayTokenizer(4, "V", 1, IntegerTokenizer(params.base)),
        "answer_tokenizer": IntegerTokenizer(params.base),
        "generator": FractionDeterminantGenerator(params),
    }


def build_fraction_compare(params):
    return {
        "problem_tokenizer": NumberArrayTokenizer(4, "V", 1, IntegerTokenizer(params.base)),
        "answer_tokenizer": SymbolicIntTokenizer(0, 1),
        "generator": FractionCompareGenerator(params),
    }


def build_modular_add(params):
    return {
        "problem_tokenizer": NumberArrayTokenizer(2, "V", 1, IntegerTokenizer(params.base)),
        "answer_tokenizer": IntegerTokenizer(params.base),
        "generator": ModularAddGenerator(params),
    }


def build_modular_product(params):
    return {
        "problem_tokenizer": NumberArrayTokenizer(2, "V", 1, IntegerTokenizer(params.base)),
        "answer_tokenizer": IntegerTokenizer(params.base),
        "generator": ModularMulGenerator(params),
    }


def register_int_args(parser):
    parser.add_argument("--base", type=int, default=10)
    parser.add_argument("--minint", type=int, default=1)
    parser.add_argument("--maxint", type=int, default=100)


def register_gcd_args(parser):
    register_int_args(parser)
    parser.add_argument("--operand_distribution", type=str, default="uniform", help="Operand sampling: uniform, log_uniform")
    parser.add_argument(
        "--outcome_distribution", type=str, default="uniform", help="GCD outcome distribution: uniform, log_uniform, natural, inv_sqrt, mixed"
    )
    parser.add_argument("--max_gcd", type=int, default=100, help="Max GCD value for outcome-balanced sampling")
    parser.add_argument("--mixed_pct", type=float, default=0.05, help="Fraction of balanced samples in mixed mode")


def register_modular_args(parser):
    register_int_args(parser)
    parser.add_argument("--modulus", type=int, default=97)


OPERATIONS = {
    "gcd": {"build": build_gcd, "register_args": register_gcd_args},
    "fraction_simplify": {"build": build_fraction_simplify, "register_args": register_int_args},
    "fraction_round": {"build": build_fraction_round, "register_args": register_int_args},
    "fraction_add": {"build": build_fraction_add, "register_args": register_int_args},
    "fraction_product": {"build": build_fraction_product, "register_args": register_int_args},
    "fraction_determinant": {"build": build_fraction_determinant, "register_args": register_int_args},
    "fraction_compare": {"build": build_fraction_compare, "register_args": register_int_args},
    "modular_add": {"build": build_modular_add, "register_args": register_modular_args},
    "modular_product": {"build": build_modular_product, "register_args": register_modular_args},
}
