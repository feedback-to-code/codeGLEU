# Copyright (c) Microsoft Corporation.
# Copyright (c) 2023 Konstantin Chernyshev.
# Licensed under the MIT license.
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from . import dataflow_match, gleu, syntax_match
from .utils import AVAILABLE_LANGS, get_tree_sitter_language

PACKAGE_DIR = Path(__file__).parent


def calc_codegleu(
    sources: List[str],
    references: Union[List[str], List[List[str]]],
    predictions: List[str],
    lang: str,
    weights: tuple[float, ...] = (0.25,) * 4,
    penalty: float = 1,
    tokenizer: Optional[Callable[[str], list[str]]] = None,
    keywords_dir: Path = PACKAGE_DIR / "keywords",
) -> Dict[str, float]:
    """Calculate codegleu score

    Args:
        predictions: list of predictions
        references: list of lists with references
        lang: input language, one of AVAILABLE_LANGS
        weights: weights of the ngram_match, weighted_ngram_match, syntax_match, and dataflow_match respectively
        tokenizer: tokenizer function, Defaults to lambda s: s.split()
        keywords_dir: path to the directory with keywords files
        lang_so_file: path to the .so file with the parser for the language

    Return:
        Scores dict
    """
    assert len(references) == len(predictions), "Number of references and predictions should be the same"
    assert lang in AVAILABLE_LANGS, f"Language {lang} is not supported (yet). Available languages: {AVAILABLE_LANGS}"
    assert len(weights) == 4, "weights should be a tuple of 4 floats (alpha, beta, gamma, theta)"
    assert keywords_dir.exists(), f"keywords_dir {keywords_dir} does not exist"

    # get the tree-sitter language for a given language
    tree_sitter_language = get_tree_sitter_language(lang)

    # preprocess inputs
    references = [[x.strip() for x in ref] if isinstance(ref, list) else [ref.strip()] for ref in references]
    hypotheses = [x.strip() for x in predictions]
    sources = [x.strip() for x in sources]

    # calculate ngram match (BLEU)
    if tokenizer is None:

        def tokenizer(s: str) -> list[str]:
            return s.split()

    tokenized_srcs = [tokenizer(x) for x in sources]
    tokenized_hyps = [tokenizer(x) for x in hypotheses]
    tokenized_refs = [[tokenizer(x) for x in reference] for reference in references]

    ngram_match_score = gleu.corpus_gleu(tokenized_srcs, tokenized_refs, tokenized_hyps, penalty=penalty)

    # calculate weighted ngram match
    with open(keywords_dir / (lang + ".txt"), "r", encoding="utf-8") as f:
        keywords = [x.strip() for x in f.readlines()]

    key_weights = {f"key_{key}": 1 for key in keywords} | {"default": 0.2}
    weighted_ngram_match_score = gleu.corpus_gleu(tokenized_srcs, tokenized_refs, tokenized_hyps, key_weights=key_weights, penalty=penalty)

    # calculate syntax match
    syntax_match_score = syntax_match.corpus_syntax_match(sources, references, hypotheses, penalty, lang, tree_sitter_language=tree_sitter_language)

    # calculate dataflow match
    dataflow_match_score = dataflow_match.corpus_dataflow_match(
        sources, references, hypotheses, penalty, lang, tree_sitter_language=tree_sitter_language
    )
    alpha, beta, gamma, theta = weights
    if dataflow_match_score == -1:
        dataflow_match_score = 0
        code_gleu_score = alpha * ngram_match_score + beta * weighted_ngram_match_score + gamma * syntax_match_score + theta * dataflow_match_score
    else:
        code_gleu_score = alpha * ngram_match_score + beta * weighted_ngram_match_score + gamma * syntax_match_score + theta * dataflow_match_score

    return {
        "codegleu": code_gleu_score,
        "ngram_match_score": ngram_match_score,
        "weighted_ngram_match_score": weighted_ngram_match_score,
        "syntax_match_score": syntax_match_score,
        "dataflow_match_score": dataflow_match_score,
    }
