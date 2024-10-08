# Copyright (c) Microsoft Corporation.
# Copyright (c) 2023 Konstantin Chernyshev.
# Licensed under the MIT license.
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from codegleu.codegleu import dataflow_match, ngram_match, syntax_match
from codegleu.utils import AVAILABLE_LANGS, get_tree_sitter_language

PACKAGE_DIR = Path(__file__).parent.parent


def calc_codegleu(
    sources: List[str],
    references: Union[List[str], List[List[str]]],
    hypotheses: List[str],
    lang: str,
    weights: tuple[float, ...] = (0.25,) * 4,
    penalty: tuple[float, float, float, float] = (1, 1, 1, 1),
    tokenizer: Optional[Callable[[str], list[str]]] = None,
    keywords_dir: Path = PACKAGE_DIR / "keywords",
    ret_intermediates: bool = False,
    intermediates: dict = {},
    n_weights: tuple[float, ...] = (0.25,) * 4,
) -> Dict[str, dict | float]:
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
    assert len(references) == len(hypotheses), "Number of references and predictions should be the same"
    assert lang in AVAILABLE_LANGS, f"Language {lang} is not supported (yet). Available languages: {AVAILABLE_LANGS}"
    assert len(weights) == 4, "weights should be a tuple of 4 floats (alpha, beta, gamma, theta)"
    assert keywords_dir.exists(), f"keywords_dir {keywords_dir} does not exist"

    # get the tree-sitter language for a given language
    tree_sitter_language = get_tree_sitter_language(lang)

    # calculate weighted ngram match
    with open(keywords_dir / (lang + ".txt"), "r", encoding="utf-8") as f:
        keywords = [x.strip() for x in f.readlines()]

    key_weights: dict[str, float] = {f"key_{key}": 1 for key in keywords} | {"default": 0.05}

    n = len(n_weights)
    n_weights = tuple(w / sum(n_weights) for w in n_weights)
    if not intermediates:
        for i, ref in enumerate(references):
            references[i] = [x.strip() for x in ref] if isinstance(ref, list) else [ref.strip()]  # type: ignore
        for i, x in enumerate(hypotheses):
            hypotheses[i] = x.strip()
        for i, x in enumerate(sources):
            sources[i] = x.strip()
        intermediates = {
            "ngram": ngram_match.corpus_gleu_intermediate(sources, references, hypotheses, lang, tokenizer, n),  # type: ignore
            "syntax": syntax_match.corpus_syntax_intermediate(sources, references, hypotheses, lang, tree_sitter_language),  # type: ignore
            "dataflow": dataflow_match.corpus_dataflow_intermediate(sources, references, hypotheses, lang, tree_sitter_language),  # type: ignore
        }
    ngram_match_score, p_n = ngram_match.corpus_gleu_score(intermediates["ngram"], n_weights, {}, penalty[0])
    weighted_ngram_match_score, wp_n = ngram_match.corpus_gleu_score(intermediates["ngram"], n_weights, key_weights, penalty[1])
    syntax_match_score = syntax_match.corpus_syntax_score(intermediates["syntax"], penalty[2])
    dataflow_match_score = dataflow_match.corpus_dataflow_score(intermediates["dataflow"], penalty[3])

    scores = [ngram_match_score, weighted_ngram_match_score, syntax_match_score, dataflow_match_score]
    zeroed_scores = [0 if score == -1 else score for score in scores]
    usablescores = [(s, w) for (s, w) in zip(scores, weights) if s != -1]
    uweighttotal = sum([w for (s, w) in usablescores])
    code_gleu_score = sum([s * w / (uweighttotal or 1) for (s, w) in usablescores])
    return {
        "codegleu": code_gleu_score,
        "ngram_match_score": zeroed_scores[0],
        "weighted_ngram_match_score": zeroed_scores[1],
        "syntax_match_score": zeroed_scores[2],
        "dataflow_match_score": zeroed_scores[3],
    } | (
        {
            "intermediates": intermediates,
        }
        if ret_intermediates
        else {}
    )
