# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
from collections import Counter

from .utils import ngrams


def counter_diff(a, b):
    diff = Counter(a)
    for k in set(a) & set(b):
        del diff[k]
    return diff


def calc_gleu(
    source: list[str],
    references: list[list[str]],
    hypothesis: list[str],
    n_weights: tuple[float, ...] = (0.25,) * 4,
    key_weights: dict[str, float] = {},
    penalty: float = 1,
):
    return corpus_gleu([source], [references], [hypothesis], n_weights, key_weights, penalty)


def get_all_ngrams(l: list[str], n: int) -> list[Counter]:
    return [Counter(ngrams(l, i)) for i in range(1, n + 1)]


def corpus_gleu_intermediate(
    sources: list[list[str]],
    references: list[list[list[str]]],
    hypotheses: list[list[str]],
    n_weights: tuple[float, ...] = (0.25,) * 4,
    key_weights: dict[str, float] = {},
    penalty: float = 1,
    debug: bool = False,
) -> dict[str, list]:

    intermediates: dict[str, list] = {
        "s_interm": [],
        "h_interm": [],
        "r_interms": [],
    }

    for source, references_sample, hypothesis in zip(sources, references, hypotheses):
        intermediates["s_interm"] += [get_all_ngrams(source, len(n_weights))]
        intermediates["h_interm"] += [get_all_ngrams(hypothesis, len(n_weights))]

        refs = []
        for reference in references_sample:
            refs += [get_all_ngrams(reference, len(n_weights))]
        intermediates["r_interms"] += [refs]
    return intermediates


# very similar to dataflow match, might merge later
def corpus_gleu(
    sources: list[list[str]],
    references: list[list[list[str]]],
    hypotheses: list[list[str]],
    n_weights: tuple[float, ...] = (0.25,) * 4,
    key_weights: dict[str, float] = {},
    penalty: float = 1,
    debug: bool = False,
    intermediates: dict[str, list] = {},
) -> float:
    match_count = 0
    total_count = 0

    if key_weights:
        default_key_weight = key_weights.pop("default", 1)
        key_weights = {key.removeprefix("key_"): value for key, value in key_weights.items()}
    else:
        default_key_weight = 1
        key_weights = {}

    intermediates = intermediates or corpus_gleu_intermediate(sources, references, hypotheses, n_weights, key_weights, penalty, debug)

    for source_interm, reference_interms, hypothesis_interm in zip(intermediates["s_interm"], intermediates["r_interms"], intermediates["h_interm"]):
        for reference_interm in reference_interms:
            for n in range(0, len(n_weights)):
                source_subexp_diff = counter_diff(source_interm[n], reference_interm[n])

                def weighted_value(ngram, count):
                    if ngram[0] in key_weights.keys():
                        return count * key_weights[ngram[0]]
                    return count * default_key_weight

                weighted_count = lambda mydict: sum([weighted_value(ngram, count) for ngram, count in mydict.items()])
                matching_subexp = weighted_count(hypothesis_interm[n] & reference_interm[n])
                penalty_subexp = weighted_count(hypothesis_interm[n] & source_subexp_diff)

                match_count += n_weights[n] * max(matching_subexp - penalty * penalty_subexp, 0)
                total_count += n_weights[n] * max(weighted_count(reference_interm[n]), 1)

    if total_count == 0:
        logging.warning(
            "WARNING: There is no reference ngrams extracted from the whole corpus, "
            "and the ngram match score degenerates to 0. Please consider ignoring this score."
        )
        return 0
    score = match_count / total_count
    return score
