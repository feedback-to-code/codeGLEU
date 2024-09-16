# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
import math
from collections import Counter
from typing import Callable, Optional

from .parser import try_remove_comments_and_docstrings
from .utils import ngrams, multirefscores


def counter_diff(a, b):
    diff = Counter(a)
    for k in set(a) & set(b):
        del diff[k]
    return diff


def brevity_penalty(closest_ref_len, hyp_len):
    if hyp_len > closest_ref_len:
        return 1
    # If hypothesis is empty, brevity penalty = 0 should result in BLEU = 0.0
    elif hyp_len == 0:
        return 0
    else:
        return math.exp(1 - closest_ref_len / hyp_len)


def smoothing_function(p_n, epsilon=0.1):
    """
    Smoothing method 1: Add *epsilon* counts to precision with 0 counts.
    """
    return [((p_i[0] + epsilon), p_i[1]) if p_i[0] == 0 else p_i for p_i in p_n]


def calc_gleu(
    source: str,
    references: list[str],
    hypothesis: str,
    lang: str,
    tokenizer: Optional[Callable[[str], list[str]]] = None,
    n_weights: tuple[float, ...] = (0.25,) * 4,
    key_weights: dict[str, float] = {},
    penalty: float = 1,
):
    return corpus_gleu_score(
        corpus_gleu_intermediate([source], [references], [hypothesis], lang, tokenizer, len(n_weights)), n_weights, key_weights, penalty
    )


def get_all_ngrams(l: list[str], n: int) -> list[Counter]:
    return [Counter(map(str, ngrams(l, i))) for i in range(1, n + 1)]


def corpus_gleu_intermediate(
    sources: list[str],
    references: list[list[str]],
    hypotheses: list[str],
    lang: str,
    tokenizer: Optional[Callable[[str], list[str]]] = None,
    n: int = 4,
) -> dict[str, list]:

    intermediates: dict[str, list] = {
        "s_interm": [],
        "h_interm": [],
        "r_interms": [],
    }

    if tokenizer is None:

        def tokenizer(s: str) -> list[str]:
            return try_remove_comments_and_docstrings(s, lang).split()

    for source, references_sample, hypothesis in zip(sources, references, hypotheses):
        tokenized_src = tokenizer(source)
        tokenized_hyp = tokenizer(hypothesis)
        intermediates["s_interm"] += [get_all_ngrams(tokenized_src, n)]
        intermediates["h_interm"] += [get_all_ngrams(tokenized_hyp, n)]

        refs = []
        for reference in references_sample:
            tokenized_ref = tokenizer(reference)
            refs += [get_all_ngrams(tokenized_ref, n)]
        intermediates["r_interms"] += [refs]
    return intermediates


# very similar to dataflow match, might merge later
def corpus_gleu_score(
    intermediates: dict[str, list],
    n_weights: tuple[float, ...] = (0.25,) * 4,
    key_weights: dict[str, float] = {},
    penalty: float = 1,
) -> tuple[float, list[list[int]]]:
    hyp_lengths = 0
    ref_lengths = 0
    if key_weights:
        default_key_weight = key_weights.pop("default", 1)
        key_weights = {key.removeprefix("key_"): value for key, value in key_weights.items()}
    else:
        default_key_weight = 1
        key_weights = {}
    
    source_interm = intermediates["s_interm"][0]
    hypothesis_interm = intermediates["h_interm"][0]
    reference_interms = [ri[0] for ri in intermediates["r_interms"]]
    scores = []

    hyp_len = Counter(hypothesis_interm[0]).total()
    hyp_lengths += hyp_len
    ref_lens = (Counter(reference_interm[0]).total() for reference_interm in reference_interms)
    ref_lengths += min(ref_lens, key=lambda ref_len: (abs(ref_len - hyp_len), ref_len))
    for reference_interm in reference_interms:
        p_n = [[0, 0] for _ in range(0, len(n_weights))]
        for n in range(0, len(n_weights)):
            source_interm_n = Counter(source_interm[n])
            reference_interm_n = Counter(reference_interm[n])
            hypothesis_interm_n = Counter(hypothesis_interm[n])

            def weighted_value(ngram, count):
                for key in key_weights:
                    if f"('{key}'," in ngram:
                        return count * key_weights[key]
                return count * default_key_weight

            weighted_count = lambda mydict: sum([weighted_value(ngram, count) for ngram, count in mydict.items()])

            ref_added = reference_interm_n - source_interm_n
            ref_removed = source_interm_n - reference_interm_n
            hyp_added = hypothesis_interm_n - source_interm_n
            hyp_removed = source_interm_n - hypothesis_interm_n

            correct_changes = weighted_count((ref_added & hyp_added) + (ref_removed & hyp_removed))
            wrong_changes = weighted_count((hyp_added - ref_added) + (hyp_removed - ref_removed))
            total_changes = weighted_count(ref_added + ref_removed)
            p_n[n][0] += max(0, correct_changes - penalty * wrong_changes)
            p_n[n][1] += max(0, total_changes)

        if p_n[0][1] != 0:
            bp = brevity_penalty(ref_lengths, hyp_lengths)
            p_n = smoothing_function(p_n)
            sgen = (w_i * math.log(p_i[0] / p_i[1]) for w_i, p_i in zip(n_weights, p_n))
            s: float = bp * math.exp(math.fsum(sgen))
            scores.append(s)

    if scores == []:
        return -1.0
    
    return multirefscores(scores)
