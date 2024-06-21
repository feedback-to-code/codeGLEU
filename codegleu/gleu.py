# Courtney Napoles
# <napoles@cs.jhu.edu>
# 21 June 2015
# ##
# gleu.py
#
# This script calculates the GLEU score of a sentence, as described in
# our ACL 2015 paper, Ground Truth for Grammatical Error Correction Metrics
# by Courtney Napoles, Keisuke Sakaguchi, Matt Post, and Joel Tetreault.
#
# For instructions on how to get the GLEU score, call "compute_gleu -h"
#
# Updated 2 May 2016: This is an updated version of GLEU that has been
# modified to handle multiple references more fairly.
#
# Updated 6 9 2017: Fixed inverse brevity penalty
#
# This script was adapted from bleu.py by Adam Lopez.
# <https://github.com/alopez/en600.468/blob/master/reranker/>

import math
from collections import Counter

import numpy as np
import scipy.stats


class GLEU:
    refs: list[list[list[str]]]
    reflens: list[list[int]]

    def __init__(
        self,
        sources: list[list[str]],
        references: list[list[list[str]]],
        n_weights: tuple[float, ...] = (0.25,) * 4,
        key_weights: dict[str, float] = {"default": 1},
        penalty: float = 1,
    ):
        """
        :param sources: source sentences
        :type sources: list(str)
        :param references: reference sentences, row-wise
        :type references: list(list(str))
        :param n_weights: weights for unigrams, bigrams, trigrams and so on
        :type n_weights: list(float)
        :param key_weights: weights for keywords
        :type key_weights: list(float)
        """
        self.order = len(n_weights)
        self.penalty = penalty

        total = sum(n_weights)
        n_weights = tuple(weight / total for weight in n_weights if weight != 0)
        self.n_weights = n_weights

        if key_weights:
            self.default_key_weight = key_weights.pop("default", 1)
            self.key_weights = {key.removeprefix("key_"): value for key, value in key_weights.items()}
        else:
            self.default_key_weight = 1
            self.key_weights = {}
        self.refs = [[] for _ in range(len(sources))]
        self.reflens = [[] for _ in range(len(sources))]
        self.load_sources(sources)
        self.load_references(references)

    def load_sources(self, sources):
        self.all_source_ngrams = [[self.get_ngram_counts(line, n) for n in range(1, self.order + 1)] for line in sources]

    def load_references(self, references):
        self.refs = references
        self.reflens = [[len(ref) for ref in refs] for refs in references]

        # count number of references each n-gram appears in
        self.all_r_ngrams_freq = [Counter() for i in range(self.order)]

        self.all_r_ngrams = []
        for refset in self.refs:
            all_ngrams = []
            self.all_r_ngrams.append(all_ngrams)

            for n in range(1, self.order + 1):
                ngrams = self.get_ngram_counts(refset[0], n)
                all_ngrams.append(ngrams)

                for k in ngrams.keys():
                    self.all_r_ngrams_freq[n - 1][k] += 1

                for ref in refset[1:]:
                    new_ngrams = self.get_ngram_counts(ref, n)
                    for nn in new_ngrams.elements():
                        if new_ngrams[nn] > ngrams.get(nn, 0):
                            ngrams[nn] = new_ngrams[nn]

    def get_ngram_counts(self, sentence, n):
        return Counter([tuple(sentence[i : i + n]) for i in range(len(sentence) + 1 - n)])

    # returns ngrams in a but not in b
    def counter_diff(self, a, b):
        diff = Counter(a)
        for k in set(a) & set(b):
            del diff[k]
        return diff

    def normalization(self, ngram, n):
        return 1.0 * self.all_r_ngrams_freq[n - 1][ngram] / len(self.reflens[0])

    # Collect BLEU-relevant statistics for a single hypothesis/reference pair.
    # Return value is a generator yielding:
    # (c, r, numerator1, denominator1, ... numerator4, denominator4)
    # Summing the columns across calls to this function on an entire corpus
    # will produce a vector of statistics that can be used to compute GLEU
    def gleu_stats(self, h, i, r_ind=None):

        self.hypothesislen = len(h)
        self.this_hypothesis_ngrams = [self.get_ngram_counts(h, n) for n in range(1, self.order + 1)]
        yield self.hypothesislen
        yield self.reflens[i][r_ind]

        for n in range(self.order):
            hypothesis_ngrams = self.this_hypothesis_ngrams[n]
            source_ngrams = self.all_source_ngrams[i][n]
            reference_ngrams = self.get_ngram_counts(self.refs[i][r_ind], n + 1)

            source_ngram_diff = source_ngrams - reference_ngrams  # + (reference_ngrams - source_ngrams)

            def weighted_value(ngram, count):
                if ngram[0] in self.key_weights.keys():
                    return count * self.key_weights[ngram[0]]
                return count * self.default_key_weight

            weighted_count = lambda mydict: sum([weighted_value(ngram, count) for ngram, count in mydict.items()])
            correct_ngrams = weighted_count(hypothesis_ngrams & reference_ngrams)
            penalty_ngrams = weighted_count(hypothesis_ngrams & source_ngram_diff)
            yield max([correct_ngrams - self.penalty * penalty_ngrams, 0])

            ngram_count = weighted_count(hypothesis_ngrams)
            yield max([ngram_count, 0])

    # Compute GLEU from collected statistics obtained by call(s) to gleu_stats
    def gleu(self, stats, smooth=False):
        # smooth 0 counts for sentence-level scores
        if smooth:
            stats = [s if s != 0 else 1 for s in stats]
        if len([stat for stat in stats if stat == 0]) > 0:
            return 0
        (c, r) = stats[:2]
        log_gleu_prec = sum([self.n_weights[i] * math.log(float(x) / y) for i, (x, y) in enumerate(zip(stats[2::2], stats[3::2]))])
        return math.exp(min([0, 1 - float(r) / c]) + log_gleu_prec)


def get_gleu_stats(scores):
    mean = np.mean(scores)
    std = np.std(scores) if len(scores) > 1 else 0
    ci = scipy.stats.norm.interval(0.95, loc=mean, scale=std) if std > 0 else (0, 0)
    return [mean, std, ci[0], ci[1]]


def corpus_gleu(
    sources: list[list[str]],
    references: list[list[list[str]]],
    hypotheses: list[list[str]],
    n_weights: tuple[float, ...] = (0.25,) * 4,
    key_weights: dict[str, float] = {},
    penalty: float = 1,
    debug: bool = False,
) -> float:
    n = len(n_weights)
    gleu_calculator = GLEU(sources, references, n_weights, key_weights, penalty)

    def dbprint(*args, **kwargs):
        if debug:
            print(*args, **kwargs)

    dbprint("===== Sentence-level scores =====")
    dbprint("SID Mean Stdev 95%CI GLEU")

    refnum = len(references[0])
    iter_stats = [[0, 0] for i in range((n + 1) * refnum)]
    for i, h in enumerate(hypotheses):
        stats_by_ref: list[list[int]] = [[] for i in range(refnum)]
        for ref in range(refnum):
            stats_by_ref[ref] = list(gleu_calculator.gleu_stats(h, i, r_ind=ref))
            dbprint(stats_by_ref[ref])
            iter_stats[ref] = [sum(scores) for scores in zip(iter_stats[ref], stats_by_ref[ref])]

        # sentence-level GLEU is the mean GLEU of the hypothesis
        # compared to each reference
        dbprint(i, h)
        stats = get_gleu_stats([gleu_calculator.gleu(stats, smooth=True) for stats in stats_by_ref])
        dbprint(" ".join([str(stat) for stat in stats]))

    stats = get_gleu_stats([gleu_calculator.gleu(stats) for stats in iter_stats])

    dbprint("\n==== Overall score =====")
    dbprint("Mean Stdev 95%CI GLEU")
    dbprint(" ".join([str(stat) for stat in stats]))

    return stats[0]
