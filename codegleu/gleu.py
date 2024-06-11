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


class GLEU:
    refs: list[list[list[str]]]
    reflens: list[list[int]]

    def __init__(self, sources: list[str], references: list[list[str]], order: int = 4):
        self.order = order
        self.load_sources(sources)
        self.load_references(references)

    def load_hypothesis_sentence(self, hypothesis):
        self.hypothesislen = len(hypothesis)
        self.this_hypothesis_ngrams = [self.get_ngram_counts(hypothesis, n) for n in range(1, self.order + 1)]

    def load_sources(self, sources):
        self.all_source_ngrams = [
            [self.get_ngram_counts(line.split(), n) for n in range(1, self.order + 1)] for line in sources
        ]

    def load_references(self, references):
        self.refs = [[] for _ in range(len(self.all_source_ngrams))]
        self.reflens = [[] for _ in range(len(self.all_source_ngrams))]
        for reference in references:
            for i, line in enumerate(reference):
                self.refs[i].append(line.split())
                self.reflens[i].append(len(line.split()))

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
    def get_ngram_diff(self, a, b):
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
    def gleu_stats(self, i, r_ind=None):
        yield self.hypothesislen
        yield self.reflens[i][r_ind]

        for n in range(1, self.order + 1):
            hypothesis_ngrams = self.this_hypothesis_ngrams[n - 1]
            source_ngrams = self.all_source_ngrams[i][n - 1]
            reference_ngrams = self.get_ngram_counts(self.refs[i][r_ind], n)

            source_ngram_diff = self.get_ngram_diff(source_ngrams, reference_ngrams)

            yield max([sum((hypothesis_ngrams & reference_ngrams).values()) - sum((hypothesis_ngrams & source_ngram_diff).values()), 0])

            yield max([self.hypothesislen + 1 - n, 0])

    # Compute GLEU from collected statistics obtained by call(s) to gleu_stats
    def gleu(self, stats, smooth=False):
        # smooth 0 counts for sentence-level scores
        if smooth:
            stats = [s if s != 0 else 1 for s in stats]
        if len([stat for stat in stats if stat == 0]) > 0:
            return 0
        (c, r) = stats[:2]
        log_gleu_prec = sum([math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]) / 4
        return math.exp(min([0, 1 - float(r) / c]) + log_gleu_prec)
