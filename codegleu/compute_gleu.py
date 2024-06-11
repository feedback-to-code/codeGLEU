#!/usr/bin/env python

# Courtney Napoles
# <courtneyn@jhu.edu>
# 21 June 2015
# ##
# compute_gleu
#
# This script calls gleu.py to calculate the GLEU score of a sentence, as
# described in our ACL 2015 paper, Ground Truth for Grammatical Error
# Correction Metrics by Courtney Napoles, Keisuke Sakaguchi, Matt Post,
# and Joel Tetreault.
#
# For instructions on how to get the GLEU score, call "compute_gleu -h"
#
# Updated 2 May 2016: This is an updated version of GLEU that has been
# modified to handle multiple references more fairly.
#
# This script was adapted from compute-bleu by Adam Lopez.
# <https://github.com/alopez/en600.468/blob/master/reranker/>

import argparse
import sys
import os
from gleu import GLEU
import scipy.stats
import numpy as np
import random


def get_gleu_stats(scores):
    mean = np.mean(scores)
    std = np.std(scores) if len(scores) > 1 else 0
    ci = scipy.stats.norm.interval(0.95, loc=mean, scale=std) if len(scores) > 1 else (0,0)
    return ["%f" % mean, "%f" % std, "(%.3f,%.3f)" % (ci[0], ci[1])]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--reference",
        help="Target language reference sentences. Multiple " "files for multiple references.",
        nargs="*",
        dest="reference",
        required=True,
    )
    parser.add_argument("-s", "--source", help="Source language source sentences", dest="source", required=True)
    parser.add_argument(
        "-o",
        "--hypothesis",
        help="Target language hypothesis sentences to evaluate "
        "(can be more than one file--the GLEU score of each "
        "file will be output separately). Use '-o -' to read "
        "hypotheses from stdin.",
        nargs="*",
        dest="hypothesis",
        required=True,
    )
    parser.add_argument("-n", help="Maximum order of ngrams", type=int, default=4)
    parser.add_argument("-d", "--debug", help="Debug; print sentence-level scores", default=False, action="store_true")

    args = parser.parse_args()

    source = open(args.source).readlines()
    references = [open(ref).readlines() for ref in args.reference]

    gleu_calculator = GLEU(source, references, args.n)

    for hpath in args.hypothesis:
        instream = sys.stdin if hpath == "-" else open(hpath)
        hyp = [line.split() for line in instream]

        if not args.debug:
            print(os.path.basename(hpath))

        if args.debug:
            print("===== Sentence-level scores =====")
            print("SID Mean Stdev 95%CI GLEU")

        refnum = len(args.reference)
        iter_stats = [[0,0]*(args.n+1)] * refnum

        for i, h in enumerate(hyp):
            gleu_calculator.load_hypothesis_sentence(h)

            stats_by_ref = [[]] * refnum
            for ref in range(refnum):
                stats_by_ref[ref] = list(gleu_calculator.gleu_stats(i, r_ind=ref))
                iter_stats[ref] = [sum(scores) for scores in zip(iter_stats[ref], stats_by_ref[ref])]

            if args.debug:
                # sentence-level GLEU is the mean GLEU of the hypothesis
                # compared to each reference
                for r in range(refnum):
                    if stats_by_ref[r] is None:
                        stats_by_ref[r] = [s for s in gleu_calculator.gleu_stats(i, r_ind=r)]

                print(i, h)
                print(" ".join(get_gleu_stats([gleu_calculator.gleu(stats, smooth=True) for stats in stats_by_ref])))

        if args.debug:
            print("\n==== Overall score =====")
            print("Mean Stdev 95%CI GLEU")
            print(" ".join(get_gleu_stats([gleu_calculator.gleu(stats) for stats in iter_stats])))
        else:
            print(get_gleu_stats([gleu_calculator.gleu(stats) for stats in iter_stats])[0])
