# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# -*- coding:utf-8 -*-
import argparse
from pathlib import Path
from typing import List, Tuple

from . import calc_codegleu

PACKAGE_DIR = Path(__file__).parent


def main(
    src_file: str,
    ref_files: List[str],
    hyp_file: str,
    lang: str,
    weights: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
) -> None:
    pre_references = [[x.strip() for x in open(file, "r", encoding="utf-8").readlines()] for file in ref_files]
    hypotheses = [x.strip() for x in open(hyp_file, "r", encoding="utf-8").readlines()]
    sources = [x.strip() for x in open(src_file, "r", encoding="utf-8").readlines()]

    for i in range(len(pre_references)):
        assert len(hypotheses) == len(pre_references[i]) == len(sources)

    references = []
    for i in range(len(hypotheses)):
        ref_for_instance = []
        for j in range(len(pre_references)):
            ref_for_instance.append(pre_references[j][i])
        references.append(ref_for_instance)
    assert len(references) * len(references[0]) == len(pre_references) * len(
        hypotheses
    ), "References must be converted from column-wise to row-wise"

    code_gleu_score = calc_codegleu(
        sources,
        references,
        hypotheses,
        lang,
        weights=weights,
    )

    print(
        f"ngram_match: {code_gleu_score['ngram_match_score']}",
        f"weighted_ngram_match: {code_gleu_score['weighted_ngram_match_score']}",
        f"syntax_match: {code_gleu_score['syntax_match_score']}",
        f"dataflow_match: {code_gleu_score['dataflow_match_score']}",
    )

    print("codegleu score: ", code_gleu_score["codegleu"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--refs", type=str, nargs="+", required=True, help="reference files")
    parser.add_argument("--hyp", type=str, required=True, help="hypothesis file")
    parser.add_argument("--src", type=str, required=True, help="source file")
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        choices=["java", "js", "c_sharp", "php", "go", "python", "ruby", "rust"],
    )
    parser.add_argument("--params", type=str, default="0.25,0.25,0.25,0.25", help="alpha, beta, gamme and delta")

    args = parser.parse_args()

    lang = args.lang
    alpha, beta, gamma, theta = [float(x) for x in args.params.split(",")]

    main(
        args.src,
        args.refs,
        args.hyp,
        args.lang,
        weights=(alpha, beta, gamma, theta),
    )
