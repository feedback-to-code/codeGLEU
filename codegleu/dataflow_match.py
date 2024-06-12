# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
from collections import Counter

from tree_sitter import Parser

from .parser import (
    DFG_csharp,
    DFG_go,
    DFG_java,
    DFG_javascript,
    DFG_php,
    DFG_python,
    DFG_ruby,
    DFG_rust,
    index_to_code_token,
    tree_to_token_index,
    try_remove_comments_and_docstrings,
)
from .utils import get_tree_sitter_language

dfg_function = {
    "python": DFG_python,
    "java": DFG_java,
    "ruby": DFG_ruby,
    "go": DFG_go,
    "php": DFG_php,
    "javascript": DFG_javascript,
    "c_sharp": DFG_csharp,
    "c": DFG_csharp,  # XLCoST uses C# parser for C
    "cpp": DFG_csharp,  # XLCoST uses C# parser for C++
    "rust": DFG_rust,
}


def counter_diff(a, b):
    diff = Counter(a)
    for k in set(a) & set(b):
        del diff[k]
    return diff


def calc_dataflow_match(source: str, references: list[str], hypothesis: str, lang: str, langso_so_file):
    return corpus_dataflow_match([source], [references], [hypothesis], lang, langso_so_file)


# very similar to syntax match, might merge later
def corpus_dataflow_match(
    sources: list[str], references: list[list[str]], hypotheses: list[str], lang: str, tree_sitter_language=None
) -> float:
    if not tree_sitter_language:
        tree_sitter_language = get_tree_sitter_language(lang)

    parser = Parser()
    parser.language = tree_sitter_language
    parser = [parser, dfg_function[lang]]  # type: ignore[assignment]
    match_count = 0
    total_count = 0

    for source, references_sample, hypothesis in zip(sources, references, hypotheses):
        source = try_remove_comments_and_docstrings(source, lang)
        source_dfg = get_data_flow(source, parser)
        source_dfg_norm = Counter(map(str, normalize_dataflow(source_dfg)))

        hypothesis = try_remove_comments_and_docstrings(hypothesis, lang)
        hypothesis_dfg = get_data_flow(hypothesis, parser)
        hypothesis_dfg_norm = Counter(map(str, normalize_dataflow(hypothesis_dfg)))

        for reference in references_sample:

            reference = try_remove_comments_and_docstrings(reference, lang)
            reference_dfg = get_data_flow(reference, parser)
            reference_dfg_norm = Counter(map(str, normalize_dataflow(reference_dfg)))

            source_dfg_diff = counter_diff(source_dfg_norm, reference_dfg_norm)

            matching_dataflow = (hypothesis_dfg_norm & reference_dfg_norm).total()
            penalty_dataflow = (hypothesis_dfg_norm & source_dfg_diff).total()
            score = matching_dataflow - penalty_dataflow

            match_count += max(score, 0)
            total_count += reference_dfg_norm.total()
    if total_count == 0:
        logging.warning(
            "WARNING: There is no reference data-flows extracted from the whole corpus, "
            "and the data-flow match score degenerates to 0. Please consider ignoring this score."
        )
        return 0
    score = match_count / total_count
    return score


def get_data_flow(code, parser):
    try:
        tree = parser[0].parse(bytes(code, "utf8"))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
        code = code.split("\n")
        code_tokens = [index_to_code_token(x, code) for x in tokens_index]
        index_to_code = {}
        for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx, code)
        try:
            DFG, _ = parser[1](root_node, index_to_code, {})
        except Exception:
            DFG = []
        DFG = sorted(DFG, key=lambda x: x[1])
        indexs = set()
        for d in DFG:
            if len(d[-1]) != 0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG = []
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg = new_DFG
    except Exception:
        code.split()
        dfg = []
    # merge nodes
    dic = {}
    for d in dfg:
        if d[1] not in dic:
            dic[d[1]] = d
        else:
            dic[d[1]] = (
                d[0],
                d[1],
                d[2],
                list(set(dic[d[1]][3] + d[3])),
                list(set(dic[d[1]][4] + d[4])),
            )
    DFG = []
    for d in dic:
        DFG.append(dic[d])
    dfg = DFG
    return dfg


def normalize_dataflow(dataflow):
    var_dict = {}
    i = 0
    normalized_dataflow = []
    for item in dataflow:
        var_name = item[0]
        relationship = item[2]
        par_vars_name_list = item[3]
        for name in par_vars_name_list:
            if name not in var_dict:
                var_dict[name] = "var_" + str(i)
                i += 1
        if var_name not in var_dict:
            var_dict[var_name] = "var_" + str(i)
            i += 1
        normalized_dataflow.append(
            (
                var_dict[var_name],
                relationship,
                [var_dict[x] for x in par_vars_name_list],
            )
        )
    return normalized_dataflow
