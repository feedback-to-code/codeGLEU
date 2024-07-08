# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
from collections import Counter

from tree_sitter import Parser

from codegleu.parser import (
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
from codegleu.utils import get_tree_sitter_language

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


def calc_dataflow_match(source: str, references: list[str], hypothesis: str, penalty: float, lang: str, langso_so_file):
    return corpus_dataflow_score(corpus_dataflow_intermediate([source], [references], [hypothesis], lang, langso_so_file), penalty)


def corpus_dataflow_intermediate(
    sources: list[str],
    references: list[list[str]],
    hypotheses: list[str],
    lang: str,
    tree_sitter_language=None,
) -> dict[str, list]:
    if not tree_sitter_language:
        tree_sitter_language = get_tree_sitter_language(lang)

    parser = Parser()
    parser.language = tree_sitter_language
    parser = [parser, dfg_function[lang]]  # type: ignore[assignment]

    intermediates: dict[str, list] = {
        "s_interm": [],
        "h_interm": [],
        "r_interms": [],
    }

    for source, references_sample, hypothesis in zip(sources, references, hypotheses):
        source = try_remove_comments_and_docstrings(source, lang)
        source_dfg = get_data_flow(source, parser)
        source_dfg_norm = normalize_dataflow(source_dfg)
        source_dfg_count = Counter(map(str, source_dfg))

        hypothesis = try_remove_comments_and_docstrings(hypothesis, lang)
        hypothesis_dfg = get_data_flow(hypothesis, parser)
        hypothesis_dfg_norm = normalize_dataflow(hypothesis_dfg)
        hypothesis_dfg_count = Counter(map(str, hypothesis_dfg))

        intermediates["s_interm"] += [source_dfg_count]
        intermediates["h_interm"] += [hypothesis_dfg_count]
        refs = []
        for reference in references_sample:
            reference = try_remove_comments_and_docstrings(reference, lang)
            reference_dfg = get_data_flow(reference, parser)
            reference_dfg_norm = normalize_dataflow(reference_dfg)
            reference_dfg_count = Counter(map(str, reference_dfg))
            refs += [reference_dfg_count]
        intermediates["r_interms"] += [refs]
    return intermediates


# very similar to syntax match, might merge later
def corpus_dataflow_score(
    intermediates: dict[str, list],
    penalty: float = 1,
) -> float:
    match_count = 0
    total_count = 0

    for source_interm, reference_interms, hypothesis_interm in zip(intermediates["s_interm"], intermediates["r_interms"], intermediates["h_interm"]):
        for reference_interm in reference_interms:
            source_interm = Counter(source_interm)
            reference_interm = Counter(reference_interm)
            hypothesis_interm = Counter(hypothesis_interm)

            source_subexp_diff = counter_diff(source_interm, reference_interm)
            matching_subexp = (hypothesis_interm & reference_interm).total()
            penalty_subexp = (hypothesis_interm & source_subexp_diff).total()
            score = matching_subexp - penalty * penalty_subexp

            match_count += max(0, score)
            total_count += max(0, reference_interm.total())

    if total_count == 0:
        # logging.warning(
        #     "WARNING: There is no reference data-flows extracted from the whole corpus, "
        #     "and the data-flow match score degenerates to 0. Please consider ignoring this score."
        # )
        return -1.0
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
