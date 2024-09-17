# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
from collections import Counter

from tree_sitter import Parser

from .parser import DFG_csharp, DFG_go, DFG_java, DFG_javascript, DFG_php, DFG_python, DFG_ruby, DFG_rust, try_remove_comments_and_docstrings
from .utils import get_tree_sitter_language, multirefscores

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


def get_all_sub_trees(root_node):
    node_stack = []
    sub_tree_sexp_list = []
    depth = 1
    node_stack.append([root_node, depth])
    while len(node_stack) != 0:
        cur_node, cur_depth = node_stack.pop()
        sub_tree_sexp_list.append(str(cur_node))
        for child_node in cur_node.children:
            if len(child_node.children) != 0:
                depth = cur_depth + 1
                node_stack.append([child_node, depth])
    return sub_tree_sexp_list


def calc_dataflow_match(source: str, references: list[str], hypothesis: str, penalty: float, lang: str, langso_so_file):
    return corpus_syntax_score(corpus_syntax_intermediate([source], [references], [hypothesis], lang, langso_so_file), penalty)


def corpus_syntax_intermediate(
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

    intermediates: dict[str, list] = {
        "s_interm": [],
        "h_interm": [],
        "r_interms": [],
    }

    for source, references_sample, hypothesis in zip(sources, references, hypotheses):
        source = try_remove_comments_and_docstrings(source, lang)
        source_tree = parser.parse(bytes(source, "utf8")).root_node
        source_subexp = Counter(map(str, get_all_sub_trees(source_tree)))

        hypothesis = try_remove_comments_and_docstrings(hypothesis, lang)
        hypothesis_tree = parser.parse(bytes(hypothesis, "utf8")).root_node
        hypothesis_subexp = Counter(map(str, get_all_sub_trees(hypothesis_tree)))

        intermediates["s_interm"] += [source_subexp]
        intermediates["h_interm"] += [hypothesis_subexp]
        refs = []
        for reference in references_sample:
            reference = try_remove_comments_and_docstrings(reference, lang)
            reference_tree = parser.parse(bytes(reference, "utf8")).root_node
            reference_subexp = Counter(map(str, get_all_sub_trees(reference_tree)))
            refs += [reference_subexp]
        intermediates["r_interms"] += [refs]
    return intermediates


# very similar to dataflow match, might merge later
def corpus_syntax_score(
    intermediates: dict[str, list],
    penalty: float = 1,
) -> float:
    refs = {}
    
    for source, references_sample, hypothesis in zip(intermediates["s_interm"], intermediates["r_interms"], intermediates["h_interm"]):
        for index, reference in enumerate(references_sample):
            source_interm = Counter(source)
            reference_interm = Counter(reference)
            hypothesis_interm = Counter(hypothesis)

            ref_added = reference_interm - source_interm
            ref_removed = source_interm - reference_interm
            hyp_added = hypothesis_interm - source_interm
            hyp_removed = source_interm - hypothesis_interm
            
            correct_changes = ((ref_added & hyp_added) + (ref_removed & hyp_removed)).total()
            wrong_changes = ((hyp_added - ref_added) + (hyp_removed - ref_removed)).total()
            total_changes = (ref_added + ref_removed).total()

            if index not in refs: refs[index] = [0, 0]
            refs[index][0] += max(0, correct_changes - penalty * wrong_changes)
            refs[index][1] += total_changes

    scores = [(v[0] / v[1]) if v[1] else -1 for _, v in sorted(refs.items())]

    return multirefscores(scores)
