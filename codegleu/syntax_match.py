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
    try_remove_comments_and_docstrings,
    tree_to_token_index,
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


def calc_syntax_match(source: str, references: list[str], candidate: str, lang: str):
    return corpus_syntax_match([source], [references], [candidate], lang)


# very similar to dataflow match, might merge later
def corpus_syntax_match(
    sources: list[str], references: list[list[str]], hypotheses: list[str], lang: str, tree_sitter_language=None
) -> float:
    if not tree_sitter_language:
        tree_sitter_language = get_tree_sitter_language(lang)

    parser = Parser()
    parser.language = tree_sitter_language

    match_count = 0
    total_count = 0

    for source, references_sample, hypothesis in zip(sources, references, hypotheses):
        source = try_remove_comments_and_docstrings(source, lang)
        source_tree = parser.parse(bytes(source, "utf8")).root_node
        source_subexp = Counter(get_all_sub_trees(source_tree))

        hypothesis = try_remove_comments_and_docstrings(hypothesis, lang)
        hypothesis_tree = parser.parse(bytes(hypothesis, "utf8")).root_node
        hypothesis_subexp = Counter(get_all_sub_trees(hypothesis_tree))

        for reference in references_sample:

            reference = try_remove_comments_and_docstrings(reference, lang)
            reference_tree = parser.parse(bytes(reference, "utf8")).root_node
            reference_subexp = Counter(get_all_sub_trees(reference_tree))

            source_subexp_diff = counter_diff(source_subexp, reference_subexp)

            matching_subexp = (hypothesis_subexp & reference_subexp).total()
            penalty_subexp = (hypothesis_subexp & source_subexp_diff).total()
            score = matching_subexp - penalty_subexp

            match_count += max(score, 0)
            total_count += reference_subexp.total()
    if total_count == 0:
        logging.warning(
            "WARNING: There is no reference syntax extracted from the whole corpus, "
            "and the syntax match score degenerates to 0. Please consider ignoring this score."
        )
        return 0
    score = match_count / total_count
    return score
