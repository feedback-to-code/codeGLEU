# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

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
}


def counter_diff(a, b):
    diff = Counter(a)
    for k in set(a) & set(b):
        del diff[k]
    return diff


def calc_syntax_match(source: str, references: list[str], candidate: str, lang: str):
    return corpus_syntax_match([source], [references], [candidate], lang)


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

        hypothesis = try_remove_comments_and_docstrings(hypothesis, lang)
        hypothesis_tree = parser.parse(bytes(hypothesis, "utf8")).root_node

        for reference in references_sample:

            reference = try_remove_comments_and_docstrings(reference, lang)
            reference_tree = parser.parse(bytes(reference, "utf8")).root_node

            def get_all_sub_trees(root_node):
                node_stack = []
                sub_tree_sexp_list = []
                depth = 1
                node_stack.append([root_node, depth])
                while len(node_stack) != 0:
                    cur_node, cur_depth = node_stack.pop()
                    sub_tree_sexp_list.append([str(cur_node), cur_depth])
                    for child_node in cur_node.children:
                        if len(child_node.children) != 0:
                            depth = cur_depth + 1
                            node_stack.append([child_node, depth])
                return sub_tree_sexp_list
            
            nodes_only = lambda t: [x[0] for x in get_all_sub_trees(t)]

            source_subexp = Counter(nodes_only(source_tree))
            hypothesis_subexp = Counter(nodes_only(hypothesis_tree))
            reference_subexp = Counter(nodes_only(reference_tree))

            source_subexp_diff = counter_diff(source_subexp, reference_subexp)

            matching_subexp = (hypothesis_subexp & reference_subexp).total()
            penalty_subexp = (hypothesis_subexp & source_subexp_diff).total()
            score = matching_subexp - penalty_subexp

            match_count += max(score, 0)
            total_count += len(reference_subexp)
    score = match_count / total_count
    return score
