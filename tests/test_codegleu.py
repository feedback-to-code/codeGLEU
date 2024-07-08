import inspect
import logging
from typing import Any, List

import pytest

from codegleu.diffsim import AVAILABLE_LANGS, calc_diffsim


@pytest.mark.parametrize(
    ["sources", "predictions", "references", "codegleu"],
    [
        (
            ["def foo ( x ) :\n    return x"],
            ["some rannnndom words in length more than 3"],
            ["def test ( ) :\n pass"],
            0.25,
        ),  # cause data_flow=1
        (
            ["def foo ( x ) :\n    return x"],
            ["def bar ( y , x ) :\n    a = x * x\n    return a"],
            ["def foo ( x ) :\n    return x"],
            0.3,
        ),
        (
            ["def foo ( x ) :\n    return x"],
            ["def foo ( x ) :\n    return x * x"],
            ["def bar ( x ) :\n    return x"],
            0.37,
        ),
        (["def foo ( x ) :\n    return x"], ["def bar ( x ) :\n    return x"], ["def foo ( x ) :\n    return x"], 0.80),
        (["def foo ( y ) :\n    return y"], ["def foo ( x ) :\n    return x"], ["def foo ( x ) :\n    return x"], 1.0),
    ],
)
def test_simple_cases(sources: List[Any], predictions: List[Any], references: List[Any], codegleu: float) -> None:
    result = calc_diffsim(sources, references, predictions, "python")
    logging.debug(result)
    assert result["codegleu"] == pytest.approx(codegleu, 0.01)


@pytest.mark.parametrize(["lang"], [(lang,) for lang in AVAILABLE_LANGS])
def test_exact_match_works_for_all_langs(lang: str) -> None:
    source = predictions = references = ["some matching string a couple of times"]
    assert calc_diffsim(source, references, predictions, lang)["codegleu"] == 1.0


@pytest.mark.parametrize(
    ["lang", "sources", "predictions", "references"],
    [
        (
            "python",
            ["def foo ( x ) :\n    return x"],
            ["def foo ( x ) :\n    return x"],
            ["def bar ( y ) :\n    return y"],
        ),
        (
            "java",
            ["public function foo ( x ) { return x }"],
            ["public function foo ( x ) { return x }"],
            ["public function bar ( y ) {\n   return y\n}"],
        ),
        (
            "javascript",
            ["function foo ( x ) { return x }"],
            ["function foo ( x ) { return x }"],
            ["function bar ( y ) {\n   return y\n}"],
        ),
        (
            "c",
            ["int foo ( int x ) { return x }"],
            ["int foo ( int x ) { return x }"],
            ["int bar ( int y ) {\n   return y\n}"],
        ),
        (
            "c_sharp",
            ["public int foo ( int x ) { return x }"],
            ["public int foo ( int x ) { return x }"],
            ["public int bar ( int y ) {\n   return y\n}"],
        ),
        (
            "cpp",
            ["int foo ( int x ) { return x }"],
            ["int foo ( int x ) { return x }"],
            ["int bar ( int y ) {\n   return y\n}"],
        ),
        (
            "php",
            ["function foo ( x ) { return x }"],
            ["function foo ( x ) { return x }"],
            ["function bar ( y ) {\n   return y\n}"],
        ),
        ("go", ["func foo ( x ) { return x }"], ["func foo ( x ) { return x }"], ["func bar ( y ) {\n   return y\n}"]),
        (
            "ruby",
            ["def foo ( x ) :\n    return x"],
            ["def foo ( x ) :\n    return x"],
            ["def bar ( y ) :\n    return y"],
        ),
        ("rust", ["fn foo ( x ) -> i32 { x }"], ["fn foo ( x ) -> i32 { x }"], ["fn bar ( y ) -> i32 { y }"]),
    ],
)
def test_simple_cases_work_for_all_langs(lang: str, sources: list[Any], predictions: List[Any], references: List[Any]) -> None:
    result = calc_diffsim(sources, references, predictions, lang)
    logging.debug(result)
    assert result["codegleu"] == pytest.approx(0.55, 0.1)


def test_error_when_lang_not_supported() -> None:
    with pytest.raises(AssertionError):
        calc_diffsim(["def foo : pass"], ["def foo : pass"], ["def bar : pass"], "not_supported_lang")


def test_error_when_input_length_mismatch() -> None:
    with pytest.raises(AssertionError):
        calc_diffsim(["def foo : pass"], ["def foo : pass"], ["def bar : pass", "def buz : pass"], "python")


@pytest.mark.parametrize(
    ["sources", "predictions", "references", "codegleu"],
    [
        (["def foo ( x ) : pass"], ["def foo ( x ) : pass"], ["def foo ( x ) : pass"], 1.0),
        (["def foo ( x ) : pass"], ["def foo ( x ) : pass"], [["def foo ( x ) : pass"]], 1.0),
        (["def foo ( x ) : pass"], ["def foo ( x ) : pass"], [["def bar ( x ) : pass", "def foo ( x ) : pass"]], 0.75),
        (["def foo ( x ) : pass"], ["def foo ( x ) : pass"], [["def foo ( x ) : pass", "def bar ( x ) : pass"]], 0.75),
    ],
)
def test_input_variants(sources: List[Any], predictions: List[Any], references: List[Any], codegleu: float) -> None:
    assert calc_diffsim(sources, references, predictions, "python")["codegleu"] == pytest.approx(codegleu, 0.01)


# TODO: fix this test
# @pytest.mark.timeout(1)
def test_finite_processing_time_in_bug_testcase() -> None:
    dummy_source = inspect.cleandoc(
        """
        def bar(n):
            pass
    """
    )
    dummy_true_code = inspect.cleandoc(
        """
        def foo(n):
            pass
    """
    )
    generated_code = inspect.cleandoc(
        """
        def foo(n):
           for i in range(n):
               for j in range(n):
                   for k in range(n):
                       for l in range(n):
                           for m in range(n):
                               for n in range(n):
                                   for o in range(n):
                                       for p in range(n):
                                           for q in range(n):
                                               for r in range(n):
                                                   for s in range(n):
                                                       for t in range(n):
                                   #                         for u in range(n):
                                   #                             for v in range(n):
                                   #                                 for w in range(n):
                                   #                                     for x in range(n):
                                   #                                         for y in range(n):
                                   #                                             for z in range(n):
                                   #                                                 for a in range(n):
                                   #                                                     for b in range(n):
                                   #                                                         for c in range(n):
                                   #                                                             for d in range(n):
                                   #                                                               for e in range(n):
                                   #                                                               for f in range(n):
                                   #                                                               for g in range(n):
                                   #                                                               for h in range(n):
                                   #                                                               for i in range(n):
                                   #                                                               for j in range(n):
                                   #                                                               for k in range(n):
                                   #                                                               for l in range(n):
                                   #                                                               for m in range(n):
                                   #                                                               for n in range(n):
                                   #                                                               for o in range(n):
                                   #                                                               for p in range(n):
                                   #                                                               for q in range(n):
                                   #                                                               for r in range(n):
                                   #                                                               for s
    """
    )

    # just test finite processing time
    calc_diffsim([dummy_source], [dummy_true_code], [generated_code], "python")


# TODO: add tests with direct comparison with XLCoST and CodeXGlue results
