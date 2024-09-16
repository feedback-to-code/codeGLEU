from enum import Enum
from typing import Tuple

import regex as re
import tiktoken


# TODO: this does not seem very adaptable
class BlockType(int, Enum):
    IMPORTS = 1
    CLOSED = 2
    FUNCTION = 3
    FREECODE = 4  # code outside solid blocks
    CLASS = 5
    METHOD = 6  # a method inside a class
    CLASS_FREECODE = 7
    BLOCK_IMPORT = 8  # see (+)
    LARGE_FUNCTION = 9
    LARGE_FREECODE = 10
    LARGE_CLASS = 11
    LARGE_METHOD = 12
    LARGE_CLASS_FREECODE = 13


# TODO: simplify this method - flake8 C901
def generate_snippets(
    file: str,
    create_class_subsnippets: bool = True,
    include_class: bool = False,
) -> Tuple[list[str], list[BlockType]]:
    lines: list[str] = file.split("\n")  # files contain header
    lastline = next((l for l in lines.__reversed__() if l != ""), lines[-1])
    indentlevel = len(lastline) - len(lastline.lstrip())
    for i in range(0, indentlevel + 1):
        lines.append(" " * (indentlevel - i))
    snippets: list[str] = []
    types: list[BlockType] = []
    current_snippet: str = ""
    # subsnippets are for functions in classes
    current_subsnippet: str = ""
    subsnippet_indentation_level: int = 0
    current_block_type: BlockType = BlockType.IMPORTS  # we assume file starts with imports

    def append_values(subsnippet: bool) -> None:
        nonlocal current_snippet
        nonlocal current_subsnippet
        nonlocal current_block_type
        nonlocal snippets
        nonlocal types

        content = current_subsnippet if subsnippet else current_snippet
        enc = tiktoken.get_encoding("cl100k_base")
        parts = []
        # we don't want comment snippets
        if content.strip().startswith('"""') or content.strip().startswith("'''") or content == "":
            if subsnippet:
                current_subsnippet = ""
            else:
                current_snippet = ""

            current_block_type = BlockType.CLOSED
            return None

        # we don't want snippets >8000 tokens
        if len(enc.encode(content)) > 8000:
            current_part = ""
            for line in content.split("\n"):
                if (len(enc.encode(current_part)) + len(enc.encode(line))) > 8000:
                    parts.append(current_part)
                    current_part = ""

                current_part += line + "\n"
        else:
            parts.append(content)
        for part in parts:
            snippets.append(part)
            if len(parts) > 1:
                types.append(BlockType(current_block_type.value + 6))  # make LARGE_<type>
            else:
                types.append(current_block_type)

        if subsnippet:
            current_subsnippet = ""
        else:
            current_snippet = ""

        current_block_type = BlockType.CLOSED

    # TODO: I d guess we are not the first ones who want to parse python... why not use a potential library?
    for line in lines:
        if line == "":
            continue

        # case that method is also end of class
        if current_block_type == BlockType.METHOD and line.lstrip() == line:
            append_values(True)
            # include entire class as snippet
            if include_class:
                current_block_type = BlockType.CLASS
                append_values(False)

        # ending conditions for top-level-blocks
        if (
            (current_block_type == BlockType.IMPORTS and "import" not in line)
            or (current_block_type == BlockType.FUNCTION and line.lstrip() == line)
            or (current_block_type == BlockType.FREECODE and "def" in line)
            or (current_block_type == BlockType.FREECODE and "class" in line)
            or (current_block_type == BlockType.CLASS and line.lstrip() == line)
        ):
            append_values(False)

        """
        (+)
        from ... import (
            a,
            b,
            c
        )
        """
        if current_block_type == BlockType.BLOCK_IMPORT and ")" in line:
            current_block_type = BlockType.IMPORTS

        # methods inside classes are weird because the indentations can be messed up
        if (
            current_block_type == BlockType.METHOD
            and len(line) - len(line.lstrip()) == subsnippet_indentation_level
            and ("):" not in line or "def" in line)
        ):
            append_values(True)
            current_block_type = BlockType.CLASS

        # end class_freecode block
        if current_block_type == BlockType.CLASS_FREECODE and "def" in line:
            append_values(True)
            current_block_type = BlockType.CLASS

        # see (+)
        if current_block_type == BlockType.IMPORTS and "(" in line:
            current_block_type = BlockType.BLOCK_IMPORT

        current_snippet += line + "\n"

        # determine the type of following top-level-block
        if current_block_type == BlockType.CLOSED:
            if "def" in line:
                current_block_type = BlockType.FUNCTION
            elif "class" in line:
                current_block_type = BlockType.CLASS
            else:
                current_block_type = BlockType.FREECODE

        # determine sub-level-block
        if current_block_type == BlockType.CLASS:
            if "def" in line and create_class_subsnippets:
                current_block_type = BlockType.METHOD
                subsnippet_indentation_level = len(line) - len(line.lstrip())
            elif "def" not in line and "class" not in line and create_class_subsnippets:
                current_block_type = BlockType.CLASS_FREECODE

        if current_block_type == BlockType.METHOD or current_block_type == BlockType.CLASS_FREECODE:
            current_subsnippet += line + "\n"

    if len(snippets) == 0:
        current_block_type = BlockType.FREECODE
        append_values(False)

    return snippets, types
