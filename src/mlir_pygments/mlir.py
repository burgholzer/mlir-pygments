# Copyright (c) 2025 Lukas Burgholzer
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Custom pygments lexer for MLIR.

Adapted from https://gist.github.com/oowekyala/110dacc9343dbc1b86b452902d7dc553
Released under the MIT License.
"""

from typing import Any, ClassVar

from pygments.lexer import RegexLexer, bygroups, include
from pygments.token import Comment, Keyword, Literal, Name, Punctuation, Whitespace

__all__ = ["MLIRLexer"]


def __dir__() -> list[str]:
    return __all__


digit = r"[0-9]"
hex_digit = r"[0-9a-fA-F]"
letter = r"[a-zA-Z]"

decimal_literal = rf"{digit}+"

bare_id = rf"(?:{letter}|_)(?:{letter}|{digit}|[_$.])*"
bare_id_with_ns = rf"((?:{letter}|_)(?:{letter}|{digit}|[_$])*)(\.)((?:{letter}|{digit}|[_$.])+)"

integer_type = rf"[su]?i{digit}+"
float_type = r"(?:f(?:16|32|64|80|128)|bf16|tf32|f8E5M2|f8E4M3FN|f8E5M2FNUZ|f8E5M3FNUZ|f8E4M3B11FNUZ)"


class MLIRLexer(RegexLexer):
    """A Pygments lexer for MLIR."""

    name: ClassVar[str] = "MLIR"
    aliases: ClassVar[list[str]] = ["mlir"]
    filenames: ClassVar[list[str]] = ["*.mlir"]

    tokens: ClassVar[dict[str, Any]] = {
        "comments": [
            (r"//.*?\n", Comment),
            (r"\.\.\.", Comment),  # pretend ellipsis is comment
        ],
        "literals": [
            (r"[-+]?[0-9]+[.][0-9]*([eE][-+]?[0-9]+)?", Literal.Number),
            (rf"0x{hex_digit}+", Literal.Number),
            (decimal_literal, Literal.Number),
            (r'"[^"\n\f\v\r]*"', Literal.String),
            (r"[^\S\r\n]+", Whitespace),
        ],
        "punctuation": [
            (r"[()\[\],*+?{}<>-\|:]|->|<=?|>=?|==?|::", Punctuation),
        ],
        "sigils": [
            (rf"\^{bare_id}", Name.Label),
            (rf"%{bare_id}", Name.Variable),
            (rf"%{decimal_literal}", Name.Variable),
            (rf"@{bare_id}", Name.Variable.Global),
            (rf"!{bare_id}|!{bare_id_with_ns}", Name.Type),
            (rf"#{bare_id}|#{bare_id_with_ns}", Name.Attribute),
            (rf"{bare_id_with_ns}", bygroups(Name.Namespace, Punctuation, Name.Function)),
            (rf"({integer_type}|{float_type}|index|tensor|memref)\b", Keyword.Type),
            (rf"(x)({decimal_literal})", bygroups(Punctuation, Literal.Number)),
            (rf"(x)({integer_type}|{float_type})\b", bygroups(Punctuation, Keyword.Type)),
            (rf"{bare_id}", Name.Identifier),
        ],
        "root": [
            include("sigils"),
            include("punctuation"),
            include("literals"),
            include("comments"),
        ],
    }
