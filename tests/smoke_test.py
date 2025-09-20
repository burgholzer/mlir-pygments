# Copyright (c) 2025 Lukas Burgholzer
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Check that basic features work."""

from mlir_pygments import MLIRLexer


def test_smoke() -> None:
    """A smoke test to check that the lexer can be instantiated and used."""
    code = """// This is a hello world quantum program that creates a Bell state and measures it.
module {
    func.func @bellState() {
        %qreg = "mqtref.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtref.QubitRegister
        %q0 = "mqtref.extractQubit"(%qreg) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
        %q1 = "mqtref.extractQubit"(%qreg) <{index_attr = 1 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit

        mqtref.h() %q0
        mqtref.x() %q1 ctrl %q0
        %m0 = mqtref.measure %q0
        %m1 = mqtref.measure %q1

        "mqtref.deallocQubitRegister"(%qreg) : (!mqtref.QubitRegister) -> ()

        return
    }
}"""
    lexer = MLIRLexer()
    tokens = list(lexer.get_tokens(code))
    assert len(tokens) > 0
