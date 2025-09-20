# Pygments syntax highlighting for MLIR

Pygments lexer for the Multi-Level Intermediate Representation (MLIR) used in compiler infrastructures.
Provides syntax highlighting in terminals, HTML, Sphinx, and other Pygments-powered tools.

## Features

- Highlights MLIR core syntax: operations (incl. dialect ops), types, attributes, SSA values, blocks/regions, and comments.
- Works with pygmentize (CLI), Python APIs, and Sphinx code blocks.
- Lightweight dependency on Pygments.

## Installation

Install the latest release from PyPI:

```bash
uv add pygments-mlir
```

This also installs the `pygments` package if not already installed.

Quick check that the lexer is registered:

```bash
pygmentize -L lexers | grep -i mlir
```

## Usage

- CLI (terminal colors):

```bash
pygmentize -l mlir -f terminal16m example.mlir
```

- CLI (HTML output):

```bash
pygmentize -l mlir -f html -O full,style=default -o example.html example.mlir
```

- Python API:

```python
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter

code = """
module {
  func.func @add(%a: i32, %b: i32) -> i32 {
    %0 = arith.addi %a, %b : i32
    return %0 : i32
  }
}
""".strip()

lexer = get_lexer_by_name("mlir")
formatter = HtmlFormatter(full=True, style="default")
html = highlight(code, lexer, formatter)
# write html to a file or display it
```

- Sphinx:
  - Ensure `pygments-mlir` is in your build environment (requirements).
  - In docs, use the `mlir` language in code blocks:

````
```mlir
module {
  func.func @add(%a: i32, %b: i32) -> i32 {
    %0 = arith.addi %a, %b : i32
    return %0 : i32
  }
}
```
````

## Example

```mlir
module {
  func.func @add(%a: i32, %b: i32) -> i32 {
    %0 = arith.addi %a, %b : i32
    return %0 : i32
  }
}
```

## Development

- Install from source for local testing:

```bash
uv sync
pygmentize -L lexers | grep -i mlir
```

- Style/coverage/tests: contributions welcome via PR.

## Acknowledgments

This package is adapted from https://gist.github.com/oowekyala/110dacc9343dbc1b86b452902d7dc553 (MIT License).
It was also inspired by the [openqasm-pygments](https://github.com/openqasm/openqasm-pygments) project (Apache 2.0).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.
