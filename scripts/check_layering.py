#!/usr/bin/env python3
"""Checks that the crate's modules only depend downwards.

Layers, lowest first. A module may use its own layer and the ones below it, never one above.
Test code (anything under `#[cfg(test)]`) is exempt: tests may reach across layers.

    python3 scripts/check_layering.py [--src DIR] [--graph]

Exit status 0 if the layering holds, 1 if not. `--graph` prints the observed dependencies
between layers (non-test code only), to compare with the diagram in the book.

What counts as a dependency: any `crate::<module>` path in non-test code, whether in a `use`
(including the grouped form `use crate::{a::X, b::Y}`) or written inline. Comments, doc comments
and string literals are ignored, so doc links never count.
"""

import argparse
import re
import sys
from pathlib import Path

# Rank of each top-level module of the crate. Equal ranks may depend on each other freely.
# `diagnostics` is a leaf shared by all layers, so it sits next to `utils`.
LAYERS = [
    ("utils", 0),
    ("diagnostics", 0),
    ("modelling", 1),
    ("constraints", 2),
    ("mdd", 3),
    ("sampling", 4),
    ("bp", 5),
    ("destroy", 6),
    ("learning", 7),
    ("local_search", 8),
    ("pyaicad", 9),
]
RANK = dict(LAYERS)


def blank_comments_and_strings(src: str) -> str:
    """Replaces comments and string/char literals by spaces, keeping newlines and offsets."""
    out = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        two = src[i : i + 2]
        if two == "//":
            j = src.find("\n", i)
            j = n if j == -1 else j
            out.append(" " * (j - i))
            i = j
        elif two == "/*":
            depth, j = 1, i + 2
            while j < n and depth:
                if src[j : j + 2] == "/*":
                    depth, j = depth + 1, j + 2
                elif src[j : j + 2] == "*/":
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            out.append("".join(ch if ch == "\n" else " " for ch in src[i:j]))
            i = j
        elif c == "r" and re.match(r'r#*"', src[i:]) and (i == 0 or not (src[i - 1].isalnum() or src[i - 1] == "_")):
            hashes = re.match(r"r(#*)\"", src[i:]).group(1)
            end = src.find('"' + hashes, i + 2 + len(hashes))
            j = n if end == -1 else end + 1 + len(hashes)
            out.append("".join(ch if ch == "\n" else " " for ch in src[i:j]))
            i = j
        elif c == '"':
            j = i + 1
            while j < n and src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            j = min(j + 1, n)
            out.append("".join(ch if ch == "\n" else " " for ch in src[i:j]))
            i = j
        elif c == "'":
            m = re.match(r"'(\\.[^']*|[^\\'])'", src[i:])  # char literal, not a lifetime
            if m:
                out.append(" " * len(m.group(0)))
                i += len(m.group(0))
            else:
                out.append(c)
                i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def blank_test_code(code: str) -> str:
    """Blanks every item that follows a `#[cfg(test)]` attribute (a block item or a `;` item)."""
    out = list(code)
    for m in re.finditer(r"#\s*\[\s*cfg\s*\(\s*test\s*\)\s*\]", code):
        i = m.end()
        # Skip further attributes on the same item.
        while True:
            a = re.match(r"\s*#\s*\[", code[i:])
            if not a:
                break
            depth, j = 0, i + a.end() - 1
            while j < len(code):
                depth += code[j] == "["
                depth -= code[j] == "]"
                j += 1
                if depth == 0:
                    break
            i = j
        # The item ends at its matching `}` if a `{` comes before any `;`, else at the `;`.
        semi = code.find(";", i)
        brace = code.find("{", i)
        if brace != -1 and (semi == -1 or brace < semi):
            depth, j = 0, brace
            while j < len(code):
                depth += code[j] == "{"
                depth -= code[j] == "}"
                j += 1
                if depth == 0:
                    break
            end = j
        else:
            end = semi + 1 if semi != -1 else len(code)
        for k in range(m.start(), end):
            if out[k] != "\n":
                out[k] = " "
    return "".join(out)


def crate_dependencies(code: str):
    """Yields (line, module) for each top-level module of the crate named in `code`."""
    line_of = lambda pos: code.count("\n", 0, pos) + 1
    for m in re.finditer(r"\bcrate\s*::\s*(\w+)", code):
        yield line_of(m.start()), m.group(1)
    # Grouped form: `crate::{a::X, b::{Y, Z}, c}`: the head of each top-level item.
    for m in re.finditer(r"\bcrate\s*::\s*\{", code):
        depth, j, start = 1, m.end(), m.end()
        items = []
        while j < len(code) and depth:
            ch = code[j]
            depth += ch == "{"
            depth -= ch == "}"
            if (ch == "," and depth == 1) or depth == 0:
                items.append((start, code[start:j]))
                start = j + 1
            j += 1
        for pos, item in items:
            h = re.match(r"\s*(\w+)", item)
            if h:
                yield line_of(pos + h.start(1)), h.group(1)


def layer_of(path: Path, src: Path):
    top = path.relative_to(src).parts[0]
    top = top[:-3] if top.endswith(".rs") else top
    return top if top in RANK else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(Path(__file__).resolve().parent.parent / "src"))
    ap.add_argument("--graph", action="store_true", help="print observed layer dependencies")
    args = ap.parse_args()
    src = Path(args.src)

    violations, edges = [], {}
    for path in sorted(src.rglob("*.rs")):
        layer = layer_of(path, src)
        if layer is None:
            continue  # lib.rs and the like
        code = blank_test_code(blank_comments_and_strings(path.read_text(encoding="utf-8")))
        for line, dep in crate_dependencies(code):
            if dep not in RANK or dep == layer:
                continue
            edges.setdefault((layer, dep), []).append(f"{path.relative_to(src.parent)}:{line}")
            if RANK[dep] > RANK[layer]:
                violations.append(f"{path.relative_to(src.parent)}:{line}: `{layer}` (layer {RANK[layer]}) uses `{dep}` (layer {RANK[dep]})")

    if args.graph:
        for (a, b), where in sorted(edges.items(), key=lambda e: (RANK[e[0][0]], e[0])):
            print(f"{a} -> {b}  ({len(where)} uses)")
    violations.sort()
    if violations:
        print("Layering violations (a module may only use its own layer or one below):", file=sys.stderr)
        for v in violations:
            print("  " + v, file=sys.stderr)
        return 1
    print("layering ok: " + " < ".join(name for name, _ in LAYERS if name != "diagnostics"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
