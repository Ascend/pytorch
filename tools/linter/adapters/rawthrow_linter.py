"""
Linter for raw `throw` statements in C++ code.

Unlike grep_linter.py with a plain regex, this linter performs lightweight
C++ lexical analysis to skip:
  - single-line comments (//...)
  - block comments (/* ... */)
  - string literals ("...")
  - character literals ('...')
  - raw string literals (R"delim(...)delim")
before searching for the `throw` keyword. This avoids false positives from
`throw` appearing inside log messages, comments, or other non-code contexts.

The `@allow-raw-throw` allowlist marker on a file is still honored.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from enum import Enum
from typing import NamedTuple


MAX_FILE_SIZE: int = 1024 * 1024 * 1024  # 1GB
MAX_MATCHES_PER_FILE: int = 100


class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"


LINTER_NAME: str = ""
ERROR_DESCRIPTION: str | None = None


class LintMessage(NamedTuple):
    path: str | None
    line: int | None
    char: int | None
    code: str
    severity: LintSeverity
    name: str
    original: str | None
    replacement: str | None
    description: str | None


def print_lint_message(
    name: str,
    severity: LintSeverity = LintSeverity.ERROR,
    path: str | None = None,
    line: int | None = None,
    description: str | None = None,
) -> None:
    msg = LintMessage(
        path=path,
        line=line,
        char=None,
        code=LINTER_NAME,
        severity=severity,
        name=name,
        original=None,
        replacement=None,
        description=description or ERROR_DESCRIPTION,
    )
    print(json.dumps(msg._asdict()), flush=True)


THROW_RE = re.compile(r"\bthrow\b")
ALLOW_RE = re.compile(r"@allow-raw-throw")


def strip_cpp_noncode(source: str) -> str:
    """
    Return a copy of `source` with comments and string/char literals replaced
    by spaces (preserving line numbers and total length, so line offsets are
    accurate for the caller).
    """
    out: list[str] = []
    i = 0
    n = len(source)

    def blank(span: str) -> str:
        # Preserve newlines so line numbers line up; blank out everything else.
        return "".join(c if c == "\n" else " " for c in span)

    while i < n:
        c = source[i]
        nxt = source[i + 1] if i + 1 < n else ""

        # Raw string literal: R"delim(...)delim"
        # Allow a u8/u/U/L prefix before R.
        if c == "R" and nxt == '"':
            prefix_end = i + 2
            delim_end = source.find("(", prefix_end)
            if delim_end != -1:
                delim = source[prefix_end:delim_end]
                close = f'){delim}"'
                close_idx = source.find(close, delim_end + 1)
                if close_idx != -1:
                    end = close_idx + len(close)
                    out.append(blank(source[i:end]))
                    i = end
                    continue
            # Malformed; fall through to treat as regular char.

        # Optional prefix (u8, u, U, L) before R"..." or "..."
        if c in ("u", "U", "L") and i + 1 < n:
            # Check u8R"..." / uR"..." / u"..." etc.
            j = i + 1
            if c == "u" and source[j : j + 1] == "8":
                j += 1
            if source[j : j + 1] == "R" and source[j + 1 : j + 2] == '"':
                # Let the R-string branch handle it on the next iteration by
                # fast-forwarding i to j.
                out.append(source[i:j])
                i = j
                continue
            if source[j : j + 1] == '"':
                out.append(source[i:j])
                i = j
                continue

        # Block comment /* ... */
        if c == "/" and nxt == "*":
            end = source.find("*/", i + 2)
            end = n if end == -1 else end + 2
            out.append(blank(source[i:end]))
            i = end
            continue

        # Line comment // ...
        if c == "/" and nxt == "/":
            end = source.find("\n", i + 2)
            end = n if end == -1 else end
            out.append(blank(source[i:end]))
            i = end
            continue

        # String literal "..."
        if c == '"':
            j = i + 1
            while j < n:
                ch = source[j]
                if ch == "\\" and j + 1 < n:
                    j += 2
                    continue
                if ch == '"':
                    j += 1
                    break
                if ch == "\n":
                    # Unterminated string on this line; stop to avoid runaway.
                    break
                j += 1
            out.append(blank(source[i:j]))
            i = j
            continue

        # Char literal '...'
        if c == "'":
            j = i + 1
            while j < n:
                ch = source[j]
                if ch == "\\" and j + 1 < n:
                    j += 2
                    continue
                if ch == "'":
                    j += 1
                    break
                if ch == "\n":
                    break
                j += 1
            out.append(blank(source[i:j]))
            i = j
            continue

        out.append(c)
        i += 1

    return "".join(out)


def lint_file(filename: str, error_name: str) -> None:
    try:
        with open(filename, encoding="utf-8", errors="replace") as f:
            source = f.read()
    except OSError as err:
        print_lint_message(
            path=filename,
            name="file-access-error",
            description=f"Failed to read file: {err}",
        )
        return

    # Respect file-level allowlist.
    if ALLOW_RE.search(source):
        return

    stripped = strip_cpp_noncode(source)

    matches: list[int] = []
    for m in THROW_RE.finditer(stripped):
        line_no = stripped.count("\n", 0, m.start()) + 1
        matches.append(line_no)
        if len(matches) >= MAX_MATCHES_PER_FILE + 1:
            break

    report = matches[:MAX_MATCHES_PER_FILE]
    for line_no in report:
        print_lint_message(path=filename, line=line_no, name=error_name)

    if len(matches) > MAX_MATCHES_PER_FILE:
        print_lint_message(
            path=filename,
            name="too-many-matches",
            description=(
                f"File has more than {MAX_MATCHES_PER_FILE} matches, "
                "only showing the first ones"
            ),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Raw `throw` linter for C++ sources.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument("--linter-name", required=True)
    parser.add_argument("--error-name", required=True)
    parser.add_argument("--error-description", required=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("filenames", nargs="+")
    args = parser.parse_args()

    global LINTER_NAME, ERROR_DESCRIPTION
    LINTER_NAME = args.linter_name
    ERROR_DESCRIPTION = args.error_description

    logging.basicConfig(
        format="<%(threadName)s:%(levelname)s> %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
        stream=sys.stderr,
    )

    for filename in args.filenames:
        try:
            size = os.path.getsize(filename)
        except OSError as err:
            print_lint_message(
                path=filename,
                name="file-access-error",
                description=f"Failed to stat file: {err}",
            )
            continue

        if size > MAX_FILE_SIZE:
            print_lint_message(
                path=filename,
                severity=LintSeverity.WARNING,
                name="file-too-large",
                description=(
                    f"File size ({size} bytes) exceeds {MAX_FILE_SIZE} bytes, skipping"
                ),
            )
            continue

        lint_file(filename, args.error_name)


if __name__ == "__main__":
    main()
