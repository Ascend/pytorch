import token
from collections.abc import Sequence
from tokenize import TokenInfo

from . import NO_TOKEN, ParseError


FSTRING_START: int = getattr(token, "FSTRING_START", NO_TOKEN)
FSTRING_END: int = getattr(token, "FSTRING_END", NO_TOKEN)

BRACKETS = {"{": "}", "(": ")", "[": "]"}
BRACKETS_INV = {j: i for i, j in BRACKETS.items()}


_FSTRING_SENTINEL = ("__fstring_start__",)


def bracket_pairs(tokens: Sequence[TokenInfo]) -> dict[int, int]:
    """Returns a dictionary mapping opening to closing brackets"""
    braces: dict[int, int] = {}
    stack: list[object] = []
    in_fstring = False

    t: TokenInfo | None = None
    for i, t in enumerate(tokens):
        if t.type == token.OP and not in_fstring:
            if t.string in BRACKETS:
                stack.append(i)
            elif inv := BRACKETS_INV.get(t.string):
                if not stack:
                    raise ParseError(t, "Never opened")
                begin = stack.pop()
                if not isinstance(begin, int) or not 0 <= begin < len(tokens):
                    raise ParseError(t, f"Mismatched braces at {begin}")

                if not (stack and stack[-1] is _FSTRING_SENTINEL):
                    braces[begin] = i

                b = tokens[begin].string
                if b != inv:
                    raise ParseError(t, f"Mismatched braces '{b}' at {begin}")
        elif t.type == FSTRING_START:
            stack.append(_FSTRING_SENTINEL)
            in_fstring = True
        elif t.type == FSTRING_END:
            if not stack or stack.pop() is not _FSTRING_SENTINEL:
                raise ParseError(t, "Mismatched FSTRING_START/FSTRING_END")
            in_fstring = False
    if stack:
        raise ParseError(
            t if t is not None else tokens[-1], "Left open"
        )
    return braces
