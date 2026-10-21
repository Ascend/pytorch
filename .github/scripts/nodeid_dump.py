"""pytest plugin: dump collected nodeids as JSON Lines to a dedicated file.

Loaded via ``-p nodeid_dump`` with ``PYTEST_NODEID_FILE`` pointing at the
output path. Writing to a file (not stdout) decouples nodeids from terminal
output: names containing newlines/``::``/edge whitespace survive verbatim,
and nothing printed to stdout (warnings, C-level fd writes from CANN/HAL,
tracebacks) can corrupt the channel.

Records are serialized with json.dumps defaults (ensure_ascii=True): every
non-ASCII or control character becomes a \\uXXXX escape, so each record is
one line of printable ASCII terminated by exactly one ``\\n`` — the reader
splits on ``"\\n"`` only (never str.splitlines(), which would also split on
\\v \\f \\x1c \\x1d \\x1e \\x85 \\u2028 \\u2029 and resurrect the very
line-splitting bug this channel exists to fix).

A trailing ``{"count": N}`` record lets the reader reconcile the record
count; the file is written to a .tmp sibling and os.replace()d into place,
so a crashed collector never leaves a half file.

Inert when PYTEST_NODEID_FILE is unset, so manual pytest runs are
unaffected. Registers exactly one hook and never touches session.items.
"""
import json
import os

# Characters that break line-oriented transport or single-line display.
_LINEBREAK_CHARS = "\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029"


def pytest_collection_finish(session):
    path = os.environ.get("PYTEST_NODEID_FILE")
    if not path:
        return

    items = session.items  # final selection, post-modifyitems, in run order
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for item in items:
            # A nodeid is only usable as a pytest selection argument if no
            # collector name in its chain contains "::": pytest splits
            # selection args on every "::" and matches level by level.
            chain = [n.name or "" for n in item.listchain()]
            selectable = all("::" not in n for n in chain)
            hazards = []
            if any(c in item.nodeid for c in _LINEBREAK_CHARS):
                hazards.append("linebreak")
            if item.name != item.name.strip():
                hazards.append("edge_whitespace")
            f.write(json.dumps({
                "nodeid": item.nodeid,
                "selectable": selectable,
                "hazards": hazards,
                "location": list(item.location),
            }) + "\n")
        f.write(json.dumps({"count": len(items)}) + "\n")
    os.replace(tmp, path)  # atomic: a crashed collector never leaves a half file
