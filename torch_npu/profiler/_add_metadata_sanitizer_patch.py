import json
import warnings


_REPLACEMENTS = {
    '"': "'",
    "\\": "/",
    "\n": " ",
    "\r": " ",
    "\t": " ",
    "\b": " ",
    "\f": " ",
}

_DESCRIPTIONS = {
    '"': "double quote",
    "\\": "backslash",
    "\n": "newline",
    "\r": "carriage return",
    "\t": "tab",
    "\b": "backspace",
    "\f": "form feed",
}


def _describe(char):
    return _DESCRIPTIONS.get(char, "control character U+%04X" % ord(char))


def _sanitize_value(value):
    """Replace every character whose JSON encoding would contain a backslash.

    Returns the safe value and a list describing what changed, empty when the
    value passed through untouched.
    """
    if not isinstance(value, str):
        value = str(value)

    parts = []
    changed = {}
    for char in value:
        if char in _REPLACEMENTS:
            parts.append(_REPLACEMENTS[char])
            changed[char] = changed.get(char, 0) + 1
        elif ord(char) < 0x20:
            parts.append(" ")
            changed[char] = changed.get(char, 0) + 1
        else:
            parts.append(char)

    notes = ["%d %s" % (count, _describe(char)) for char, count in sorted(changed.items())]
    return "".join(parts), notes


def _sanitize_json_value(value):
    """Sanitise a pre-serialised JSON document.

    Every string inside the document is sanitised and the result is
    re-serialised with ensure_ascii=False, so no \\uXXXX escapes are introduced
    either. A value that does not parse as JSON is treated as a plain string.
    """
    if not isinstance(value, str):
        value = str(value)

    try:
        document = json.loads(value)
    except (ValueError, TypeError):
        return _sanitize_value(value)

    collected = []

    def walk(node):
        if isinstance(node, str):
            safe, notes = _sanitize_value(node)
            collected.extend(notes)
            return safe
        if isinstance(node, dict):
            return {walk(key): walk(item) for key, item in node.items()}
        if isinstance(node, list):
            return [walk(item) for item in node]
        return node

    return json.dumps(walk(document), ensure_ascii=False), collected


def _warn_altered(api, key, notes, what="value"):
    warnings.warn(
        "%s(%r): the %s contained characters that cannot survive the trace writer "
        "and was altered (%s). Writing it unchanged would produce an unparseable "
        "trace file." % (api, key, what, ", ".join(notes)),
        UserWarning,
        stacklevel=3,
    )


def _apply_metadata_sanitizer_patch():
    """Guard profiler metadata against characters that corrupt the trace.

    Two defects combine on the kineto path. torch.profiler's add_metadata wraps
    the value with '"' + value.replace('"', '\\\\"') + '"', escaping quotes but
    not backslashes, so a value such as C:\\temp is already malformed JSON. The
    trace writer then replaces every backslash with a forward slash, turning a
    correctly escaped \\" into /" and ending the JSON string early, which makes
    the entire trace file unparseable. add_metadata_json is affected the same
    way, so routing a value through it is not a workaround.

    Neither can be fixed in torch_npu: the writer belongs to PyTorch's
    libkineto, which is not vendored here. This removes the offending
    characters before the value is handed over and warns when a value had to be
    altered, so a quote costs the user an approximation of their string instead
    of the whole recording. Values that would have been written correctly are
    passed through untouched and raise no warning.

    Keys are sanitised the same way. A key is written as a JSON object key in
    the same document, so an unescapable character in it corrupts the trace
    just as one in the value does. Keys are usually hard-coded, but nothing
    enforces that.
    """
    from torch.profiler import profiler as torch_profiler

    target = getattr(torch_profiler, "_KinetoProfile", None)
    if target is None:
        return
    if getattr(target, "_npu_metadata_sanitized", False):
        return

    original_add_metadata = target.add_metadata
    original_add_metadata_json = target.add_metadata_json

    def add_metadata(self, key, value):
        # The key is written as a JSON object key in the same document, so an
        # unescapable character in it corrupts the trace exactly as one in the
        # value does. Keys are usually hard-coded, but nothing enforces that.
        safe_key, key_notes = _sanitize_value(key)
        if key_notes:
            _warn_altered("add_metadata", key, key_notes, "key")
        safe, notes = _sanitize_value(value)
        if notes:
            _warn_altered("add_metadata", key, notes)
        return original_add_metadata(self, safe_key, safe)

    def add_metadata_json(self, key, value):
        safe_key, key_notes = _sanitize_value(key)
        if key_notes:
            _warn_altered("add_metadata_json", key, key_notes, "key")
        safe, notes = _sanitize_json_value(value)
        if notes:
            _warn_altered("add_metadata_json", key, notes)
        return original_add_metadata_json(self, safe_key, safe)

    add_metadata.__doc__ = original_add_metadata.__doc__
    add_metadata_json.__doc__ = original_add_metadata_json.__doc__

    target.add_metadata = add_metadata
    target.add_metadata_json = add_metadata_json
    target._npu_metadata_sanitized = True
