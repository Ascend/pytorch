"""
Tests for the profiler metadata sanitiser.

CPU activity only, deliberately. The defect being guarded against is in
PyTorch's own trace writer and reproduces with no device activity at all, so
these tests need neither an NPU nor the PrivateUse1 backend and run anywhere
torch_npu can be imported.

Run:  python test/profiler/test_metadata_sanitizer.py
"""

import json
import os
import tempfile
import unittest
import warnings

import torch
import torch_npu  # noqa: F401
from torch.profiler import profile, ProfilerActivity

from torch_npu.profiler._add_metadata_sanitizer_patch import (
    _apply_metadata_sanitizer_patch,
    _sanitize_json_value,
    _sanitize_value,
)


def _cpu_work():
    tensor = torch.randn(64, 64)
    for _ in range(3):
        (tensor @ tensor).sum().item()


def _profile_with(metadata_calls):
    """Profile briefly, apply the given metadata calls, return the raw trace."""
    with tempfile.TemporaryDirectory(prefix="metadata_") as tmpdir:
        path = os.path.join(tmpdir, "trace.json")
        prof = profile(activities=[ProfilerActivity.CPU])
        prof.start()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            metadata_calls(prof)
            warned = [str(w.message) for w in caught
                      if issubclass(w.category, UserWarning)
                      and "altered" in str(w.message)]
        _cpu_work()
        prof.stop()
        prof.export_chrome_trace(path)
        with open(path, errors="replace") as fh:
            return fh.read(), warned


class TestMetadataSanitizerValues(unittest.TestCase):
    """The transformation itself, with no profiling involved."""

    def test_clean_value_is_untouched(self):
        safe, notes = _sanitize_value("plain value")
        self.assertEqual(safe, "plain value")
        self.assertEqual(notes, [])

    def test_quotes_are_replaced_and_reported(self):
        safe, notes = _sanitize_value('has "quotes" inside')
        self.assertNotIn('"', safe)
        self.assertTrue(any("double quote" in note for note in notes), notes)

    def test_backslashes_are_replaced(self):
        safe, _ = _sanitize_value("C:\\temp\\file")
        self.assertNotIn("\\", safe)

    def test_control_characters_are_replaced(self):
        safe, _ = _sanitize_value("line\nbreak\tand tab")
        self.assertNotIn("\n", safe)
        self.assertNotIn("\t", safe)

    def test_non_ascii_survives(self):
        safe, notes = _sanitize_value("kept: \u4f60\u597d")
        self.assertIn("\u4f60\u597d", safe)
        self.assertEqual(notes, [])

    def test_sanitised_values_survive_upstream_wrapping(self):
        # upstream wraps the value itself, escaping quotes but not backslashes,
        # so a sanitised value must still produce valid JSON after wrapping
        for raw in ('has "quotes"', "C:\\temp", 'mixed "a\\b"', "tab\there"):
            safe, _ = _sanitize_value(raw)
            wrapped = '"' + safe.replace('"', '\\"') + '"'
            json.loads(wrapped)


class TestMetadataSanitizerJson(unittest.TestCase):
    """The JSON variant, which sanitises strings inside the document."""

    def test_clean_json_keeps_its_meaning(self):
        safe, notes = _sanitize_json_value('{"run": 7, "tags": ["a", "b"]}')
        self.assertEqual(json.loads(safe), {"run": 7, "tags": ["a", "b"]})
        self.assertEqual(notes, [])

    def test_quoted_string_inside_json_is_sanitised(self):
        safe, _ = _sanitize_json_value(json.dumps({"note": 'has "quotes"'}))
        self.assertNotIn("\\", safe)
        self.assertNotIn('"', json.loads(safe)["note"])

    def test_nested_structures_are_walked(self):
        document = {"outer": {"inner": ['a "quoted" item']}}
        safe, notes = _sanitize_json_value(json.dumps(document))
        self.assertNotIn("\\", safe)
        self.assertNotIn('"', json.loads(safe)["outer"]["inner"][0])
        self.assertTrue(notes)

    def test_non_json_falls_back_to_plain_sanitising(self):
        safe, _ = _sanitize_json_value("not json at all")
        self.assertEqual(safe, "not json at all")


class TestMetadataSanitizerAgainstExport(unittest.TestCase):
    """What the patch is for: the trace survives a value containing a quote."""

    @classmethod
    def setUpClass(cls):
        # applied at import of torch_npu; calling again must be harmless
        _apply_metadata_sanitizer_patch()
        _apply_metadata_sanitizer_patch()

    def test_underlying_defect_is_still_present(self):
        # The patch wraps add_metadata, so the unpatched path is no longer
        # reachable through it. Reproduce the defect through the underlying API
        # that add_metadata funnels into, using the wrapping upstream performs.
        raw_value = 'has "quotes" inside'
        wrapped = '"' + raw_value.replace('"', '\\"') + '"'

        def calls(_prof):
            torch.autograd._add_metadata_json("unsanitised", wrapped)

        raw_text, _ = _profile_with(calls)
        with self.assertRaises(ValueError):
            json.loads(raw_text)

    def test_trace_parses_with_a_quoted_value(self):
        def calls(prof):
            prof.add_metadata("quoted", 'has "quotes" inside')

        raw_text, _ = _profile_with(calls)
        data = json.loads(raw_text)
        self.assertIn("traceEvents", data)

    def test_metadata_key_and_value_reach_the_trace(self):
        def calls(prof):
            prof.add_metadata("quoted", 'has "quotes" inside')

        raw_text, _ = _profile_with(calls)
        data = json.loads(raw_text)
        self.assertIn("quoted", data)
        self.assertIn("quotes", str(data["quoted"]))

    def test_json_metadata_reaches_the_trace(self):
        def calls(prof):
            prof.add_metadata_json("doc", json.dumps({"note": 'also "quoted"'}))

        raw_text, _ = _profile_with(calls)
        data = json.loads(raw_text)
        self.assertIn("doc", data)

    def test_the_user_is_warned_when_a_value_is_altered(self):
        def calls(prof):
            prof.add_metadata("quoted", 'has "quotes" inside')

        _, warned = _profile_with(calls)
        self.assertGreaterEqual(len(warned), 1)

    def test_a_key_containing_a_quote_does_not_corrupt_the_trace(self):
        # the key is written as a JSON object key in the same document, so an
        # unescapable character in it corrupts the trace as a bad value does
        def calls(prof):
            prof.add_metadata('bad"key', "clean value")

        raw_text, warned = _profile_with(calls)
        data = json.loads(raw_text)
        self.assertNotIn('bad"key', data)
        self.assertTrue(any("key" in message for message in warned), warned)
        sanitised = [k for k in data if k.startswith("bad") and k.endswith("key")]
        self.assertEqual(len(sanitised), 1, sorted(data))
        self.assertEqual(data[sanitised[0]], "clean value")

    def test_a_key_containing_a_backslash_does_not_corrupt_the_trace(self):
        def calls(prof):
            prof.add_metadata("bad\\key", "clean value")

        raw_text, _ = _profile_with(calls)
        data = json.loads(raw_text)
        self.assertNotIn("bad\\key", data)

    def test_a_bad_key_on_the_json_variant_is_sanitised(self):
        def calls(prof):
            prof.add_metadata_json('bad"key', json.dumps({"run": 7}))

        raw_text, warned = _profile_with(calls)
        data = json.loads(raw_text)
        self.assertNotIn('bad"key', data)
        self.assertTrue(any("key" in message for message in warned), warned)

    def test_a_clean_key_is_untouched(self):
        def calls(prof):
            prof.add_metadata("clean_key", "clean value")

        raw_text, warned = _profile_with(calls)
        data = json.loads(raw_text)
        self.assertEqual(data.get("clean_key"), "clean value")
        self.assertEqual(warned, [])

    def test_a_clean_value_raises_no_warning(self):
        def calls(prof):
            prof.add_metadata("clean", "nothing to sanitise here")

        raw_text, warned = _profile_with(calls)
        data = json.loads(raw_text)
        self.assertEqual(data.get("clean"), "nothing to sanitise here")
        self.assertEqual(warned, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
