# Copyright (c) 2026, Huawei Technologies Co., Ltd
#
"""Unit tests for the TeSourcePinPass FxGraphCache key pin.

Covers the three properties the pass must hold, at unit level (no NPU
compile needed):

- digest semantics: deterministic on an unchanged tree, sensitive to .py
  content edits and to new .py files, blind to non-.py files and
  __pycache__ contents.  Note ``get_hash_for_files`` is lru_cached per
  process (frozen at first read -- faithful to what the process
  imported), so the content-sensitivity cases model a process restart
  via ``cache_clear()``;
- uuid contract: stable bytes, independent of pass instantiation;
- install composition: TeSourcePin is installed unconditionally (not
  gated on ``elide_int_float_int`` -- source-key pinning is unrelated to
  the elide pass), Elide is appended only when its flag is on, re-install
  is idempotent, an existing CustomGraphPass composes, and a third-party
  RAW callable aborts the install untouched (upstream keeps its bypass;
  no uuid is fabricated for foreign callables).
"""

import os
import tempfile
import unittest

import torch_npu  # noqa: F401  (package import order for fx_passes)

from torch._inductor import config as inductor_config
from torch._inductor.custom_graph_pass import (
    CustomGraphPass,
    get_custom_graph_passes,
    get_hash_for_files,
)

from torch_npu._inductor.triton_experimental import config as ncfg
from torch_npu._inductor.triton_experimental.fx_passes import (
    ElideIntFloatIntPass,
    TeSourcePinPass,
    _install_elide_int_float_int_pass,
    _te_source_digest,
)


class TestTeSourceDigest(unittest.TestCase):
    @staticmethod
    def _mk_tree(root, files):
        for rel, content in files.items():
            p = os.path.join(root, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as f:
                f.write(content)

    @staticmethod
    def _fresh_digest(base):
        # get_hash_for_files freezes its result per process (lru_cache);
        # a fresh read = a new process.  Production semantics: an edited
        # .py only takes effect after restart, and the frozen digest
        # matches exactly the code the process imported.
        get_hash_for_files.cache_clear()
        return _te_source_digest(base)

    def test_deterministic_and_content_sensitive(self):
        with tempfile.TemporaryDirectory() as root:
            self._mk_tree(root, {"a.py": "x = 1\n", os.path.join("sub", "b.py"): "y = 2\n"})
            d1 = self._fresh_digest(root)
            d2 = self._fresh_digest(root)
            self.assertEqual(d1, d2)
            # same file set, changed content -> changed digest (fresh process)
            self._mk_tree(root, {"a.py": "x = 2\n"})
            self.assertNotEqual(self._fresh_digest(root), d1)

    def test_new_py_file_changes_digest(self):
        with tempfile.TemporaryDirectory() as root:
            self._mk_tree(root, {"a.py": "x = 1\n"})
            d1 = _te_source_digest(root)
            self._mk_tree(root, {"c.py": "z = 3\n"})
            self.assertNotEqual(_te_source_digest(root), d1)

    def test_non_py_and_pycache_are_blind(self):
        with tempfile.TemporaryDirectory() as root:
            self._mk_tree(root, {
                "a.py": "x = 1\n",
                "notes.txt": "junk",
                os.path.join("__pycache__", "a.cpython.311.pyc"): "\x00\x01",
            })
            d1 = _te_source_digest(root)
            # non-.py edits (incl. pycache churn) must not flip the key
            self._mk_tree(root, {"notes.txt": "changed"})
            self._mk_tree(root, {os.path.join("__pycache__", "a.cpython.311.pyc"): "\x00\x02"})
            self.assertEqual(_te_source_digest(root), d1)

    def test_uuid_is_stable_bytes_across_instances(self):
        d1 = _te_source_digest()
        d2 = TeSourcePinPass().uuid()
        d3 = TeSourcePinPass().uuid()
        self.assertIsInstance(d1, bytes)
        self.assertEqual(d1, d2)
        self.assertEqual(d2, d3)


class TestInstallComposition(unittest.TestCase):
    def setUp(self):
        self._prev_pass = inductor_config.post_grad_custom_post_pass
        self._prev_elide = ncfg.elide_int_float_int

    def tearDown(self):
        inductor_config.post_grad_custom_post_pass = self._prev_pass
        ncfg.elide_int_float_int = self._prev_elide

    @staticmethod
    def _chain():
        return get_custom_graph_passes(inductor_config.post_grad_custom_post_pass)

    def test_pin_installed_unconditionally(self):
        # elide OFF must NOT suppress the source-key pin
        ncfg.elide_int_float_int = False
        inductor_config.post_grad_custom_post_pass = None
        _install_elide_int_float_int_pass()
        self.assertEqual([type(p) for p in self._chain()], [TeSourcePinPass])

    def test_elide_appended_when_flag_on(self):
        ncfg.elide_int_float_int = True
        inductor_config.post_grad_custom_post_pass = None
        _install_elide_int_float_int_pass()
        self.assertEqual(
            [type(p) for p in self._chain()], [ElideIntFloatIntPass, TeSourcePinPass])

    def test_reinstall_is_idempotent(self):
        ncfg.elide_int_float_int = True
        inductor_config.post_grad_custom_post_pass = None
        _install_elide_int_float_int_pass()
        _install_elide_int_float_int_pass()
        self.assertEqual(len(self._chain()), 2)

    def test_composes_with_existing_custom_pass(self):
        class OtherPass(CustomGraphPass):
            def __call__(self, graph):
                pass

            def uuid(self):
                return b"other"

        ncfg.elide_int_float_int = False
        inductor_config.post_grad_custom_post_pass = OtherPass()
        _install_elide_int_float_int_pass()
        self.assertEqual([type(p) for p in self._chain()], [OtherPass, TeSourcePinPass])

    def test_third_party_raw_callable_aborts_untouched(self):
        def third_party(graph):
            pass

        ncfg.elide_int_float_int = True
        inductor_config.post_grad_custom_post_pass = third_party
        _install_elide_int_float_int_pass()
        # no fabricated uuid for foreign callables: upstream keeps its
        # cache bypass, the raw callable stays as-is
        self.assertIs(inductor_config.post_grad_custom_post_pass, third_party)


if __name__ == "__main__":
    unittest.main()
