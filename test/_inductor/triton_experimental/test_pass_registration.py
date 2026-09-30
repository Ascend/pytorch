# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
import unittest
from unittest import mock

from torch._inductor import config
from torch._inductor.codecache import FxGraphHashDetails
from torch._inductor.custom_graph_pass import CustomGraphPass

from torch_npu._inductor.triton_experimental import fx_passes


class RecordingPass(CustomGraphPass):
    def __init__(self, name):
        self.name = name

    def __call__(self, graph):
        graph.append(self.name)

    def uuid(self):
        return self.name


class TestPassRegistration(unittest.TestCase):
    def test_preserves_previous_passes_and_order(self):
        first, second = RecordingPass("first"), RecordingPass("second")
        for previous, expected in (
            (None, []), (first, [first]),
            ([first, second], [first, second]),
            ((first, second), [first, second]),
        ):
            with self.subTest(previous=previous), config.patch(
                post_grad_custom_post_pass=previous
            ), mock.patch.object(fx_passes.ncfg, "elide_int_float_int", True), mock.patch.object(
                fx_passes, "_elide_int_float_int_roundtrip_pass",
                side_effect=lambda graph: graph.append("elide"),
            ):
                fx_passes._install_elide_int_float_int_pass()
                passes = config.post_grad_custom_post_pass
                self.assertIsInstance(passes, list)
                self.assertEqual(passes[:-1], expected)
                self.assertIsInstance(passes[-1], fx_passes.ElideIntFloatIntPass)
                graph = []
                for pass_ in passes:
                    pass_(graph)
                self.assertEqual(graph, [p.name for p in expected] + ["elide"])
                fx_passes._install_elide_int_float_int_pass()
                self.assertIs(config.post_grad_custom_post_pass, passes)

    def test_disabled_leaves_registration_unchanged(self):
        previous = RecordingPass("previous")
        with config.patch(post_grad_custom_post_pass=previous), mock.patch.object(
            fx_passes.ncfg, "elide_int_float_int", False
        ):
            fx_passes._install_elide_int_float_int_pass()
            self.assertIs(config.post_grad_custom_post_pass, previous)

    def test_uuid_covers_implementation_and_flag(self):
        pass_ = fx_passes.ElideIntFloatIntPass()
        with mock.patch.object(fx_passes.ncfg, "elide_int_float_int", True):
            enabled = pass_.uuid()
            self.assertTrue(enabled)
            self.assertEqual(enabled, fx_passes.ElideIntFloatIntPass().uuid())
        with mock.patch.object(fx_passes.ncfg, "elide_int_float_int", False):
            self.assertNotEqual(enabled, pass_.uuid())
        with mock.patch.object(fx_passes, "get_hash_for_files", return_value=b"changed") as hash_files:
            self.assertEqual(pass_.uuid(), b"changed")
            self.assertEqual(hash_files.call_args.args[0], (fx_passes.__file__,))

    def test_unknown_previous_pass_is_not_given_a_uuid(self):
        def previous(graph):
            return None

        with config.patch(post_grad_custom_post_pass=previous), mock.patch.object(
            fx_passes.ncfg, "elide_int_float_int", True
        ):
            fx_passes._install_elide_int_float_int_pass()
            self.assertIs(config.post_grad_custom_post_pass[0], previous)
            self.assertNotIsInstance(previous, CustomGraphPass)

    def test_inductor_collects_each_uuid_in_order(self):
        details = object.__new__(FxGraphHashDetails)
        previous = RecordingPass("previous-v1")
        elide = fx_passes.ElideIntFloatIntPass()
        identity = details._get_custom_pass_detail([previous, elide])
        self.assertEqual(identity, (previous.uuid(), elide.uuid()))
        self.assertNotEqual(identity, details._get_custom_pass_detail([elide, previous]))
        previous.name = "previous-v2"
        self.assertNotEqual(identity, details._get_custom_pass_detail([previous, elide]))


if __name__ == "__main__":
    unittest.main()
