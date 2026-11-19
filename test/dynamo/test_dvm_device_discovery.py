# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# lintrunner: skip PYFMT
# Owner(s): ["module: dynamo"]
"""Target discovery for a DVM compilation: ``npu_devices``.

Before any cache key exists,
:func:`torch_npu.dynamo._package_compat.context.npu_devices`
decides whether a compilation has an NPU target at all: the key is built from
the devices found here, a graph with no NPU tensor keeps native dispatch, and a
target whose ordinal is unresolved (``torch.device("npu")``) must stay
unresolved instead of being guessed, so that ``make_context`` bypasses
persistence rather than keying on the wrong card.

The inputs alone are not enough: a graph may return CPU tensors after creating
NPU tensors internally, so the scan also reads each node's tensor metadata and
the device arguments of ``to``/``npu`` calls -- the branches covered here.
"""
import unittest

import torch
import torch_npu  # noqa: F401  - registers torch.npu
from torch.testing._internal.common_utils import TestCase, run_tests

from torch_npu.dynamo._package_compat.context import npu_devices

RUN_NPU = torch.npu.is_available()


class _Holder(torch.nn.Module):
    """A non-FX module wrapping a GraphModule, as ``torch.compile`` sees one."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return self.inner(x)


def _graph_module(graph_builder):
    graph = torch.fx.Graph()
    graph_builder(graph)
    return torch.fx.GraphModule(torch.nn.Module(), graph)


@unittest.skipIf(not RUN_NPU, "requires npu")
class TestNpuDeviceDiscovery(TestCase):
    def test_npu_inputs_are_discovered(self):
        x = torch.randn(2).npu()

        devices = npu_devices((x,))

        self.assertEqual(devices, (torch.device("npu", 0),))

    def test_cpu_only_inputs_yield_nothing(self):
        x = torch.randn(2)

        self.assertEqual(npu_devices((x,)), ())

    def test_kwargs_are_scanned(self):
        x = torch.randn(2)
        w = torch.randn(2).npu()

        devices = npu_devices((x,), {"weight": w})

        self.assertEqual(devices, (torch.device("npu", 0),))

    def test_module_parameters_and_buffers_are_discovered(self):
        def build(graph):
            graph.output(graph.placeholder("x"))

        holder = _Holder(_graph_module(build))
        holder.register_parameter(
            "weight", torch.nn.Parameter(torch.zeros(2).npu()))
        holder.register_buffer("bias", torch.zeros(2).npu())

        devices = npu_devices((torch.randn(2),), module=holder)

        self.assertEqual(devices, (torch.device("npu", 0),))

    def test_intermediate_node_metadata_is_discovered(self):
        # A graph whose inputs are CPU but which creates an NPU tensor in the
        # middle: only the node's tensor metadata can reveal the target.
        def build(graph):
            x = graph.placeholder("x")
            y = graph.call_function(torch.add, (x, x))
            graph.output(y)

        gm = _graph_module(build)
        intermediate = next(
            node for node in gm.graph.nodes if node.op == "call_function")
        intermediate.meta["example_value"] = torch.randn(2).npu()
        holder = _Holder(gm)

        devices = npu_devices((torch.randn(2),), module=holder)

        self.assertEqual(devices, (torch.device("npu", 0),))

    def test_a_to_call_names_its_target(self):
        def build(graph):
            x = graph.placeholder("x")
            graph.output(
                graph.call_method("to", (x,), {"device": torch.device("npu", 1)}))

        holder = _Holder(_graph_module(build))

        devices = npu_devices((torch.randn(2),), module=holder)

        self.assertEqual(devices, (torch.device("npu", 1),))

    def test_a_device_argument_on_a_call_function_is_discovered(self):
        def build(graph):
            x = graph.placeholder("x")
            graph.output(
                graph.call_function(torch.ones, (2,), {"device": "npu:1"}))

        holder = _Holder(_graph_module(build))

        devices = npu_devices((torch.randn(2),), module=holder)

        self.assertEqual(devices, (torch.device("npu", 1),))

    def test_an_unindexed_target_stays_unresolved(self):
        # ``torch.device("npu")`` without an ordinal must survive as-is: the
        # request is bypassed downstream instead of guessing card 0.
        def build(graph):
            x = graph.placeholder("x")
            graph.output(graph.call_method("to", (x, torch.device("npu"))))

        holder = _Holder(_graph_module(build))

        devices = npu_devices((torch.randn(2),), module=holder)

        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0].type, "npu")
        self.assertIsNone(devices[0].index)


if __name__ == "__main__":
    run_tests()
