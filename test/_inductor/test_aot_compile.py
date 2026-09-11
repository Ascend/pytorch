# Copyright (c) 2026 Huawei Technologies Co., Ltd
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Add validation cases for torch._export.aot_compile on NPU.

1. PyTorch community tests do not sufficiently cover all arguments of
   torch._export.aot_compile through this entry API.
2. This file validates modules, callables, kwargs, dynamic_shapes,
   options, non-default boolean arguments, and invalid arguments on NPU.
"""

import os
import unittest

import torch
from torch.testing._internal.common_utils import TestCase, run_tests
import torch_npu._inductor  # noqa: F401

_NOT_SET = object()

device_type = (
    acc.type
    if (acc := torch.accelerator.current_accelerator())
    else "cpu"
)


class AddModel(torch.nn.Module):
    def forward(self, x, y):
        return torch.relu(x + y)


class KwargsModel(torch.nn.Module):
    def forward(self, x, bias=None):
        return torch.relu(x + bias)


def callable_add(x, y):
    return torch.relu(x + y)


@unittest.skipUnless(
    device_type == "npu" and torch.npu.is_available(),
    "requires NPU",
)
class TestExportAotCompile(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        torch.manual_seed(1234)
        torch.npu.set_device(0)

    def _make_inputs(self):
        x = torch.randn(4, 8, device=device_type)
        y = torch.randn(4, 8, device=device_type)
        return x, y

    def _dynamic_shapes_dict(self):
        batch = torch.export.Dim("batch", min=2, max=8)
        return {
            "x": {0: batch},
            "y": {0: batch},
        }

    def _dynamic_shapes_tuple(self):
        batch = torch.export.Dim("batch", min=2, max=8)
        return (
            {0: batch},
            {0: batch},
        )

    def _assert_valid_artifact(self, artifact):
        if isinstance(artifact, (str, os.PathLike)):
            artifacts = [artifact]
        else:
            self.assertIsInstance(artifact, (list, tuple))
            artifacts = list(artifact)

        self.assertGreater(len(artifacts), 0)

        for artifact in artifacts:
            self.assertIsInstance(artifact, (str, os.PathLike))
            artifact_path = os.fspath(artifact)

            self.assertTrue(
                os.path.isfile(artifact_path),
                f"AOT artifact does not exist: {artifact_path}",
            )
            self.assertGreater(
                os.path.getsize(artifact_path),
                0,
                f"AOT artifact is empty: {artifact_path}",
            )

    def _compile_and_check(
        self,
        function,
        args,
        kwargs=_NOT_SET,
        **compile_kwargs,
    ):
        if isinstance(function, torch.nn.Module):
            function = function.to(device_type).eval()

        with torch.no_grad():
            if kwargs is _NOT_SET:
                artifact = torch._export.aot_compile(
                    function,
                    args,
                    **compile_kwargs,
                )
            else:
                artifact = torch._export.aot_compile(
                    function,
                    args,
                    kwargs=kwargs,
                    **compile_kwargs,
                )

        self._assert_valid_artifact(artifact)
        return artifact

    def test_module_with_args_and_omitted_optional_arguments(self):
        x, y = self._make_inputs()

        self._compile_and_check(
            AddModel(),
            (x, y),
        )

    def test_plain_callable(self):
        x, y = self._make_inputs()

        self._compile_and_check(
            callable_add,
            (x, y),
        )

    def test_kwargs_none(self):
        x, y = self._make_inputs()

        self._compile_and_check(
            AddModel(),
            (x, y),
            kwargs=None,
        )

    def test_kwargs_empty(self):
        x, y = self._make_inputs()

        self._compile_and_check(
            AddModel(),
            (x, y),
            kwargs={},
        )

    def test_kwargs_nonempty(self):
        x, bias = self._make_inputs()

        self._compile_and_check(
            KwargsModel(),
            (x,),
            kwargs={"bias": bias},
        )

    def test_dynamic_shapes_dict(self):
        x, y = self._make_inputs()

        self._compile_and_check(
            AddModel(),
            (x, y),
            dynamic_shapes=self._dynamic_shapes_dict(),
        )

    def test_dynamic_shapes_tuple(self):
        x, y = self._make_inputs()

        self._compile_and_check(
            AddModel(),
            (x, y),
            dynamic_shapes=self._dynamic_shapes_tuple(),
        )

    def test_options_empty(self):
        x, y = self._make_inputs()

        self._compile_and_check(
            AddModel(),
            (x, y),
            options={},
        )

    def test_options_nonempty(self):
        x, y = self._make_inputs()

        self._compile_and_check(
            AddModel(),
            (x, y),
            options={"max_autotune": False},
        )

    def test_remove_runtime_assertions_true(self):
        x, y = self._make_inputs()

        self._compile_and_check(
            AddModel(),
            (x, y),
            dynamic_shapes=self._dynamic_shapes_dict(),
            remove_runtime_assertions=True,
        )

    def test_disable_constraint_solver_true(self):
        x, y = self._make_inputs()

        self._compile_and_check(
            AddModel(),
            (x, y),
            dynamic_shapes=self._dynamic_shapes_dict(),
            disable_constraint_solver=True,
        )

    def test_same_signature_false(self):
        x, y = self._make_inputs()

        self._compile_and_check(
            AddModel(),
            (x, y),
            same_signature=False,
        )

    def test_invalid_keyword_argument(self):
        model = AddModel().to(device_type).eval()
        x, y = self._make_inputs()
        invalid_kwargs = {"not_a_real_argument": True}

        with self.assertRaises(TypeError):
            torch._export.aot_compile(
                model,
                (x, y),
                **invalid_kwargs,
            )

    def test_invalid_option_name(self):
        x, y = self._make_inputs()

        with self.assertRaisesRegex(AttributeError, "does not exist"):
            self._compile_and_check(
                AddModel(),
                (x, y),
                options={"not_a_real_inductor_option": True},
            )


if __name__ == "__main__":
    run_tests()
