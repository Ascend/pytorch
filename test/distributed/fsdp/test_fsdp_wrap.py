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
Add validation cases for torch.distributed.fsdp.wrap APIs on NPU:
1. PyTorch community tests exercise wrap() only through CUDA-bound FSDP
   scenarios and lack direct checks of the annotation and override contract.
2. This file validates torch.distributed.fsdp.wrap.wrap together with the
   enable_wrap context (extendable for the auto-wrap policies).
3. wrap() itself is device-agnostic; the canonical FSDP wrapper path on NPU
   is covered by test/distributed/fsdp/ (FSDP1 auto_wrap_policy and FSDP2
   suites), so a lightweight recording wrapper_cls is used here.
"""

import torch
from torch import nn
from torch.distributed.fsdp.wrap import enable_wrap, wrap
from torch.testing._internal.common_utils import TestCase, run_tests

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class _RecordingWrapper(nn.Module):
    # Minimal wrapper_cls recording construction kwargs for assertions.
    def __init__(self, module, **kwargs):
        super().__init__()
        self.module = module
        self.wrapper_kwargs = kwargs

    def forward(self, x):
        return self.module(x)


class _NestedWrapModule(nn.Module):
    # Children call wrap() during construction; the annotation only takes
    # effect inside an enable_wrap context (community SkipModel pattern).
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 3, bias=True)
        self.fc1 = wrap(self.fc1)
        self.fc2 = nn.Linear(3, 2, bias=True)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class TestFsdpWrapAPI(TestCase):
    def _model(self):
        return _NestedWrapModule().to(device_type)

    def test_wrap_outside_context_returns_module(self):
        # Outside enable_wrap, wrap() is a no-op annotation returning the same
        # module object whether or not overrides are passed.
        model = self._model()
        self.assertIs(wrap(model), model)
        self.assertIs(wrap(model, foo=1), model)
        self.assertNotIsInstance(model.fc1, _RecordingWrapper)

    def test_wrap_inside_context_applies_wrapper_cls(self):
        # Inside enable_wrap, wrap() constructs wrapper_cls around the module;
        # modules built inside the context wrap their children too, and
        # parameters stay on the device after wrapping.
        with enable_wrap(wrapper_cls=_RecordingWrapper):
            model = self._model()
            wrapped = wrap(model)
            self.assertIsInstance(wrapped, _RecordingWrapper)
            self.assertIs(wrapped.module, model)
            self.assertIsInstance(model.fc1, _RecordingWrapper)
        self.assertEqual(wrapped.wrapper_kwargs, {})
        param = next(wrapped.parameters())
        self.assertEqual(param.device.type, device_type)
        out = wrapped(torch.ones(2, 4, device=device_type))
        self.assertEqual(out.device.type, device_type)

    def test_wrap_overrides_take_priority_over_context(self):
        # Context kwargs act as defaults for every wrap() call; same-name
        # kwargs passed to wrap() take priority and reach wrapper_cls.
        with enable_wrap(wrapper_cls=_RecordingWrapper, foo=1, bar=2):
            first = wrap(self._model(), bar=3)
            second = wrap(self._model())
        self.assertEqual(first.wrapper_kwargs, {"foo": 1, "bar": 3})
        self.assertEqual(second.wrapper_kwargs, {"foo": 1, "bar": 2})

    def test_module_wrap_overrides_take_precedence(self):
        # A module carrying a _wrap_overrides attribute forces the highest
        # priority over both context and wrap() kwargs (FSDP uses this to
        # disable mixed precision for BatchNorm submodules).
        model = self._model()
        model._wrap_overrides = {"bar": 4}
        with enable_wrap(wrapper_cls=_RecordingWrapper, foo=1, bar=2):
            wrapped = wrap(model, bar=3)
        self.assertEqual(wrapped.wrapper_kwargs, {"foo": 1, "bar": 4})

    def test_nested_enable_wrap_rejected(self):
        # Nested enable_wrap contexts are rejected with NotImplementedError.
        with enable_wrap(wrapper_cls=_RecordingWrapper):
            with self.assertRaises(NotImplementedError):
                with enable_wrap(wrapper_cls=_RecordingWrapper):
                    pass

    def test_context_state_reset_after_exit(self):
        # The autowrap state is cleared on exit: wrap() no-ops again and a
        # new context can be entered with fresh kwargs.
        model = self._model()
        with enable_wrap(wrapper_cls=_RecordingWrapper, foo=1):
            self.assertIsInstance(wrap(model), _RecordingWrapper)
        self.assertIs(wrap(model), model)
        with enable_wrap(wrapper_cls=_RecordingWrapper):
            wrapped = wrap(model)
        self.assertIsInstance(wrapped, _RecordingWrapper)
        self.assertEqual(wrapped.wrapper_kwargs, {})

    def test_missing_wrapper_cls_rejected(self):
        # A context without a usable wrapper_cls is rejected when wrap() is
        # called inside it.
        model = self._model()
        with enable_wrap(wrapper_cls=None):
            with self.assertRaises(AssertionError):
                wrap(model)


if __name__ == "__main__":
    run_tests()
