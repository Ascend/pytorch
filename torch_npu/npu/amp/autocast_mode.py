# Copyright (c) 2021 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION.
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

import warnings
from torch.cuda.amp import autocast
from .common import amp_definitely_not_available


class NpuAutocast(autocast):
    r"""
    Instances of :class:`NpuAutocast` serve as context managers or decorators that
    allow regions of your script to run in mixed precision.

    In these regions, NPU ops run in an op-specific dtype chosen by NpuAutocast
    to improve performance while maintaining accuracy.
    See the :ref:`Autocast Op Reference<autocast-op-reference>` for details.

    When entering an NpuAutocast-enabled region, Tensors may be any type.
    You should not call ``.half()`` on your model(s) or inputs when using autocasting.

    :class:`NpuAutocast` should wrap only the forward pass(es) of your network, including the loss
    computation(s).  Backward passes under NpuAutocast are not recommended.
    Backward ops run in the same type that NpuAutocast used for corresponding forward ops.

    Example::

        # Creates model and optimizer in default precision
        model = Net().npu()
        optimizer = optim.SGD(model.parameters(), ...)

        for input, target in data:
            optimizer.zero_grad()

            # Enables autocasting for the forward pass (model + loss)
            with NpuAutocast():
                output = model(input)
                loss = loss_fn(output, target)

            # Exits the context manager before backward()
            loss.backward()
            optimizer.step()

    See the :ref:`Automatic Mixed Precision examples<amp-examples>` for usage (along with gradient scaling)
    in more complex scenarios (e.g., gradient penalty, multiple models/losses, custom autograd functions).

    :class:`NpuAutocast` can also be used as a decorator, e.g., on the ``forward`` method of your model::

        class AutocastModel(nn.Module):
            ...
            @NpuAutocast()
            def forward(self, input):
                ...

    Floating-point Tensors produced in an NpuAutocast-enabled region may be ``float16``.
    After returning to an NpuAutocast-disabled region, using them with floating-point
    Tensors of different dtypes may cause type mismatch errors.  If so, cast the Tensor(s)
    produced in the NpuAutocast region back to ``float32`` (or other dtype if desired).
    If a Tensor from the NpuAutocast region is already ``float32``, the cast is a no-op,
    and incurs no additional overhead.  Example::

        # Creates some tensors in default dtype (here assumed to be float32)
        a_float32 = torch.rand((8, 8), device="npu")
        b_float32 = torch.rand((8, 8), device="npu")
        c_float32 = torch.rand((8, 8), device="npu")
        d_float32 = torch.rand((8, 8), device="npu")

        with NpuAutocast():
            # torch.mm is on NpuAutocast's list of ops that should run in float16.
            # Inputs are float32, but the op runs in float16 and produces float16 output.
            # No manual casts are required.
            e_float16 = torch.mm(a_float32, b_float32)
            # Also handles mixed input types
            f_float16 = torch.mm(d_float32, e_float16)

        # After exiting NpuAutocast, calls f_float16.float() to use with d_float32
        g_float32 = torch.mm(d_float32, f_float16.float())

    Type mismatch errors *in* an NpuAutocast-enabled region are a bug; if this is what you observe,
    please file an issue.

    ``NpuAutocast(enabled=False)`` subregions can be nested in NpuAutocast-enabled regions.
    Locally disabling NpuAutocast can be useful, for example, if you want to force a subregion
    to run in a particular ``dtype``.  Disabling NpuAutocast gives you explicit control over
    the execution type.  In the subregion, inputs from the surrounding region
    should be cast to ``dtype`` before use::

        # Creates some tensors in default dtype (here assumed to be float32)
        a_float32 = torch.rand((8, 8), device="npu")
        b_float32 = torch.rand((8, 8), device="npu")
        c_float32 = torch.rand((8, 8), device="npu")
        d_float32 = torch.rand((8, 8), device="npu")

        with NpuAutocast():
            e_float16 = torch.mm(a_float32, b_float32)

            with NpuAutocast(enabled=False):
                # Calls e_float16.float() to ensure float32 execution
                # (necessary because e_float16 was created in an autocasted region)
                f_float32 = torch.mm(c_float32, e_float16.float())

            # No manual casts are required when re-entering the NpuAutocast-enabled region.
            # torch.mm again runs in float16 and produces float16 output, regardless of input types.
            g_float16 = torch.mm(d_float32, f_float32)

    The NpuAutocast state is thread-local.  If you want it enabled in a new thread, the context manager or decorator
    must be invoked in that thread.  This affects :class:`torch.nn.DataParallel` and
    :class:`torch.nn.parallel.DistributedDataParallel` when used with more than one NPU per process
    (see :ref:`Working with Multiple GPUs<amp-multigpu>`).

    Args:
        enabled(bool, optional, default=True):  Whether autocasting should be enabled in the region.
    """
    def __init__(self, enabled=True):
        if enabled and amp_definitely_not_available():
            warnings.warn("torch.npu.amp.NpuAutocast only affects NPU ops, but NPU is not available.  Disabling.")
            self._enabled = False
        else:
            self._enabled = enabled