# Copyright (c) 2026 Huawei Technologies Co., Ltd
# All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific Language governing permissions and
# limitations under the License.
"""
Add validation cases for
torch.distributed.algorithms.Join.notify_join_context
on NPU.

This file validates disabled Join behavior, Joinable initialization,
first/non-first Joinable behavior, asynchronous HCCL all-reduce
notification, and the throw_on_early_termination execution path.
"""

import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.distributed.algorithms.join import (
    Join,
    Joinable,
    JoinHook,
)
from torch_npu.testing.testcase import TestCase, run_tests


WORLD_SIZE = 2


def _find_free_port():
    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM,
    )

    sock.bind(
        ("127.0.0.1", 0)
    )

    port = sock.getsockname()[1]

    sock.close()

    return port


def _init_process_group(
    rank,
    world_size,
    port,
):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)

    torch.npu.set_device(rank)

    dist.init_process_group(
        backend="hccl",
        rank=rank,
        world_size=world_size,
    )


class _DummyJoinHook(JoinHook):
    def main_hook(self):
        pass

    def post_hook(
        self,
        is_last_joiner,
    ):
        pass


class _DummyJoinable(Joinable):
    def __init__(
        self,
        device,
        process_group,
    ):
        super().__init__()

        self._device = device
        self._process_group = process_group

    def join_hook(
        self,
        **kwargs,
    ):
        return _DummyJoinHook()

    @property
    def join_device(self):
        return self._device

    @property
    def join_process_group(self):
        return self._process_group


class _InvalidJoinable(Joinable):
    def __init__(self):
        # Intentionally do NOT call:
        #
        # super().__init__()
        #
        # This validates the _join_config check.
        self._device = torch.device("npu:0")
        self._process_group = None

    def join_hook(
        self,
        **kwargs,
    ):
        return _DummyJoinHook()

    @property
    def join_device(self):
        return self._device

    @property
    def join_process_group(self):
        return self._process_group


def _worker_notify_join_context(
    rank,
    world_size,
    port,
):
    _init_process_group(
        rank,
        world_size,
        port,
    )

    device = torch.device(
        f"npu:{rank}"
    )

    process_group = dist.group.WORLD

    first_joinable = _DummyJoinable(
        device,
        process_group,
    )

    second_joinable = _DummyJoinable(
        device,
        process_group,
    )

    # Join.__init__() configures:
    #
    # first_joinable:
    #     is_first_joinable = True
    #
    # second_joinable:
    #     is_first_joinable = False
    Join(
        [
            first_joinable,
            second_joinable,
        ],
        enable=True,
        throw_on_early_termination=False,
    )

    # ---------------------------------------------------------
    # First Joinable
    #
    # notify_join_context() should create:
    #
    # torch.ones(1, device=npu)
    #
    # and issue:
    #
    # dist.all_reduce(..., async_op=True)
    #
    # Therefore a Work handle should be returned.
    # ---------------------------------------------------------

    work = Join.notify_join_context(
        first_joinable
    )

    assert work is not None

    work.wait()

    torch.npu.synchronize()


    # ---------------------------------------------------------
    # Non-first Joinable
    #
    # The API should be vacuous and return None.
    # ---------------------------------------------------------

    result = Join.notify_join_context(
        second_joinable
    )

    assert result is None


    # Make sure both ranks complete the test
    # before destroying the process group.
    dist.barrier(device_ids=[rank])

    dist.destroy_process_group()


def _worker_throw_on_early_termination_enabled(
    rank,
    world_size,
    port,
):
    _init_process_group(
        rank,
        world_size,
        port,
    )

    device = torch.device(
        f"npu:{rank}"
    )

    process_group = dist.group.WORLD


    joinable = _DummyJoinable(
        device,
        process_group,
    )


    # Enable the early-termination detection branch.
    #
    # Both ranks are still active, so:
    #
    # zeros = torch.zeros(1, device=device)
    # all_reduce(zeros)
    #
    # remains zero.
    #
    # No RuntimeError should be raised.
    Join(
        [joinable],
        enable=True,
        throw_on_early_termination=True,
    )


    work = Join.notify_join_context(
        joinable
    )

    assert work is not None

    work.wait()

    torch.npu.synchronize()


    dist.barrier(device_ids=[rank])

    dist.destroy_process_group()


class TestJoinNotifyJoinContext(TestCase):
    def test_notify_join_context_disabled(self):
        """
        Joinable.__init__() creates a disabled _join_config.

        Since Join(...) has not been created for this object,
        notify_join_context() should directly return None.
        """

        joinable = _DummyJoinable(
            torch.device("npu:0"),
            None,
        )

        result = Join.notify_join_context(
            joinable
        )

        self.assertIsNone(result)


    def test_notify_join_context_invalid_joinable(self):
        """
        Verify that a Joinable implementation must call
        Joinable.__init__().
        """

        joinable = _InvalidJoinable()

        with self.assertRaisesRegex(
            AssertionError,
            "constructor calls",
        ):
            Join.notify_join_context(
                joinable
            )


    def test_notify_join_context_hccl(self):
        """
        Validate the real HCCL path on two NPUs.

        First Joinable:
            returns asynchronous Work handle.

        Second Joinable:
            returns None.
        """

        if torch.npu.device_count() < WORLD_SIZE:
            self.skipTest(
                "This test requires at least 2 NPU devices."
            )

        port = _find_free_port()

        mp.spawn(
            _worker_notify_join_context,
            args=(
                WORLD_SIZE,
                port,
            ),
            nprocs=WORLD_SIZE,
            join=True,
        )


    def test_throw_on_early_termination_enabled_hccl(self):
        """
        Validate the extra collective executed when
        throw_on_early_termination=True.

        Both ranks are active, so the API should not
        report early termination.
        """

        if torch.npu.device_count() < WORLD_SIZE:
            self.skipTest(
                "This test requires at least 2 NPU devices."
            )

        port = _find_free_port()

        mp.spawn(
            _worker_throw_on_early_termination_enabled,
            args=(
                WORLD_SIZE,
                port,
            ),
            nprocs=WORLD_SIZE,
            join=True,
        )


if __name__ == "__main__":
    run_tests()
