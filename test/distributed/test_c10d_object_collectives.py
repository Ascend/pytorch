from functools import partial, wraps
import os
import types
import unittest

import torch
import torch.distributed as dist

import torch_npu  # noqa: F401
from torch_npu._compat.version import CURRENT_VERSION
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize


def with_comms(func=None):
    if func is None:
        return partial(
            with_comms,
        )

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        self.dist_init()
        func(self, *args, **kwargs)
        self.destroy_comms()

    return wrapper


# COMPAT(< 2.14): upstream pytorch#189353 added weights_only to object
# collectives in 2.14. On 2.13 the upstream APIs have no such kwarg, so only
# generate the weights_only=False variants there.
_WEIGHTS_ONLY_PARAMS = [True, False] if CURRENT_VERSION >= (2, 14) else [False]


class TestObjectCollectives(TestCase):
    MAIN_PROCESS_RANK = -1

    def join_or_run(self, fn):
        @wraps(fn)
        def wrapper(self):
            if self.rank == self.MAIN_PROCESS_RANK:
                for p in self.processes:
                    p.join()
            else:
                fn()

        return types.MethodType(wrapper, self)

    def __init__(self, method_name: str = "runTest") -> None:
        super().__init__(method_name)
        fn = getattr(self, method_name)
        setattr(self, method_name, self.join_or_run(fn))

    def setUp(self):
        super(TestCase, self).setUp()
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29501'
        os.environ["BACKEND"] = dist.Backend.HCCL
        self.processes = []
        self.rank = self.MAIN_PROCESS_RANK
        proc = torch.multiprocessing.get_context("spawn").Process

        for rank in range(int(self.world_size)):
            process = proc(
                target=self.__class__._run,
                name="process " + str(rank),
                args=(rank, self._current_test_name()),
            )
            process.start()
            self.processes.append(process)

    def tearDown(self):
        super().tearDown()
        for p in self.processes:
            p.terminate()
        self.processes = []

    def _current_test_name(self) -> str:
        # self.id() == e.g. '__main__.TestDistributed.TestAdditive.test_get_rank'
        return self.id().split(".")[-1]

    @property
    def world_size(self) -> int:
        return 4

    @classmethod
    def _run(cls, rank: int, test_name: str) -> None:
        self = cls(test_name)
        self.rank = rank
        getattr(self, test_name)()

    def destroy_comms(self):
        # Wait for all ranks to reach here before starting shutdown.
        dist.barrier()
        dist.destroy_process_group()

    def dist_init(self):
        torch.npu.set_device(self.rank)
        dist.init_process_group(backend="hccl", rank=self.rank, world_size=self.world_size)

    @skipIfUnsupportMultiNPU(4)
    @parametrize("weights_only", _WEIGHTS_ONLY_PARAMS)
    @with_comms()
    def test_all_gather_object(self, weights_only):
        gather_objects = ["foo", 12, {1: 2}, ["foo", 12, {1: 2}]]
        output = [None for _ in gather_objects]
        if CURRENT_VERSION >= (2, 14):
            dist.all_gather_object(output, gather_objects[dist.get_rank()], weights_only=weights_only)
        else:
            dist.all_gather_object(output, gather_objects[dist.get_rank()])
        self.assertEqual(output, gather_objects)

    @skipIfUnsupportMultiNPU(4)
    @parametrize("weights_only", _WEIGHTS_ONLY_PARAMS)
    @with_comms()
    def test_broadcast_object_list(self, weights_only):
        expected_objects = ["foo", 12, {1: 2}, ["foo", 12, {1: 2}]]
        if dist.get_rank() == 0:
            objects = expected_objects  # any picklable object
        else:
            objects = [None, None, None, None]
        if CURRENT_VERSION >= (2, 14):
            dist.broadcast_object_list(objects, src=0, weights_only=weights_only)
        else:
            dist.broadcast_object_list(objects, src=0)
        self.assertEqual(objects, expected_objects)

    @skipIfUnsupportMultiNPU(4)
    @parametrize("weights_only", _WEIGHTS_ONLY_PARAMS)
    @with_comms()
    def test_scatter_object_list(self, weights_only):
        input_list = list(range(dist.get_world_size())) if self.rank == 0 else None
        output_list = [None]
        if CURRENT_VERSION >= (2, 14):
            dist.scatter_object_list(
                scatter_object_output_list=output_list,
                scatter_object_input_list=input_list,
                weights_only=weights_only)
        else:
            dist.scatter_object_list(
                scatter_object_output_list=output_list,
                scatter_object_input_list=input_list)

        self.assertEqual(self.rank, output_list[0])

    @skipIfUnsupportMultiNPU(4)
    @parametrize("weights_only", _WEIGHTS_ONLY_PARAMS)
    @with_comms()
    def test_gather_object(self, weights_only):
        gather_objects = ["foo", 12, {1: 2}, ["foo", 12, {1: 2}]]
        obj = gather_objects[dist.get_rank()]
        output = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
        dist.gather_object(obj=obj, object_gather_list=output, dst=0, weights_only=weights_only)
        if dist.get_rank() == 0:
            self.assertEqual(output, gather_objects)

    @skipIfUnsupportMultiNPU(4)
    @parametrize("weights_only", _WEIGHTS_ONLY_PARAMS)
    @with_comms()
    def test_gather_object_group_dst(self, weights_only):
        # group_dst is the dst rank within `group`. For the default group
        # (== the global group here), group_dst=0 is equivalent to dst=0.
        gather_objects = ["foo", 12, {1: 2}, ["foo", 12, {1: 2}]]
        obj = gather_objects[dist.get_rank()]
        output = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
        dist.gather_object(obj=obj, object_gather_list=output, group_dst=0,
                           weights_only=weights_only)
        if dist.get_rank() == 0:
            self.assertEqual(output, gather_objects)

    @skipIfUnsupportMultiNPU(4)
    @with_comms()
    def test_gather_object_subgroup(self):
        # Sub-group [2, 3] with group_dst=0 (group-local view of global rank
        # 2). _gather_object canonicalizes group_dst to the global dst, and
        # torch_npu._gather must convert it back to a group-local rootRank
        # before calling into HCCL, whose assertRootRank validates in group
        # space (without the conversion HCCL raises "invalid root rank: 2").
        # use_compatible_impl must be on so _gather takes the real HCCL
        # gather path instead of the allgather fallback.
        torch_npu.npu.use_compatible_impl(True)
        subgroup = dist.new_group(ranks=[2, 3])
        if self.rank in (2, 3):
            output = [None, None] if self.rank == 2 else None
            dist.gather_object(self.rank, object_gather_list=output,
                               group=subgroup, group_dst=0)
            if self.rank == 2:
                self.assertEqual(output, [2, 3])

    @skipIfUnsupportMultiNPU(4)
    @with_comms()
    def test_gather_object_dst_and_group_dst_conflict(self):
        # dst and group_dst are mutually exclusive. The check raises on all
        # ranks before any communication happens, so assertRaises is safe
        # (no barrier deadlock).
        with self.assertRaises(ValueError):
            dist.gather_object(obj="foo", object_gather_list=[None], dst=0, group_dst=0)

    @skipIfUnsupportMultiNPU(4)
    @unittest.skipIf(
        CURRENT_VERSION < (2, 14),
        "weights_only added to object collectives in torch 2.14 (upstream pytorch#189353)",
    )
    @with_comms()
    def test_weights_only_rejects_unsafe_object(self):
        output = [None] * dist.get_world_size()
        with self.assertRaises(Exception):
            dist.all_gather_object(output, obj=with_comms, weights_only=True)

    @skipIfUnsupportMultiNPU(4)
    @unittest.skipIf(
        CURRENT_VERSION < (2, 14),
        "weights_only added to object collectives in torch 2.14 (upstream pytorch#189353)",
    )
    @with_comms()
    def test_gather_object_weights_only_rejects_unsafe_object(self):
        # gather_object goes through torch_npu's own _gather_object. Unlike
        # all_gather_object (where every rank deserializes and fails), only
        # the dst rank deserializes here, so only dst raises. Catch per rank
        # and assert after the collective: a bare assertRaises would either
        # fail on non-dst ranks or deadlock them at the barrier in
        # destroy_comms.
        obj = with_comms
        output = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
        err = None
        try:
            dist.gather_object(obj=obj, object_gather_list=output, dst=0, weights_only=True)
        except Exception as e:
            err = e
        if dist.get_rank() == 0:
            self.assertIsNotNone(err)
        else:
            self.assertIsNone(err)


instantiate_parametrized_tests(TestObjectCollectives)

if __name__ == "__main__":
    run_tests()
