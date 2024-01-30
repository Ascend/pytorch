import copy
import os
import sys
import unittest
from contextlib import nullcontext
from typing import Any, cast, List

import numpy as np

import torch
import torch.distributed as dist

from torch.distributed.algorithms.ddp_comm_hooks.ddp_zero_hook import (
    hook_with_zero_step,
    hook_with_zero_step_interleaved,
)
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook
from torch.distributed.algorithms.join import Join, Joinable, JoinHook
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.distributed.optim.zero_redundancy_optimizer import _broadcast_object
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW, SGD
from torch.testing._internal import common_distributed
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_WINDOWS,
    parametrize,
    run_tests,
    TEST_WITH_ASAN,
    TEST_WITH_DEV_DBG_ASAN,
)

try:
    import torchvision

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

import torch_npu
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


def _get_backend_for_tests():
    return dist.Backend.HCCL



BACKEND = _get_backend_for_tests()


class TestZeroRedundancyOptimizer(common_distributed.MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        os.environ["WORLD_SIZE"] = str(self.world_size)
        self._spawn_processes()

    @property
    def device(self):
        return (
            torch.device("npu") if torch.npu.is_available() else torch.device("cpu")
        )

    @property
    def world_size(self):
        return 1

    def tearDown(self):
        try:
            torch.distributed.destroy_process_group()
        except AssertionError:
            pass
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def dist_init(self, rank, world_size=-1, backend=BACKEND):
        if world_size < 1:
            world_size = self.world_size
        store = dist.FileStore(self.file_name, world_size)
        return dist.init_process_group(
            backend=backend,
            store=store,
            rank=rank,
            world_size=world_size,
        )


class TestZeroRedundancyOptimizerSingleRank(TestZeroRedundancyOptimizer):
    @skipIfUnsupportMultiNPU(2)
    def test_state_dict(self):
        torch.npu.set_device(self.rank)
        self.dist_init(self.rank)
        LR1 = 0.1
        LR2 = 0.01
        MOMENTUM = 0.9
        RECIPIENT_RANK = 0  # rank 0 is the only rank since the world size is 1
        x = torch.tensor([1.0], device=self.device, requires_grad=True)
        optim = ZeroRedundancyOptimizer(
            [x],
            optimizer_class=SGD,
            lr=LR1,
            momentum=MOMENTUM,
        )
        x.backward()
        optim.step()
        self.assertEqual(x, torch.tensor([0.9], device=self.device))
        self.assertEqual(
            optim.optim.state[x]["momentum_buffer"],
            torch.tensor([1.0], device=self.device),
        )

        optim.zero_grad()
        optim.consolidate_state_dict(to=RECIPIENT_RANK)
        state_dict = optim.state_dict()

        # Check that the state dict has keys compliant with PyTorch
        self.assertIn("param_groups", state_dict.keys())
        self.assertIn("state", state_dict.keys())

        # Check that the state has the expected keys
        self.assertEqual(state_dict["param_groups"][0]["lr"], 0.1)
        self.assertEqual(state_dict["param_groups"][0]["momentum"], 0.9)
        self.assertFalse(state_dict["param_groups"][0]["nesterov"])
        self.assertEqual(state_dict["param_groups"][0]["weight_decay"], 0.0)
        self.assertEqual(state_dict["param_groups"][0]["dampening"], 0.0)

        # Check that the state and the `param_groups` attribute are in sync
        for k in state_dict["param_groups"][0]:
            if k != "params":
                self.assertEqual(
                    state_dict["param_groups"][0][k],
                    optim.param_groups[0][k],
                )

        # Check that the state is reloaded with the correct values and device
        optim = ZeroRedundancyOptimizer([x], optimizer_class=SGD, lr=LR2)
        optim.load_state_dict(state_dict)
        self.assertEqual(
            optim.optim.state[x]["momentum_buffer"],
            torch.tensor([1.0], device=self.device),
        )

        # We should we using `LR1` and not `LR2` after reloading, both within
        # the optimizer and as exposed by the `param_groups` attribute
        self.assertEqual(optim.param_groups[0]["lr"], LR1)
        x.backward()
        optim.step()
        self.assertEqual(x, torch.tensor([0.71], device=self.device))
        self.assertEqual(
            optim.optim.state[x]["momentum_buffer"],
            torch.tensor([1.9], device=self.device),
        )

        # Check that the exposed `param_groups`` are on the proper device
        self.assertEqual(optim.param_groups[0]["params"][0].device, x.device)

    @skipIfUnsupportMultiNPU(2)
    def test_lr_scheduler(self):
        torch.npu.set_device(self.rank)
        self.dist_init(self.rank)
        NUM_ITERS = 5
        LR = 0.01
        x = torch.tensor([1.0], device=self.device, requires_grad=True)
        x2 = torch.tensor([1.0], device=self.device, requires_grad=True)
        o1 = ZeroRedundancyOptimizer([x], optimizer_class=SGD, lr=LR)
        o2 = torch.optim.SGD([x2], lr=LR)
        s = torch.optim.lr_scheduler.StepLR(o1, 1)
        s2 = torch.optim.lr_scheduler.StepLR(o2, 1)
        for _ in range(NUM_ITERS):
            x.backward()
            o1.zero_grad()
            o1.step()
            s.step()
            x2.backward()
            o2.zero_grad()
            o2.step()
            s2.step()
            self.assertEqual(x, x2)

    @skipIfUnsupportMultiNPU(2)
    def test_step_with_kwargs(self):
        torch.npu.set_device(self.rank)
        self.dist_init(self.rank)
        LR = 0.1

        class SGDWithStepKWArg(torch.optim.SGD):
            def step(self, closure=None, kwarg=None):
                super().step()
                kwarg.append(5)

        kwarg: List[Any] = []
        x = torch.tensor([1.0], device=self.device, requires_grad=True)
        optim = ZeroRedundancyOptimizer(
            [x],
            optimizer_class=SGDWithStepKWArg,
            lr=LR,
        )
        x.backward()
        optim.step(0, kwarg=kwarg)
        self.assertEqual(kwarg, [5])
        self.assertEqual(x, torch.tensor([0.9], device=self.device))

    @skipIfUnsupportMultiNPU(2)
    def test_step_with_extra_inner_key(self):
        torch.npu.set_device(self.rank)
        self.dist_init(self.rank)
        LR = 0.1

        class SGDWithNewKey(torch.optim.SGD):
            # Dummy optimizer which adds a new key to the param groups
            def step(self, closure=None):
                super().step()
                self.param_groups[0]["new_key"] = 0.1

        x = torch.tensor([1.0], device=self.device, requires_grad=True)
        optim = ZeroRedundancyOptimizer([x], optimizer_class=SGDWithNewKey, lr=LR)
        x.backward()
        optim.step()
        self.assertEqual(optim.param_groups[0]["new_key"], 0.1)
        self.assertEqual(x, torch.tensor([0.9], device=self.device))

    @skipIfUnsupportMultiNPU(2)
    def test_step_without_closure(self):
        torch.npu.set_device(self.rank)
        self.dist_init(self.rank)
        LR = 0.1

        class SGDWithoutClosure(torch.optim.SGD):
            def step(self):
                return super().step()

        x = torch.tensor([1.0], device=self.device, requires_grad=True)
        optim = ZeroRedundancyOptimizer(
            [x],
            optimizer_class=SGDWithoutClosure,
            lr=LR,
        )
        x.backward()
        optim.step()
        self.assertEqual(x, torch.tensor([0.9], device=self.device))

    @skipIfUnsupportMultiNPU(2)
    def test_zero_grad(self):
        torch.npu.set_device(self.rank)
        self.dist_init(self.rank)
        LR = 0.01
        x = torch.rand(1)
        m = torch.nn.Linear(1, 1)
        optim = ZeroRedundancyOptimizer(m.parameters(), optimizer_class=SGD, lr=LR)
        y = m(x)
        y.backward(x)
        self.assertNotEqual(m.weight.grad, torch.zeros_like(m.weight))
        self.assertNotEqual(m.weight.grad, torch.zeros_like(m.weight))
        optim.zero_grad()
        self.assertIsNone(m.weight.grad)
        self.assertIsNone(m.bias.grad)

    @skipIfUnsupportMultiNPU(2)
    def test_constructor(self):
        torch.npu.set_device(self.rank)
        self.dist_init(self.rank)
        LR = 0.01
        m = torch.nn.Sequential(
            torch.nn.Linear(5, 10),
            torch.nn.Linear(10, 10),
            torch.nn.Linear(10, 10),
        )
        # Test various constructor inputs in the form: (input, expected error)
        ctor_inputs = [
            ([], ValueError),  # empty parameter list
            (torch.randn(1), TypeError),  # non-iterable: `torch.Tensor`
            (1.2, TypeError),  # non-iterable: `float`
            (
                [
                    {"params": [l1.weight for l1 in m]},
                    {"params": [l2.bias for l2 in m]},
                ],
                None,
            ),  # iterable of dict
            (
                list(m.parameters()) + [42],
                TypeError,
            ),  # iterable containing invalid type
            (m.parameters(), None),  # `params` as a generator
            (list(m.parameters()), None),  # `params` as a list
        ]
        for ctor_input, error in ctor_inputs:
            context = self.assertRaises(error) if error else nullcontext()
            with context:
                ZeroRedundancyOptimizer(
                    ctor_input,
                    optimizer_class=SGD,
                    lr=LR,
                )

        # Test constructing with multiple parameter groups more thoroughly
        WD = 0.01
        BETAS = (0.9, 0.999)
        EPS = 1e-8
        params = [
            {"params": [l1.weight for l1 in m], "weight_decay": 0.0},
            {"params": [l2.bias for l2 in m], "weight_decay": WD},
        ]
        optim = ZeroRedundancyOptimizer(
            params,
            optimizer_class=AdamW,
            lr=LR,
            betas=BETAS,
            eps=EPS,
        )
        if len(optim.param_groups) != 2:
            raise AssertionError(f"Expected 2 ZeRO param groups, but got {len(optim.param_groups)}")
        if len(optim.optim.param_groups) != 2:
            raise AssertionError("Expected 2 local optimizer param groups, but got "
                                 f"{len(optim.optim.param_groups)}")




class TestZeroRedundancyOptimizerDistributed(TestZeroRedundancyOptimizer):
    @property
    def device(self):
        return (
            torch.device(self.rank)
            if torch.npu.is_available()
            else torch.device("cpu")
        )

    @property
    def world_size(self):
        return min(2, max(2, torch.npu.device_count()))

    @property
    def context(self):
        return (
            nullcontext()
            if not torch.npu.is_available()
            else torch.npu.device(self.rank)
        )

    def _check_same_model_params(
        self,
        model_a: torch.nn.Module,
        model_b: torch.nn.Module,
        message: str = "",
    ) -> None:
        # Check that model parameters match
        for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
            torch.testing.assert_close(
                p_a,
                p_b,
                atol=1e-3,
                rtol=1e-5,
                msg=f"Model parameters differ:\n{p_a} {p_b}\n" + message,
            )
        # Check that model buffers match
        for b_a, b_b in zip(model_a.buffers(), model_b.buffers()):
            torch.testing.assert_close(
                b_a,
                b_b,
                msg=f"Model buffers differ:\n{b_a} {b_b}\n" + message,
            )


    @skipIfUnsupportMultiNPU(2)
    def test_step(self):
        torch.npu.set_device(self.rank)
        self.dist_init(self.rank, world_size=self.world_size)
        LR = 0.01

        with self.context:
            x = torch.tensor([float(self.rank + 1)], device=self.device)
            m = torch.nn.Linear(1, 1)
            m.weight.data = torch.tensor([[1.0]])
            m.bias.data = torch.tensor([2.0])
            m = m.to(self.device)
            m_zero = copy.deepcopy(m).to(self.device)

            optim = SGD(m.parameters(), lr=LR)
            o_zero = ZeroRedundancyOptimizer(
                m_zero.parameters(),
                optimizer_class=SGD,
                lr=LR,
            )

            y = m(x)
            y.backward(x)
            y_zero = m_zero(x)
            y_zero.backward(x)

            for p in m.parameters():
                dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                p.grad.data /= self.world_size
            optim.step()
            for p in m_zero.parameters():
                dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                p.grad.data /= self.world_size
            o_zero.step()

            self.assertEqual(m.weight, m_zero.weight)
            self.assertEqual(m.bias, m_zero.bias)


    @skipIfUnsupportMultiNPU(2)
    def test_step_with_closure(self):
        torch.npu.set_device(self.rank)
        self.dist_init(self.rank, world_size=self.world_size)

        with self.context:
            for bucket_view in [False, True]:
                x_val = self.rank + 1
                weight = 1.0
                bias = 2.0
                error = 1.0
                target = torch.tensor(
                    [x_val * weight + bias + error],
                    device=self.device,
                )
                loss_fn = torch.nn.L1Loss()

                x = torch.tensor([float(x_val)], device=self.device)
                m = torch.nn.Linear(1, 1)
                m.weight.data = torch.tensor([[weight]])
                m.bias.data = torch.tensor([bias])
                m.to(self.device)

                optim = ZeroRedundancyOptimizer(
                    m.parameters(),
                    optimizer_class=SGD,
                    parameters_as_bucket_view=bucket_view,
                    lr=0.1,
                )

                y = m(x)
                y.backward(x)
                for p in m.parameters():
                    dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                    p.grad.data /= self.world_size

                def closure():
                    optim.zero_grad()
                    output = m(x)
                    loss = loss_fn(output, target)
                    loss.backward()
                    return loss

                loss = optim.step(closure=closure)

                self.assertEqual(loss, torch.tensor(error))
                self.assertEqual(m.weight, torch.tensor([[1.1]]))
                self.assertEqual(m.bias, torch.tensor([2.1]))

    @skipIfUnsupportMultiNPU(2)
    def test_lr_scheduler(self):
        torch.npu.set_device(self.rank)
        self.dist_init(self.rank)
        x = torch.tensor([1.0], device=self.device, requires_grad=True)
        x2 = torch.tensor([1.0], device=self.device, requires_grad=True)
        optim = ZeroRedundancyOptimizer([x], optimizer_class=SGD, lr=0.01)
        o2 = torch.optim.SGD([x2], lr=0.01)
        s = torch.optim.lr_scheduler.StepLR(optim, 1)
        s2 = torch.optim.lr_scheduler.StepLR(o2, 1)
        for _ in range(5):
            x.backward()
            optim.zero_grad()
            optim.step()
            s.step()
            x2.backward()
            o2.zero_grad()
            o2.step()
            s2.step()
            self.assertEqual(x, x2)

    @skipIfUnsupportMultiNPU(2)
    def test_sharding(self):
        torch.npu.set_device(self.rank)
        self.dist_init(self.rank)
        LR = 0.01
        sizes = [9, 7, 5, 3]
        params = []
        for size in sizes * self.world_size:
            params.append(torch.rand(size, 1))
        optim = ZeroRedundancyOptimizer(params, optimizer_class=SGD, lr=LR)
        self.assertEqual(
            sum([x.numel() for x in optim.optim.param_groups[0]["params"]]),
            sum(sizes),
        )

    @skipIfUnsupportMultiNPU(2)
    def test_add_param_group(self):
        torch.npu.set_device(self.rank)
        self.dist_init(self.rank)
        LR = 0.01

        # Test with all parameters trainable to begin with
        def all_trainable():
            params = []
            sizes = [9, 7, 5, 3]
            sizes_world = sizes * self.world_size
            for size in sizes_world[:-1]:
                params.append(torch.rand(size, 1))

            # Make sure that the params are trainable so that they are factored
            # into the size-based parameter partitioning
            for p in params:
                p.requires_grad = True

            optim = ZeroRedundancyOptimizer(params, optimizer_class=SGD, lr=LR)
            self.assertEqual(len(optim.param_groups), 1)
            optim.add_param_group({"params": [torch.rand(3, 1)]})
            # Verify that new group is added to the correct partition, making
            # all partitions have the same elements
            self.assertEqual(len(optim.param_groups), 2)
            self.assertEqual(
                sum([x.numel() for g in optim.optim.param_groups for x in g["params"]]),
                sum(sizes),
            )
            self.assertEqual(len(optim.optim.param_groups), 2)

        # Test a pathological config with a first big non-trainable param
        def some_trainable():
            params = []
            for size in [100, 3, 5, 2, 6, 4]:
                params.append(torch.rand(size, 1))

            # Make sure that all but the first param are trainable so that they
            # are factored into the size-based parameter partitioning
            for p in params[1:]:
                p.requires_grad = True

            optim = ZeroRedundancyOptimizer(params, optimizer_class=SGD, lr=LR)
            self.assertEqual(len(optim.param_groups), 1)
            optim.add_param_group({"params": [torch.rand(3, 1)]})
            self.assertEqual(len(optim.param_groups), 2)
            self.assertEqual(len(optim.optim.param_groups), 2)

        all_trainable()
        some_trainable()

    @skipIfUnsupportMultiNPU(2)
    def test_multiple_param_groups(self):
        torch.npu.set_device(self.rank)
        self.dist_init(self.rank)
        BATCH_SIZE, NUM_ITERS = 8, 3
        INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM = 5, 10, 5
        WD, LR = 0.01, 0.01
        model1 = torch.nn.Sequential(
            torch.nn.Linear(INPUT_DIM, HIDDEN_DIM),
            torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            torch.nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
        )
        model2 = copy.deepcopy(model1)
        model3 = copy.deepcopy(model1)
        model1 = model1.to(self.device)
        model2 = model2.to(self.device)
        model3 = model3.to(self.device)
        inputs = [
            torch.randn(BATCH_SIZE, INPUT_DIM).to(self.device) for _ in range(NUM_ITERS)
        ]
        # Construct `optim1` with both parameter groups upfront
        optim1 = ZeroRedundancyOptimizer(
            [
                {"params": [l1.weight for l1 in model1], "weight_decay": 0.0},
                {"params": [l2.bias for l2 in model1], "weight_decay": WD},
            ],
            optimizer_class=AdamW,
            lr=LR,
        )
        # Construct `optim2` by adding the second parameter after
        optim2 = ZeroRedundancyOptimizer(
            [l1.weight for l1 in model2],
            optimizer_class=AdamW,
            lr=LR,
            weight_decay=0.0,
        )
        optim2.add_param_group({"params": [l1.bias for l1 in model2], "weight_decay": WD})
        # Construct `optim3` as a non-sharded optimizer
        optim3 = AdamW(
            [
                {"params": [l1.weight for l1 in model3], "weight_decay": 0.0},
                {"params": [l2.bias for l2 in model3], "weight_decay": WD},
            ],
            lr=LR,
        )
        # Check parity over a few iterations
        for input_ in inputs:
            for model, optim in (
                (model1, optim1),
                (model2, optim2),
                (model3, optim3),
            ):
                optim.zero_grad()
                out = model(input_)
                loss = out.sum()
                loss.backward()
                optim.step()
            for layer1, layer2, layer3 in zip(model1, model2, model3):
                torch.testing.assert_close(layer1.weight, layer2.weight)
                torch.testing.assert_close(layer1.weight, layer3.weight)
                torch.testing.assert_close(layer1.bias, layer2.bias)
                torch.testing.assert_close(layer1.bias, layer3.bias)


    @skipIfUnsupportMultiNPU(2)
    def test_collect_shards(self):
        torch.npu.set_device(self.rank)
        self.dist_init(self.rank)
        LR = 1e-3
        MOMENTUM = 0.99
        BATCH_SIZE, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM = 3, 20, 10, 5
        REFERENCE_RANK = 0
        target = torch.rand((BATCH_SIZE, OUTPUT_DIM), device=self.device)
        inputs = torch.rand((BATCH_SIZE, INPUT_DIM), device=self.device)
        model = torch.nn.Sequential(
            torch.nn.Linear(INPUT_DIM, HIDDEN_DIM),
            torch.nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
        ).to(self.device)
        loss_fn = torch.nn.L1Loss()
        loss_fn.to(self.device)
        optimizer = ZeroRedundancyOptimizer(
            model.parameters(),
            optimizer_class=SGD,
            lr=LR,
            momentum=MOMENTUM,  # ensure there exists state to shard
        )

        def closure():
            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_fn(output, target)
            loss.backward()
            return loss

        # Run a dummy step so that the optimizer state dict exists
        _ = optimizer.step(closure=closure)

        # Get the optimizer state on the reference rank
        optimizer.consolidate_state_dict(to=REFERENCE_RANK)
        if self.rank == REFERENCE_RANK:
            # Check that the state has the correct size
            optimizer_state_dict = optimizer.state_dict()
            self.assertEqual(
                len(optimizer_state_dict["state"]),
                len(list(model.parameters())),
            )
        else:
            optimizer_state_dict = {}

        # Load the optimizer state on all ranks without any exceptions
        optimizer_state_dict = _broadcast_object(
            optimizer_state_dict,
            src_rank=REFERENCE_RANK,
            group=dist.group.WORLD,
            device=self.device,
        )
        optimizer.load_state_dict(optimizer_state_dict)

    @skipIfUnsupportMultiNPU(2)
    def test_nondefault_process_group(self):
        MIN_WORLD_SIZE = 4
        if self.world_size < MIN_WORLD_SIZE:
            common_distributed.logger.info(
                "Skipping `test_nondefault_process_group()` since world size "
                "of %s is less than %s",
                self.world_size, MIN_WORLD_SIZE
            )
            return

        torch.npu.set_device(self.rank)
        self.dist_init(self.rank, self.world_size, "hccl")

        device = torch.device(self.rank)

        # Create a new process group consisting of the even ranks to exercise
        # the case where the global and local ranks do not necessarily match
        subgroup_ranks = [r for r in range(self.world_size) if r % 2 == 0]
        process_group = dist.new_group(
            ranks=subgroup_ranks,
            backend="hccl",
        )
        # Ranks not participating in the new process group are no longer needed
        if self.rank not in subgroup_ranks:
            return

        # Set different seeds across ranks so that each rank gets different
        # training data and hence the model sync check is meaningful
        torch.manual_seed(self.rank)
        np.random.seed(self.rank)

        EPOCHS, BATCH_SIZE, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM = 5, 3, 20, 10, 5
        LR = 1e-3
        MOMENTUM = 0.99
        REFERENCE_RANK = 0
        if REFERENCE_RANK not in subgroup_ranks:
            raise AssertionError("Reference rank must be in the new process group")
        loss_fn = torch.nn.L1Loss().to(device)

        def check(optimizer):
            for _ in range(EPOCHS):
                target = torch.rand((BATCH_SIZE, OUTPUT_DIM), device=device)
                inputs = torch.rand((BATCH_SIZE, INPUT_DIM), device=device)

                def closure():
                    optimizer.zero_grad()
                    output = model(inputs)
                    loss = loss_fn(output, target)
                    loss /= self.world_size
                    loss.backward()
                    dist.all_reduce(loss, group=process_group)
                    return loss

                _ = optimizer.step(closure=closure)

                # Check that the parameters match across ranks after a step
                for pg in optimizer.param_groups:
                    for p in pg["params"]:
                        receptacle = []
                        if self.rank == REFERENCE_RANK:
                            receptacle.extend([p.clone() for _ in subgroup_ranks])
                        else:
                            receptacle.extend([torch.empty(p.size(), device=p.device) for _ in subgroup_ranks])
                        dist.all_gather(
                            receptacle,
                            p,
                            group=process_group,
                        )
                        if self.rank == REFERENCE_RANK:
                            reference_param = receptacle[0]
                            for param in receptacle[1:]:
                                torch.testing.assert_close(
                                    reference_param,
                                    param,
                                    msg="Models differ between ranks",
                                )

        model = torch.nn.Sequential(
            torch.nn.Linear(INPUT_DIM, HIDDEN_DIM),
            torch.nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
        ).to(device)
        optimizer = ZeroRedundancyOptimizer(
            model.parameters(),
            optimizer_class=SGD,
            lr=LR,
            momentum=MOMENTUM,  # ensure there exists state to shard
            process_group=process_group,
        )
        check(optimizer)

    @parametrize(
        "optimizer_class_str",
        ["Adam", "AdamW", "SGD"],
        # Use string to appease the internal test name parser
    )
    @parametrize(
        "maximize",
        [False, True],
    )
    @skipIfUnsupportMultiNPU(2)
    def test_local_optimizer_parity(
        self,
        optimizer_class_str: str,
        maximize: bool,
    ):
        torch.npu.set_device(self.rank)
        self.dist_init(self.rank)
        BATCHES = 20
        BATCH_SIZE = 64
        LR = 1e-3
        INPUT_DIM = 2
        HIDDEN_DIM = 3
        OUTPUT_DIM = 3
        torch.manual_seed(self.rank)
        np.random.seed(self.rank)
        if optimizer_class_str == "Adam":
            optimizer_class = torch.optim.Adam
        elif optimizer_class_str == "AdamW":
            optimizer_class = torch.optim.AdamW
        elif optimizer_class_str == "SGD":
            optimizer_class = torch.optim.SGD
        else:
            raise AssertionError(f"Unsupported optimizer class: {optimizer_class_str}")

        with self.context:
            # Define a base model with a different buffer for each rank
            model = torch.nn.Sequential(
                torch.nn.Linear(INPUT_DIM, HIDDEN_DIM),
                torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
                torch.nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
            ).to(self.device)
            model.register_buffer(
                "test_buffer",
                torch.ones((1), device=self.device) * self.rank,
            )
            # Define models/optimizers for DDP with ZeRO and DDP with local
            # optimizer
            defaults = {"maximize": True} if maximize else {}
            sharded_optimizer = ZeroRedundancyOptimizer(
                params=model.parameters(),
                optimizer_class=optimizer_class,
                lr=LR,
                **defaults,
            )
            sharded_ddp_model = DDP(
                module=model,
                device_ids=[self.rank],
                broadcast_buffers=True,
                find_unused_parameters=True,
            )
            local_model = copy.deepcopy(model).to(self.device)
            ddp_optimizer = optimizer_class(
                local_model.parameters(),
                lr=LR,
                **defaults,
            )
            ddp_model = DDP(
                local_model,
                device_ids=[self.rank],
                broadcast_buffers=True,
                find_unused_parameters=True,
            )
            # Check that the model is properly synchronized between ranks
            # at construction time
            self._check_same_model_params(
                sharded_ddp_model,
                ddp_model,
                "Models differ from the start",
            )

            def check_step():
                input_tensor = torch.rand((BATCH_SIZE, INPUT_DIM))

                def closure_ddp(input_tensor=input_tensor):
                    ddp_optimizer.zero_grad()
                    ddp_loss = ddp_model(input_tensor).abs().sum()
                    ddp_loss.backward()
                    return ddp_loss

                def closure_sharded(input_tensor=input_tensor):
                    sharded_optimizer.zero_grad()
                    sharded_loss = sharded_ddp_model(input_tensor).abs().sum()
                    sharded_loss.backward()
                    return sharded_loss

                loss_ddp = cast(
                    torch.Tensor,
                    ddp_optimizer.step(closure=closure_ddp),
                )
                loss_sharded_optim = cast(
                    torch.Tensor,
                    sharded_optimizer.step(closure=closure_sharded),
                )
                torch.testing.assert_close(
                    loss_ddp,
                    loss_sharded_optim,
                    msg="Losses differ between local optimizer and ZeRO",
                )
                self._check_same_model_params(
                    sharded_ddp_model,
                    ddp_model,
                    "Models differ after a step",
                )

            # Check that parity is maintained
            for i in range(BATCHES):
                check_step()
                # For the second half of batches, change the parameter
                # trainability to further test parity
                if i > BATCHES // 2:
                    next(ddp_model.parameters()).requires_grad = bool(i % 2)
                    next(sharded_ddp_model.parameters()).requires_grad = bool(i % 2)

            # Check that the `state_dict` checkpoints are compatible between
            # the local optimizer and ZeRO
            REFERENCE_RANK = 0
            # - Get states
            ddp_state_dict = ddp_optimizer.state_dict()
            sharded_optimizer.consolidate_state_dict(to=REFERENCE_RANK)
            sharded_optim_state_dict = [
                sharded_optimizer.state_dict() if self.rank == REFERENCE_RANK else {}
            ]
            dist.broadcast_object_list(
                sharded_optim_state_dict,
                src=REFERENCE_RANK,
                group=dist.group.WORLD,
            )
            sharded_optim_state_dict_ = sharded_optim_state_dict[0]

            # - Cross-load the states
            # Run one step and check that the models are still the same
            ddp_state_dict_ref = copy.deepcopy(ddp_state_dict)
            ddp_optimizer.load_state_dict(sharded_optim_state_dict_)
            sharded_optimizer.load_state_dict(ddp_state_dict)
            check_step()

            # - Reload their respective states
            # Run one step and check that the models are still the same
            ddp_optimizer.load_state_dict(ddp_state_dict_ref)
            sharded_optimizer.load_state_dict(sharded_optim_state_dict_)
            check_step()

    def _test_zero_join(self, device):
        NUM_INPUTS = 3
        NUM_EPOCHS = 2
        LR = 0.01
        torch.manual_seed(0)
        torch.npu.manual_seed(0)

        rank = self.rank
        world_size = self.world_size
        is_npu = device.type == "npu"
        backend = _get_backend_for_tests() if is_npu else dist.Backend.GLOO
        torch.npu.set_device(self.rank)
        self.dist_init(rank, world_size, backend)

        model = torch.nn.Sequential(
            torch.nn.Linear(2, 3),
            torch.nn.Linear(3, 3),
            torch.nn.Linear(3, 3),
        )
        model.to(device)

        # DDP ensures correct gradients in data parallel training, so DDP with
        # local optimizers on uneven inputs should be equivalent to ZeRO on
        # uneven inputs with gradients being manually set
        ddp_model = DDP(model, device_ids=[rank]) if is_npu else DDP(model)
        local_optim = torch.optim.Adam(ddp_model.parameters(), lr=LR)
        zero_model = copy.deepcopy(model)
        zero_model.to(device)
        zero_optim = ZeroRedundancyOptimizer(
            zero_model.parameters(),
            torch.optim.Adam,
            lr=LR,
        )
        loss_fn = torch.nn.MSELoss()

        # Use uneven inputs: rank i has i extra inputs
        inputs = [torch.randn(20, 2).to(device) for _ in range(NUM_INPUTS + rank)]
        labels = torch.randn(20, 3).to(device)

        # Save the gradients and parameters from DDP as the ground truth; do
        # so on the last-joining rank (in this case, the largest rank)
        grads_at_each_iter = []
        params_at_each_iter = []
        with ddp_model.join():
            for _ in range(NUM_EPOCHS):
                for input_ in inputs:
                    output = ddp_model(input_)
                    loss_fn(output, labels).backward()
                    if rank == world_size - 1:
                        grads = []
                        for p in ddp_model.parameters():
                            grads.append(p.grad.detach().clone().to(device))
                    local_optim.step()
                    if rank == world_size - 1:
                        params = []
                        for p in ddp_model.parameters():
                            params.append(p.detach().clone().to(device))
                        grads_at_each_iter.append(grads)
                        params_at_each_iter.append(params)

        # Broadcast the saved gradients and parameters to all of the other
        # ranks (which joined early)
        grads_and_params = [grads_at_each_iter, params_at_each_iter]
        grads_and_params = _broadcast_object(
            grads_and_params,
            src_rank=world_size - 1,
            group=dist.group.WORLD,
            device=device,
        )
        grads_at_each_iter = grads_and_params[0]
        params_at_each_iter = grads_and_params[1]
        # once the latter supports loading to the destination device instead
        # of the source device

        # A process must still set the remaining gradients after joining, so we
        # define a join hook to do this before the ZeRO join hook
        class _JoinGradInfo:
            def __init__(self, grads):
                self.grads = grads  # remaining gradients to set (in order)
                self.index = 0

        class _SetGradsJoinHook(JoinHook):
            def __init__(self, zero_optim, grads):
                zero_optim._join_grad_info = _JoinGradInfo(grads)
                self.zero = zero_optim
                super().__init__()

            def main_hook(self):
                join_grad_info = self.zero._join_grad_info
                grads = self.zero._join_grad_info.grads[join_grad_info.index]
                join_grad_info.index += 1
                for p, grad in zip(self.zero._all_params, grads):
                    p.grad = grad.detach().clone().to(device)

        class _GradientSetter(Joinable):
            def __init__(self):
                super().__init__()

            def join_hook(self, **kwargs):
                if "zero_optim" not in kwargs:
                    raise AssertionError("zero_optim not in kwargs")
                if "grads" not in kwargs:
                    raise AssertionError("grads not in kwargs")
                zero_optim = kwargs["zero_optim"]
                grads = kwargs["grads"]
                return _SetGradsJoinHook(zero_optim, grads)

            @property
            def join_device(self):
                return device

            @property
            def join_process_group(self):
                return dist.group.WORLD

        num_grads_after_joining = NUM_EPOCHS * (world_size - rank - 1)
        grads = grads_at_each_iter[-num_grads_after_joining:]
        gradient_setter = _GradientSetter()
        iter_ = 0
        with Join(
            [gradient_setter, zero_optim],
            zero_optim=zero_optim,
            grads=grads,
        ):
            for _ in range(NUM_EPOCHS):
                for _ in inputs:
                    # Notify join context that this process has not joined
                    Join.notify_join_context(gradient_setter)
                    # Set gradients manually
                    for p, grad in zip(
                        zero_model.parameters(),
                        grads_at_each_iter[iter_],
                    ):
                        p.grad = grad.detach().clone().to(device)
                    # Perform optimizer step and check parity
                    zero_optim.step()
                    for p, ddp_p in zip(
                        zero_model.parameters(),
                        params_at_each_iter[iter_],
                    ):
                        torch.testing.assert_close(
                            p,
                            ddp_p,
                            msg="Parameters differ between using ZeRO and "
                            "local optimizer",
                        )
                    iter_ += 1


    @skipIfUnsupportMultiNPU(2)
    def test_zero_join_npu(self):
        self._test_zero_join(self.device)


    def _test_zero_model_parallel(self, parameters_as_bucket_view: bool):
        if self.rank >= 2:
            raise AssertionError("rank greater or equal to 2.")
        NUM_EPOCHS = 2
        NUM_INPUTS = 4
        LR = 0.01
        torch.manual_seed(0)
        torch.npu.manual_seed(0)

        class ModelParallelModel(torch.nn.Module):
            def __init__(self, dev0, dev1):
                super().__init__()
                self.dev0 = dev0
                self.dev1 = dev1
                self.net0 = torch.nn.Linear(10, 10).to(dev0)
                self.relu = torch.nn.ReLU()
                self.net1 = torch.nn.Linear(10, 5).to(dev1)

            def forward(self, x):
                x = x.to(self.dev0)
                x = self.relu(self.net0(x))
                x = x.to(self.dev1)
                return self.net1(x)

        class LocalModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net0 = torch.nn.Linear(10, 10)
                self.relu = torch.nn.ReLU()
                self.net1 = torch.nn.Linear(10, 5)

            def forward(self, x):
                return self.net1(self.relu(self.net0(x)))

        dev0 = torch.device(2 * self.rank)
        dev1 = torch.device(2 * self.rank + 1)
        mp_model = ModelParallelModel(dev0, dev1)
        ddp_model = DDP(mp_model)
        local_model = LocalModel().to(dev0)

        # Ensure the parameters are the same across the two models
        def copy_param(p):
            return torch.nn.Parameter(p.detach().clone().to(dev0))

        local_model.net0.weight = copy_param(mp_model.net0.weight)
        local_model.net0.bias = copy_param(mp_model.net0.bias)
        local_model.net1.weight = copy_param(mp_model.net1.weight)
        local_model.net1.bias = copy_param(mp_model.net1.bias)

        # Compare parity between DDP with model parallelism using ZeRO and
        # a local model using a local optimizer
        zero_optim = ZeroRedundancyOptimizer(
            ddp_model.parameters(),
            optimizer_class=torch.optim.Adam,
            parameters_as_bucket_view=parameters_as_bucket_view,
            lr=LR,
        )
        local_optim = torch.optim.Adam(local_model.parameters(), lr=LR)
        inputs = [torch.randn(20, 10).to(dev0) for _ in range(NUM_INPUTS)]

        for _ in range(NUM_EPOCHS):
            for input_ in inputs:

                def closure_local():
                    local_optim.zero_grad()
                    local_loss = local_model(input_).abs().sum()
                    local_loss.backward()
                    return local_loss

                def closure_ddp():
                    zero_optim.zero_grad()
                    ddp_loss = ddp_model(input_).abs().sum()
                    ddp_loss.backward()
                    return ddp_loss

                local_loss = cast(torch.Tensor, local_optim.step(closure=closure_local))
                ddp_loss = cast(torch.Tensor, zero_optim.step(closure=closure_ddp))

                # Increased tolerances are needed to pass when using TF32
                torch.testing.assert_close(
                    local_loss.cpu(),
                    ddp_loss.cpu(),
                    rtol=1e-03,
                    atol=1e-08,
                ), "Losses differ between local optimizer and ZeRO"

                for local_p, ddp_p in zip(
                    local_model.parameters(), ddp_model.parameters()
                ):
                    torch.testing.assert_close(
                        local_p.cpu(),
                        ddp_p.cpu(),
                        rtol=1e-03,
                        atol=1e-04,
                    ), "Models differ after a step"

    @parametrize(
        "parameters_as_bucket_view",
        [False, True],
    )
    @skipIfUnsupportMultiNPU(2)
    def test_zero_model_parallel(
        self,
        parameters_as_bucket_view: bool,
    ):
        if self.rank >= 2:
            return
        torch.npu.set_device(self.rank)
        self.dist_init(self.rank, world_size=2)
        self._test_zero_model_parallel(parameters_as_bucket_view)
        dist.barrier()

    def _test_ddp_zero_overlap(
        self,
        device,
        hook_constructor,
        gradient_as_bucket_view,
        static_graph,
        **kwargs,
    ):
        SGD_LR = 0.01
        SGD_MOMENTUM = 0.9
        SGD_WEIGHT_DECAY = 0.001
        NUM_INPUTS = 5
        torch.manual_seed(0)
        torch.npu.manual_seed(0)
        torch.use_deterministic_algorithms(True)

        rank = self.rank
        is_npu = device.type == "npu"
        models_to_test = [
            (
                torch.nn.Sequential(
                    torch.nn.Linear(1000, 2000),
                    torch.nn.Linear(2000, 500),
                ),
                [torch.randn(1, 1000).to(device) for _ in range(NUM_INPUTS)],
            )
        ]

        for (model, inputs) in models_to_test:
            device_ids = [rank] if is_npu else None
            # Set up the DDP model overlapping with ZeRO
            ddp_model_overlap = DDP(
                copy.deepcopy(model).to(device),
                device_ids=device_ids,
                gradient_as_bucket_view=gradient_as_bucket_view,
            )
            if static_graph:
                ddp_model_overlap._set_static_graph()
            zero_optim = ZeroRedundancyOptimizer(
                ddp_model_overlap.parameters(),
                optimizer_class=torch.optim.SGD,
                overlap_with_ddp=True,
                lr=SGD_LR,
                momentum=SGD_MOMENTUM,
                weight_decay=SGD_WEIGHT_DECAY,
            )
            ddp_model_overlap.register_comm_hook(
                None,
                hook_constructor(
                    allreduce_hook,
                    ddp_model_overlap,
                    zero_optim,
                    **kwargs,
                ),
            )

            # Set up the DDP model with local optimizer
            ddp_model_local = DDP(
                copy.deepcopy(model).to(device),
                device_ids=device_ids,
                gradient_as_bucket_view=gradient_as_bucket_view,
            )
            if static_graph:
                ddp_model_local._set_static_graph()
            local_optim = torch.optim.SGD(
                ddp_model_local.parameters(),
                lr=SGD_LR,
                momentum=SGD_MOMENTUM,
                weight_decay=SGD_WEIGHT_DECAY,
            )

            # Check that the parameters match initially
            for p1, p2 in zip(
                ddp_model_overlap.parameters(), ddp_model_local.parameters()
            ):
                self.assertEqual(p1, p2)

            # Save the parameters to ensure they were updated
            init_params_overlap = copy.deepcopy(
                list(ddp_model_overlap.parameters())
            )

            # Ensure that this test runs independently
            dist.barrier()

            # Run the DDP model overlapping with ZeRO
            num_warmup_inputs = 2 if not static_graph else 3
            for input_ in inputs[:num_warmup_inputs]:
                output = ddp_model_overlap(input_)
                loss = output.sum()
                loss.backward()
            for input_ in inputs:
                zero_optim.zero_grad()
                output = ddp_model_overlap(input_)
                loss = output.sum()
                loss.backward()

            # Run the DDP model with local optimizer
            for input_ in inputs:
                local_optim.zero_grad()
                output = ddp_model_local(input_)
                loss = output.sum()
                loss.backward()
                local_optim.step()
            dist.barrier()

            # Check that the parameters are equal
            for p1, p2 in zip(
                ddp_model_overlap.parameters(), ddp_model_local.parameters()
            ):
                self.assertEqual(p1, p2)

            # Check that the parameters were updated
            self.assertNotEqual(
                init_params_overlap,
                list(ddp_model_overlap.parameters()),
            )

            # Ensure that this test runs independently
            dist.barrier()


    @parametrize(
        "use_npu",
        [True],
    )
    @parametrize(
        "use_interleaved_hook",
        [False, True],
    )
    @parametrize(
        "gradient_as_bucket_view",
        [False, True],
    )
    @parametrize(
        "static_graph",
        [False, True],
    )
    @parametrize(
        "shard_buckets",
        [False, True],
    )
    @skipIfUnsupportMultiNPU(2)
    def test_ddp_zero_overlap(
        self,
        use_npu: bool,
        use_interleaved_hook: bool,
        gradient_as_bucket_view: bool,
        static_graph: bool,
        shard_buckets: bool,
    ):
        device = torch.device(self.rank) if use_npu else torch.device("cpu")
        backend = _get_backend_for_tests()
        torch.npu.set_device(self.rank)
        self.dist_init(self.rank, self.world_size, backend)
        hook_constructor = (
            hook_with_zero_step
            if not use_interleaved_hook
            else hook_with_zero_step_interleaved
        )

        self._test_ddp_zero_overlap(
            device,
            hook_constructor,
            gradient_as_bucket_view,
            static_graph,
            shard_buckets=shard_buckets,
        )


instantiate_parametrized_tests(TestZeroRedundancyOptimizerSingleRank)
instantiate_parametrized_tests(TestZeroRedundancyOptimizerDistributed)

if __name__ == "__main__":
    run_tests()