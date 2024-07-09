from contextlib import nullcontext
import numpy as np
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.reinplace import reinplace
from torch.library import Library, impl
from torch.utils._pytree import tree_map, tree_map_only, tree_flatten
from torch._dispatch.python import enable_crossref_functionalize, enable_python_dispatcher
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


# meta register implementation
m = Library("npu", "IMPL", "Meta")


def _functionalize(f, *, reapply_views: bool, crossref: bool):
    def to_fun(t: torch.Tensor):
        func_t = torch._to_functional_tensor(t)
        func_t.requires_grad = t.requires_grad
        return func_t

    def wrapped(*inputs):
        ctx = nullcontext()
        if crossref:
            ctx = enable_crossref_functionalize()
        with ctx:
            inputs_functional = tree_map_only(torch.Tensor, to_fun, inputs)
            torch._enable_functionalization(reapply_views=reapply_views)
            try:
                out = f(*inputs_functional)
            finally:
                torch._disable_functionalization()
            flat_inputs, _ = tree_flatten(inputs)
            flat_inputs_functional, _ = tree_flatten(inputs_functional)
            for inpt, input_functional in zip(flat_inputs, flat_inputs_functional):
                torch._sync(input_functional)
                inpt_new = torch._from_functional_tensor(input_functional)
                if inpt_new is not inpt:
                    # Existing deficiency in functionalize():
                    # we don't correctly mutate input metadata (yet?)
                    if inpt_new.shape == inpt.shape:
                        inpt.copy_(inpt_new)
            tree_map(torch._sync, out)
            out_unwrapped = tree_map(torch._from_functional_tensor, out)
            return out_unwrapped

    return wrapped


class TestFunctionalization(TestCase):

    crossref = False

    def get_logs(self, func, *inpts, reapply_views=False, run_reinplace=False):
        inpts_clone = tree_map_only(torch.Tensor, torch.clone, inpts)
        traced_f = make_fx(_functionalize(func, reapply_views=reapply_views, crossref=self.crossref))(*inpts)
        if run_reinplace:
            traced_f = reinplace(traced_f, *inpts_clone)
        return traced_f.code

    def assert_functionalization(self, func, *inpts, reapply_views=False, mutated_input_metadata=False):
        clones1 = tree_map_only(torch.Tensor, torch.clone, inpts)
        clones2 = tree_map_only(torch.Tensor, torch.clone, inpts)
        clones3 = tree_map_only(torch.Tensor, torch.clone, inpts)

        # Compare outputs (and mutated inputs), with and without functionalization.
        out_ref = func(*inpts)
        out_functional = _functionalize(func, reapply_views=reapply_views, crossref=self.crossref)(*clones1)

        # The reinplacing is only valid to run with reapply_views=True.
        functional_func = make_fx(_functionalize(func, reapply_views=True, crossref=self.crossref))(*clones2)
        reinplace_func = reinplace(functional_func, *clones2)

        # NOTE: for now, need to in fresh inputs here, because make_fx
        # will directly mutate the inputs that you trace with.
        # Once this is fixed we can clean this up.
        out_reinplace = reinplace_func(*clones3)

        # functionalize() deficiency: input metadata mutations aren't propagated properly,
        # so we just need to skip checks here for the tests that exercise that.
        if not mutated_input_metadata:
            flat_inpts, _ = tree_flatten(inpts)
            flat_clones1, _ = tree_flatten(clones1)
            flat_clones3, _ = tree_flatten(clones3)
            for inpt, input_clone, input_clone3 in zip(flat_inpts, flat_clones1, flat_clones3):
                self.assertEqual(inpt, input_clone)  # input mutations should still occur
                self.assertEqual(inpt, input_clone3)

        # Handle tests with multi-tensor outputs
        if isinstance(out_ref, tuple):
            out_refs, out_functionals, out_reinplaces = list(out_ref), list(out_functional), list(out_reinplace)
        else:
            out_refs, out_functionals, out_reinplaces = [out_ref], [out_functional], [out_reinplace]

        for out_ref_, out_functional_, out_reinplace_ in zip(out_refs, out_functionals, out_reinplaces):
            self.assertEqual(out_ref_, out_functional_)
            self.assertEqual(out_ref_, out_reinplace_)

    def test_scatter_update(self):
        def f(iself, indices, updates):
            return torch.ops.npu.scatter_update_(iself, indices, updates, -2)
        in_self = torch.randn(4, 4, 32, 256, dtype=torch.float16).npu()
        in_indices = torch.tensor([1, 1, 1, 1]).npu()
        in_updates = torch.randn(4, 4, 1, 256, dtype=torch.float16).npu()

        logs = self.get_logs(f, in_self, in_indices, in_updates)
        self.assertExpectedInline(logs, """\



def forward(self, arg0_1, arg1_1, arg2_1):
    scatter_update = torch.ops.npu.scatter_update.default(arg0_1, arg1_1, arg2_1, -2);  arg1_1 = arg2_1 = None
    copy_ = torch.ops.aten.copy_.default(arg0_1, scatter_update);  arg0_1 = None
    return scatter_update
    """)

        self.assert_functionalization(f, in_self, in_indices, in_updates)

    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_scatter(self):
        def f(fake_var, fake_indices, fake_updates, fake_quant_scales):
            return torch.ops.npu.npu_quant_scatter_(fake_var, fake_indices, fake_updates, fake_quant_scales,
                                                None, -2, -1, "update")

        data_var = np.random.uniform(0, 1, [1, 1, 32]).astype(np.int8)
        in_var = torch.from_numpy(data_var).to(torch.int8).npu()
        data_indices = np.random.uniform(0, 1, [1]).astype(np.int32)
        in_indices = torch.from_numpy(data_indices).to(torch.int32).npu()
        data_updates = np.random.uniform(1, 2, [1, 1, 32]).astype(np.float16)
        in_updates = torch.from_numpy(data_updates).to(torch.bfloat16).npu()
        data_quant_scales = np.random.uniform(0, 1, [1, 1, 32]).astype(np.float16)
        in_quant_scales = torch.from_numpy(data_quant_scales).to(torch.bfloat16).npu()

        logs = self.get_logs(f, in_var, in_indices, in_updates, in_quant_scales)
        self.assertExpectedInline(logs, """\



def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
    npu_quant_scatter = torch.ops.npu.npu_quant_scatter.default(arg0_1, arg1_1, arg2_1, arg3_1, None, -2, -1);  arg1_1 = arg2_1 = arg3_1 = None
    copy_ = torch.ops.aten.copy_.default(arg0_1, npu_quant_scatter);  arg0_1 = None
    return npu_quant_scatter
    """)

        self.assert_functionalization(f, in_var, in_indices, in_updates, in_quant_scales)

    def test_npu_scatter_nd_update(self):
        def f(var, indices, updates):
            return torch_npu.npu_scatter_nd_update_(var, indices, updates)

        data_var = np.random.uniform(0, 1, [24, 128]).astype(np.float16)
        var = torch.from_numpy(data_var).to(torch.float16).npu()
        data_indices = np.random.uniform(0, 12, [12, 1]).astype(np.int32)
        indices = torch.from_numpy(data_indices).to(torch.int32).npu()
        data_updates = np.random.uniform(1, 2, [12, 128]).astype(np.float16)
        updates = torch.from_numpy(data_updates).to(torch.float16).npu()

        logs = self.get_logs(f, var, indices, updates)
        self.assertExpectedInline(logs, """\



def forward(self, arg0_1, arg1_1, arg2_1):
    npu_scatter_nd_update = torch.ops.npu.npu_scatter_nd_update.default(arg0_1, arg1_1, arg2_1);  arg1_1 = arg2_1 = None
    copy_ = torch.ops.aten.copy_.default(arg0_1, npu_scatter_nd_update);  arg0_1 = None
    return npu_scatter_nd_update
    """)

        self.assert_functionalization(f, var, indices, updates)

    def test_npu_silu_functionalize(self):
        @impl(m, "npu_silu")
        def npu_silu(self_):
            return torch.empty_like(self_)

        @impl(m, "npu_silu_")
        def npu_silu_(self_):
            return self_

        def f(self_):
            return torch.ops.npu.npu_silu_(self_)

        a = torch.randn(1, 2).npu()
        logs = self.get_logs(f, a)
        self.assertExpectedInline(logs, """\



def forward(self, arg0_1):
    npu_silu = torch.ops.npu.npu_silu.default(arg0_1)
    copy_ = torch.ops.aten.copy_.default(arg0_1, npu_silu);  arg0_1 = None
    return npu_silu
    """)
        self.assert_functionalization(f, a)


if __name__ == "__main__":
    run_tests()
