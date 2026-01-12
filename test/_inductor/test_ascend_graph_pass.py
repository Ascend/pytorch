import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu
import torch_npu._inductor


class CastStandardModel(torch.nn.Module):
    def forward(self, arg1_1):
        cast_1 = torch.ops.npu._npu_dtype_cast.default(arg1_1, torch.int64)
        output = torch.ops.aten.relu.default(cast_1)
        return output


class CatSliceCatModel(torch.nn.Module):
    def forward(self, first_element):
        cat1 = torch.cat(
            [
                first_element[:, 0:1, :, :],
                first_element[:, 1:2, :, :],
                first_element[:, 2:3, :, :],
            ],
            dim=1,
        )
        cat2 = torch.cat(
            [
                cat1[:, 0:1, :, :],   # 0~1
                cat1[:, 1:2, :, :],   # 1~2
                cat1[:, 2:3, :, :],   # 2~3
            ],
            dim=1,
        )
        return cat2


class FoldAddModel(torch.nn.Module):
    def forward(self, first_element):
        add = torch.ops.aten.add.Tensor(first_element, 0)
        add_output = torch.ops.aten.relu.default(add)
        return add_output


class FoldCatModel(torch.nn.Module):
    def forward(self, t1, t2, t3, t4, t5):
        cat1 = torch.ops.aten.cat.default([t1, t2], 1)
        cat2 = torch.ops.aten.cat.default([cat1, t3], 1)
        cat3 = torch.ops.aten.cat.default([cat2, t4], 1)
        cat4 = torch.ops.aten.cat.default([cat3, t5], 1)
        return cat4


class FoldCloneModel(torch.nn.Module):
    def forward(self, t1):
        clone_1 = torch.ops.aten.clone.default(t1)
        relu_1 = torch.ops.aten.relu.default(clone_1)
        return relu_1


class FoldDetachModel(torch.nn.Module):
    def forward(self, t1):
        detach_x = torch.ops.aten.detach.default(t1)
        output = torch.ops.aten.relu.default(detach_x)
        return output


class FoldDivModel(torch.nn.Module):
    def forward(self, t1, t2, t3):
        div_1 = torch.ops.aten.div(t1, 1)
        div_output = torch.relu(div_1)
        return div_output


class FoldExpandModel(torch.nn.Module):
    def forward(self, t1):
        add = torch.ops.aten.expand.default(t1, [256, 128, 1])
        add_output = torch.ops.aten.relu.default(add)
        return add_output


class FoldMulModel(torch.nn.Module):
    def forward(self, t1):
        add = torch.ops.aten.mul.Tensor(t1, 1)
        add_output = torch.ops.aten.relu.default(add)
        return add_output


class FoldReduceModel(torch.nn.Module):
    def forward(self, t1):
        sum_1 = torch.ops.aten.sum.dim_IntList(t1, [1, 3])
        return sum_1


class FoldMultiShapeUnchangeModel(torch.nn.Module):
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1):
        embedding = torch.ops.aten.embedding.default(arg0_1, arg1_1)
        view = torch.ops.aten.view.default(embedding, [-1, 1, 64])
        squeeze = torch.ops.aten.squeeze.dim(view, 1)
        embedding_1 = torch.ops.aten.embedding.default(arg2_1, arg3_1)
        view_1 = torch.ops.aten.view.default(embedding_1, [-1, 1, 32])
        squeeze_1 = torch.ops.aten.squeeze.dim(view_1, 1)
        embedding_2 = torch.ops.aten.embedding.default(arg4_1, arg5_1)
        view_2 = torch.ops.aten.view.default(embedding_2, [-1, 1, 16])
        squeeze_2 = torch.ops.aten.squeeze.dim(view_2, 1)
        permute_1 = torch.ops.aten.permute.default(arg6_1, [1, 0])
        permute_2 = torch.ops.aten.permute.default(permute_1, [1, 0])
        relu = torch.ops.aten.relu.default(arg7_1)
        addmm_1 = torch.ops.aten.addmm.default(arg8_1, relu, permute_2)
        relu_1 = torch.ops.aten.relu.default(addmm_1)
        return {"squeeze": squeeze, "squeeze_1": squeeze_1, "squeeze_2": squeeze_2, "permute_2": permute_2, "relu_1": relu_1}


class FoldSinkViewModel(torch.nn.Module):
    def forward(self, t1):
        view_1 = torch.ops.aten.view.default(t1, [1, -1])
        output = torch.ops.aten.relu.default(view_1)
        return output


class FoldSliceModel(torch.nn.Module):
    def forward(self, base, view, t1, t2, t3):
        end = 16
        slice_1 = torch.ops.aten.slice_scatter.default(base, view, 1, 0, end)
        result = view + slice_1
        slice_2 = torch.ops.aten.slice_scatter.default(t1, t2, 1, 0, 3)
        b = torch.ops.aten.slice.Tensor(t3, 1, 0, None)
        result_c = torch.ops.aten.add.Tensor(b, b)
        return {"result": result, "slice_2": slice_2, "result_c": result_c}


class FoldSqueezeModel(torch.nn.Module):
    def forward(self, t1, t2,):
        squeeze_1 = torch.ops.aten.squeeze.default(t1)
        squeeze_2 = torch.ops.aten.squeeze.default(squeeze_1)
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(t2, 1)
        squeeze_3 = torch.ops.aten.squeeze.dim(unsqueeze_1, 1)
        return {"squeeze_2": squeeze_2, "squeeze_3": squeeze_3}


class FoldSubModel(torch.nn.Module):
    def forward(self, t1):
        sub = torch.ops.aten.sub.Tensor(t1, torch.ops.aten.zeros_like.default(t1))
        sub_output = torch.ops.aten.relu.default(sub)
        rsub = torch.ops.aten.rsub.Tensor(torch.ops.aten.zeros_like.default(t1), t1)
        rsub_output = torch.ops.aten.relu.default(rsub)
        return sub_output + rsub_output


class FoldToCopyModel(torch.nn.Module):
    def forward(self, t1):
        copy_1 = torch.ops.aten._to_copy.default(t1)
        result = torch.ops.aten.add.Tensor(t1, copy_1)
        return result


class FoldViewModel(torch.nn.Module):
    def forward(self, t1, t2):
        squeeze_1 = torch.ops.aten.squeeze.dim(t1, 2)
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(squeeze_1, 0)
        view_1 = torch.ops.aten.view.default(unsqueeze_1, [1, -1])
        output = torch.ops.aten.view.default(t2, [128, 64])
        return {"view_1": view_1, "output": output}


class FoldWhereModel(torch.nn.Module):
    def forward(self, t1):
        mask = t1 > 0
        return torch.ops.aten.where.self(mask, t1, t1)


class FoldPadSliceModel(torch.nn.Module):
    def forward(self, t1):
        inputPad = torch._C._nn.pad(t1, [0, 0, 0, 50], "constant", 0.0)
        inputSlice = inputPad[:, :50]
        output = torch.relu(inputSlice)
        return output


class TestAscendGraphPass(TestUtils):
    def cast_standard_op_calc(self, first_element):
        cast_1 = torch.ops.npu._npu_dtype_cast.default(first_element, torch.int64)
        output = torch.ops.aten.relu.default(cast_1)
        return output
    
    
    @parametrize('shape', [(256, 5)])
    @parametrize('dtype', ['int64'])
    def test_cast_standard_compile_cases(self, shape, dtype):
        first_element = self._generate_tensor(shape, dtype)
        std_result = self.cast_standard_op_calc(first_element)
        compiled_op_calc = torch.compile(self.cast_standard_op_calc, backend="inductor")
        inductor_result = compiled_op_calc(first_element)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)
        

    @parametrize('shape', [(256, 5)])
    @parametrize('dtype', ['int64'])
    def test_cast_standard_ut_cases(self, shape, dtype):
        first_element = self._generate_tensor(shape, dtype)
        model = CastStandardModel()
        graph_module = fx.symbolic_trace(model)
        ShapeProp(graph_module).propagate(first_element)
        from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import fold_cast
        fold_cast(graph_module.graph)
        graph_module.recompile()
        std_result = model(first_element)
        inductor_result = graph_module(first_element)

        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


    def cat_slice_cat_op_calc(self, first_element):
        # 第一次 cat：在 dim=1 上拼接 3 个部分 → [1, 3, 64, 1024]
        cat1 = torch.cat(
            [
                first_element[:, 0:1, :, :],   # 第0个channel
                first_element[:, 1:2, :, :],   # 第1个channel
                first_element[:, 2:3, :, :],   # 第2个channel
            ],
            dim=1,
        )
        # 第二次 cat：在同一个 dim=1 上，对 cat1 的连续切片再做一次 cat
        # → 完全等价于 cat1，Inductor 应该直接 erase 掉第二次 cat 和所有 getitem
        cat2 = torch.cat(
            [
                cat1[:, 0:1, :, :],   # 0~1
                cat1[:, 1:2, :, :],   # 1~2
                cat1[:, 2:3, :, :],   # 2~3
            ],
            dim=1,
        )
        return cat2


    @parametrize('shape', [(1, 3, 64, 1024)])
    @parametrize('dtype', ['float32'])
    def test_cat_slice_cat_compile_cases(self, shape, dtype):
        first_element = self._generate_tensor(shape, dtype)

        std_result = self.cat_slice_cat_op_calc(first_element)

        compiled_op_calc = torch.compile(self.cat_slice_cat_op_calc, backend="inductor")
        inductor_result = compiled_op_calc(first_element)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)
        
    
    @parametrize('shape', [(1, 3, 64, 1024)])
    @parametrize('dtype', ['float32'])
    def test_cat_slice_cat_ut_cases(self, shape, dtype):
        first_element = self._generate_tensor(shape, dtype)
        model = CatSliceCatModel()
        graph_module = fx.symbolic_trace(model)
        ShapeProp(graph_module).propagate(first_element)
        from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import cat_slice_cat_fold_pass
        cat_slice_cat_fold_pass(graph_module.graph)
        graph_module.recompile()
        std_result = model(first_element)
        inductor_result = graph_module(first_element)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


    def fold_add_op_calc(self, first_element):
        add = torch.add(first_element, 0)
        add_output = torch.relu(add)
        return add_output


    @parametrize('shape', [(1, 2, 3)])
    @parametrize('dtype', ['float32'])
    def test_fold_add_compile_cases(self, shape, dtype):
        first_element = self._generate_tensor(shape, dtype)

        std_result = self.fold_add_op_calc(first_element)

        compiled_op_calc = torch.compile(self.fold_add_op_calc, backend="inductor")
        inductor_result = compiled_op_calc(first_element)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)
        
    
    @parametrize('shape', [(1, 2, 3)])
    @parametrize('dtype', ['float32'])
    def test_fold_add_ut_cases(self, shape, dtype):
        first_element = self._generate_tensor(shape, dtype)
        model = FoldAddModel()
        graph_module = fx.symbolic_trace(model)
        ShapeProp(graph_module).propagate(first_element)
        from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import fold_four_op_pass
        fold_four_op_pass(graph_module.graph)
        graph_module.recompile()
        std_result = model(first_element)
        inductor_result = graph_module(first_element)

        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


    def fold_cat_op_calc(self, t1, t2, t3, t4, t5):
        cat1 = torch.cat([t1, t2], dim=1)
        cat2 = torch.cat([cat1, t3], dim=1)
        cat3 = torch.cat([cat2, t4], dim=1)
        cat4 = torch.cat([cat3, t5], dim=1)
        return cat4


    @parametrize('shape', [(2, 4)])
    @parametrize('dtype', ['float32'])
    def test_fold_cat_compile_cases(self, shape, dtype):
        t1 = self._generate_tensor(shape, dtype)
        t2 = self._generate_tensor(shape, dtype)
        t3 = self._generate_tensor(shape, dtype)
        t4 = self._generate_tensor(shape, dtype)
        t5 = self._generate_tensor(shape, dtype)

        std_result = self.fold_cat_op_calc(t1, t2, t3, t4, t5)

        compiled_op_calc = torch.compile(self.fold_cat_op_calc, backend="inductor")
        inductor_result = compiled_op_calc(t1, t2, t3, t4, t5)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)
        
    
    @parametrize('shape', [(2, 4)])
    @parametrize('dtype', ['float32'])
    def test_fold_cat_ut_cases(self, shape, dtype):
        t1 = self._generate_tensor(shape, dtype)
        t2 = self._generate_tensor(shape, dtype)
        t3 = self._generate_tensor(shape, dtype)
        t4 = self._generate_tensor(shape, dtype)
        t5 = self._generate_tensor(shape, dtype)
        model = FoldCatModel()
        graph_module = fx.symbolic_trace(model)
        ShapeProp(graph_module).propagate(t1, t2, t3, t4, t5)
        from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import fold_cat
        fold_cat(graph_module.graph)
        graph_module.recompile()
        std_result = model(t1, t2, t3, t4, t5)
        inductor_result = graph_module(t1, t2, t3, t4, t5)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


    def fold_clone_op_calc(self, t1):
        clone_1 = torch.clone(t1)
        output = torch.relu(clone_1)
        return output


    @parametrize('shape', [(1, 2, 3)])
    @parametrize('dtype', ['float32'])
    def test_fold_clone_compile_cases(self, shape, dtype):
        t1 = self._generate_tensor(shape, dtype)
        
        std_result = self.fold_clone_op_calc(t1)

        compiled_op_calc = torch.compile(self.fold_clone_op_calc, backend="inductor")
        inductor_result = compiled_op_calc(t1)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)
        
    
    @parametrize('shape', [(1, 2, 3)])
    @parametrize('dtype', ['float32'])
    def test_fold_clone_ut_cases(self, shape, dtype):
        t1 = self._generate_tensor(shape, dtype)
        model = FoldCloneModel()
        graph_module = fx.symbolic_trace(model)
        ShapeProp(graph_module).propagate(t1)
        from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import fold_clone
        fold_clone(graph_module.graph)
        graph_module.recompile()
        std_result = model(t1)
        inductor_result = graph_module(t1)

        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


    def fold_detach_op_calc(self, t1):
        detach_x = torch.ops.aten.detach.default(t1)
        output = torch.ops.aten.relu.default(detach_x)
        return output


    @parametrize('shape', [(3, 3)])
    @parametrize('dtype', ['float32'])
    def test_fold_detach_compile_cases(self, shape, dtype):
        t1 = self._generate_tensor(shape, dtype)
        std_result = self.fold_detach_op_calc(t1)
        compiled_op_calc = torch.compile(self.fold_detach_op_calc, backend="inductor")
        inductor_result = compiled_op_calc(t1)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)
        
    
    @parametrize('shape', [(3, 3)])
    @parametrize('dtype', ['float32'])
    def test_fold_detach_ut_cases(self, shape, dtype):
        t1 = self._generate_tensor(shape, dtype)
        model = FoldDetachModel()
        graph_module = fx.symbolic_trace(model)
        ShapeProp(graph_module).propagate(t1)
        from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import fold_detach
        fold_detach(graph_module.graph)
        graph_module.recompile()
        std_result = model(t1)
        inductor_result = graph_module(t1)

        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


    def fold_div_op_calc(self, t1):
        div_1 = torch.ops.aten.div(t1, 1)
        div_output = torch.relu(div_1)
        return div_output


    @parametrize('shape', [(1, 2, 3)])
    @parametrize('dtype', ['float32'])
    def test_fold_div_compile_cases(self, shape, dtype):
        t1 = self._generate_tensor(shape, dtype)
        std_result = self.fold_div_op_calc(t1)
        compiled_op_calc = torch.compile(self.fold_div_op_calc, backend="inductor")
        inductor_result = compiled_op_calc(t1)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)
        
    
    @parametrize('shape', [(2, 4, 8)])
    @parametrize('dtype', ['float32'])
    def test_fold_div_ut_cases(self, shape, dtype):
        t1 = self._generate_tensor(shape, dtype)
        t2 = self._generate_tensor(shape, dtype)
        t3 = self._generate_tensor(shape, dtype)
        model = FoldDivModel()
        graph_module = fx.symbolic_trace(model)
        ShapeProp(graph_module).propagate(t1, t2, t3)
        from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import fold_four_op_pass
        fold_four_op_pass(graph_module.graph)
        graph_module.recompile()
        std_result = model(t1, t2, t3)
        inductor_result = graph_module(t1, t2, t3)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


    def fold_expand_op_calc(self, t1):
        expand = torch.ops.aten.expand.default(t1, [256, 128, 1])
        relu = torch.relu(expand)
        return relu


    @parametrize('shape', [(256, 128, 1)])
    @parametrize('dtype', ['float32'])
    def test_fold_expand_compile_cases(self, shape, dtype):
        t1 = self._generate_tensor(shape, dtype)
        
        std_result = self.fold_expand_op_calc(t1)

        compiled_op_calc = torch.compile(self.fold_expand_op_calc, backend="inductor")
        inductor_result = compiled_op_calc(t1)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)
        
    
    @parametrize('shape', [(256, 128, 1)])
    @parametrize('dtype', ['float32'])
    def test_fold_expand_ut_cases(self, shape, dtype):
        t1 = self._generate_tensor(shape, dtype)
        model = FoldExpandModel()
        graph_module = fx.symbolic_trace(model)
        ShapeProp(graph_module).propagate(t1)
        from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import fold_expand
        fold_expand(graph_module.graph)
        graph_module.recompile()
        std_result = model(t1)
        inductor_result = graph_module(t1)

        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


    def fold_mul_op_calc(self, t1):
        mul_1 = torch.ops.aten.mul.Tensor(t1, 1)
        mul_output = torch.relu(mul_1)
        return mul_output


    @parametrize('shape', [(1, 2, 3)])
    @parametrize('dtype', ['float32'])
    def test_fold_mul_compile_cases(self, shape, dtype):
        t1 = self._generate_tensor(shape, dtype)
        std_result = self.fold_mul_op_calc(t1)
        compiled_op_calc = torch.compile(self.fold_mul_op_calc, backend="inductor")
        inductor_result = compiled_op_calc(t1)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)
        
    
    @parametrize('shape', [(1, 2, 3)])
    @parametrize('dtype', ['float32'])
    def test_fold_mul_ut_cases(self, shape, dtype):
        t1 = self._generate_tensor(shape, dtype)
        model = FoldMulModel()
        graph_module = fx.symbolic_trace(model)
        ShapeProp(graph_module).propagate(t1)
        from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import fold_four_op_pass
        fold_four_op_pass(graph_module.graph)
        graph_module.recompile()
        std_result = model(t1)
        inductor_result = graph_module(t1)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


    def fold_reduce_op_calc(self, t1):
        output = torch.sum(t1, dim=(1, 3), keepdim=False)
        return output


    @parametrize('shape', [(128, 1, 64, 1)])
    @parametrize('dtype', ['float32'])
    def test_fold_reduce_compile_cases(self, shape, dtype):
        t1 = self._generate_tensor(shape, dtype)
        std_result = self.fold_reduce_op_calc(t1)
        compiled_op_calc = torch.compile(self.fold_reduce_op_calc, backend="inductor")
        inductor_result = compiled_op_calc(t1)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)
        
    
    @parametrize('shape', [(128, 1, 64, 1)])
    @parametrize('dtype', ['float32'])
    def test_fold_reduce_ut_cases(self, shape, dtype):
        t1 = self._generate_tensor(shape, dtype)
        model = FoldReduceModel()
        graph_module = fx.symbolic_trace(model)
        ShapeProp(graph_module).propagate(t1)
        from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import fold_reduce
        fold_reduce(graph_module.graph)
        graph_module.recompile()
        std_result = model(t1)
        inductor_result = graph_module(t1)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


    def fold_redundant_op_calc(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1):
        embedding = torch.ops.aten.embedding.default(arg0_1, arg1_1)
        view = torch.ops.aten.view.default(embedding, [-1, 1, 64])
        squeeze = torch.ops.aten.squeeze.dim(view, 1)
        embedding_1 = torch.ops.aten.embedding.default(arg2_1, arg3_1)
        view_1 = torch.ops.aten.view.default(embedding_1, [-1, 1, 32])
        squeeze_1 = torch.ops.aten.squeeze.dim(view_1, 1)
        embedding_2 = torch.ops.aten.embedding.default(arg4_1, arg5_1)
        view_2 = torch.ops.aten.view.default(embedding_2, [-1, 1, 16])
        squeeze_2 = torch.ops.aten.squeeze.dim(view_2, 1)
        permute_1 = torch.ops.aten.permute.default(arg6_1, [1, 0])
        permute_2 = torch.ops.aten.permute.default(permute_1, [1, 0])
        relu = torch.ops.aten.relu.default(arg7_1)
        addmm_1 = torch.ops.aten.addmm.default(arg8_1, relu, permute_2)
        relu_1 = torch.ops.aten.relu.default(addmm_1)
        return {"squeeze": squeeze, "squeeze_1": squeeze_1, "squeeze_2": squeeze_2, "permute_2": permute_2, "relu_1": relu_1}


    def test_fold_redundant_compile_cases(self):
        arg0_1 = torch.randn(289094, 64, dtype=torch.float32)
        arg1_1 = torch.randint(0, 289094, (128,), dtype=torch.int64)
        arg2_1 = torch.randn(98, 32, dtype=torch.float32)
        arg3_1 = torch.randint(0, 98, (128,), dtype=torch.int64)
        arg4_1 = torch.randn(14, 16, dtype=torch.float32)
        arg5_1 = torch.randint(0, 14, (128,), dtype=torch.int64)
        arg6_1 = torch.randn(6144, 6144, dtype=torch.float32)
        arg7_1 = torch.randn(128, 6144, dtype=torch.float32)
        arg8_1 = torch.randn(6144, dtype=torch.float32)
        std_result = self.fold_redundant_op_calc(arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1)

        compiled_op_calc = torch.compile(self.fold_redundant_op_calc, backend="inductor")
        inductor_result = compiled_op_calc(arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)
        
        
    def test_fold_redundant_ut_cases(self):
        arg0_1 = torch.randn(289094, 64, dtype=torch.float32)
        arg1_1 = torch.randint(0, 289094, (128,), dtype=torch.int64)
        arg2_1 = torch.randn(98, 32, dtype=torch.float32)
        arg3_1 = torch.randint(0, 98, (128,), dtype=torch.int64)
        arg4_1 = torch.randn(14, 16, dtype=torch.float32)
        arg5_1 = torch.randint(0, 14, (128,), dtype=torch.int64)
        arg6_1 = torch.randn(6144, 6144, dtype=torch.float32)
        arg7_1 = torch.randn(128, 6144, dtype=torch.float32)
        arg8_1 = torch.randn(6144, dtype=torch.float32)
        model = FoldMultiShapeUnchangeModel()
        graph_module = fx.symbolic_trace(model)
        ShapeProp(graph_module).propagate(arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1)
        from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import fold_redundant_ops
        fold_redundant_ops(graph_module.graph)
        graph_module.recompile()
        std_result = model(arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1)
        inductor_result = graph_module(arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


    def fold_sink_view_op_calc(self, t1):
        x = t1.view(1, -1) # reshape
        return torch.relu(x)


    @parametrize('shape', [(2, 3, 4)])
    @parametrize('dtype', ['float32'])
    def test_fold_sink_viewcompile_cases(self, shape, dtype):
        t1 = self._generate_tensor(shape, dtype)
        std_result = self.fold_sink_view_op_calc(t1)
        compiled_op_calc = torch.compile(self.fold_sink_view_op_calc, backend="inductor")
        inductor_result = compiled_op_calc(t1)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)
        
    
    @parametrize('shape', [(2, 3, 4)])
    @parametrize('dtype', ['float32'])
    def test_fold_sink_viewut_cases(self, shape, dtype):
        t1 = self._generate_tensor(shape, dtype)
        model = FoldSinkViewModel()
        graph_module = fx.symbolic_trace(model)
        ShapeProp(graph_module).propagate(t1)
        from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import fold_sink_view
        fold_sink_view(graph_module.graph)
        graph_module.recompile()
        std_result = model(t1)
        inductor_result = graph_module(t1)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


    def fold_slice_op_calc(self, base, view, t1, t2, t3):
        end = view.shape[1]
        result = torch.slice_scatter(base, view, 1, 0, end)
        result = result + view
        data = t1.slice_scatter(t2, dim=1, start=0, end=t2.shape[1])
        b = t3[:, 0:]
        result_c = b + b
        return {"result": result, "data": data, "result_c": result_c}


    def test_fold_slice_compile_cases(self):
        base = torch.randn(8, 16, 32)
        view = torch.ones(8, 16, 32)
        t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
        t2 = torch.tensor([[9, 9, 9], [8, 8, 8]])
        t3 = torch.randn(4, 16, 32, 64)
        std_result = self.fold_slice_op_calc(base, view, t1, t2, t3)
        compiled_op_calc = torch.compile(self.fold_slice_op_calc, backend="inductor")
        inductor_result = compiled_op_calc(base, view, t1, t2, t3)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)
        
        
    def test_fold_slice_ut_cases(self):
        base = torch.randn(8, 16, 32)
        view = torch.ones(8, 16, 32)
        t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
        t2 = torch.tensor([[9, 9, 9], [8, 8, 8]])
        t3 = torch.randn(4, 16, 32, 64)
        model = FoldSliceModel()
        graph_module = fx.symbolic_trace(model)
        ShapeProp(graph_module).propagate(base, view, t1, t2, t3)
        from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import fold_slice
        fold_slice(graph_module.graph)
        graph_module.recompile()
        std_result = model(base, view, t1, t2, t3)
        inductor_result = graph_module(base, view, t1, t2, t3)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


    def fold_squeeze_op_calc(self, t1, t2):
        squeeze_1 = torch.squeeze(t1)
        squeeze_2 = torch.squeeze(squeeze_1)
        unsqueeze_1 = torch.unsqueeze(t2, 1)
        squeeze_3 = torch.squeeze(unsqueeze_1, 1)
        return {"squeeze_2": squeeze_2, "squeeze_3": squeeze_3}


    def test_fold_squeeze_compile_cases(self):
        t1 = torch.randn(2, 4)
        t2 = torch.randn(2, 1, 1, 4)
        std_result = self.fold_squeeze_op_calc(t1, t2)
        compiled_op_calc = torch.compile(self.fold_squeeze_op_calc, backend="inductor")
        inductor_result = compiled_op_calc(t1, t2)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)
        
        
    def test_fold_squeeze_ut_cases(self):
        t1 = torch.randn(2, 4)
        t2 = torch.randn(2, 1, 1, 4)
        model = FoldSqueezeModel()
        graph_module = fx.symbolic_trace(model)
        ShapeProp(graph_module).propagate(t1, t2)
        from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import fold_squeeze
        fold_squeeze(graph_module.graph)
        graph_module.recompile()
        std_result = model(t1, t2)
        inductor_result = graph_module(t1, t2)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


    def fold_sub_op_calc(self, t1):
        sub = torch.sub(t1, torch.ops.aten.zeros_like.default(t1))
        sub_output = torch.relu(sub)
        rsub = torch.rsub(torch.ops.aten.zeros_like.default(t1), t1)
        rsub_output = torch.relu(rsub)
        return sub_output + rsub_output


    @parametrize('shape', [(1, 2, 3)])
    @parametrize('dtype', ['float32'])
    def test_fold_sub_compile_cases(self, shape, dtype):
        t1 = self._generate_tensor(shape, dtype)
        std_result = self.fold_sub_op_calc(t1)
        compiled_op_calc = torch.compile(self.fold_sub_op_calc, backend="inductor")
        inductor_result = compiled_op_calc(t1)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)
        
    
    @parametrize('shape', [(1, 2, 3)])
    @parametrize('dtype', ['float32'])
    def test_fold_sub_ut_cases(self, shape, dtype):
        t1 = self._generate_tensor(shape, dtype)
        model = FoldSubModel()
        graph_module = fx.symbolic_trace(model)
        ShapeProp(graph_module).propagate(t1)
        from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import fold_four_op_pass
        fold_four_op_pass(graph_module.graph)
        graph_module.recompile()
        std_result = model(t1)
        inductor_result = graph_module(t1)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


    def fold_to_copy_op_calc(self, t1):
        copy_1 = torch.ops.aten._to_copy.default(t1)
        result = copy_1 + t1
        return result


    @parametrize('shape', [(2, 4)])
    @parametrize('dtype', ['float32'])
    def test_fold_to_copy_compile_cases(self, shape, dtype):
        t1 = self._generate_tensor(shape, dtype)
        std_result = self.fold_to_copy_op_calc(t1)
        compiled_op_calc = torch.compile(self.fold_to_copy_op_calc, backend="inductor")
        inductor_result = compiled_op_calc(t1)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)
        
    
    @parametrize('shape', [(2, 4)])
    @parametrize('dtype', ['float32'])
    def test_fold_to_copy_ut_cases(self, shape, dtype):
        t1 = self._generate_tensor(shape, dtype)
        model = FoldToCopyModel()
        graph_module = fx.symbolic_trace(model)
        ShapeProp(graph_module).propagate(t1)
        from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import fold_to_copy
        fold_to_copy(graph_module.graph)
        graph_module.recompile()
        std_result = model(t1)
        inductor_result = graph_module(t1)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


    def fold_view_op_calc(self, t1, t2):
        squeeze_1 = t1.squeeze(2)
        unsqueeze_1 = squeeze_1.unsqueeze(0)
        view_1 = unsqueeze_1.view(1, -1)
        output = t2.view(128, 64)
        return {"view_1": view_1, "output": output}


    def test_fold_view_compile_cases(self):
        t1 = torch.randn(1, 3, 1, 5)
        t2 = torch.randn(128, 64)
        std_result = self.fold_view_op_calc(t1, t2)
        compiled_op_calc = torch.compile(self.fold_view_op_calc, backend="inductor")
        inductor_result = compiled_op_calc(t1, t2)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)
        
        
    def test_fold_view_ut_cases(self):
        t1 = torch.randn(1, 3, 1, 5)
        t2 = torch.randn(128, 64)
        model = FoldViewModel()
        graph_module = fx.symbolic_trace(model)
        ShapeProp(graph_module).propagate(t1, t2)
        from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import view_fold_pass
        view_fold_pass(graph_module.graph)
        graph_module.recompile()
        std_result = model(t1, t2)
        inductor_result = graph_module(t1, t2)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


    def fold_where_op_calc(self, t1):
        mask = t1 > 0
        return torch.where(mask, t1, t1) # 两个分支完全相同


    @parametrize('shape', [(3, 4)])
    @parametrize('dtype', ['float32'])
    def test_fold_where_compile_cases(self, shape, dtype):
        t1 = self._generate_tensor(shape, dtype)
        std_result = self.fold_where_op_calc(t1)
        compiled_op_calc = torch.compile(self.fold_where_op_calc, backend="inductor")
        inductor_result = compiled_op_calc(t1)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)
        
    
    @parametrize('shape', [(3, 4)])
    @parametrize('dtype', ['float32'])
    def test_fold_where_ut_cases(self, shape, dtype):
        t1 = self._generate_tensor(shape, dtype)
        model = FoldWhereModel()
        graph_module = fx.symbolic_trace(model)
        ShapeProp(graph_module).propagate(t1)
        from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import fold_where
        fold_where(graph_module.graph)
        graph_module.recompile()
        std_result = model(t1)
        inductor_result = graph_module(t1)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


    def pad_slice_op_calc(self, t1):
        inputPad = torch._C._nn.pad(t1, [0, 0, 0, 50], "constant", 0.0)
        inputSlice = inputPad[:, :50]
        output = torch.relu(inputSlice)
        return output


    @parametrize('shape', [(128, 50, 128)])
    @parametrize('dtype', ['float32'])
    def test_pad_slice_compile_cases(self, shape, dtype):
        t1 = self._generate_tensor(shape, dtype)
        std_result = self.pad_slice_op_calc(t1)
        compiled_op_calc = torch.compile(self.pad_slice_op_calc, backend="inductor")
        inductor_result = compiled_op_calc(t1)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)
        
    
    @parametrize('shape', [(128, 50, 128)])
    @parametrize('dtype', ['float32'])
    def test_pad_slice_ut_cases(self, shape, dtype):
        t1 = self._generate_tensor(shape, dtype)
        model = FoldPadSliceModel()
        graph_module = fx.symbolic_trace(model)
        ShapeProp(graph_module).propagate(t1)
        from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import pad_slice_fold
        pad_slice_fold(graph_module.graph)
        graph_module.recompile()
        std_result = model(t1)
        inductor_result = graph_module(t1)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


instantiate_parametrized_tests(TestAscendGraphPass)


if __name__ == "__main__":
    run_tests()