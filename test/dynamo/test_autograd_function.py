# Owner(s): ["module: dynamo"]

import copy
import math

import torch
import torch_npu

import torch._dynamo.test_case
import torch._dynamo.testing
import torch._dynamo.utils


class CustomFunc1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, foo):
        return foo + foo

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class CustomFunc3(torch.autograd.Function):
    # Test there is graph break in forward function
    @staticmethod
    def forward(ctx, foo):
        result = foo + foo
        torch._dynamo.graph_break()
        result = result + foo
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        (result,) = ctx.saved_tensors
        return grad_output * math.sqrt(result.numel())


class Module1(torch.nn.Module):
    def forward(self, foo):
        return CustomFunc1().apply(foo)


class Module2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fn = CustomFunc1.apply

    def forward(self, foo):
        return self.fn(foo)


class Module3(torch.nn.Module):
    def forward(self, foo):
        return CustomFunc1().apply(foo)


class Module4(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fn = CustomFunc1.apply

    def forward(self, foo):
        return self.fn(foo)


class Module5(torch.nn.Module):
    def forward(self, foo):
        return CustomFunc3().apply(foo)


class Module6(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fn = CustomFunc3.apply

    def forward(self, foo):
        return self.fn(foo)


class LinearFunction(torch.autograd.Function):
    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(ipt, weight, bias):
        output = ipt.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    # inputs is a Tuple of all of the inputs passed to forward.
    # output is the output of the forward().
    def setup_context(ctx, inputs, output):
        ipt, weight, bias = inputs
        ctx.save_for_backward(ipt, weight, bias)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        ipt, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(ipt)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias


class ModuleLinear(torch.nn.Module):
    def forward(self, ipt, weight, bias=None):
        return LinearFunction.apply(ipt, weight, bias)


class MaterializingGradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.set_materialize_grads(False)
        return x.clone(), x.clone()

    @staticmethod
    def backward(ctx, grad_out1, grad_out2):
        return grad_out1, grad_out2


class MaterializingGradModule(torch.nn.Module):
    def forward(self, x):
        return MaterializingGradFunction.apply(x)


class CustomFuncBwdPrintGraphBreak(torch.autograd.Function):
    @staticmethod
    def forward(ctx, foo):
        return torch.add(foo, foo)

    @staticmethod
    def backward(ctx, grad_output):
        print("graph break!")
        return grad_output


class CustomFuncBwdPrintModule(torch.nn.Module):
    def forward(self, x):
        return CustomFuncBwdPrintGraphBreak.apply(x)


class CustomFuncStrideBwd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, foo):
        return torch.add(foo, foo)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.stride()


class CustomFuncStrideModule(torch.nn.Module):
    def forward(self, x):
        return CustomFuncStrideBwd.apply(x)


class CustomFuncSaveForBwd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, foo):
        result = foo + foo
        result = result + foo
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        (result,) = ctx.saved_tensors
        return grad_output * math.sqrt(result.numel())


class SaveForBwdModule(torch.nn.Module):
    def forward(self, foo):
        return CustomFuncSaveForBwd().apply(foo)


class ContextSaveAndMark(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        with torch.no_grad():
            ctx.save_for_backward(x)
            ctx.mark_non_differentiable(x)
            return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class ContextMarkAndSave(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        with torch.no_grad():
            ctx.mark_non_differentiable(x)
            ctx.save_for_backward(x)
            return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class ModuleWithGradFunc(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.f = func.apply

    def forward(self, x):
        return self.f(x)


class AutogradFunctionTests(torch._dynamo.test_case.TestCase):
    # Sound behaviors, tested for working capture
    def test_autograd_function_equivalence(self):
        for grad in [True, False]:
            for i in range(1, 5):
                torch._dynamo.reset()
                model = globals()[f"Module{i}"]()
                opt_model = torch._dynamo.optimize("eager")(model)
                self.assertTrue(
                    torch.allclose(
                        opt_model(torch.ones(2, 3, requires_grad=grad)),
                        torch.tensor([2.0], requires_grad=grad),
                    )
                )

    def test_autograd_function_has_graph_break(self):
        for grad in [True, False]:
            x = torch.randn(10, requires_grad=grad)
            for model in [Module5(), Module6()]:
                torch._dynamo.reset()
                cnts = torch._dynamo.testing.CompileCounter()
                opt_model = torch._dynamo.optimize(cnts)(model)
                for _ in range(3):
                    ref = model(x)
                    res = opt_model(x)
                    self.assertTrue(torch.allclose(ref, res))
                self.assertEqual(cnts.frame_count, 2)

    def test_linear_setup_context(self):
        model = ModuleLinear()
        opt_model = torch._dynamo.optimize("eager")(model)
        ipt = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        weight = torch.randn(3, 2, dtype=torch.double, requires_grad=True)
        optim_result = opt_model(ipt, weight)
        eager_result = model(ipt, weight)
        self.assertEqual(optim_result, eager_result)

    def test_materialize_grad(self):
        model = MaterializingGradModule()
        opt_model = torch._dynamo.optimize("eager")(model)
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        optim_result = opt_model(x)
        eager_result = model(x)
        self.assertEqual(optim_result, eager_result)

    def test_print_in_bwd(self):
        model = CustomFuncBwdPrintModule()
        opt_model = torch._dynamo.optimize("eager", nopython=True)(model)
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, ".*BuiltinVariable\\(print\\).*"
        ):
            opt_model(x)

    def test_stride_in_bwd(self):
        model = CustomFuncStrideModule()
        opt_model = torch._dynamo.optimize("eager", nopython=True)(model)
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "Illegal getattr invocation stride in strict mod",
        ):
            opt_model(x)

    def test_save_for_bwd(self):
        model = SaveForBwdModule()
        opt_model = torch._dynamo.optimize("eager", nopython=True)(model)
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        opt_model(x)

    def test_allow_in_graph(self):
        torch._dynamo.utils.counters.clear()
        cnt = torch._dynamo.testing.CompileCounter()

        @torch._dynamo.allow_in_graph
        class AllowInGraphFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                torch._dynamo.graph_break()
                ctx.x0 = x.size(0)
                return x * 2

            @staticmethod
            def backward(ctx, grad_out):
                return grad_out * ctx.x0

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x):
            return AllowInGraphFunc.apply(x)

        x = torch.rand(2, 3, requires_grad=True)
        result = fn(x)

        self.assertEqual(result, AllowInGraphFunc.apply(x))
        self.assertEqual(cnt.frame_count, 1)

    def test_once_differentiable(self):
        from torch.autograd.function import once_differentiable

        torch._dynamo.utils.counters.clear()
        cnt = torch._dynamo.testing.CompileCounter()

        class ScaleGradient(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            @once_differentiable
            def backward(ctx, grad):
                return grad * 0.5

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x):
            return ScaleGradient.apply(x)

        x = torch.randn(3, requires_grad=True)
        result = fn(x)

        self.assertEqual(result, ScaleGradient.apply(x))
        self.assertEqual(cnt.frame_count, 1)

    def test_classmethod(self):
        class Shake(torch.autograd.Function):
            @classmethod
            def forward(cls, ctx, foo):
                return foo + foo

            @classmethod
            def backward(cls, ctx, grad_output):
                return grad_output

        def f(x):
            return Shake.apply(x)

        x = torch.randn(4, 4, 4, 4, requires_grad=True)
        opt_m = torch.compile(backend="eager")(f)
        opt_m(x)

    def test_function_context_save_and_mark(self):
        mod = ModuleWithGradFunc(ContextSaveAndMark)
        args, kwargs = ([torch.rand([1])], {})
        before = mod(*args, **kwargs)

        torch._dynamo.reset()
        compiled_model = torch._dynamo.optimize("eager")(mod)
        after = compiled_model(*args, **kwargs)
        self.assertEqual(before, after)

    def test_function_context_mark_and_save(self):
        mod = ModuleWithGradFunc(ContextMarkAndSave)
        args, kwargs = ([torch.rand([1])], {})
        before = mod(*args, **kwargs)

        torch._dynamo.reset()
        compiled_model = torch._dynamo.optimize("eager")(mod)
        after = compiled_model(*args, **kwargs)
        self.assertEqual(before, after)

    def test_multi_output(self):
        torch._dynamo.utils.counters.clear()
        cnt = torch._dynamo.testing.CompileCounter()

        class Foo(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x.clone(), x.clone()

            @staticmethod
            def backward(ctx, grad1, grad2):
                return grad1 + grad2

        @torch.compile(backend=cnt, fullgraph=True)
        def f(x):
            return Foo.apply(x)

        x = torch.randn(3, requires_grad=True)
        result = f(x)

        self.assertEqual(result, Foo.apply(x))
        self.assertEqual(cnt.frame_count, 1)

    def test_amp_custom_fwd_bwd(self):
        torch._dynamo.utils.counters.clear()
        cnt = torch._dynamo.testing.CompileCounter()

        class MyMM(torch.autograd.Function):
            @staticmethod
            @torch.npu.amp.custom_fwd
            def forward(ctx, a, b):
                ctx.save_for_backward(a, b)
                return a.mm(b)

            @staticmethod
            @torch.npu.amp.custom_bwd
            def backward(ctx, grad):
                a, b = ctx.saved_tensors
                return grad.mm(b.t()), a.t().mm(grad)

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(a, b):
            return MyMM.apply(a, b)

        a = torch.randn([64, 64], dtype=torch.float32, requires_grad=True)
        grad = a.clone()
        res = fn(a, a)
        res.backward(grad)

        self.assertEqual(res, MyMM.apply(a, a))
        self.assertEqual(cnt.frame_count, 1)

    def test_graph_break_if_lifted_free_variable(self):
        torch._dynamo.utils.counters.clear()
        cnt = torch._dynamo.testing.CompileCounter()
        delta = torch.randn(3)

        class Foo(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x.clone(), (x + delta).clone()

            @staticmethod
            def backward(ctx, grad1, grad2):
                return grad1 + grad2

        @torch.compile(backend=cnt)
        def f(x):
            return Foo.apply(x)

        x = torch.randn(3, requires_grad=True)
        result = f(x)

        self.assertEqual(result, Foo.apply(x))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(
            list(torch._dynamo.utils.counters["graph_break"].values()), [1]
        )

    def test_function_with_bound_free_variable(self):
        class LowerBound(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inputs, bound):
                ctx.save_for_backward(inputs, inputs.new_ones(1) * bound)
                return inputs.clamp(min=bound)

            @staticmethod
            def backward(ctx, grad_output):
                inputs, bound = ctx.saved_tensors
                return (inputs >= bound) * grad_output, None

        class MyMod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gamma = torch.nn.Parameter(torch.rand([4, 128, 32, 32]))

            def forward(self, x):
                gamma = LowerBound.apply(self.gamma, 1)
                return x + gamma

        mod = MyMod()
        args, kwargs = ([torch.rand([4, 128, 32, 32])], {})
        before = mod(*args, **kwargs)

        compiled_model = torch._dynamo.optimize("eager")(mod)
        after = compiled_model(*args, **kwargs)
        self.assertEqual(before, after)

    # I pulled all of these test cases from test_autograd.py
    # In the future, we should make the Dynamo test suite actually
    # run on test_autograd.py (it's disabled right now) and delete these.
    def test_smoke_from_test_autograd(self):
        class Func(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                out0 = x.clone()
                out1 = x.clone()
                ctx.mark_non_differentiable(out1)
                ctx._materialize_non_diff_grads = False
                return out0, out1

            @staticmethod
            def backward(ctx, g0, g1):
                assert g1 is None
                return g0

        def mult1(x):
            return x.prod(dim=-1).prod(dim=-1)

        class Mult(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                y = mult1(x)
                ctx.save_for_backward(x, y)
                return y

            @staticmethod
            def backward(ctx, grad_output):
                x, y = ctx.saved_tensors
                return (grad_output * y)[:, None, None] / x

        mult2 = Mult.apply

        class Double(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                y = x**2
                ctx.save_for_backward(x, y)
                return y

            @staticmethod
            def backward(ctx, grad_output):
                x, _ = ctx.saved_tensors
                return grad_output * 2 * x

        # this is equivalent, but uses the output of .forward() in .backward()
        class Double2(Double):
            @staticmethod
            def backward(ctx, grad_output):
                x, y = ctx.saved_tensors
                return grad_output * 2 * y / x

        double = Double.apply
        double2 = Double2.apply

        class Identity(torch.autograd.Function):
            @staticmethod
            def forward(ctx, a, b):
                return a, a + b

            @staticmethod
            def backward(ctx, grad_a, grad_b):
                return grad_a + grad_b, grad_b

        class MyFunc2(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inp):
                return inp.clone()

            @staticmethod
            def backward(ctx, gO):
                return torch.tensor(float("nan")).expand(10, 10)

        def run_fn(a):
            out = MyFunc2.apply(a)
            return out.sum()

        class MyFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inp):
                return inp.view_as(inp)

            @staticmethod
            def backward(ctx, grad):
                return grad

        class MyAdder(torch.autograd.Function):
            @staticmethod
            def forward(ctx, a, b):
                a.add_(b)
                ctx.mark_dirty(a)
                return a

            @staticmethod
            def backward(ctx, grad):
                return grad, grad

        class InplaceMul(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                result = x.mul_(2)
                ctx.mark_dirty(result)
                return result

            @staticmethod
            def backward(ctx, grad_output):
                pass

            @staticmethod
            def jvp(ctx, x_t):
                if jvp_err:
                    return x_t
                else:
                    return x_t.mul_(2)

        class MyFn2(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, y):
                return x + y, x

            @staticmethod
            def vjp(ctx, gO1, gO2):
                return gO1 + gO2, gO1

            @staticmethod
            def jvp(ctx, x_t, y_t):
                return x_t + y_t, fn(x_t)

        class MyFn3(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inp, inplace):
                view = inp.clone()[:3]
                if inplace:
                    view += 2
                return view

            @staticmethod
            def backward(ctx, grad):
                return grad, None

        def test():
            a = torch.tensor(1.0, requires_grad=True)
            out = Func.apply(a)[0]
            out.backward()

            x = torch.ones(2, 4, 4).requires_grad_()
            mult2(x)

            x = torch.tensor(2).double().requires_grad_()
            double(x)
            double2(x)

            x = torch.randn(5, 5, requires_grad=True)
            y = torch.randn(5, 5, requires_grad=True)
            q, p = Identity.apply(x, y)

            a = torch.rand(1, 2)
            b = torch.rand(1, requires_grad=True)
            view_a = MyFn.apply(a)

            a = torch.ones(2, requires_grad=True)
            b = torch.ones(2, requires_grad=True)
            c = MyAdder.apply(a.clone(), b)
            c.sum().backward()

            z = torch.tensor(1.0, requires_grad=True)
            x = z.clone()
            y = InplaceMul.apply(x)

            a = torch.tensor(1.0, dtype=torch.double, requires_grad=True)
            b = torch.tensor(1.0, dtype=torch.double, requires_grad=True)
            c = torch.tensor(1.0, dtype=torch.double)
            d = torch.tensor(1.0, dtype=torch.double)
            MyFn2.apply(a, b)
            MyFn2.apply(c, d)

            base = torch.rand(10, requires_grad=True)
            foo = MyFn3.apply(base, False)

        test()
        opt_test = torch._dynamo.optimize("eager")(test)
        opt_test()

    def test_tensor_subclass_intermediary_input(self):
        class FooTensor(torch.Tensor):
            @staticmethod
            def __new__(cls, data, config, scale):
                self = torch.Tensor._make_wrapper_subclass(
                    cls,
                    config[0],
                    strides=config[1],
                    storage_offset=config[2],
                    dtype=config[3],
                    layout=config[4],
                    requires_grad=config[5],
                    device=data.device,
                )
                self._data = data
                self._config = config
                self._scale = scale
                return self

            def __repr__(self):
                return "FooTensor"

            def __tensor_flatten__(self):
                return ("_data",), (
                    self._config,
                    self._scale,
                )

            @staticmethod
            def __tensor_unflatten__(tensors, metadatas):
                return FooTensor(tensors["_data"], metadatas[0], metadatas[1])

            @classmethod
            def __torch_dispatch__(cls, func, types, args, kwargs=None):
                # handling clone and view is so dynamo fakefication passes, it's not
                # intended to be handling user code
                if func == torch.ops.aten.clone.default:
                    return FooTensor(
                        args[0]._data.clone(), args[0]._config, args[0]._scale
                    )
                elif func == torch.ops.aten.view.default:
                    new_data = args[0]._data.view(*args[1:])
                    return FooTensor(new_data, args[0]._config, args[0]._scale)

                raise NotImplementedError()

            __torch_function__ = torch._C._disabled_torch_function_impl

        class foo_autograd_fn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                # access some data from `x`, where `x` is a tensor subclass
                x2 = x._data + 1.0
                # create and return a tensor subclass from within a torch.autograd.Function
                x3 = FooTensor(x2, x._config, x._scale)
                return x3._data

            @staticmethod
            def backward(ctx, g):
                return g

        x_ref = torch.randn(4, 4).requires_grad_(True)
        x = copy.deepcopy(x_ref)
        scale = torch.tensor(1.0)
        # Weird that this is needed, but not having this breaks a lot of things
        torch._dynamo.allow_in_graph(FooTensor)

        def foo(x, scale):
            config = (
                x.size(),
                x.stride(),
                x.storage_offset(),
                x.dtype,
                x.layout,
                x.requires_grad,
            )
            x = FooTensor(x, config, scale)
            x = foo_autograd_fn.apply(x)
            return x

        y_ref = foo(x_ref, scale)
        y_ref.sum().backward()

        foo_opt = torch.compile(foo, backend="eager")
        y = foo_opt(x, scale)
        y.sum().backward()

        self.assertEqual(y, y_ref)
        self.assertEqual(x.grad, x_ref.grad)

    def test_smuggle_symint_issue_111031(self):
        from torch.autograd import Function

        class Foo(Function):
            @staticmethod
            def forward(ctx, x):
                ctx.x0 = x.size(0)
                return x * 2

            @staticmethod
            def backward(ctx, grad_out):
                return grad_out * ctx.x0

        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True, dynamic=True)
        def foo(x):
            return Foo.apply(x)

        foo(torch.randn(2, requires_grad=True))
        self.assertEqual(cnts.frame_count, 1)

    def test_smuggle_tensor_and_complex_structures(self):
        from torch.autograd import Function

        class Foo(Function):
            @staticmethod
            def forward(ctx, x):
                ctx.x0 = x
                ctx.x1 = [1, 2, 3]
                return x * 2

            @staticmethod
            def backward(ctx, grad_out):
                x0mul = grad_out * ctx.x0
                for i in ctx.x1:
                    x0mul = (x0mul * i) + x0mul
                return x0mul

        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True, dynamic=True)
        def foo(x):
            return Foo.apply(x)

        foo(torch.randn(2, requires_grad=True))
        self.assertEqual(cnts.frame_count, 1)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
