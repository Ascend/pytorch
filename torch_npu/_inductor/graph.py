from typing import (
     Any,
     List,
     Tuple,
     Union,
 )
import itertools
import operator

import sympy
import torch
from torch.fx.node import Node
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from torch.utils._sympy.numbers import int_oo
from torch._inductor import config, metrics
from torch._inductor import graph as inductor_graph
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo.utils import defake, dynamo_timed
from torch._inductor.virtualized import NullHandler, V


LazyString = inductor_graph.LazyString
OrderedSet = inductor_graph.OrderedSet
Pointwise = inductor_graph.Pointwise
Reduction = inductor_graph.Reduction
StorageBox = inductor_graph.StorageBox
TensorBox = inductor_graph.TensorBox
constrain_to_fake_tensors = inductor_graph.constrain_to_fake_tensors
constrain_to_fx_strides = inductor_graph.constrain_to_fx_strides
fallback_handler = inductor_graph.fallback_handler
fallback_node_due_to_unsupported_type = inductor_graph.fallback_node_due_to_unsupported_type
gather_origins = inductor_graph.gather_origins
ir = inductor_graph.ir
is_magic_method = inductor_graph.is_magic_method
log = inductor_graph.log
make_channels_last_strides_for = inductor_graph.make_channels_last_strides_for
needs_realized_inputs = inductor_graph.needs_realized_inputs
resolve_unbacked_bindings = inductor_graph.resolve_unbacked_bindings
GraphLowering = inductor_graph.GraphLowering


def patch_codegen_with_cpp_wrapper():
    def npu_codegen_with_cpp_wrapper(self) -> Tuple[str, List[Tuple[int, Node]]]:
        # add "npu" support
        if any(device in self.device_types for device in ["cuda", "xpu", "npu"]):
            if config.triton.autotune_at_compile_time:
                # If autotune_at_compile_time is True, we can do the codegen in one-pass
                return self.codegen()
            else:
                # first pass
                self.cpp_wrapper = False
                compiled = self.compile_to_module().call

                def materialize(
                    x: Union[torch.SymInt, torch.SymFloat, torch.Tensor]
                ) -> Union[int, float, torch.Tensor]:
                    if x is None:
                        return None
                    elif isinstance(x, (torch.SymInt, torch.SymFloat)):
                        # Need concrete value to run dynamic shapes and tune the result
                        return x.node.hint
                    elif isinstance(x, FakeTensor):
                        return defake(x)
                    else:
                        if not isinstance(x, torch.Tensor):
                            raise AssertionError("Unknown type when creating real inputs" + str(type(x)))
                        return x

                tracing_context = torch._guards.TracingContext.try_get()
                if tracing_context is not None and not isinstance(
                    V.real_inputs, NullHandler
                ):
                    if tracing_context.output_strides:
                        tracing_context.output_strides.clear()

                    params_flat = [
                        param
                        for param in tracing_context.params_flat  # type: ignore[union-attr]
                        if param is not None
                    ]
                    real_inputs = [
                        materialize(x)
                        for x in itertools.chain(params_flat, V.real_inputs)
                    ]
                else:
                    # In the backward pass, V.real_inputs is not OrderedSet.
                    # Generating random inputs based on self.example_inputs sometimes can be problematic,
                    # e.g. illegal memory access. A comprehensive fix is to autotune in a separate process.
                    real_inputs = [
                        materialize(x)  # type:ignore[arg-type]
                        for x in (
                            self.example_inputs  # type:ignore[union-attr]
                            if isinstance(V.real_inputs, NullHandler)
                            else V.real_inputs
                        )
                    ]

                if self.mutated_inputs:
                    from .compile_fx import clone_preserve_strides

                    mutated_input_idxs = [
                        idx
                        for idx, name in enumerate(self.graph_inputs)
                        if name in self.mutated_inputs
                        and isinstance(real_inputs[idx], torch.Tensor)
                    ]
                    for idx in mutated_input_idxs:
                        # clone mutated Tensor inputs to avoid mutating them in
                        # the first pass of the CPP wrapper-based compilation, as
                        # this will lead to a side effect on the example inputs:
                        # e.g. if torch.compile(f)(x) if called on input-mutating
                        # f, the inputs x will be mutated twice in the process:
                        # once here, and again when running the compiled model;
                        # this will also lead to a numerically incorrect output
                        mutated_inp = real_inputs[idx]
                        if not isinstance(mutated_inp, torch.Tensor):
                            raise AssertionError
                        real_inputs[idx] = clone_preserve_strides(mutated_inp)
                        del mutated_inp

                with torch.utils._python_dispatch._disable_current_modes():
                    compiled(real_inputs)
                del real_inputs

                # second pass
                self.cpp_wrapper = True
                self.removed_buffers.clear()
                self.removed_operations.clear()
                self.inplaced_to_remove.clear()
                V.graph.sizevars.precomputed_replacements.clear()
                V.graph.sizevars.inv_precomputed_replacements.clear()
                metrics.reset()
                with config.patch({"triton.autotune_at_compile_time": False}):
                    return self.codegen()
        else:
            # cpu
            return self.codegen()
    from torch._inductor.graph import GraphLowering
    GraphLowering.codegen_with_cpp_wrapper = npu_codegen_with_cpp_wrapper


def patch_count_bytes():
    def count_bytes(self):
        total_bytes = 0
        node_counts = []
        node_runtimes = []
        for node in self.scheduler.nodes:
            try:
                num_bytes = node.get_read_write_buffers_sizes()
            except AssertionError:
                num_bytes = 0
            total_bytes += num_bytes
            node_counts.append((node, num_bytes // 4))
            node_runtimes.append((node, node.get_estimated_runtime()))

        return total_bytes, node_counts, node_runtimes
    torch._inductor.graph.GraphLowering.count_bytes = count_bytes

def patch_run_node():
    def run_node_npu(self, n: torch.fx.Node) -> object:
        def debug(msg: str) -> None:
            log.debug("lowering %s %s", LazyString(n.format_node), msg)

        from torch._inductor.compiler_bisector import CompilerBisector

        buffer_watermark = len(self.buffers)
        operation_watermark = len(self.operations)

        # origins: OrderedSet[Union[Node, ir.IRNode]] = OrderedSet([n])
        origins: OrderedSet[Any] = OrderedSet([n])
        is_call_function = n.op == "call_function"
        if is_call_function:
            args, kwargs = self.fetch_args_kwargs_from_env(n)
            origins |= gather_origins(args, kwargs)
        with (
            ir.IRNode.current_origins(origins),
            self.set_current_node(n),
            V.set_current_node(n),
        ):
            if (
                n.op == "call_function"
                and n.target is not operator.getitem
                and (
                    fallback_node_due_to_unsupported_type(n)
                    or CompilerBisector.disable_subsystem(
                        "inductor", "lowerings", lambda: repr(n)
                    )
                )
            ):
                debug("fallback_handler")
                result = fallback_handler(n.target, add_to_fallback_set=False)(
                    *args,  # type: ignore[possibly-undefined]
                    **kwargs,  # type: ignore[possibly-undefined]
                )
            elif (
                n.op == "call_function"
                and n.target is torch.ops.higher_order.triton_kernel_wrapper_mutation
                and config.triton_kernel_default_layout_constraint != "flexible_layout"
            ):
                debug("user_defined_triton_kernel_layout_constraints")
                if (
                    config.triton_kernel_default_layout_constraint
                    == "needs_fixed_stride_order"
                ):
                    old_args = args  # type: ignore[possibly-undefined]
                    old_kwargs = kwargs  # type: ignore[possibly-undefined]

                    if arg_kwarg_vals := n.meta.get("arg_kwarg_vals"):
                        inp_args = arg_kwarg_vals[0]
                        inp_kwargs = arg_kwarg_vals[1]
                        args, kwargs = constrain_to_fake_tensors(
                            args, kwargs, inp_args, inp_kwargs
                        )
                    else:
                        args, kwargs = constrain_to_fx_strides(n, *args, **kwargs)  # type: ignore[index]
                    result = self.call_function(n.target, args, kwargs)  # type: ignore[arg-type]
                    self.propagate_mutation(n, old_args, old_kwargs, args, kwargs)  # type: ignore[possibly-undefined]
                else:
                    raise RuntimeError(
                        f"Unknown triton_kernel_default_layout_constraint: {config.triton_kernel_default_layout_constraint}"
                    )
            elif is_magic_method(n.target):
                # TODO: this is sus, it probably should be handled in the
                # lowerings themselves similarly to sym_size/sym-stride
                # https://github.com/pytorch/pytorch/issues/127789
                debug("is_magic_method")
                if isinstance(
                    n.meta["val"], (torch.SymInt, torch.SymFloat, torch.SymBool)
                ):
                    result = n.meta["val"].node.expr
                else:
                    result = super(GraphLowering, self).run_node(n)
            else:
                debug("")
                result = super(GraphLowering, self).run_node(n)

            # require the same stride order for dense outputs,
            # 1. user-land view() will not throw because inductor
            # output different strides than eager
            # long term the solution is to make view() always succeed
            # with infallible strides.
            # 2: as_strided ops, we need make sure its input has same size/stride with
            # eager model to align with eager behavior.
            as_strided_ops = [
                torch.ops.aten.as_strided.default,
                torch.ops.aten.as_strided_.default,
                torch.ops.aten.as_strided_scatter.default,
                torch.ops.aten.resize.default,
                torch.ops.aten.resize_as.default,
            ]
            is_output = any(user.op == "output" for user in n.users)
            is_user_visible = n in self.user_visible_output_strides
            is_input_for_as_strided = any(
                user.target in as_strided_ops for user in n.users
            )

            if n.meta.get("inductor_realize_to_strides", False) and isinstance(
                result, TensorBox
            ):
                result.realize()
                strides = n.meta["val"].stride()
                sym_strides = torch._inductor.utils.any_is_symbolic(*strides)
                if result.maybe_get_stride() != strides and not sym_strides:
                    stride_order = ir.get_stride_order(strides)
                    result = ir.ExternKernel.require_stride_order(result, stride_order)
            if (
                is_output
                and isinstance(result, TensorBox)
                and isinstance(result.data, ir.BaseView)
            ):
                # Realize so that outputs are correctly aliased
                result.realize()

            if (is_output or is_input_for_as_strided) and isinstance(
                n.meta["val"], torch.Tensor
            ):
                if is_user_visible:
                    strides = self.user_visible_output_strides.get(n)
                else:
                    strides = n.meta["val"].stride()

                if strides is not None and len(strides) > 0:
                    allow_padding = (
                        config.pad_outputs or not is_user_visible
                    ) and not is_input_for_as_strided
                    dense = torch._prims_common.is_non_overlapping_and_dense(
                        n.meta["val"]
                    )
                    unbacked_symbols_in_strides = (
                        len(free_unbacked_symbols(strides)) > 0
                    )
                    if (
                        not unbacked_symbols_in_strides
                        and dense
                        and len(result.get_size()) == 4
                        and n in self.nodes_prefer_channels_last
                        and not is_user_visible
                        and not is_input_for_as_strided
                    ):
                        strides = ir.FlexibleLayout.stride_ordered_for_memory_format(
                            result.get_size(), torch.channels_last
                        )
                    if not unbacked_symbols_in_strides and len(strides):
                        # To avoid converting possible view ops to a copy kernel, we use the previous
                        # require_exact_strides to handle views. But ultimately it's better to require
                        # the right strides at the tensor definition.
                        if n.meta["val"]._is_view() or isinstance(
                            result.data, ir.BaseView
                        ):
                            result = ir.ExternKernel.require_stride_order(
                                result,
                                ir.get_stride_order(strides),
                                allow_padding=allow_padding,
                            )
                        else:
                            strides = [
                                s.node.expr if isinstance(s, torch.SymInt) else s
                                for s in strides
                            ]
                            result = ir.ExternKernel.require_exact_strides(
                                result, strides, allow_padding=allow_padding
                            )

            # Realize if (1) any user need inputs realized, or (2) there is
            # already too many reads and rematerializing can be bad.
            num_users = len(OrderedSet(n.users))
            if num_users > 1 and isinstance(result, TensorBox):
                for user in n.users:
                    if user.target in needs_realized_inputs:
                        result.realize_hint()
                        # This inclusion is somewhat controversial (from
                        # discussion between Horace, Natalia, and Elias).
                        # Currently, it's not very clear why this is helpful.
                        # The general idea here is that even though a node may
                        # have FlexibleLayout, we still often *treat* it as if
                        # it was contiguous. This appears to sometimes result in
                        # suboptimal behavior.
                        #
                        # When we do a better job selecting layout, we should
                        # revisit this.
                        need_fixed_layout = [
                            torch.ops.aten.convolution_backward.default,
                            torch.ops.aten.mm.default,
                            torch.ops.aten._int_mm.default,
                        ]
                        need_fixed_channels_last_layout = []
                        if not self.layout_opt:
                            need_fixed_layout.append(torch.ops.aten.convolution.default)
                        if torch._C._has_mkldnn:
                            need_fixed_layout += [
                                torch.ops.mkldnn._linear_pointwise.default,
                                torch.ops.mkldnn._linear_pointwise.binary,
                                torch.ops.aten.mkldnn_rnn_layer.default,
                                torch.ops.onednn.qlinear_pointwise.default,
                                torch.ops.onednn.qlinear_pointwise.tensor,
                                torch.ops.onednn.qlinear_pointwise.binary,
                                torch.ops.onednn.qlinear_pointwise.binary_tensor,
                            ]
                            need_fixed_channels_last_layout += [
                                torch.ops.mkldnn._convolution_pointwise.default,
                                torch.ops.mkldnn._convolution_pointwise.binary,
                                torch.ops.mkldnn._convolution_pointwise_.binary,
                                torch.ops.mkldnn._convolution_transpose_pointwise.default,
                                torch.ops.onednn.qconv2d_pointwise.default,
                                torch.ops.onednn.qconv2d_pointwise.binary,
                            ]
                            if torch._C.has_mkl:
                                need_fixed_layout += [torch.ops.mkl._mkl_linear.default]
                        if user.target in need_fixed_layout:
                            result = ir.ExternKernel.require_stride_order(
                                result,
                                ir.get_stride_order(n.meta["val"].stride()),
                                allow_padding=True,
                            )
                        if (
                            user.target in need_fixed_channels_last_layout
                            and n is user.args[0]
                        ):
                            result = ir.ExternKernel.require_stride_order(
                                result,
                                ir.get_stride_order(
                                    make_channels_last_strides_for(n.meta["val"].shape)
                                ),
                            )
                    if user.op == "output":
                        if isinstance(result.data.data, (Pointwise, Reduction)):
                            result.realize()

                # TODO(jansel): introduce a store vs inline choice
                result.mark_reuse(len(n.users))

            # Realize if the IRNode already has accumulated lots of reads
            if isinstance(result, TensorBox) and result.has_exceeded_max_reads():
                # Prevent excessive accumulation in a computed buffer, when
                # there are multiple branches each with small number of memory
                # reads, but they converge to a user.
                result.realize_hint()

            # Realize if a Pointwise has too much stuff to be inlined.
            # As this may cause RecursionError during Inductor's evaluation.
            if isinstance(result, TensorBox) and isinstance(result.data, StorageBox):
                curr = result.data.data
                if isinstance(curr, Pointwise):
                    # Use inner fn as a rough proxy. Good enough.
                    if curr.has_large_inner_fn(threshold=100):
                        result.realize()
                    
                    from .config import lowering_axis_count
                    if lowering_axis_count and len(curr.ranges) >= lowering_axis_count:
                        result.realize()

        # This is not complete, but it doesn't have to be: origin_node
        # tracking is best effort.  The logic here critically relies on direct
        # TensorBox -> StorageBox denoting a non-view; we don't bother trying
        # to get views to work.  Feel free to add any extra cases as needed.
        #
        # Note: we can't YOLO tree_map over this result, because if there are
        # buffers or a view involved, we might not be able to validly assign
        # the origin_node here.
        if isinstance(result, TensorBox) and isinstance(result.data, ir.StorageBox):
            if isinstance(result.data.data, ir.Loops):
                result.data.data._post_init_setattr("origin_node", n)
            elif isinstance(result.data.data, ir.Buffer):
                result.data.data._post_init_setattr("origin_node", n)
                if isinstance(result.data.data, ir.ComputedBuffer) and isinstance(
                    result.data.data.data, ir.Loops
                ):
                    result.data.data.data._post_init_setattr("origin_node", n)
                # Not really multi-output, can straightforwardly recurse in
                elif (
                    isinstance(result.data.data, ir.MultiOutput)
                    and not result.data.data.indices
                ):
                    if isinstance(result.data.data.inputs[0], ir.Buffer):
                        result.data.data.inputs[0]._post_init_setattr("origin_node", n)

        self.register_users_of(result)

        new_unbacked_defs = OrderedSet[sympy.Symbol]()
        for buf in self.buffers[buffer_watermark:]:
            new_unbacked_defs |= buf.get_unbacked_symbol_defs()
        for op in self.operations[operation_watermark:]:
            new_unbacked_defs |= op.get_unbacked_symbol_defs()

        def format_new_defs() -> str:
            r = [
                f"unbacked_symbol_defs={buf.get_unbacked_symbol_defs()} in:\n{buf}\n"
                for buf in self.buffers[buffer_watermark:]
            ]
            r.extend(
                f"unbacked_symbol_defs={op.get_unbacked_symbol_defs()} in:\n{op}\n"
                for op in self.operations[operation_watermark:]
            )
            return "***\n".join(r)

        if n.op != "placeholder":
            # Note [Backwards runtime asserts]
            # Backwards poses an interesting problem for deferred runtime
            # asserts.  In the easy case, we may solely close over data
            # dependent sized tensors, and there are no binding sites for
            # unbacked SymInts.  In this case, we can just drop all the
            # runtime asserts on the floor: no non-placeholder bindings, no
            # problem.
            #
            # However, it is *possible* for a fresh runtime assert to show up
            # between forwards and backwards.  Right now, the freezing process
            # that happens when we lower forwards means that we will freeze
            # runtime asserts, and then the moment the backwards lowering
            # process attempts to add a new deferred runtime assert, we will
            # fail.  Let's say you remove that assert.  Now when we get here,
            # we need to make sure we actually emit these asserts (because we
            # can't emit them in forwards, we already compiled it).  So we
            # have to do something here.  But we don't want to reemit ALL
            # deferred runtime asserts, we only want to emit the NEW ones.
            # Therefore needing some sort of stratification in the ShapeEnv.
            # This is all doable, it just hasn't been done yet.
            shape_env = V.graph.sizevars.shape_env

            def make_assert(expr, msg: str) -> None:
                assert_op = ir.AssertScalar(expr, msg)
                self.register_buffer(assert_op, set_name=True)
                self.register_operation(assert_op)

            for i0 in new_unbacked_defs:
                ras = self.ras_by_symbol.pop(i0, [])
                # NB: size-like not needed, we won't retrace
                vr = shape_env.var_to_range[i0]
                if not shape_env._default_unspecified_value_range().issubset(vr):

                    def is_convertible(s) -> bool:
                        if s in (int_oo, -int_oo):
                            return False
                        try:
                            int(s)
                            return True
                        except TypeError:
                            return False

                    if is_convertible(vr.lower):
                        make_assert(i0 >= vr.lower, f"{i0} >= {vr.lower}")
                    if is_convertible(vr.upper):
                        make_assert(i0 <= vr.upper, f"{i0} <= {vr.upper}")

                for ra in ras:
                    fvs = free_unbacked_symbols(ra.expr)
                    missing = fvs - self.bound_unbacked_symbols
                    if missing:
                        i1 = min(missing, key=str)
                        self.ras_by_symbol.setdefault(i1, []).append(ra)
                    else:
                        make_assert(ra.expr, f"{ra.expr}")

            self.bound_unbacked_symbols |= new_unbacked_defs

            unbacked_bindings = resolve_unbacked_bindings(
                V.graph.sizevars.shape_env, n.meta.get("unbacked_bindings", {})
            )
            # When we do lowering, it is possible we reallocate unbacked SymInts.
            # So we need to line up the unbacked SymInts when performing the test
            # here
            #
            # In principle, we could permit lowering to introduce MORE unbacked
            # SymInts: as long as all the old unbacked ones are accounted for,
            # it's fine for inductor to introduce extra calls to item()/unbacked()
            # whatever.  This actually happens in practice when an unbacked SymInt
            # gets memoized away; naively, when Inductor reprocesses a kernel, it
            # doesn't know that the memo still applies, and ends up allocating a
            # new symbol.  However, this is generally a bad thing: we may still
            # end up needing to test equalities on the symbols, and a fresh
            # symbol is likely to hit lots of GuardOnDataDependent errors that
            # we already know facts for.
            renamed_unbacked_bindings = OrderedSet(
                V.fake_mode.shape_env.unbacked_renamings.get(s, s)
                for s in unbacked_bindings.keys()
            )

        return result

    torch._inductor.graph.GraphLowering.run_node = run_node_npu
