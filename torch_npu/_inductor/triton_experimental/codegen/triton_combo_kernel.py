# Copyright (c) 2026, Huawei Technologies Co., Ltd
#
# Licensed under the Apache-2.0 License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/pytorch/pytorch/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Code generation for TE combo kernels and their pointwise members.

Each member uses a local tile index and slot-specific range-tree names.
``TEComboKernel`` emits the shared dispatch over all members' tiles.
"""

import sympy

from torch._inductor.codegen.common import ArgName, ConstexprArg
from torch._inductor.codegen.simd import IterationRangesEntry, IterationRangesRoot
from torch._inductor.codegen.triton_combo_kernel import ComboKernel
from torch._inductor.codegen.triton_utils import config_of, signature_to_meta
from torch._inductor.runtime.hints import DeviceProperties
from torch._inductor.utils import IndentedBuffer, Placeholder
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

from ..compat import IS_TRITON_36_PLUS
from ..device_props import get_npu_vector_core_count
from .triton import NPUTritonKernel


__all__ = ["TEComboSubKernel", "TEComboKernel"]


def _te_combo_rekey_entry(
    tree: IterationRangesRoot,
    node: IterationRangesEntry,
    old_symbol: sympy.Symbol,
) -> None:
    """Update the symbol maps populated by ``tree.lookup`` after renaming.

    ``tree.nodes`` is keyed by the index expression, which does not change.
    Other generated names are derived from ``node.name`` after this call.
    """
    new_symbol = node.symbol()
    if old_symbol == new_symbol:
        return

    range_tree_nodes = V.kernel.range_tree_nodes
    range_tree_nodes.pop(old_symbol, None)
    range_tree_nodes[new_symbol] = node

    # Find the entry registered by lookup; do not rely on its position.
    for i in range(len(tree.var_list) - 1, -1, -1):
        if tree.var_list[i] == old_symbol:
            tree.var_list[i] = new_symbol
            break

    if old_symbol in tree.var_ranges:
        tree.var_ranges[new_symbol] = tree.var_ranges.pop(old_symbol)


def _te_combo_suffix_tree(tree: IterationRangesRoot, num: int) -> None:
    """Suffix range-tree entries as ``lookup`` creates them.

    Keep ``tree.prefix`` unchanged because axis selection depends on it.
    Entries are created lazily, so the hook is installed before body codegen.
    """
    original = tree.lookup
    if getattr(original, "_te_combo_suffixed", False):
        # Avoid wrapping lookup twice and appending the slot suffix twice.
        return

    def lookup(divisor, length):
        known = len(tree.nodes)
        node = original(divisor, length)
        if len(tree.nodes) == known:
            # Existing entries already have their slot suffix.
            return node

        old_symbol = node.symbol()
        # set_name() replaces codegen(), which reduction entries still need
        # to emit their index assignment. Rename directly and clear its cache.
        node.name = f"{node.name}_{num}"
        node.codegen.cache_clear()
        _te_combo_rekey_entry(tree, node, old_symbol)
        return node

    lookup._te_combo_suffixed = True  # type: ignore[attr-defined]
    tree.lookup = lookup  # type: ignore[method-assign]


def _te_combo_suffix_nodes(kernel: NPUTritonKernel, num: int) -> None:
    """Install slot-specific naming on the member's range trees."""
    for tree in kernel.range_trees:
        _te_combo_suffix_tree(tree, num)


class TEComboSubKernel(NPUTritonKernel):
    """A TE pointwise kernel using slot-specific names and local tile indices."""

    def __init__(self, *args, **kwargs):
        self._combo_num = None
        self._combo_local_tile = None
        super().__init__(*args, **kwargs)

    def set_combo_slot(self, num: int, local_tile_name: str) -> None:
        """Bind the member's slot before its range-tree entries are created."""
        self._combo_num = num
        self._combo_local_tile = local_tile_name
        _te_combo_suffix_nodes(self, num)

    def iteration_ranges_get_pid(self, entry: IterationRangesRoot) -> str:
        """Use the member-local tile for linearized combo dispatch."""
        if self._combo_local_tile is not None and self._npu_linearize:
            return self._combo_local_tile
        return super().iteration_ranges_get_pid(entry)


class TEComboKernel(ComboKernel):
    """Assemble linearized TE members over one concatenated tile space."""

    def __init__(self, xblock: int):
        super().__init__(TEComboSubKernel, enable_autotune=False)
        self.xblock = xblock

    def codegen_kernel(self, name=None):
        if len(self.sub_kernels) < 2:
            raise AssertionError("TE combo needs at least two members")

        code = IndentedBuffer()
        code.splice(NPUTritonKernel.gen_common_triton_imports())
        code.splice(self.sub_kernels[0].gen_triton_ext_imports())
        seen_helpers = OrderedSet()
        for sub in self.sub_kernels:
            for helper in sub.helper_functions:
                if helper not in seen_helpers:
                    code.writeline("")
                    code.splice(helper)
                    seen_helpers.add(helper)

        argdefs, _, signature, _ = self.args.python_argdefs()
        self.triton_meta = {
            "signature": signature_to_meta(
                signature, size_dtype=self.sub_kernels[0].index_dtype,
                argdefs=argdefs,
            ),
            "device": DeviceProperties.create(V.graph.get_current_device_or_throw()),
            "constants": {},
            "mix_mode": "aiv",
            "configs": [config_of(signature)],
        }
        if IS_TRITON_36_PLUS:
            signature.append(ConstexprArg("XBLOCK"))
        argdefs.append(ArgName("XBLOCK", is_constexpr=True))

        counts = []
        for sub in self.sub_kernels:
            free_nodes = [
                node for tree in sub.range_trees
                if not tree.is_reduction and not tree.is_loop
                for node in tree.nodes.values()
                if node.name not in tree.tree_node_mapping
            ]
            count = 1
            for node in free_nodes:
                length = int(node.length)
                count *= (length + min(length, self.xblock) - 1) // min(length, self.xblock) if length else 0
            counts.append(count)
        offsets = []
        total = 0
        for count in counts:
            offsets.append(total)
            total += count

        self.inductor_meta = {
            "grid_type": "Grid1D",
            "autotune_hints": set(),
            "kernel_name": str(Placeholder.DESCRIPTIVE_NAME),
            "mutated_arg_names": [],
            "optimize_mem": V.graph.is_inference or V.graph.is_backward,
            "no_x_dim": False,
            "num_load": sum(sub.num_load for sub in self.sub_kernels),
            "num_reduction": 0,
            "npu_num_x_nodes": sum(len(sub._npu_free_blocks_names()) for sub in self.sub_kernels),
            "npu_linearize": False,
            "te_combo_meta": {
                "combo_num_kernels": len(self.sub_kernels),
                "combo_tile_offsets": offsets,
                "combo_tile_counts": counts,
                "combo_no_x_dim": [sub.no_x_dim for sub in self.sub_kernels],
                "combo_dynamic_args": [],
                "combo_dispatch_kind": "group_dispatch",
                "combo_xblock": self.xblock,
            },
            **NPUTritonKernel.inductor_meta_common(),
        }
        kernel_name = name or str(Placeholder.KERNEL_NAME)
        size_hints = {"x": max(1, total * self.xblock)}
        code.splice(
            f"@npu_triton_heuristics.foreach(size_hints={size_hints!r}, "
            f"filename=__file__, triton_meta={self.triton_meta!r}, "
            f"inductor_meta={self.inductor_meta!r})\n@triton.jit"
        )
        code.writeline(
            f"def {kernel_name}({', '.join(arg.full_name() for arg in argdefs)}):"
        )
        with code.indent():
            code.writeline(f"total_thread = {get_npu_vector_core_count()}")
            code.writeline("group_id = tl.program_id(0)")
            for old, new in self.args.aliases():
                code.writeline(f"{old} = {new}")
            for num, sub in enumerate(self.sub_kernels):
                for tree in sub.range_trees:
                    if tree.is_reduction or tree.is_loop:
                        continue
                    if not hasattr(tree, "pre_loop_code"):
                        raise AssertionError("TE combo member has no linearized header")
                    code.splice(tree.pre_loop_code)
                blocks = sub._npu_emit_total_blocks(code, f"num_blocks_{num}")
                if blocks != f"num_blocks_{num}":
                    code.writeline(f"num_blocks_{num} = {blocks}")
                code.writeline(
                    f"S_{num + 1} = "
                    f"{'0' if num == 0 else f'S_{num}'} + num_blocks_{num}"
                )
            code.writeline(f"tiles_per_core = S_{len(self.sub_kernels)} // total_thread")
            code.writeline(f"core_tail = S_{len(self.sub_kernels)} % total_thread")
            code.writeline("group_size = tiles_per_core + (group_id < core_tail)")
            code.writeline("group_base = group_id * tiles_per_core + tl.minimum(group_id, core_tail)")
            code.writeline("for i in range(group_size):")
            with code.indent():
                code.writeline("tile = group_base + i")
                for num, sub in enumerate(self.sub_kernels):
                    lo = "0" if num == 0 else f"S_{num}"
                    code.writeline(
                        f"{'if' if num == 0 else 'elif'} tile < S_{num + 1}:"
                    )
                    with code.indent():
                        code.writeline(f"local_tile_{num} = tile - {lo}")
                        sub.codegen_prologue(sub.body)
                        sub.codegen_body()
                        sub._filter_pdl(sub.body)
                        code.splice(sub.body)
        return code.getvalue()

    def call_kernel(self, name):
        # The TE launcher takes its grid from te_combo_meta, not the upstream
        # CUDA dispatch_class.
        _, call_args, _, arg_types = self.args.python_argdefs()
        V.graph.wrapper_code.generate_kernel_call(
            name, call_args, triton=True, arg_types=arg_types,
            triton_meta=self.triton_meta, inductor_meta=self.inductor_meta,
        )
