import types

from .library import *


class GemmOperation:
    #
    def __init__(
        self,
        gemm_kind,
        arch,
        dispatch_policy,
        tile_description,
        A,
        B,
        C,
        element_epilogue,
        block_swizzle=None,
        block_epilogue=None,
        D=None,
        allow_hf32=False,
    ):
        self.gemm_kind = gemm_kind
        self.arch = arch
        self.dispatch_policy = dispatch_policy
        self.tile_description = tile_description
        self.A = A
        self.B = B
        self.C = C
        self.D = D

        if self.D is None:
            self.D = self.C

        self.hf32_mode = False
        if (
            allow_hf32
            and self.A.element == DataType.f32
            and self.B.element == DataType.f32
        ):
            self.hf32_mode = True
        self.element_epilogue = element_epilogue
        self.block_swizzle = block_swizzle
        self.block_epilogue = block_epilogue

    def accumulator_type(self):
        return self.element_epilogue

    def arch_name(self):
        return ArchTypeNames[self.arch]

    # Generates a short string representing the AB layout tags (e.g., nt or tn)
    def layout_name(self):
        return "%s%s" % (
            ShortLayoutTypeNames[self.A.layout],
            ShortLayoutTypeNames[self.B.layout],
        )

    def dispatch_policy_name(self):
        res = f"{ShortDispatchPolicyNames[self.dispatch_policy]}"
        if self.hf32_mode:
            res += "-hf32"
        return res

    def block_swizzle_name(self):
        return f"{ShortBlockSwizzleNames[self.block_swizzle]}"

    # Generates a string representing the element type.
    def extended_name(self):
        extended_name = (
            "{element_a}_{element_b}_{element_c}_{element_epi}_{element_d}".format(
                element_a=DataTypeNames[self.A.element],
                element_b=DataTypeNames[self.B.element],
                element_c=DataTypeNames[self.C.element],
                element_epi=DataTypeNames[self.element_epilogue],
                element_d=DataTypeNames[self.D.element],
            )
        )
        return extended_name

    # Generate the full kernel function name
    def procedural_name(self):
        """The full procedural name indicates architecture, extended name, tile size, and layout."""
        tile_desc = self.tile_description.procedural_name()
        swizzle_name = (
            "" if self.block_swizzle is None else f"_{self.block_swizzle_name()}"
        )
        return "catlass_{p}_{op}_{dp}{sw}_{ex}_{td}_{l}".format(
            p=self.arch_name(),
            op=self.gemm_typename(),
            dp=self.dispatch_policy_name(),
            sw=swizzle_name,
            ex=self.extended_name(),
            td=tile_desc,
            l=self.layout_name(),
        )

    def configuration_name(self):
        return self.procedural_name()

    def arch_typename(self):
        return ArchTypeTag[self.arch]

    def gemm_typename(self):
        return GemmKindNames[self.gemm_kind]

    def swizzle_typename(self):
        return "" if self.block_swizzle is None else BlockSwizzleTag[self.block_swizzle]

    def dispatch_policy_typename(self):
        res = DispatchPolicyTag[self.dispatch_policy]
        if "HF32_MODE" in res:
            hf32_str = "true" if self.hf32_mode else "false"
            res = res.replace("HF32_MODE", hf32_str)
        return res

    def swap_as_evg_kernel(self):
        if self.gemm_kind not in EVGGemmKindMap:
            return
        self.gemm_kind = EVGGemmKindMap[self.gemm_kind]


def _make_layouttypname_func(attr_name: str):
    def _func(self):
        return LayoutTag[getattr(self, attr_name).layout]

    return _func


for name in ["A", "B", "C", "D"]:
    setattr(GemmOperation, f"layout{name}_typename", _make_layouttypname_func(name))
