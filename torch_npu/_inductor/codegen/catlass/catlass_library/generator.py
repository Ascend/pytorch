from torch_npu.npu import matmul
from torch_npu._inductor.config import num_cube_core
from .library import *
from .gemm_operation import GemmOperation
from .gemm_autotune import generate_configs, _gemm_template_heuristics


def CreateGemmOperator(
    manifest,
    arch,
    gemm_kinds,
    layouts,
    dispatch_policies,
    data_type,
    shape_desc=None,
):
    operations = []
    element_a, element_b, element_c, element_epilogue = data_type
    perferred_gemms = _gemm_template_heuristics(
        gemm_kinds, shape_desc, num_cube_core, element_c
    )
    for gemm_kind in perferred_gemms:
        for dispatch_policy in dispatch_policies:
            configs = generate_configs(
                arch, gemm_kind, dispatch_policy, data_type[0], shape_desc
            )
            for cfg in configs:
                for layout in layouts:
                    A = TensorDescription(element_a, layout[0])
                    B = TensorDescription(element_b, layout[1])
                    C = TensorDescription(element_c, layout[2])

                    new_operation = GemmOperation(
                        gemm_kind,
                        arch,
                        dispatch_policy,
                        cfg.tile_desc,
                        A,
                        B,
                        C,
                        element_epilogue,
                        cfg.block_swizzle,
                        allow_hf32=matmul.allow_hf32,
                    )
                    manifest.append(new_operation, shape_desc)
                    operations.append(new_operation)
    return operations


def Generate910B(manifest, shape_desc=None):
    if manifest.get_ops(shape_desc) is not None:
        # use cached ops
        return

    Generate910B_MM(manifest, shape_desc)
    Generate910B_GEMM(manifest, shape_desc)


def Generate910B_MM(manifest, shape_desc):
    layouts = [
        (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.RowMajor),
        (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.RowMajor),
        (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.RowMajor),
        (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.RowMajor),
    ]

    dispatch_policies = [
        DispatchPolicyType.MmadPingpongEUnitFlag,
    ]

    data_types = [
        (DataType.f16, DataType.f16, DataType.f32, DataType.f32),
        (DataType.f16, DataType.f16, DataType.f16, DataType.f32),
        (DataType.bf16, DataType.bf16, DataType.f32, DataType.f32),
        (DataType.bf16, DataType.bf16, DataType.bf16, DataType.f32),
        (DataType.f32, DataType.f32, DataType.f32, DataType.f32),
    ]

    gemm_kinds = [
        GemmKind.BasicMatmulTla,
    ]

    arch = ArchType.A2
    # MatmulTla
    for data_type in data_types:
        CreateGemmOperator(
            manifest,
            arch,
            gemm_kinds,
            layouts,
            dispatch_policies,
            data_type,
            shape_desc,
        )


def Generate910B_GEMM(manifest, shape_desc):
    layouts = [
        (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.RowMajor),
        (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.RowMajor),
        (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.RowMajor),
        (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.RowMajor),
    ]

    dispatch_policies = [
        DispatchPolicyType.GemmA2EUnitFlagEShuffleKEABBA,
    ]

    # dtype of A, B, C and epilogue
    # GEMM does not support bfloat16
    data_types = [
        (DataType.f16, DataType.f16, DataType.f32, DataType.f32),
        (DataType.f16, DataType.f16, DataType.f16, DataType.f32),
        (DataType.f32, DataType.f32, DataType.f32, DataType.f32),
    ]

    arch = ArchType.A2
    gemm_kinds = [GemmKind.Gemm]
    for data_type in data_types:
        CreateGemmOperator(
            manifest,
            arch,
            gemm_kinds,
            layouts,
            dispatch_policies,
            data_type,
            shape_desc,
        )


def Generate910D(manifest, shape_desc=None):
    if manifest.get_ops(shape_desc) is not None:
        # use cached ops
        return

    Generate910D_MM(manifest, shape_desc)


def Generate910D_MM(manifest, shape_desc):
    layouts = [
        (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.RowMajor),
        (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.RowMajor),
        (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.RowMajor),
        (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.RowMajor),
    ]

    dispatch_policies = [
        # DispatchPolicyType.MmadPingpongDUnitFlag,
        DispatchPolicyType.MmadPingpongEUnitFlag,
        # DispatchPolicyType.MmadPreloadAsyncWithCallback_12221EUnitFlagDShuffleK,
    ]

    data_types = [
        (DataType.f16, DataType.f16, DataType.f32, DataType.f32),
        (DataType.f16, DataType.f16, DataType.f16, DataType.f32),
        (DataType.bf16, DataType.bf16, DataType.f32, DataType.f32),
        (DataType.bf16, DataType.bf16, DataType.bf16, DataType.f32),
        (DataType.f32, DataType.f32, DataType.f32, DataType.f32),
    ]

    gemm_kinds = [
        GemmKind.BasicMatmulTla,
        GemmKind.StreamkMatmulTla,
        GemmKind.MultiCoreSplitkMatmulTla,
        GemmKind.TailMultiCoreSplitkMatmulTla,
    ]

    arch = ArchType.A5
    # MatmulTla
    for data_type in data_types:
        CreateGemmOperator(
            manifest,
            arch,
            gemm_kinds,
            layouts,
            dispatch_policies,
            data_type,
            shape_desc,
        )

    # BatchedMatmulTla
    gemm_kinds = [
        GemmKind.BatchedMatmulTla,
    ]
    dispatch_policies = [
        DispatchPolicyType.MmadMultiBatch,
    ]
    for data_type in data_types:
        operations = CreateGemmOperator(
            manifest,
            arch,
            gemm_kinds,
            layouts,
            dispatch_policies,
            data_type,
            shape_desc,
        )
        if len(operations) == 0:
            dispatch_policies = [
                DispatchPolicyType.MmadPingpongEUnitFlag,
            ]
            CreateGemmOperator(
                manifest,
                arch,
                gemm_kinds,
                layouts,
                dispatch_policies,
                data_type,
                shape_desc,
            )

    # GroupedMatmulTla
    data_types = [
        (DataType.f16, DataType.f16, DataType.f32, DataType.f32),
        (DataType.f16, DataType.f16, DataType.f16, DataType.f32),
        (DataType.bf16, DataType.bf16, DataType.f32, DataType.f32),
        (DataType.bf16, DataType.bf16, DataType.bf16, DataType.f32),
    ]
    gemm_kinds = [
        GemmKind.GroupedMatmulSliceMTla,
    ]
    dispatch_policies = [
        DispatchPolicyType.MmadPingpongEUnitFlag,
    ]
    for data_type in data_types:
        CreateGemmOperator(
            manifest,
            arch,
            gemm_kinds,
            layouts,
            dispatch_policies,
            data_type,
            shape_desc,
        )
