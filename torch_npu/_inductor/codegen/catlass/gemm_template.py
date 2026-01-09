from __future__ import annotations

import copy
import enum
import math
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import sympy
import torch
from torch._inductor import ir
from torch._inductor.codegen.common import IndentedBuffer
from torch._inductor.ir import (
    Buffer,
    ChoiceCaller,
    ComputedBuffer,
    FixedLayout,
    IRNode,
    Layout,
    Pointwise,
    ReinterpretView,
)
from torch._inductor.scheduler import BaseSchedulerNode
from torch._inductor.utils import is_dynamic
from torch._inductor.virtualized import V

from ...config import catlass as catlass_config
from . import catlass_utils
from .catlass_python_evg import CatlassEVGCodegen
from .catlass_library import library as catlass_lib
from .catlass_library.gemm_operation import GemmOperation
from .catlass_utils import torch_dtype_to_catlass_type, get_npu_arch, _normalize_npu_arch
from .catlass_kernel import CATLASSTemplateBuffer, CATLASSTemplateKernel
from .catlass_template import CATLASSTemplate

log = logging.getLogger("torch._inductor")

EVGArgRenames = Any


# TLA Matmul template
TLA_MM_TEMPLATE_CATLASS_1X = r"""
{{template.header().getvalue()}}
{{template.globals().getvalue()}}
// When workspace_size is not a nullptr, populates requested workspace_size and returns.
// Otherwise, computes the Gemm kernel using the given workspace ptr.

{{template.render_gemm_arguments(op_instance, argument_template,
                                 X, W, Bias, Y, alpha, beta, relu_enabled, kernel)}}

extern "C" {
PT_EXPORT {{kernel_call_signature}} {
    try {
    uint32_t B = {{kernel.size(Y, 0, -3, default_value=1)}};
    uint32_t m = {{kernel.size(X, -2)}};
    uint32_t k = {{kernel.size(X, -1)}};
    uint32_t n = {{kernel.size(W, -1)}};

    GemmCoord problemShape{m, n, k};

    // Define the layout of each matrix
    LayoutTagA tagA{m, k};
    LayoutTagB tagB{k, n};
    LayoutTagC tagC{m, n};
    auto layoutA = tla::MakeLayoutFromTag(tagA);
    auto layoutB = tla::MakeLayoutFromTag(tagB);
    auto layoutC = tla::MakeLayoutFromTag(tagC);

    uint8_t* deviceA = {{template.catlass_type_cast(X, kernel.ptr(X))}};
    uint8_t* deviceB = {{template.catlass_type_cast(W, kernel.ptr(W))}};
    uint8_t* deviceBias = {{template.catlass_type_cast(Bias, kernel.ptr(Bias))}};
    uint8_t* deviceC = {{template.catlass_type_cast(Y, kernel.ptr(Y))}};

    using BlockScheduler = {{op_instance.swizzle_typename()}};
    // epilogue visitor graph definition may need m, n, k
    // so we put it in here
    {{epilogue_visitor_graph}}
    {{epilogue_visitor_args}}
    {{epilogue_arguments}}

    using GemmKernel = Gemm::Kernel::{{op_instance.gemm_typename()}}<GemmBlock, BlockEpilogue, BlockScheduler>;
    using GemmAdapter = Gemm::Device::DeviceGemm<GemmKernel>;

    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    {{kernel_arguments}}
    GemmAdapter gemm_op;

    if (workspace_size) {
        *workspace_size = gemm_op.GetWorkspaceSize(arguments);
        return 0;
    }

    {{ffts_addr_prepare}}
    {
        auto status = gemm_op.CanImplement(arguments);
        CATLASS_CHECK(status);
    }
    {
        auto status = gemm_op.Initialize(arguments, workspace);
        CATLASS_CHECK(status);
    }
    {
        auto status = {{kernel_call}}
        CATLASS_CHECK(status);
    }

    }
    catch (std::exception& e) {
        std::cerr << "Runtime error: " << e.what() << std::endl;
        return -1;
    }
    catch (...) {
        return -1;
    }
    return 0;
}
}
"""


TLA_GROUP_MM_TEMPLATE_CATLASS_1X = r"""
{{template.header().getvalue()}}
{{template.globals().getvalue()}}
// When workspace_size is not a nullptr, populates requested workspace_size and returns.
// Otherwise, computes the Gemm kernel using the given workspace ptr.

{{template.render_gemm_arguments(op_instance, argument_template,
                                 X, W, Bias, Y, alpha, beta, relu_enabled, kernel)}}

extern "C" {
PT_EXPORT {{kernel_call_signature}} {
    try {
    uint32_t B = {{kernel.size(Y, 0, -3, default_value=1)}};
    uint32_t m = {{kernel.size(X, -2)}};
    uint32_t k = {{kernel.size(X, -1)}};
    uint32_t n = {{kernel.size(W, -1)}};
    uint32_t problemCount = {{template.offsets_size}};

    GemmCoord problemShape{m, n, k};

    // Define the layout of each matrix
    LayoutTagA tagA{m, k};
    LayoutTagB tagB{k, n};
    LayoutTagC tagC{m, n};
    auto layoutA = tla::MakeLayoutFromTag(tagA);
    auto layoutB = tla::MakeLayoutFromTag(tagB);
    auto layoutC = tla::MakeLayoutFromTag(tagC);

    uint8_t* deviceA = {{template.catlass_type_cast(X, kernel.ptr(X))}};
    uint8_t* deviceB = {{template.catlass_type_cast(W, kernel.ptr(W))}};
    uint8_t* deviceBias = {{template.catlass_type_cast(Bias, kernel.ptr(Bias))}};
    uint8_t* deviceC = {{template.catlass_type_cast(Y, kernel.ptr(Y))}};
    uint8_t* deviceGroupList = (uint8_t*) offsets;

    using BlockScheduler = {{op_instance.swizzle_typename()}};
    // epilogue visitor graph definition may need m, n, k
    // so we put it in here
    {{epilogue_visitor_graph}}
    {{epilogue_visitor_args}}
    {{epilogue_arguments}}

    using GemmKernel = Gemm::Kernel::{{op_instance.gemm_typename()}}<GemmBlock, BlockEpilogue, BlockScheduler, {{DataTypeTag[template.offsets_type]}}>;
    using GemmAdapter = Gemm::Device::DeviceGemm<GemmKernel>;

    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    {{kernel_arguments}}
    GemmAdapter gemm_op;

    if (workspace_size) {
        *workspace_size = gemm_op.GetWorkspaceSize(arguments);
        return 0;
    }

    {{ffts_addr_prepare}}
    {
        auto status = gemm_op.CanImplement(arguments);
        CATLASS_CHECK(status);
    }
    {
        auto status = gemm_op.Initialize(arguments, workspace);
        CATLASS_CHECK(status);
    }
    {
        auto status = {{kernel_call}}
        CATLASS_CHECK(status);
    }

    }
    catch (std::exception& e) {
        std::cerr << "Runtime error: " << e.what() << std::endl;
        return -1;
    }
    catch (...) {
        return -1;
    }
    return 0;
}
}
"""


TLA_MM_ARGS_CATLASS_1X = r"""
    // Initialize GemmUniversal1xInstance arguments.

    // Define ArchTag
    using ArchTag = {{op_instance.arch_typename()}};

    using ElementA = {{kernel.catlass_dtype(X)}};
    using ElementB = {{kernel.catlass_dtype(W)}};
    using ElementC = {{kernel.catlass_dtype(Y)}};
    using ElementBias = {{kernel.catlass_dtype(Bias)}};

    // Define the layout
    using LayoutTagA = {{op_instance.layoutA_typename()}};
    using LayoutTagB = {{op_instance.layoutB_typename()}};
    using LayoutTagC = {{op_instance.layoutC_typename()}};

    using DispatchPolicy = {{op_instance.dispatch_policy_typename()}};
    using L1TileShape = {{op_instance.tile_description.l1_tile_typename(True)}};
    using L0TileShape = {{op_instance.tile_description.l0_tile_typename(True)}};

    using TileCopy = 
        Gemm::Tile::PackedTileCopyTla<ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC, ElementBias, {{relu_enabled}}>;
    using GemmBlock = 
        Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape,
                                  ElementA, ElementB, ElementC, ElementBias, TileCopy>;
"""


# =============== A2/3 Specific Catlass GemmTemplate ===============
# Optimized Matmul template
OPT_MM_TEMPLATE_CATLASS_1X = r"""
{{template.header().getvalue()}}
{{template.globals().getvalue()}}
// When workspace_size is not a nullptr, populates requested workspace_size and returns.
// Otherwise, computes the Gemm kernel using the given workspace ptr.

template<
    class ArchTag,
    class AType,
    class BType,
    class CType,
    class BiasType = void
>
struct TileCopyOpt : public Catlass::Gemm::Tile::TileCopy<ArchTag, AType, BType, CType, BiasType> {
    using Base = Catlass::Gemm::Tile::TileCopy<ArchTag, AType, BType, CType, BiasType>;
    using ElementA = typename Base::ElementA;
    using ElementB = typename Base::ElementB;
    using ElementAccumulator = typename Base::ElementAccumulator;

    // When matrix A is row-major, if the number of rows in matrix A is less than 16, 
    // using the CopyGmToL1IntervalDataCopy method can improve the transfer efficiency.
    // The situation is similar for matrix B. If the above conditions are met, 
    // please uncomment the following and comment out the original matrix A transfer method

    // using CopyGmToL1A = Gemm::Tile::CopyGmToL1IntervalDataCopy<ArchTag, AType>;

    using CopyGmToL1A = typename Base::CopyGmToL1A;
    using CopyGmToL1B = typename Base::CopyGmToL1B;

    using CopyL1ToL0A = typename Base::CopyL1ToL0A;
    using CopyL1ToL0B = typename Base::CopyL1ToL0B;

    using CopyL0CToGm = typename Base::CopyL0CToGm; 
    using BiasTypeSelector = typename Base::BiasTypeSelector; 
    using CopyGmToL1Bias = typename Base::CopyGmToL1Bias;
    using CopyL1ToBT = typename Base::CopyL1ToBT;
};

{{template.render_gemm_arguments(op_instance, argument_template, epilogue_template,
                                 X, W, Bias, Y, alpha, beta, kernel)}}

template <bool NeedPaddingA, bool NeedPaddingB, bool isMGreaterN,
          typename LayoutA, typename LayoutB>
int LaunchGemmKernelImpl(
    const GemmCoord& problemShape,
    const LayoutA& layoutA, const LayoutB& layoutB,
    uint8_t* deviceA, uint8_t* deviceB, uint8_t* deviceC,
    size_t* workspace_size, uint8_t* workspace, aclrtStream stream)
{
    using TileCopy = TileCopyOpt<ArchTag,
        std::conditional_t<NeedPaddingA, ATypePadding, AType>,
        std::conditional_t<NeedPaddingB, BTypePadding, BType>,
        CType>;
    
    using BlockMmadOpt = Gemm::Block::BlockMmad<
        DispatchPolicy, L1TileShape, L0TileShape,
        std::conditional_t<NeedPaddingA, ATypePadding, AType>,
        std::conditional_t<NeedPaddingB, BTypePadding, BType>,
        CType, void, TileCopy>;
    
    using GemmKernel = Gemm::Kernel::{{op_instance.gemm_typename()}}<
        std::conditional_t<NeedPaddingA, GlobalPaddingA, void>,
        std::conditional_t<NeedPaddingB, GlobalPaddingB, void>,
        BlockMmadOpt, BlockEpilogue,
        std::conditional_t<isMGreaterN, BlockScheduler30, BlockScheduler31>>;
    
    {{kernel_arguments}}

    using GemmAdapter = Gemm::Device::DeviceGemm<GemmKernel>;
    GemmAdapter gemm_op;

    if (workspace_size) {
        *workspace_size = gemm_op.GetWorkspaceSize(arguments);
        return 0;
    }

    {{ffts_addr_prepare}}
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    {
        auto status = gemm_op.CanImplement(arguments);
        CATLASS_CHECK(status);
    }
    {
        auto status = gemm_op.Initialize(arguments, workspace);
        CATLASS_CHECK(status);
    }
    {
        auto status = {{kernel_call}}
        CATLASS_CHECK(status);
    }

    return 0;
}

extern "C" {
PT_EXPORT {{kernel_call_signature}} {
    try {
    uint32_t B = {{kernel.size(Y, 0, -3, default_value=1)}};
    uint32_t m = {{kernel.size(X, -2)}};
    uint32_t k = {{kernel.size(X, -1)}};
    uint32_t n = {{kernel.size(W, -1)}};

    GemmCoord problemShape{m, n, k};

    // Define the layout of each matrix
    LayoutA layoutA = LayoutA::template MakeLayout<ElementA>(m, k);
    LayoutB layoutB = LayoutB::template MakeLayout<ElementB>(k, n);
    LayoutC layoutC = LayoutC::template MakeLayout<ElementC>(m, n);

    uint8_t* deviceA = {{template.catlass_type_cast(X, kernel.ptr(X))}};
    uint8_t* deviceB = {{template.catlass_type_cast(W, kernel.ptr(W))}};
    uint8_t* deviceBias = {{template.catlass_type_cast(Bias, kernel.ptr(Bias))}};
    uint8_t* deviceC = {{template.catlass_type_cast(Y, kernel.ptr(Y))}};

    bool isNeedPaddingA = IsNeedPadding(layoutA, alignByElement);
    bool isNeedPaddingB = IsNeedPadding(layoutB, alignByElement);

    if (m > n) {
        if (isNeedPaddingA && isNeedPaddingB) {
            LaunchGemmKernelImpl<true, true, true>(problemShape, layoutA, layoutB,
                deviceA, deviceB, deviceC, workspace_size, workspace, stream);
        } else if (isNeedPaddingA) {
            LaunchGemmKernelImpl<true, false, true>(problemShape, layoutA, layoutB,
                deviceA, deviceB, deviceC, workspace_size, workspace, stream);
        } else if (isNeedPaddingB) {
            LaunchGemmKernelImpl<false, true, true>(problemShape, layoutA, layoutB,
                deviceA, deviceB, deviceC, workspace_size, workspace, stream);
        } else {
            LaunchGemmKernelImpl<false, false, true>(problemShape, layoutA, layoutB,
                deviceA, deviceB, deviceC, workspace_size, workspace, stream);
        }
    } else {
        if (isNeedPaddingA && isNeedPaddingB) {
            LaunchGemmKernelImpl<true, true, false>(problemShape, layoutA, layoutB,
                deviceA, deviceB, deviceC, workspace_size, workspace, stream);
        } else if (isNeedPaddingA) {
            LaunchGemmKernelImpl<true, false, false>(problemShape, layoutA, layoutB,
                deviceA, deviceB, deviceC, workspace_size, workspace, stream);
        } else if (isNeedPaddingB) {
            LaunchGemmKernelImpl<false, true, false>(problemShape, layoutA, layoutB,
                deviceA, deviceB, deviceC, workspace_size, workspace, stream);
        } else {
            LaunchGemmKernelImpl<false, false, false>(problemShape, layoutA, layoutB,
                deviceA, deviceB, deviceC, workspace_size, workspace, stream);
        }
    }
    }
    catch (std::exception& e) {
        std::cerr << "Runtime error: " << e.what() << std::endl;
        return -1;
    }
    catch (...) {
        return -1;
    }
    return 0;
}
}
"""


OPT_MM_ARGS_CATLASS_1X = r"""
    // Initialize GemmUniversal1xInstance arguments.

    // Define ArchTag
    using ArchTag = {{op_instance.arch_typename()}};

    using ElementA = {{kernel.catlass_dtype(X)}};
    using ElementB = {{kernel.catlass_dtype(W)}};
    using ElementC = {{kernel.catlass_dtype(Y)}};

    constexpr uint32_t alignByByte = 512;
    constexpr uint32_t alignByElement = alignByByte / sizeof(ElementC);

    // Define the Layout
    using LayoutA = {{op_instance.layoutA_typename()}};
    using LayoutB = {{op_instance.layoutB_typename()}};
    using LayoutC = {{op_instance.layoutC_typename()}};
    using LayoutBias = Catlass::layout::VectorLayout;

    using AType = Gemm::GemmType<ElementA, LayoutA>;
    using BType = Gemm::GemmType<ElementB, LayoutB>;
    using CType = Gemm::GemmType<ElementC, LayoutC>;

    // Define padding layout
    static const uint32_t COMPUTE_LENGTH_A = 96 * 1024 / sizeof(ElementA);
    static const uint32_t COMPUTE_LENGTH_B = 96 * 1024 / sizeof(ElementB);
    using PaddingTag = Catlass::Gemm::Kernel::PaddingTag;
    constexpr PaddingTag paddingTagA = (std::is_same_v<LayoutA, layout::zN> || std::is_same_v<LayoutA, layout::nZ>) ?
        PaddingTag::NO_PADDING : PaddingTag::PADDING_BLOCK_ND;
    constexpr PaddingTag paddingTagB = (std::is_same_v<LayoutB, layout::zN> || std::is_same_v<LayoutB, layout::nZ>) ?
        PaddingTag::NO_PADDING : PaddingTag::PADDING_BLOCK_ND;
    using PaddingBuilderA = Catlass::Gemm::Kernel::PaddingBuilder<
        ArchTag, ElementA, LayoutA, COMPUTE_LENGTH_A, paddingTagA>;
    using GlobalPaddingA = PaddingBuilderA::Padding;
    using PaddingBuilderB = Catlass::Gemm::Kernel::PaddingBuilder<
        ArchTag, ElementB, LayoutB, COMPUTE_LENGTH_B, paddingTagB>;
    using GlobalPaddingB = PaddingBuilderB::Padding;

    using LayoutMmadA = typename PaddingBuilderA::LayoutAfterPadding;
    using LayoutMmadB = typename PaddingBuilderB::LayoutAfterPadding;
    using ATypePadding = Gemm::GemmType<ElementA, LayoutMmadA>;
    using BTypePadding = Gemm::GemmType<ElementB, LayoutMmadB>;

    using DispatchPolicy = {{op_instance.dispatch_policy_typename()}};
    using L1TileShape = {{op_instance.tile_description.l1_tile_typename()}};
    using L0TileShape = {{op_instance.tile_description.l0_tile_typename()}};
    using BlockScheduler30 = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
    using BlockScheduler31 = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
    using BiasType = {{template.catlass_elem_type(kernel.catlass_dtype(Bias), "LayoutBias")}};

    {{epilogue_arguments}}
"""


# ------ Basic Matmul -------

MM_TEMPLATE_CATLASS_1X = r"""
{{template.header().getvalue()}}
{{template.globals().getvalue()}}
// When workspace_size is not a nullptr, populates requested workspace_size and returns.
// Otherwise, computes the Gemm kernel using the given workspace ptr.

{{template.render_gemm_arguments(op_instance, argument_template, epilogue_template,
                                 X, W, Bias, Y, alpha, beta, kernel)}}

extern "C" {
PT_EXPORT {{kernel_call_signature}} {
    try {
    uint32_t B = {{kernel.size(Y, 0, -3, default_value=1)}};
    uint32_t m = {{kernel.size(X, -2)}};
    uint32_t k = {{kernel.size(X, -1)}};
    uint32_t n = {{kernel.size(W, -1)}};

    GemmCoord problemShape{m, n, k};

    // Define the layout of each matrix
    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutC layoutC{m, n};

    uint8_t* deviceA = {{template.catlass_type_cast(X, kernel.ptr(X))}};
    uint8_t* deviceB = {{template.catlass_type_cast(W, kernel.ptr(W))}};
    uint8_t* deviceBias = {{template.catlass_type_cast(Bias, kernel.ptr(Bias))}};
    uint8_t* deviceC = {{template.catlass_type_cast(Y, kernel.ptr(Y))}};

    {{kernel_arguments}}
    GemmAdapter gemm_op;
    
    if (workspace_size) {
        *workspace_size = gemm_op.GetWorkspaceSize(arguments);
        return 0;
    }

    {{ffts_addr_prepare}}
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    {
        auto status = gemm_op.CanImplement(arguments);
        CATLASS_CHECK(status);
    }
    {
        auto status = gemm_op.Initialize(arguments, workspace);
        CATLASS_CHECK(status);
    }
    {
        auto status = {{kernel_call}}
        CATLASS_CHECK(status);
    }
    }
    catch (std::exception& e) {
        std::cerr << "Runtime error: " << e.what() << std::endl;
        return -1;
    }
    catch (...) {
        return -1;
    }
    return 0;
}
}
"""


MM_KERNEL_CALL_CATLASS_1X = r"""gemm_op(stream, aicCoreNum);"""


FFTS_ADDR_PREPARE = r"""
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));
"""

MM_FFTS_KERNEL_CALL_CATLASS_1X = r"""gemm_op(stream, aicCoreNum, fftsAddr);"""


MM_ARGS_CATLASS_1X = r"""
    // Initialize GemmUniversal1xInstance arguments.

    // Define ArchTag
    using ArchTag = {{op_instance.arch_typename()}};

    // Define the Layout
    using LayoutA = {{op_instance.layoutA_typename()}};
    using LayoutB = {{op_instance.layoutB_typename()}};
    using LayoutC = {{op_instance.layoutC_typename()}};
    using LayoutBias = Catlass::layout::VectorLayout;

    using DispatchPolicy = {{op_instance.dispatch_policy_typename()}};
    using L1TileShape = {{op_instance.tile_description.l1_tile_typename()}};
    using L0TileShape = {{op_instance.tile_description.l0_tile_typename()}};

    using AType = {{template.catlass_elem_type(kernel.catlass_dtype(X), "LayoutA")}};
    using BType = {{template.catlass_elem_type(kernel.catlass_dtype(W), "LayoutB")}};
    using CType = {{template.catlass_elem_type(kernel.catlass_dtype(Y), "LayoutC")}};
    using BiasType = {{template.catlass_elem_type(kernel.catlass_dtype(Bias), "LayoutBias")}};

    using GemmBlock = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType, BiasType>;
    using BlockScheduler = {{op_instance.swizzle_typename()}};
    {{epilogue_arguments}}

    using GemmKernel = Gemm::Kernel::{{op_instance.gemm_typename()}}<GemmBlock, BlockEpilogue, BlockScheduler>;
    using GemmAdapter = Gemm::Device::DeviceGemm<GemmKernel>;
"""


MM_ARGS_CATLASS_1X_VOID_EPILOGUE = r"""
    using BlockEpilogue = void;
"""


MM_ARGS_CATLASS_EVG_EPILOGUE = r"""
    using BlockEpilogue = Catlass::Epilogue::Block::BlockEpilogue<
        Catlass::Epilogue::EpilogueVisitor<>,
        {{op_instance.arch_typename()}},
        Int<computeLength>,
        {{evg_name}},
        ElementC
    >;
"""


MM_KERNEL_ARGUMENTS_CATLASS_1X = r"""
    typename GemmKernel::Arguments arguments{problemShape, deviceA, deviceB, deviceC};
"""


BMM_KERNEL_ARGUMENTS_CATLASS_1X = r"""
    typename GemmKernel::Arguments arguments{B, problemShape, deviceA, deviceB, deviceC};
"""


MMBIAS_KERNEL_ARGUMENTS_CATLASS_1X = r"""
    typename GemmKernel::Arguments arguments{problemShape, deviceA, deviceB, deviceC, deviceBias};
"""


TLA_MM_KERNEL_ARGUMENTS_CATLASS_1X = r"""
    typename GemmKernel::Arguments arguments{problemShape, deviceA, layoutA, deviceB, layoutB, deviceC, layoutC, deviceBias};
"""


TLA_MM_KERNEL_WITH_CORE_NUM_ARGUMENTS_CATLASS_1X = r"""
    typename GemmKernel::Arguments arguments{problemShape, deviceA, layoutA, deviceB, layoutB, deviceC, layoutC, aicCoreNum, deviceBias};
"""


TLA_BMM_KERNEL_ARGUMENTS_CATLASS_1X = r"""
    typename GemmKernel::Arguments arguments{B, problemShape, deviceA, layoutA, deviceB, layoutB, deviceC, layoutC};
"""


TLA_GROUP_MM_KERNEL_ARGUMENTS_CATLASS_1X = r"""
    typename GemmKernel::Arguments arguments{problemShape, problemCount, deviceGroupList, deviceA, layoutA, deviceB, layoutB, deviceC, layoutC};
"""


TLA_MM_EVG_KERNEL_ARGUMENTS_CATLASS_1X = r"""
    typename GemmKernel::Arguments arguments{problemShape, deviceA, layoutA, deviceB, layoutB, deviceC, layoutC, deviceBias, evg_args};
"""


GEMM_KERNEL_ARGUMENTS_CATLASS_1X = r"""
    typename EpilogueBlock::Params epilogueParams{alpha, beta, deviceBias, layoutBias, deviceC, layoutC};
    typename GemmKernel::Arguments arguments{problemShape, align, deviceA, deviceB, workspace, deviceWA, deviceWB, epilogueParams};
"""


GEMM_TEMPLATE_CATLASS_1X = r"""
{{template.header().getvalue()}}
{{template.globals().getvalue()}}
// When workspace_size is not a nullptr, populates requested workspace_size and returns.
// Otherwise, computes the Gemm kernel using the given workspace ptr.

{{template.render_gemm_arguments(op_instance, argument_template, epilogue_template,
                                 X, W, Bias, Y, alpha, beta, kernel)}}

extern "C" {
PT_EXPORT {{kernel_call_signature}} {
    try {
    uint32_t B = {{kernel.size(Y, 0, -3, default_value=1)}};
    uint32_t m = {{kernel.size(X, -2)}};
    uint32_t k = {{kernel.size(X, -1)}};
    uint32_t n = {{kernel.size(W, -1)}};

    GemmCoord problemShape{m, n, k};

    // define scalar
    float alpha = {{alpha}};
    float beta = {{beta}};

    // Define the layout of each matrix
    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutC layoutC{m, n};
    LayoutBias layoutBias{m, n};    // TODO(px): check here

    // Define Workspace layout & size
    const uint32_t align = 128;
    LayoutA layoutWA = GetWorkspaceLayout(layoutA, align);
    LayoutB layoutWB = GetWorkspaceLayout(layoutB, align);
    size_t sizeWA = GetWorkspaceLen(layoutWA) * sizeof({{kernel.catlass_dtype(X)}});
    size_t sizeWB = GetWorkspaceLen(layoutWB) * sizeof({{kernel.catlass_dtype(W)}});
    size_t sizeWorkspace = static_cast<size_t>(M) * N * sizeof({{kernel.catlass_dtype(Y)}});

    uint8_t* deviceA = {{template.catlass_type_cast(X, kernel.ptr(X))}};
    uint8_t* deviceB = {{template.catlass_type_cast(W, kernel.ptr(W))}};
    uint8_t* deviceBias = {{template.catlass_type_cast(Bias, kernel.ptr(Bias))}};
    uint8_t* deviceC = {{template.catlass_type_cast(Y, kernel.ptr(Y))}};

    if (workspace_size) {
        // TODO(px): Gemm's GetWorkspaceSize has no use
        // *workspace_size = gemm_op.GetWorkspaceSize(arguments);
        if (!IsSameStride(layoutWA, layoutA)) {
            sizeWorkspace += sizeWA;
        }
        if (!IsSameStride(layoutWB, layoutB)) {
            sizeWorkspace += sizeWB;
        }
        *workspace_size = sizeWorkspace;
        return 0;
    }

    // split the workspace to three part: workspace, deviceWA (optional), deviceWB (optional)
    uint8_t* deviceWA = deviceA;
    uint8_t* deviceWB = deviceB;
    uint8_t* offset = workspace + sizeWorkspace;
    if (!IsSameStride(layoutWA, layoutA)) {
        deviceWA = offset;
        offset += sizeWA;
    }
    if (!IsSameStride(layoutWB, layoutB)) {
        deviceWB = offset;
    }

    {{kernel_arguments}}
    GemmAdapter gemm_op;

    {{ffts_addr_prepare}}
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    {
        auto status = gemm_op.CanImplement(arguments);
        CATLASS_CHECK(status);
    }
    {
        auto status = gemm_op.Initialize(arguments, workspace);
        CATLASS_CHECK(status);
    }
    {
        auto status = {{kernel_call}}
        CATLASS_CHECK(status);
    }
    }
    catch (std::exception& e) {
        std::cerr << "Runtime error: " << e.what() << std::endl;
        return -1;
    }
    catch (...) {
        return -1;
    }
    return 0;
}
}
"""


# Jinja template for Catlass 1.x GEMM Kernel arguments, used by the CATLASSGemmTemplate class below.
GEMM_ARGS_CATLASS_1X = r"""
    // Initialize GemmUniversal1xInstance arguments.

    // Define ArchTag
    using ArchTag = {{op_instance.arch_typename()}};

    // Define the Layout
    using LayoutA = {{op_instance.layoutA_typename()}};
    using LayoutB = {{op_instance.layoutB_typename()}};
    using LayoutC = {{op_instance.layoutC_typename()}};
    using LayoutBias = {{op_instance.layoutD_typename()}};

    using DispatchPolicy = {{op_instance.dispatch_policy_typename()}};
    using L1TileShape = {{op_instance.tile_description.l1_tile_typename()}};
    using L0TileShape = {{op_instance.tile_description.l0_tile_typename()}};

    using AType = {{template.catlass_elem_type(kernel.catlass_dtype(X), "LayoutA")}};
    using BType = {{template.catlass_elem_type(kernel.catlass_dtype(W), "LayoutB")}};
    using CType = {{template.catlass_elem_type(kernel.catlass_dtype(Y), "LayoutC")}};
    using BiasType = {{template.catlass_elem_type(kernel.catlass_dtype(Bias), "LayoutBias")}};

    using GemmBlock = Gemm::Block::BlockGemm<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;

    {{epilogue_arguments}}

    using GemmKernel = Gemm::Kernel::{{op_instance.gemm_typename()}}<GemmBlock, EpilogueBlock>;
    using GemmAdapter = Gemm::Device::DeviceGemm<GemmKernel>;
"""


# Jinja template for Catlass 1.x GEMM Kernel arguments if epilogue fusion is applied,
# used by the CATLASSGemmTemplate class below.
GEMM_ARGS_CATLASS_1X_EPILOGUE = r"""
    // TODO(px): define GemmOperation's epilogue dispatch policy
    using EpilogueBlockDispatchPolicy = Catlass::Epilogue::EpilogueAtlasA2Gemm;
    using DType = BiasType;
    using ComputeType = CType;
    using TileShapeCast = MatrixShape<L1TileShape::M / 2, L1TileShape::N>;
    constexpr uint32_t computeLength = L1TileShape::MN / 2;
    using TileElemWiseAddGemm = Epilogue::Tile::TileElemWiseAdd<ArchTag, ComputeType, computeLength>;
    using TileElemWiseMulsGemm = Epilogue::Tile::TileElemWiseMuls<ArchTag, ComputeType, computeLength>;
    using TileElemWiseCastD = Epilogue::Tile::TileCast<ArchTag, DType, ComputeType, TileShapeCast>;
    using EpilogueTileCopy = Epilogue::Tile::TileCopy<ArchTag, CType, BiasType, DType>;
    using EpilogueBlock = Epilogue::Block::BlockEpilogue<EpilogueBlockDispatchPolicy, CType, BiasType, DType,
        TileElemWiseAddGemm, TileElemWiseMulsGemm, TileElemWiseCastD, EpilogueTileCopy>;
"""


class BiasShape(enum.IntEnum):
    NO_BIAS = 0
    N_BIAS = 1  # bias shape is (N,)
    MN_BIAS = 2  # bias shape is (M, N)


class CATLASSGemmTemplate(CATLASSTemplate, ABC):
    """
    CATLASS GEMM Template, which is used to generate CATLASS GEMM kernels
    including those which allow flexible fusions with epilogues.
    """

    def __init__(
        self,
        input_nodes: list[Buffer],
        layout: Layout,
        alpha: float,
        beta: float,
        input_reorder: Optional[List[int]] = None,
    ) -> None:
        super().__init__("catlass_gemm", input_nodes, layout, input_reorder)
        self.alpha = alpha
        self.beta = beta
        assert len(input_nodes) == 2 or len(input_nodes) == 3 or len(input_nodes) == 4
        if len(input_nodes) == 4:
            assert self._are_inputs_layout_compatible(
                [node.get_layout() for node in input_nodes[:2]]
            )
        else:
            assert self._are_inputs_layout_compatible(
                [node.get_layout() for node in input_nodes]
            )
        self.is_group_mm = len(input_nodes) == 4
        self.offsets_type = torch_dtype_to_catlass_type(input_nodes[3].get_dtype()) if self.is_group_mm else None
        self.offsets_size = input_nodes[3].get_size()[0] if self.is_group_mm else None
        self.is_batchmm = any(len(node.get_size()) == 3 for node in input_nodes) and not self.is_group_mm
        self.shape_desc = self.get_shape_desc(self.input_nodes)
        self.bias_shape = BiasShape.NO_BIAS
        if len(self.input_nodes) >= 3 and self.input_nodes[2]:
            bias_first_stride = self.input_nodes[2].get_stride()[-2]
            # For N = 1, cannot distinguish bias shape is (M, 1) or (1,)
            # currently use matmulBias for this case
            self.bias_shape = (
                BiasShape.MN_BIAS
                if bias_first_stride != 0 and not (self.shape_desc[1] == 1)
                else BiasShape.N_BIAS
            )
        self.shape_desc = self.shape_desc + (self.bias_shape,)
        self.init_templates_map()
        if self.is_group_mm:
            self.input_nodes = [node for node in self.input_nodes if node]

    @staticmethod
    def get_shape_desc(input_nodes) -> Tuple[int, int, int]:
        X, W = input_nodes[0], input_nodes[1]
        M = X.get_size()[-2]
        K = X.get_size()[-1]
        N = W.get_size()[-1]
        shape_desc = [M, N, K]

        for i, x in enumerate(shape_desc):
            if isinstance(x, (int, sympy.Integer)):
                shape_desc[i] = int(x)
            elif isinstance(x, (sympy.Symbol, sympy.Expr)):
                x = x.subs(V.graph.sizevars.var_to_val)
                shape_desc[i] = int(x)
            else:
                raise ValueError(f"Unknown shape dim type: {type(x)}, value: {x}")
        return tuple(shape_desc)

    @staticmethod
    @abstractmethod
    def add_catlass_gemm_choices(
        choices: List[ChoiceCaller],
        layout: ir.Layout,
        input_nodes: List[Buffer],
        alpha: Union[float, int] = 1,
        beta: Union[float, int] = 0,
        input_reorder: Optional[List[int]] = None,
        **extra_kwargs,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def init_templates_map(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _is_op_kind_supported(
        self,
        op_kind: "catlass_lib.GemmKind",  # type: ignore[name-defined]  # noqa: F821
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
    def _get_template(self, op_kind: "catlass_lib.GemmKind") -> str:
        raise NotImplementedError

    @abstractmethod
    def _get_template_args(
        self, op_kind: "catlass_lib.GemmKind", has_evg: bool = False
    ) -> Tuple[str, str]:
        raise NotImplementedError

    @abstractmethod
    def _get_kernel_arguments_and_call(
        self, op_kind: "catlass_lib.GemmKind"
    ) -> Tuple[str, str, str]:
        raise NotImplementedError

    @abstractmethod
    def _are_inputs_layout_compatible(self, layouts: List[Layout]) -> bool:
        raise NotImplementedError

    @abstractmethod
    def _shape_match(
        self,
        op: "GemmOperation",  # type: ignore[name-defined]  # noqa: F821
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
    def _set_bias_layout(
        self,
        op: "GemmOperation",  # type: ignore[name-defined]  # noqa: F821
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
    def _get_extra_inputs_and_names(
        self,
        op: "GemmOperation" = None,  # type: ignore[name-defined]  # noqa: F821
    ) -> Tuple[Optional[Buffer], List[Optional[Buffer]], List[str]]:
        raise NotImplementedError

    def _is_standard_matmul(self) -> bool:
        return self.alpha == 1.0 and (self.beta == 0.0 or self.beta == 1.0)

    def _add_catlass_gemm_choices(
        self,
        choices: List[ChoiceCaller],
        layout: ir.Layout,
        input_nodes: List[Buffer],
        alpha: Union[float, int] = 1,
        beta: Union[float, int] = 0,
        input_reorder: Optional[List[int]] = None,
        **extra_kwargs,
    ) -> None:
        """
        Adds Catlass GEMM configurations choices to the auto-tuning list.

        This function mutates the passed list of choices by appending the choices for Catlass GEMM configs to it.

        Args:
            choices (list): The list to which choices are appended.
            layout (ir.Layout): The layout configuration.
            input_nodes (list): The list of input nodes.
            alpha (float,int): Scaling factor, defaults to 1.
            beta (float,int): Offset, defaults to 0.
            input_reorder (list, optional): Order of the inputs, defaults to None.
            **extra_kwargs: Additional keyword arguments.

        """
        ops = self.gen_ops(self.shape_desc)
        for name, op in ops:
            self.maybe_append_choice(
                choices,
                description=name,
                op=op,
            )
        if len(ops) == 0:
            input_layouts = [node.get_layout() for node in self.input_nodes]
            input_strides = [node.get_stride() for node in self.input_nodes]
            output_layout = layout
            warning_msg = f"No suitable Catlass GEMM configs found, fallbacks used ( {len(ops)=}, {output_layout=}, {input_layouts=}, {input_strides=} )"  # noqa: B950
            log.warning(warning_msg)
        log.debug(
            "Added %d Catlass gemm configs.",
            len(ops),
        )

    def filter_op(
        self,
        op: "GemmOperation",  # type: ignore[name-defined]  # noqa: F821
    ) -> Optional["GemmOperation"]:  # type: ignore[name-defined]  # noqa: F821
        """
        Helper method:

        Determines whether a given Catlass GEMM op definition is suitable for the current
        input / output of the operation that this template is supposed to implement.

        Takes memory layout, dtype and support for EVG operations into account,
        and filters potentially problematic ops.

        Returns None if the op is not suitable, otherwise returns the op to be used, which might
        have been mutated.
        """

        X = self.input_nodes[0]
        W = self.input_nodes[1]

        # Filter ops according to the shape match.
        if not self._shape_match(op):
            return None

        # Filter ops by dtypes.
        accumulator_torch_dtype = catlass_utils.get_accumulator_dtype(
            [X.get_dtype(), W.get_dtype()],
        )
        if not (
            catlass_utils.dtype_match(X.get_dtype(), op.A.element)
            and catlass_utils.dtype_match(W.get_dtype(), op.B.element)
            and catlass_utils.dtype_match(
                self.output_node.get_layout().dtype, op.C.element
            )
            and catlass_utils.dtype_match(
                accumulator_torch_dtype, op.accumulator_type()
            )
        ):
            return None

        # Filter ops by input layouts.
        if not (
            self.layout_match(X.get_layout(), op.A.layout)
            and self.layout_match(W.get_layout(), op.B.layout)
        ):
            return None

        # Update op.
        op = copy.deepcopy(op)

        # Set output layout.
        op.D.layout = CATLASSGemmTemplate.catlass_layout(self.output_node.get_layout())

        op.element_epilogue = op.accumulator_type()

        # Set bias layout and alignment.
        if not self._set_bias_layout(op):
            return None

        return op

    def gen_ops(
        self, shape_desc: Tuple[int, int, int, int]  # M, N, K, bais_type
    ) -> "List[Tuple[str, GemmOperation]]":  # type: ignore[name-defined]  # noqa: F821
        """
        Creates a list of Catlass GemmOperation instances that match the operation this template is designed to represent.
        The matching is carried out with respect to the input and output specifications of the operation.

        No function arguments.

        Returns:
            List[Tuple[str, GemmOperation]]: A list of (catlass_name, GemmOperation)
            tuples that are compatible with the operation requirements of this template.
        """
        ops = catlass_utils.gen_ops(shape_desc)
        res: Dict[str, GemmOperation] = {}

        for op in ops:
            assert isinstance(op, GemmOperation)
            if not self._is_op_kind_supported(op.gemm_kind):
                continue
            try:
                filter_res = self.filter_op(op)
            except Exception as e:
                log.debug(f"{op.procedural_name()} filter op Failed: {e}")
                continue
            if (
                filter_res is not None
                and res.get(filter_res.configuration_name(), None) is None
            ):
                res[filter_res.configuration_name()] = filter_res

        log.debug("Got catlass configs: total number of ops: %d, ", len(res))
        return list(res.items())[: catlass_config.catlass_max_profiling_configs]

    def header(self) -> IndentedBuffer:
        """
        # Returns a buffer containing CUDA C++ code for the header section of the CATLASS GEMM template.
        This section primarily includes the necessary header files.

        Returns:
            IndentedBuffer: An instance of IndentedBuffer that contains the generated C++ header code.
        """
        res = super().header()
        arch = _normalize_npu_arch(get_npu_arch())
        if arch == "910B":
            res.splice(
                """
                    #include "catlass/gemm/block/block_mmad.hpp"
                    #include "catlass/gemm/block/block_swizzle.hpp"
                    #include "catlass/gemm/dispatch_policy.hpp"
                    #include "catlass/gemm/gemm_type.hpp"
                    #include "catlass/gemm/device/device_gemm.hpp"
                    #include "catlass/gemm_coord.hpp"
                    #include "catlass/matrix_coord.hpp"

                    // Epilogue
                    #include "catlass/epilogue/dispatch_policy.hpp"
                    #include "catlass/epilogue/tile/tile_copy.hpp"
                    #include "catlass/epilogue/tile/tile_elemwise_add.hpp"
                    #include "catlass/epilogue/tile/tile_elemwise_muls.hpp"
                    #include "catlass/epilogue/tile/tile_cast.hpp"
                    #include "catlass/epilogue/block/block_epilogue.hpp"

                    // Tla
                    #include "tla/layout.hpp"
                    #include "tla/tensor.hpp"

                    // kernel headers
                """
            )
        elif arch == "910D":
            res.splice(
                """
                    #include "catlass/gemm/block/block_mmad.hpp"
                    #include "catlass/gemm/block/block_swizzle.hpp"
                    #include "catlass/gemm/dispatch_policy.hpp"
                    #include "catlass/gemm/gemm_type.hpp"
                    #include "catlass/gemm/device/device_gemm.hpp"
                    #include "catlass/gemm_coord.hpp"
                    #include "catlass/matrix_coord.hpp"

                    // Epilogue
                    #include "catlass/epilogue/dispatch_policy.hpp"
                    #include "catlass/epilogue/tile/tile_copy.hpp"
                    #include "catlass/epilogue/tile/tile_elemwise_add.hpp"
                    #include "catlass/epilogue/tile/tile_elemwise_muls.hpp"
                    #include "catlass/epilogue/tile/tile_cast.hpp"
                    #include "catlass/epilogue/block/block_epilogue.hpp"
                    #include "catlass/epilogue/fusion/fusion.hpp"

                    // Tla
                    #include "tla/layout.hpp"
                    #include "tla/tensor.hpp"

                    // kernel headers
                """
            )
        else:
            raise ValueError(f"Unrecognized NPU arch: {arch}")
        if not self._is_standard_matmul() or self.bias_shape == BiasShape.MN_BIAS:
            res.splice(
                """
                    #include "catlass/gemm/kernel/gemm.hpp"
                """
            )
        else:
            if arch == "910B":
                res.splice(
                    """
                        #include "catlass/gemm/kernel/basic_matmul_tla.hpp"
                    """
                )
            elif arch == "910D":
                res.splice(
                    """
                        #include "catlass/gemm/kernel/basic_matmul.hpp"
                        #include "catlass/gemm/kernel/optimized_matmul.hpp"
                        #include "catlass/gemm/kernel/batched_matmul.hpp"
                        #include "catlass/gemm/kernel/matmul_bias.hpp"
                        #include "catlass/gemm/kernel/basic_matmul_tla.hpp"
                        #include "catlass/gemm/kernel/batched_matmul_tla.hpp"
                        #include "catlass/gemm/kernel/streamk_matmul_tla.hpp"
                        #include "catlass/gemm/kernel/multi_core_splitk_matmul_tla.hpp"
                        #include "catlass/gemm/kernel/tail_multi_core_splitk_matmul_tla.hpp"
                        #include "catlass/gemm/kernel/grouped_matmul_slice_m_tla.hpp"
                        // EVG supported kernel
                        #include "catlass/gemm/kernel/basic_matmul_tla_visitor.hpp"
                    """
                )
            else:
                raise ValueError(f"Unrecognized NPU arch: {arch}")
        return res

    def globals(self) -> IndentedBuffer:
        res = super().globals()
        if not self._is_standard_matmul() or self.bias_shape == BiasShape.MN_BIAS:
            res.splice(
                """
                    // Workspace util funcs
                    layout::RowMajor GetWorkspaceLayout(layout::RowMajor layout, uint32_t align)
                    {
                        if (align == 0) {
                            return layout;
                        }
                        return layout::RowMajor(layout.shape(0), layout.shape(1), RoundUp(layout.shape(1), align));
                    }

                    
                    layout::ColumnMajor GetWorkspaceLayout(layout::ColumnMajor layout, uint32_t align)
                    {
                        if (align == 0) {
                            return layout;
                        }
                        return layout::ColumnMajor(layout.shape(0), layout.shape(1), RoundUp(layout.shape(0), align));
                    }

                    
                    size_t GetWorkspaceLen(layout::RowMajor layout)
                    {
                        return layout.shape(0) * layout.stride(0);
                    }

                    
                    size_t GetWorkspaceLen(layout::ColumnMajor layout)
                    {
                        return layout.shape(1) * layout.stride(1);
                    }

                    bool IsSameStride(layout::RowMajor layout1, layout::RowMajor layout2)
                    {
                        return layout1.stride(0) == layout2.stride(0);
                    }

                    bool IsSameStride(layout::ColumnMajor layout1, layout::ColumnMajor layout2)
                    {
                        return layout1.stride(1) == layout2.stride(1);
                    }

                """
            )
        else:
            res.splice(
                """
                    bool IsNeedPadding(layout::RowMajor layout, uint32_t align)
                    {
                        // If the stride is greater than 65536, padding is required to reduce the stride.
                        if (layout.stride(0) < 65536) {
                            return layout.stride(0) % align != 0;
                        } else {
                            return true;
                        }
                    }

                    bool IsNeedPadding(layout::ColumnMajor layout, uint32_t align)
                    {
                        // If the stride is greater than 65536, padding is required to reduce the stride.
                        if (layout.stride(1) < 65536) {
                            return layout.stride(1) % align != 0;
                        } else {
                            return true;
                        }
                    }

                    bool IsNeedPadding(layout::zN layout, uint32_t align)
                    {
                        return false;
                    }

                    bool IsNeedPadding(layout::nZ layout, uint32_t align)
                    {
                        return false;
                    }
                """
            )
        return res

    @staticmethod
    def catlass_layout(torch_layout: ir.Layout) -> "Optional[catlass_lib.LayoutType]":  # type: ignore[name-defined]  # noqa: F821
        """
        Converts an ir.Layout instance into the corresponding catlass layout str
        (RowMajor, ColumnMajor, or None if no matching value is found ).

        Args:
            torch_layout (ir.Layout): The layout that needs to be looked up.

        Returns:
            str: The converted layout corresponding to the `torch_layout` or None if no matching
            value is found.
        """

        if V.graph.sizevars.statically_known_equals(torch_layout.stride[-1], 1):
            return catlass_lib.LayoutType.RowMajor
        elif V.graph.sizevars.statically_known_equals(torch_layout.stride[-2], 1):
            return catlass_lib.LayoutType.ColumnMajor
        else:
            return None

    @staticmethod
    def catlass_elem_type(
        catlass_dtype: str,
        catlass_layout: str,
    ) -> str:
        if catlass_dtype == "void":
            return "void"
        else:
            return f"Gemm::GemmType<{catlass_dtype}, {catlass_layout}>"

    @staticmethod
    def layout_match(
        torch_layout: ir.Layout,
        catlass_layout: "catlass_lib.LayoutType",  # type: ignore[name-defined] # noqa: F821
    ) -> bool:
        """Helper Method: Determines whether a given torch layout matches a given Catlass layout"""
        return CATLASSGemmTemplate.catlass_layout(torch_layout) == catlass_layout

    def render(  # type: ignore[override]
        self,
        kernel: CATLASSTemplateKernel,
        op: "GemmOperation" = None,  # type: ignore[name-defined]  # noqa: F821
        template_buffer_node: Optional[CATLASSTemplateBuffer] = None,
        epilogue_nodes: Optional[List[BaseSchedulerNode]] = None,
        **kwargs,
    ) -> str:
        """
        The primary entry point for the code rendering process used in this template.
        Renders the Catlass based C++ code for the GEMM Kernel that this template is designed to implement,
        including potentially fused epilogues.

        Args:
            kernel (CATLASSTemplateKernel): The kernel to be rendered.
            op (GemmOperation, optional): A GEMM operation that is required to be compatible with the
                input and output definitions as well as a possible epilogue. Defaults to None.
            **kwargs: Additional keyword arguments. Currently unused.

        Returns:
            str: Catlass based C++ code fragment as a string, to be used by the current
            CATLASSTemplateKernel or autotuning code.

        Note:
            All inputs and their corresponding buffer addresses and names take precedence over previously
            passed inputs to the template at construction time. However, they should be layout compatible.
        """
        assert isinstance(
            op, GemmOperation
        ), "op argument is required and has to be an instance of GemmOperation"

        assert len(self.input_nodes) >= 2 and self.output_node is not None
        # check if we can freeze FlexibleLayout
        # since this operation could make the linearization failed
        safe_to_freeze_layout = True
        for input_node in self.input_nodes:
            if (
                not isinstance(input_node.layout, FixedLayout)
                and len(input_node.get_size()) > 2
            ):
                safe_to_freeze_layout = False
                break
        if not safe_to_freeze_layout:
            raise NotImplementedError("Layout is not fixed")

        for input_node in self.input_nodes:
            if not isinstance(input_node.layout, FixedLayout):
                input_node.freeze_layout()
        X, W = self.input_nodes[0], self.input_nodes[1]

        Y = self.output_node
        if template_buffer_node is not None:
            Y = template_buffer_node

        Bias, extra_inputs, extra_names = self._get_extra_inputs_and_names()

        # Define Kernel call signature
        # Important: This step also populates Kernel name to node mapping data structures,
        # which are required further below (for example by the template renderer)
        inputs = [X, W, Bias, *extra_inputs]
        names = ["X", "W", "Bias", *extra_names] + ["Y"]
        names_str = ",".join(names)
        input_reorder = self.input_reorder

        # The layouts might have changed between autotuning and this call if they were FlexibleLayout
        # we need to adapt, which might lead to suboptimal performance.
        op = self.fix_op_layout(op, X, W, Bias, Y)

        relu_enabled = False
        name_to_buffer = {node.get_name(): node for node in self.input_nodes}
        # handle the fake output buffer during lowering
        name_to_buffer[Y.get_name()] = Y  # type: ignore[assignment]
        # Fuse the epilogue nodes on-fly or using Catlass Epilogue Visitor Graph (EVG)
        is_evg_fusion = False
        if epilogue_nodes:
            try:
                (relu_enabled, bias_buffer, output_buffer) = self._try_fast_fusion(
                    epilogue_nodes, Y.get_name()
                )

                from .catlass_library.gemm_autotune import may_adjust_l1_tile_for_bias

                # Add bias for fusion may exceed L1 tile
                # so we will try to adjust l1 tiling before rendering
                may_adjust_l1_tile_for_bias(op)
                Bias = bias_buffer
                Y = output_buffer
                evg_name = ""
                evg_args = ""
                evg_code = ""
                inputs = [X, W, Bias, *extra_inputs]
                outputs = [Y]
            except NotImplementedError as e:
                log.debug(
                    f"Cannot fuse epilogue nodes on-fly, reason: {e}, will use EVG to fuse."
                )

                (
                    input_names,
                    output_names,
                    var_name_to_buffer_name,
                    evg_py_code,
                ) = CatlassEVGCodegen.ir_to_evg_python_code(
                    Y.get_name(), epilogue_nodes, V.kernel.removed_buffers
                )

                for name, buf in (
                    V.graph.name_to_buffer | V.graph.graph_inputs
                ).items():
                    if name not in name_to_buffer:
                        name_to_buffer[name] = buf

                D_output_name = var_name_to_buffer_name["D"]
                D_output_buffer = name_to_buffer[D_output_name]
                Y = D_output_buffer  # type: ignore[assignment]
                # Interestingly, I don't think the rest of the layout matters here since we
                # use the properties of the Y buffer to fill in D's properties in the epilogue
                # args. This is needed though because it defines types expected in the epilogue args.
                op.D.element = catlass_utils.torch_dtype_to_catlass_type(
                    D_output_buffer.get_dtype()
                )
                is_evg_fusion = True

                assert output_names, "There should be at least one write"

                epilogue_inputs = [name_to_buffer[name] for name in input_names]
                output_names.remove(Y.get_name())  # remove duplicated output
                outputs = [name_to_buffer[name] for name in output_names]

                acc_dtype = catlass_utils.get_accumulator_dtype(
                    [X.get_dtype(), W.get_dtype()]
                )
                assert acc_dtype, "Could not determine accumulator dtype"

                evg_name, evg_args, evg_code, evg_arg_renames = self._render_evg(
                    op,
                    evg_py_code,
                    var_name_to_buffer_name,
                    name_to_buffer,
                    Y.get_dtype(),
                    acc_dtype,
                )

                inputs = [
                    X,
                    W,
                    Bias,
                    *epilogue_inputs,  # type: ignore[list-item]
                    Y,
                    *extra_inputs,
                ]
                epilogue_inputs_trans = "\n"
                for name, _ in zip(input_names, epilogue_inputs):
                    epilogue_inputs_trans += f"    uint8_t* {name}_ptr = (uint8_t*)({name});\n"
                evg_code += epilogue_inputs_trans
                names_str = ",".join(
                    ["X", "W", "Bias", *input_names, "Y", *output_names, *extra_names]
                )
        else:
            # no epilogue nodes
            evg_name = ""
            outputs = [Y]
            evg_args = ""
            evg_code = ""

        # to make op mutable without affecting others
        op = copy.deepcopy(op)
        if Bias is not None:
            assert Bias.get_layout().dtype == X.get_layout().dtype
            # This might have been set to void during filtering, when the assumption was still that there's no C
            # operand
            op.C.element = op.A.element
        if is_evg_fusion:
            op.swap_as_evg_kernel()  # use EVG kernel version
            template_buffer_node.is_mix = True  # EVG template is mix template

        kernel_call_signature = kernel.def_kernel(
            inputs=inputs,
            outputs=outputs,
            names_str=names_str,
            input_reorder=input_reorder,
        )
        test_call_statement = self.test_call_statement(kernel, inputs, names_str)

        argument_template, epilogue_template = self._get_template_args(
            op.gemm_kind, has_evg=(evg_name != "")
        )
        # render epilogue arguments
        epilogue_options = dict(op_instance=op, evg_name=evg_name)
        epilogue_arguments = self._template_from_string(epilogue_template).render(
            **epilogue_options
        )
        kernel_arguments, ffts_addr_prepare, kernel_call = (
            self._get_kernel_arguments_and_call(op.gemm_kind)
        )

        options = dict(
            alpha=self.alpha,
            beta=self.beta,
            X=X,
            W=W,
            Y=Y,
            kernel_call_signature=kernel_call_signature,
            Bias=Bias,
            argument_template=argument_template,
            kernel_arguments=kernel_arguments,
            ffts_addr_prepare=ffts_addr_prepare,
            kernel_call=kernel_call,
            epilogue_visitor_graph=evg_code,
            epilogue_visitor_args=evg_args,
            epilogue_arguments=epilogue_arguments,
            template=self,
            kernel=kernel,
            op_instance=op,
            input_reorder=self.input_reorder,
            relu_enabled=relu_enabled,
            test_call_statement=test_call_statement,
            DataTypeTag=catlass_lib.DataTypeTag,
        )
        options.update(dict(zip(extra_names, extra_inputs)))
        res = self._template_from_string(self._get_template(op.gemm_kind)).render(
            **options
        )

        return res

    def fix_op_layout(
        self,
        op: "GemmOperation",  # type: ignore[name-defined] # noqa: F821
        X: Buffer,
        W: Buffer,
        Bias: Optional[Buffer],
        Y: Union[Buffer, ReinterpretView],
    ) -> "GemmOperation":  # type: ignore[name-defined]  # noqa: F821
        # This is a workaround to deal with cases where the input layouts have changed
        # between autotuning and rendering. This happens if the inputs layout
        # are FlexibleLayout instances. In this case, we need to update the
        # op's input layouts. It is a hack, because now the op
        # we benchmarked is not the same as the op we render,
        # but there is no simple way to fix this in the autotuner, since that would
        # potentially disable other optimizations.
        a_layout = X.get_layout()
        b_layout = W.get_layout()
        c_layout = Bias.get_layout() if Bias is not None else None

        d_layout = copy.deepcopy(Y.get_layout())
        match_list = [
            CATLASSGemmTemplate.layout_match(buf.get_layout(), op_layout)
            for buf, op_layout in zip(
                (X, W, Bias, Y),
                (op.A.layout, op.B.layout, op.C.layout, op.D.layout),
            )
            if buf is not None
        ]
        all_match = all(match_list)
        if all_match:
            return op
        log.warning(
            f"Catlass GEMM Layout change: Input and/or output layouts have changed between autotuning/retuning and call to render on {self}. Applying workaround. This can lead to suboptimal performance. Match List: {match_list}"  # noqa: G004, B950
        )
        new_op = copy.deepcopy(op)

        if a_layout is not None:
            new_op.A.layout = CATLASSGemmTemplate.catlass_layout(a_layout)
        if b_layout is not None:
            new_op.B.layout = CATLASSGemmTemplate.catlass_layout(b_layout)
        if c_layout is not None:
            new_op.C.layout = CATLASSGemmTemplate.catlass_layout(c_layout)
            new_op.C.element = catlass_utils.torch_dtype_to_catlass_type(c_layout.dtype)
        if d_layout is not None:
            new_op.D.layout = CATLASSGemmTemplate.catlass_layout(d_layout)
        return new_op

    def test_call_statement(
        self,
        kernel,
        input_nodes,
        names_str: str = "",
    ) -> str:
        """
        Helper method to render the Catlass C++ code required for calling the GEMM operation in the standalone
        test runner that might also be generated along with the rest of the code, if the corresponding config is
        enabled.

        Returns a C++ statement that calls the GEMM operation with the correct arguments.
        """
        _, __, arg_types = kernel.args.cpp_argdefs()
        arg_names = [name.strip() for name in names_str.strip().split(",")]
        if input_nodes[2] is None:
            del arg_names[2]
        arguments = [
            f"(({arg_type}){arg_name}_data.get())"
            for arg_type, arg_name in zip(arg_types, arg_names)
        ]
        return f"{kernel.kernel_name}({', '.join(arguments)}, workspace_size_ptr, (uint8_t*)workspace_data.get(), 0);"

    @staticmethod
    def _try_fast_fusion(
        epilogue_nodes: List[BaseSchedulerNode], template_output_name: str
    ):
        raise NotImplementedError(
            "_try_fast_fusion in CATLASSGemmTemplate not implemented"
        )

    def _render_evg(
        self,
        op: GemmOperation,
        evg_py_code: str,
        buffer_renames: dict[str, str],
        name_to_buffer: dict[str, Buffer],
        output_dtype: torch.dtype,
        accumulator_dtype: torch.dtype,
    ) -> tuple[str, str, str, EVGArgRenames]:  # type: ignore[name-defined]  # noqa: F821
        raise NotImplementedError("_render_evg in CATLASSGemmTemplate not implemented")


class CATLASS1xGemmTemplate(CATLASSGemmTemplate):
    def __init__(
        self,
        input_nodes: List[Buffer],
        layout: Layout,
        alpha: float,
        beta: float,
        input_reorder: Optional[List[int]] = None,
    ):
        super().__init__(input_nodes, layout, alpha, beta, input_reorder)

    @staticmethod
    def add_catlass_gemm_choices(
        choices: List[ChoiceCaller],
        layout: ir.Layout,
        input_nodes: List[Buffer],
        alpha: Union[float, int] = 1,
        beta: Union[float, int] = 0,
        input_reorder: Optional[List[int]] = None,
        **extra_kwargs,
    ) -> None:
        template = CATLASS1xGemmTemplate(
            input_nodes, layout, alpha, beta, input_reorder,
        )
        template._add_catlass_gemm_choices(
            choices, layout, input_nodes, alpha, beta, input_reorder, **extra_kwargs
        )

    def init_templates_map(self) -> None:
        """
        Init the specific template, template args, kernel arguments, and kernel calls.
        If there is no specific process for a gemm kind, it will use the default template.
        """
        self.template_map = {}  # main body template
        self.template_args_map = {}  # template arguments & epilogue
        self.kernel_args_map = {}  # kernel arguments
        self.kernel_calls_map = {}  # kernel calls

        # Gemm template
        self.template_map[catlass_lib.GemmKind.Gemm] = GEMM_TEMPLATE_CATLASS_1X
        self.template_args_map[catlass_lib.GemmKind.Gemm] = (
            GEMM_ARGS_CATLASS_1X,
            GEMM_ARGS_CATLASS_1X_EPILOGUE,
        )

        # Optimized Matmul template
        self.template_map[catlass_lib.GemmKind.OptimizedMatmul] = (
            OPT_MM_TEMPLATE_CATLASS_1X
        )
        self.template_args_map[catlass_lib.GemmKind.OptimizedMatmul] = (
            OPT_MM_ARGS_CATLASS_1X,
            MM_ARGS_CATLASS_1X_VOID_EPILOGUE,
        )

        # Tla Template
        gemm_kinds = [
            catlass_lib.GemmKind.BasicMatmulTla,
            catlass_lib.GemmKind.BasicMatmulTlaVisitor,  # EVG version of BasicMatmulTla
            catlass_lib.GemmKind.BatchedMatmulTla,
            catlass_lib.GemmKind.StreamkMatmulTla,
            catlass_lib.GemmKind.MultiCoreSplitkMatmulTla,
            catlass_lib.GemmKind.TailMultiCoreSplitkMatmulTla,
        ]
        template_val = TLA_MM_TEMPLATE_CATLASS_1X
        template_args_val = (
            TLA_MM_ARGS_CATLASS_1X,
            MM_ARGS_CATLASS_1X_VOID_EPILOGUE,
        )
        self.template_map.update({kind: template_val for kind in gemm_kinds})
        self.template_args_map.update({kind: template_args_val for kind in gemm_kinds})

        # Group MM Tla Template
        gemm_kind = catlass_lib.GemmKind.GroupedMatmulSliceMTla
        template_val = TLA_GROUP_MM_TEMPLATE_CATLASS_1X
        template_args_val = (
            TLA_MM_ARGS_CATLASS_1X,
            MM_ARGS_CATLASS_1X_VOID_EPILOGUE,
        )
        self.template_map.update({gemm_kind: template_val})
        self.template_args_map.update({gemm_kind: template_args_val})

        # Kernel arguments & calls
        self.kernel_args_map[catlass_lib.GemmKind.BasicMatmul] = (
            MM_KERNEL_ARGUMENTS_CATLASS_1X
        )
        self.kernel_args_map[catlass_lib.GemmKind.BatchedMatmul] = (
            BMM_KERNEL_ARGUMENTS_CATLASS_1X
        )
        self.kernel_args_map[catlass_lib.GemmKind.MatmulBias] = (
            MMBIAS_KERNEL_ARGUMENTS_CATLASS_1X
        )
        self.kernel_args_map[catlass_lib.GemmKind.Gemm] = (
            GEMM_KERNEL_ARGUMENTS_CATLASS_1X
        )
        self.kernel_args_map[catlass_lib.GemmKind.BasicMatmulTla] = (
            TLA_MM_KERNEL_ARGUMENTS_CATLASS_1X
        )
        self.kernel_args_map[catlass_lib.GemmKind.BasicMatmulTlaVisitor] = (
            TLA_MM_EVG_KERNEL_ARGUMENTS_CATLASS_1X
        )
        self.kernel_args_map[catlass_lib.GemmKind.GroupedMatmulSliceMTla] = (
            TLA_GROUP_MM_KERNEL_ARGUMENTS_CATLASS_1X
        )
        gemm_kinds = [
            catlass_lib.GemmKind.StreamkMatmulTla,
            catlass_lib.GemmKind.MultiCoreSplitkMatmulTla,
            catlass_lib.GemmKind.TailMultiCoreSplitkMatmulTla,
        ]
        kernel_args_val = TLA_MM_KERNEL_WITH_CORE_NUM_ARGUMENTS_CATLASS_1X
        self.kernel_args_map.update({kind: kernel_args_val for kind in gemm_kinds})
        self.kernel_args_map[catlass_lib.GemmKind.BatchedMatmulTla] = (
            TLA_BMM_KERNEL_ARGUMENTS_CATLASS_1X
        )
        # only Gemm and OptimizedMatmul need ffts
        gemm_kinds = [
            catlass_lib.GemmKind.Gemm,
            catlass_lib.GemmKind.OptimizedMatmul,
        ]
        kernel_calls_val = (FFTS_ADDR_PREPARE, MM_FFTS_KERNEL_CALL_CATLASS_1X)
        self.kernel_calls_map.update({kind: kernel_calls_val for kind in gemm_kinds})

    def _is_op_kind_supported(
        self,
        op_kind: "catlass_lib.GemmKind",  # type: ignore[name-defined]  # noqa: F821
    ) -> bool:
        if self.is_group_mm:
            return op_kind == catlass_lib.GemmKind.GroupedMatmulSliceMTla
        # not support D = (A @ B) * alpha + beta * C, where C's shape is (M, N)
        if not self._is_standard_matmul() or self.bias_shape == BiasShape.MN_BIAS:
            return False

        # not support D = (A @ B) + C, where A @ B is a bmm
        if self.is_batchmm and self.bias_shape != BiasShape.NO_BIAS:
            return False

        if self.is_batchmm:
            # A5 version
            return op_kind == catlass_lib.GemmKind.BatchedMatmulTla

        # simple matmul: A @ B
        supported_kinds = {
            catlass_lib.GemmKind.BasicMatmul,
            catlass_lib.GemmKind.OptimizedMatmul,
            catlass_lib.GemmKind.BasicMatmulTla,  # support mm and mm+bias
            catlass_lib.GemmKind.StreamkMatmulTla,
            catlass_lib.GemmKind.MultiCoreSplitkMatmulTla,
            catlass_lib.GemmKind.TailMultiCoreSplitkMatmulTla,
        }
        if not catlass_config.catlass_ignore_gemm_in_standard_mm:
            supported_kinds.add(catlass_lib.GemmKind.Gemm)
        return op_kind in supported_kinds

    def _get_extra_inputs_and_names(
        self,
        op: "GemmOperation" = None,  # type: ignore[name-defined]  # noqa: F821
    ) -> Tuple[Optional[Buffer], List[Optional[Buffer]], List[str]]:
        Bias = None if (len(self.input_nodes) == 2 or self.is_group_mm) else self.input_nodes[2]
        if self.is_group_mm:
            inputs: List[Optional[Buffer]] = [self.input_nodes[2]]
            names: List[str] = ["offsets"]
        else:
            inputs: List[Optional[Buffer]] = []
            names: List[str] = []
        return (Bias, inputs, names)

    def _shape_match(
        self,
        op: "GemmOperation",  # type: ignore[name-defined]  # noqa: F821
    ) -> bool:
        X, W = self.input_nodes[0], self.input_nodes[1]
        return X.get_size()[-1] == W.get_size()[-2]

    def _set_bias_layout(
        self,
        op: "GemmOperation",  # type: ignore[name-defined]  # noqa: F821
    ) -> bool:
        if not self.is_group_mm and len(self.input_nodes) >= 3 and self.input_nodes[2] is not None:
            Bias = self.input_nodes[2]
            bias_layout = CATLASSGemmTemplate.catlass_layout(Bias.get_layout())
            if bias_layout != op.D.layout:
                # bias and output layout must match
                return False
            op.C.layout = bias_layout
        else:
            op.C.element = catlass_lib.DataType.void
            op.C.layout = op.D.layout
        return True

    def _get_template(self, op_kind: "catlass_lib.GemmKind") -> str:
        return self.template_map.get(op_kind, MM_TEMPLATE_CATLASS_1X)

    def _get_template_args(
        self, op_kind: "catlass_lib.GemmKind", has_evg: bool = False
    ) -> Tuple[str, str]:
        args, epilogue_args = self.template_args_map.get(
            op_kind, (MM_ARGS_CATLASS_1X, MM_ARGS_CATLASS_1X_VOID_EPILOGUE)
        )
        return (
            (args, MM_ARGS_CATLASS_EVG_EPILOGUE) if has_evg else (args, epilogue_args)
        )

    def _get_kernel_arguments_and_call(
        self, op_kind: "catlass_lib.GemmKind"
    ) -> Tuple[str, str, str]:
        KERNEL_ARGUMENTS = self.kernel_args_map.get(
            op_kind, MM_KERNEL_ARGUMENTS_CATLASS_1X
        )
        KERNEL_CALLS = self.kernel_calls_map.get(
            op_kind, ("", MM_KERNEL_CALL_CATLASS_1X)
        )
        return (KERNEL_ARGUMENTS, *KERNEL_CALLS)

    @staticmethod
    def is_mixed_template(op: "GemmOperation") -> bool:
        return op.gemm_kind in {
            catlass_lib.GemmKind.StreamkMatmulTla,
            catlass_lib.GemmKind.MultiCoreSplitkMatmulTla,
            catlass_lib.GemmKind.TailMultiCoreSplitkMatmulTla,
            catlass_lib.GemmKind.BasicMatmulTlaVisitor,
        }

    @staticmethod
    def epilogue_fusion_type(op: "GemmOperation") -> int:
        fusion_type = 0
        if op.gemm_kind in {
            catlass_lib.GemmKind.BasicMatmulTla,
            catlass_lib.GemmKind.GroupedMatmulSliceMTla,
        }:
            # support fast fusion
            fusion_type = 1
        if op.gemm_kind == catlass_lib.GemmKind.BasicMatmulTla:
            # support EVG fusion
            fusion_type = 2

        return fusion_type

    def _are_inputs_layout_compatible(self, layouts: List[Layout]) -> bool:
        """
        Evaluates whether input layouts are compatible for set of operations supported by this class.

        Args:
            layouts (List[Layout]): List containing Layout objects representing
                                    the input matrices

        Returns:
            bool: True if layouts are GEMM compatible, otherwise False.
        """
        assert len(layouts) == 2 or len(layouts) == 3
        # Check if A and B are compatible
        A_layout, B_layout = layouts[:2]
        if len(A_layout.size) < 1:
            return False
        if len(B_layout.size) < 1:
            return False
        A_size = list(V.graph.sizevars.size_hints(A_layout.size))
        B_size = list(V.graph.sizevars.size_hints(B_layout.size))
        if len(A_size) < 2:
            A_size.insert(0, 1)
        if len(B_size) < 2:
            A_size.insert(1, 1)
        # Are batch dims broadcastable?
        while len(A_size) < len(B_size):
            A_size.insert(0, 1)
        while len(B_size) < len(A_size):
            B_size.insert(0, 1)
        K = max(A_size[-1], B_size[-2])
        M = A_size[-2]
        N = B_size[-1]
        if K != A_size[-1] and A_size[-1] != 1:
            return False
        if K != B_size[-2] and B_size[-1] != 1:
            return False
        # check batch dim broadcastable
        for i in range(len(A_size) - 2):
            if A_size[i] != B_size[i] and A_size[i] != 1 and B_size[i] != 1:
                return False
        if len(layouts) == 3:
            C_layout = layouts[2]
            C_size = [int(i) for i in C_layout.size]
            while len(C_size) < len(A_size):
                C_size.insert(0, 1)
            # check batch dims
            for i in range(len(A_size) - 2):
                bd = max(A_size[i], B_size[i])
                if bd != C_size[i] and C_size[i] != 1:
                    return False
            if len(C_size) > len(A_size):
                # This may happen if the last elements of C are contiguous and
                # their multiplied size equals the last dim size of B
                if M != C_size[len(A_size) - 2] and C_size[len(A_size) - 2] != 1:
                    return False
                remaining_size = 1
                for i in range(len(A_size) - 1, len(C_size)):
                    remaining_size *= C_size[i]
                if N != remaining_size and remaining_size != 1:
                    return False
                return True
            assert len(C_size) == len(A_size)
            if M != C_size[-2] and C_size[-2] != 1:
                return False
            if N != C_size[-1] and C_size[-1] != 1:
                return False
        return True

    @staticmethod
    def _try_fast_fusion(
        epilogue_nodes: List[BaseSchedulerNode], template_output_name: str
    ):
        if len(epilogue_nodes) > 1:
            raise NotImplementedError(
                "Do not support more than one epilogue nodes for fast-fusion."
            )

        node = epilogue_nodes[0]
        cb = node.node
        assert isinstance(cb, ComputedBuffer)
        pw = cb.data
        assert isinstance(pw, Pointwise)
        op_count_res = pw.inner_fn_opcount()
        used_ops = set(op_count_res.used_ops)
        used_ops.discard("load")
        num_ops = op_count_res.num_ops - len(op_count_res.read_buffers)
        supported_on_fly_ops = {"add", "relu"}
        if len(used_ops.difference(supported_on_fly_ops)) != 0:
            raise NotImplementedError(
                "There are ops that are not supported for fast-fusion."
            )
        if len(used_ops) > 2 or len(used_ops) != num_ops:
            raise NotImplementedError(
                "Do not support more than one add or relu for fast-fusion."
            )

        # Only support biasAdd, Relu, biasAdd+Relu on fly
        bias_buffer = None
        relu_enabled = False

        if "add" in used_ops:
            read_names = list(pw.get_read_names())
            for name in read_names:
                if name == template_output_name:
                    continue
                if name in V.graph.name_to_buffer:
                    buf = V.graph.name_to_buffer.get(name)
                elif name in V.graph.graph_inputs:
                    buf = V.graph.graph_inputs.get(name)
                else:
                    raise KeyError(
                        f"Cound not resolve buffer for name {name} (maybe removed)."
                    )

                buf_stride = buf.get_layout().stride
                if len(buf_stride) > 1 or not V.graph.sizevars.statically_known_equals(
                    buf_stride[0], 1
                ):
                    raise NotImplementedError("Do not support matrix-Add for fast-fusion.")
                assert bias_buffer is None
                bias_buffer = buf
        if "relu" in used_ops:
            relu_enabled = True

        output_name = cb.get_name()
        output_buffer = V.graph.name_to_buffer.get(output_name)
        return (relu_enabled, bias_buffer, output_buffer)

    def _render_evg(
        self,
        op: GemmOperation,
        evg_py_code: str,
        var_name_to_buffer_name: dict[str, str],
        name_to_buffer: dict[str, Buffer],
        output_dtype: torch.dtype,
        accumulator_dtype: torch.dtype,
    ) -> tuple[str, str, str, EVGArgRenames]:  # type: ignore[name-defined]  # noqa: F821
        from .catlass_library.evg_extension import create_example_tensors, trace

        acc_dtype = torch_dtype_to_catlass_type(accumulator_dtype)
        output_dtype = torch_dtype_to_catlass_type(output_dtype)

        examples = create_example_tensors(
            var_name_to_buffer_name,
            name_to_buffer,  # type: ignore[arg-type]
            V.graph.sizevars.size_hint,
        )
        evg_name, evg_args, evg_code, arg_renames = trace(
            evg_py_code,
            examples,
            acc_dtype,
            output_dtype,
            op.tile_description,  # type: ignore[attr-defined]
            {k: name_to_buffer[v] for k, v in var_name_to_buffer_name.items()},  # type: ignore[attr-type,misc]
            V.graph.sizevars.size_hint,
        )

        return (
            evg_name,
            evg_args,
            evg_code,
            arg_renames,
        )

    def render_gemm_arguments(
        self,
        op: "GemmOperation",
        argument_template: str,
        X: IRNode,
        W: IRNode,
        Bias: IRNode,
        Y: IRNode,
        alpha: float,
        beta: float,
        relu_enabled: bool,
        kernel: CATLASSTemplateKernel,
    ) -> str:
        """
        Render the Catlass C++ code required for rendering Gemm operation.

        Args:
            op (GemmOperation): GemmOperation instance.
            argument_template (str): Template for the GEMM operation arguments.
            epilogue_template (str): Template for the epilogue arguments.
            X (IRNode): The X input tensor.
            W (IRNode): The W input tensor.
            Bias (IRNode): The Bias input tensor.
            Y (IRNode): The output tensor.
            alpha (float): Scaling factor for the product of the inputs
            beta (float): Scaling factor for the output tensor.
            kernel (CATLASSTemplateKernel): NPU Template kernel for the operation.

        Returns:
            str: A block of Catlass C++ code as a string, ready to be used as arguments for the GEMM operation.
        """
        _relu_enabled = "true" if relu_enabled else "false"
        options = dict(
            op_instance=op,
            alpha=alpha,
            beta=beta,
            relu_enabled=_relu_enabled,
            X=X,
            W=W,
            Y=Y,
            Bias=Bias,
            template=self,
            kernel=kernel,
            M="M",
            N="N",
            K="K",
        )

        arguments = self._template_from_string(argument_template).render(
            **options,
        )

        return arguments
