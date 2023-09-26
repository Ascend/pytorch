FILE(GLOB _ATEN_SRCS
    *.cpp
    common/*.cpp
    mirror/*.cpp
    ops/hcom/*.cpp)

LIST(APPEND _ATEN_SRCS
${TORCHNPU_ROOT}/aten/ops/ViewKernelNpu.cpp
${TORCHNPU_ROOT}/aten/ops/DefaultKernelNpu.cpp
${TORCHNPU_ROOT}/aten/ops/_Reshape_AliasKernelNpu.cpp
${TORCHNPU_ROOT}/aten/ops/AmpForeachNonFiniteCheckAndUnscaleKernelNpu.cpp
${TORCHNPU_ROOT}/aten/ops/FastGeluKernelNpu.cpp
${TORCHNPU_ROOT}/aten/ops/RotaryMulKernelNpu.cpp
${TORCHNPU_ROOT}/aten/ops/MultiHeadAttentionKernelNpu.cpp
${TORCHNPU_ROOT}/aten/ops/GruKernelNpu.cpp
${TORCHNPU_ROOT}/aten/ops/LstmKernelNpu.cpp
${TORCHNPU_ROOT}/aten/ops/LstmCellKernelNpu.cpp
${TORCHNPU_ROOT}/aten/ops/ConfusionTransposeKernelNpu.cpp
${TORCHNPU_ROOT}/aten/ops/MaxV1KernelNpu.cpp
${TORCHNPU_ROOT}/aten/ops/MinV1KernelNpu.cpp
${TORCHNPU_ROOT}/aten/ops/PsRoiPoolingKernelNpu.cpp
${TORCHNPU_ROOT}/aten/ops/HasCompatibleShallowCopyType.cpp
${TORCHNPU_ROOT}/aten/ops/CastKernelNpu.cpp
${TORCHNPU_ROOT}/aten/ops/_AminmaxKernelNpu.cpp
${TORCHNPU_ROOT}/aten/ops/MinKernelNpu.cpp
${TORCHNPU_ROOT}/aten/ops/ArgmaxKernelNpu.cpp
${TORCHNPU_ROOT}/aten/ops/ArgminKernelNpu.cpp
${TORCHNPU_ROOT}/aten/ops/NanToNumKernelNpu.cpp
${TORCHNPU_ROOT}/aten/ops/EmbeddingBagDenseBackwardKernelNpu.cpp
${TORCHNPU_ROOT}/aten/ops/OnesLikeKernelNpu.cpp
${TORCHNPU_ROOT}/aten/ops/BroadcastKernelNpu.cpp
${TORCHNPU_ROOT}/aten/ops/FloatStatusKernelNpu.cpp
${TORCHNPU_ROOT}/aten/ops/CheckerKernelNpu.cpp
${TORCHNPU_ROOT}/aten/ops/SumKernelNpu.cpp
${TORCHNPU_ROOT}/aten/ops/MaxKernelNpu.cpp
${TORCHNPU_ROOT}/aten/ops/MinKernelNpu.cpp
${TORCHNPU_ROOT}/aten/ops/ConstantPadNdKernelNpu.cpp
${TORCHNPU_ROOT}/aten/ops/ReshapeKernelNpu.cpp
${TORCHNPU_ROOT}/aten/ops/ScatterV1KernelNpu.cpp
${TORCHNPU_ROOT}/aten/ops/SliceKernelNpu.cpp
${TORCHNPU_ROOT}/aten/ops/GatherKernelNpu.cpp
${TORCHNPU_ROOT}/aten/ops/TransposeKernelNpu.cpp
${TORCHNPU_ROOT}/aten/ops/AsStridedKernelNpu.cpp
${TORCHNPU_ROOT}/aten/ops/BmmV2KernelNpu.cpp
${TORCHNPU_ROOT}/aten/ops/IndexingKernelNpu.cpp
${TORCHNPU_ROOT}/aten/ops/StackKernelNpu.cpp
${TORCHNPU_ROOT}/aten/ops/OnesKernelNpu.cpp
${TORCHNPU_ROOT}/aten/ops/MatmulKernelNpu.cpp
${TORCHNPU_ROOT}/aten/ops/pooling/PoolingKernelNpu.cpp
)
LIST(APPEND _ATEN_SRCS
${TORCHNPU_ROOT}/aten/ops/op_api/op_api_common.cpp
${TORCHNPU_ROOT}/aten/ops/op_api/AminmaxKernelNpuOpApi.cpp
${TORCHNPU_ROOT}/aten/ops/op_api/TensorFactoriesOpApi.cpp
${TORCHNPU_ROOT}/aten/ops/op_api/CopyKernelOpApi.cpp
${TORCHNPU_ROOT}/aten/ops/op_api/TensorPropertiesOpApi.cpp
${TORCHNPU_ROOT}/aten/ops/op_api/ToKernelNpuOpApi.cpp
${TORCHNPU_ROOT}/aten/ops/op_api/FlashAttentionKernelNpuOpApi.cpp
${TORCHNPU_ROOT}/aten/ops/op_api/ArgminKernelNpuOpApi.cpp
${TORCHNPU_ROOT}/aten/ops/op_api/ArgmaxKernelNpuOpApi.cpp
${TORCHNPU_ROOT}/aten/ops/op_api/CastKernelNpuOpApi.cpp
${TORCHNPU_ROOT}/aten/ops/op_api/MultiHeadAttentionKernelNpuOpApi.cpp
${TORCHNPU_ROOT}/aten/ops/op_api/MatmulKernelNpuOpApi.cpp
)

FILE(GLOB _EXCLUDE
    VariableTypeEverything.cpp
    ADInplaceOrViewTypeEverything.cpp)

FOREACH(ITEM ${_EXCLUDE})
    LIST(REMOVE_ITEM _ATEN_SRCS ${ITEM})
ENDFOREACH()

LIST(APPEND ATEN_SRCS ${_ATEN_SRCS})

# Pass to parent
set(ATEN_SRCS ${ATEN_SRCS} PARENT_SCOPE)

if(DEFINED ENV{BSCPP_OPS_ENABLE})
  add_subdirectory(bscpp_ops)
endif()
