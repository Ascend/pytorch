import enum

from enum import auto as enum_auto


class DataType(enum.Enum):
    void = enum_auto()  # primary used to disable C tensor for epilogues
    u8 = enum_auto()
    u16 = enum_auto()
    u32 = enum_auto()
    u64 = enum_auto()
    s8 = enum_auto()
    s16 = enum_auto()
    s32 = enum_auto()
    s64 = enum_auto()
    f16 = enum_auto()
    bf16 = enum_auto()
    f32 = enum_auto()
    f64 = enum_auto()
    invalid = enum_auto()


#
DataTypeNames = {
    DataType.void: "void",
    DataType.u8: "u8",
    DataType.u16: "u16",
    DataType.u32: "u32",
    DataType.u64: "u64",
    DataType.s8: "s8",
    DataType.s16: "s16",
    DataType.s32: "s32",
    DataType.s64: "s64",
    DataType.f16: "f16",
    DataType.bf16: "bf16",
    DataType.f32: "f32",
    DataType.f64: "f64",
}


DataTypeTag = {
    DataType.void: "void",
    DataType.u8: "uint8_t",
    DataType.u16: "uint16_t",
    DataType.u32: "uint32_t",
    DataType.u64: "uint64_t",
    DataType.s8: "int8_t",
    DataType.s16: "int16_t",
    DataType.s32: "int32_t",
    DataType.s64: "int64_t",
    DataType.f16: "half",
    DataType.bf16: "bfloat16_t",
    DataType.f32: "float",
    DataType.f64: "double",
}


DataTypeSize = {
    DataType.void: 0,
    DataType.u8: 8,
    DataType.u16: 16,
    DataType.u32: 32,
    DataType.u64: 64,
    DataType.s8: 8,
    DataType.s16: 16,
    DataType.s32: 32,
    DataType.s64: 64,
    DataType.f16: 16,
    DataType.bf16: 16,
    DataType.f32: 32,
    DataType.f64: 64,
}


class LayoutType(enum.Enum):
    ColumnMajor = enum_auto()
    RowMajor = enum_auto()
    VectorLayout = enum_auto()


#
LayoutTag = {
    LayoutType.ColumnMajor: "Catlass::layout::ColumnMajor",
    LayoutType.RowMajor: "Catlass::layout::RowMajor",
    LayoutType.VectorLayout: "Catlass::layout::VectorLayout",
}


#
ShortLayoutTypeNames = {
    LayoutType.ColumnMajor: "n",
    LayoutType.RowMajor: "t",
    LayoutType.VectorLayout: "v",
}


class DispatchPolicyType(enum.Enum):
    MmadA2 = enum_auto()
    MmadA2Async = enum_auto()

    # Matmul
    MmadA2PingPongDUnitFlag = enum_auto()
    MmadA2PingPongEUnitFlag = enum_auto()

    # Optimized Matmul
    MmadA2PreloadDUnitFlagDSuffleK = enum_auto()
    MmadA2PreloadEUnitFlagESuffleK = enum_auto()
    MmadA2PreloadEUnitFlagDSuffleK = enum_auto()
    MmadA2PreloadDUnitFlagESuffleK = enum_auto()

    # MatmulBias
    MmadA2PingPongBiasDUnitFlag = enum_auto()
    MmadA2PingPongBiasEUnitFlag = enum_auto()

    # GEMM
    GemmA2EUnitFlagDShuffleKDABBA = enum_auto()
    GemmA2EUnitFlagEShuffleKDABBA = enum_auto()
    GemmA2EUnitFlagEShuffleKEABBA = enum_auto()

    # A5
    MmadPingpongDUnitFlag = enum_auto()
    MmadPingpongEUnitFlag = enum_auto()
    # disable shuffleK can get better performance in A5 (which is a little weird)
    MmadPreloadAsyncWithCallback_12221EUnitFlagDShuffleK = enum_auto()
    MmadPreloadAsyncWithCallback_12221EUnitFlagEShuffleK = enum_auto()
    MmadMultiBatch = enum_auto()


DispatchPolicyTag = {
    DispatchPolicyType.MmadA2: "Catlass::Gemm::MmadAtlasA2",
    DispatchPolicyType.MmadA2Async: "Catlass::Gemm:MmadAtlasA2Async",
    # Matmul
    DispatchPolicyType.MmadA2PingPongDUnitFlag: "Catlass::Gemm::MmadAtlasA2Pingpong<false>",
    DispatchPolicyType.MmadA2PingPongEUnitFlag: "Catlass::Gemm::MmadAtlasA2Pingpong<true>",
    DispatchPolicyType.MmadA2PreloadDUnitFlagDSuffleK: "Catlass::Gemm::MmadAtlasA2Preload<false, false>",
    DispatchPolicyType.MmadA2PreloadEUnitFlagESuffleK: "Catlass::Gemm::MmadAtlasA2Preload<true, true>",
    DispatchPolicyType.MmadA2PreloadEUnitFlagDSuffleK: "Catlass::Gemm::MmadAtlasA2Preload<true, false>",
    DispatchPolicyType.MmadA2PreloadDUnitFlagESuffleK: "Catlass::Gemm::MmadAtlasA2Preload<false, true>",
    # MatmulBias
    DispatchPolicyType.MmadA2PingPongBiasDUnitFlag: "Catlass::Gemm::MmadAtlasA2PingpongBias<false>",
    DispatchPolicyType.MmadA2PingPongBiasEUnitFlag: "Catlass::Gemm::MmadAtlasA2PingpongBias<true>",
    # GEMM
    DispatchPolicyType.GemmA2EUnitFlagDShuffleKDABBA: "Catlass::Gemm::GemmAtlasA2<true, false, false>",
    DispatchPolicyType.GemmA2EUnitFlagEShuffleKDABBA: "Catlass::Gemm::GemmAtlasA2<true, true, false>",
    DispatchPolicyType.GemmA2EUnitFlagEShuffleKEABBA: "Catlass::Gemm::GemmAtlasA2<true, true, true>",
    # A5
    DispatchPolicyType.MmadPingpongDUnitFlag: "Catlass::Gemm::MmadPingpong<ArchTag, false, HF32_MODE>",
    DispatchPolicyType.MmadPingpongEUnitFlag: "Catlass::Gemm::MmadPingpong<ArchTag, true, HF32_MODE>",
    DispatchPolicyType.MmadPreloadAsyncWithCallback_12221EUnitFlagDShuffleK: "Catlass::Gemm::MmadPreloadAsyncWithCallback<ArchTag, 1, 2, 2, 2, 1, true, false>",
    DispatchPolicyType.MmadPreloadAsyncWithCallback_12221EUnitFlagEShuffleK: "Catlass::Gemm::MmadPreloadAsyncWithCallback<ArchTag, 1, 2, 2, 2, 1, true, true>",
    # BatchedMatmul
    DispatchPolicyType.MmadMultiBatch: "Catlass::Gemm::MmadMultiBatch<ArchTag, HF32_MODE>",
}

ShortDispatchPolicyNames = {
    DispatchPolicyType.MmadA2: "dp-mmada2",
    DispatchPolicyType.MmadA2Async: "dp-mmada2-async",
    # Matmul
    DispatchPolicyType.MmadA2PingPongDUnitFlag: "dp-mmada2-pp-f",
    DispatchPolicyType.MmadA2PingPongEUnitFlag: "dp-mmada2-pp-t",
    DispatchPolicyType.MmadA2PreloadDUnitFlagDSuffleK: "dp-mmada2-pl-ff",
    DispatchPolicyType.MmadA2PreloadDUnitFlagESuffleK: "dp-mmada2-pl-ft",
    DispatchPolicyType.MmadA2PreloadEUnitFlagESuffleK: "dp-mmada2-pl-tt",
    DispatchPolicyType.MmadA2PreloadEUnitFlagDSuffleK: "dp-mmada2-pl-tf",
    # MatmulBias
    DispatchPolicyType.MmadA2PingPongBiasDUnitFlag: "dp-mmada2-ppb-f",
    DispatchPolicyType.MmadA2PingPongBiasEUnitFlag: "dp-mmada2-ppb-t",
    # GEMM
    DispatchPolicyType.GemmA2EUnitFlagDShuffleKDABBA: "dp-gemm-tff",
    DispatchPolicyType.GemmA2EUnitFlagEShuffleKDABBA: "dp-gemm-ttf",
    DispatchPolicyType.GemmA2EUnitFlagEShuffleKEABBA: "dp-gemm-ttt",
    # A5
    DispatchPolicyType.MmadPingpongDUnitFlag: "dp-mmad-pp-f",
    DispatchPolicyType.MmadPingpongEUnitFlag: "dp-mmad-pp-t",
    DispatchPolicyType.MmadPreloadAsyncWithCallback_12221EUnitFlagDShuffleK: "dp-mmad-pl-12221tf",
    DispatchPolicyType.MmadPreloadAsyncWithCallback_12221EUnitFlagEShuffleK: "dp-mmad-pl-12221tt",
    DispatchPolicyType.MmadMultiBatch: "dp-mmad-mb",
}


class BlockSwizzle(enum.Enum):
    GemmIdentityBlockSwizzle_30 = enum_auto()
    GemmIdentityBlockSwizzle_31 = enum_auto()
    GemmIdentityBlockSwizzle_40 = enum_auto()
    GemmIdentityBlockSwizzle_41 = enum_auto()
    StreamkGemmIdentityBlockSwizzle_30 = enum_auto()
    StreamkGemmIdentityBlockSwizzle_31 = enum_auto()
    SplitkGemmIdentityBlockSwizzle_30 = enum_auto()
    SplitkGemmIdentityBlockSwizzle_31 = enum_auto()
    TailSplitkGemmIdentityBlockSwizzle_30 = enum_auto()
    TailSplitkGemmIdentityBlockSwizzle_31 = enum_auto()


BlockSwizzleTag = {
    BlockSwizzle.GemmIdentityBlockSwizzle_30: "Catlass::Gemm::Block::GemmIdentityBlockSwizzle<3, 0>",
    BlockSwizzle.GemmIdentityBlockSwizzle_31: "Catlass::Gemm::Block::GemmIdentityBlockSwizzle<3, 1>",
    BlockSwizzle.GemmIdentityBlockSwizzle_40: "Catlass::Gemm::Block::GemmIdentityBlockSwizzle<4, 0>",
    BlockSwizzle.GemmIdentityBlockSwizzle_41: "Catlass::Gemm::Block::GemmIdentityBlockSwizzle<4, 1>",
    BlockSwizzle.StreamkGemmIdentityBlockSwizzle_30: "Catlass::Gemm::Block::StreamkGemmIdentityBlockSwizzle<3, 0>",
    BlockSwizzle.StreamkGemmIdentityBlockSwizzle_31: "Catlass::Gemm::Block::StreamkGemmIdentityBlockSwizzle<3, 1>",
    BlockSwizzle.SplitkGemmIdentityBlockSwizzle_30: "Catlass::Gemm::Block::SplitkGemmIdentityBlockSwizzle<3, 0>",
    BlockSwizzle.SplitkGemmIdentityBlockSwizzle_31: "Catlass::Gemm::Block::SplitkGemmIdentityBlockSwizzle<3, 1>",
    BlockSwizzle.TailSplitkGemmIdentityBlockSwizzle_30: "Catlass::Gemm::Block::TailSplitkGemmIdentityBlockSwizzle<3, 0>",
    BlockSwizzle.TailSplitkGemmIdentityBlockSwizzle_31: "Catlass::Gemm::Block::TailSplitkGemmIdentityBlockSwizzle<3, 1>",
}


ShortBlockSwizzleNames = {
    BlockSwizzle.GemmIdentityBlockSwizzle_30: "idbs30",
    BlockSwizzle.GemmIdentityBlockSwizzle_31: "idbs31",
    BlockSwizzle.GemmIdentityBlockSwizzle_40: "idbs40",
    BlockSwizzle.GemmIdentityBlockSwizzle_41: "idbs41",
    BlockSwizzle.StreamkGemmIdentityBlockSwizzle_30: "skidbs30",
    BlockSwizzle.StreamkGemmIdentityBlockSwizzle_31: "skidbs31",
    BlockSwizzle.SplitkGemmIdentityBlockSwizzle_30: "splitkidbs30",
    BlockSwizzle.SplitkGemmIdentityBlockSwizzle_31: "splitkidbs31",
    BlockSwizzle.TailSplitkGemmIdentityBlockSwizzle_30: "tsplitkidbs30",
    BlockSwizzle.TailSplitkGemmIdentityBlockSwizzle_31: "tsplitkidbs31",
}


class GemmKind(enum.Enum):
    # Standard Matmul
    BasicMatmul = enum_auto()
    OptimizedMatmul = enum_auto()
    BatchedMatmul = enum_auto()
    MatmulBias = enum_auto()
    BasicMatmulTla = enum_auto()
    BatchedMatmulTla = enum_auto()
    StreamkMatmulTla = enum_auto()
    MultiCoreSplitkMatmulTla = enum_auto()
    TailMultiCoreSplitkMatmulTla = enum_auto()
    # EVG supported kernels
    BasicMatmulTlaVisitor = enum_auto()  # EVG version of BasicMatmulTla

    # GEMM
    Gemm = enum_auto()
    Group = enum_auto()

    # Grouped Matmul
    GroupedMatmulSliceMTla = enum_auto()


GemmKindNames = {
    GemmKind.BasicMatmul: "BasicMatmul",
    GemmKind.OptimizedMatmul: "OptimizedMatmul",
    GemmKind.BatchedMatmul: "BatchedMatmul",
    GemmKind.MatmulBias: "MatmulBias",
    GemmKind.BasicMatmulTla: "BasicMatmulTla",
    GemmKind.BatchedMatmulTla: "BatchedMatmulTla",
    GemmKind.StreamkMatmulTla: "StreamkMatmulTla",
    GemmKind.MultiCoreSplitkMatmulTla: "MultiCoreSplitkMatmulTla",
    GemmKind.TailMultiCoreSplitkMatmulTla: "TailMultiCoreSplitkMatmulTla",
    GemmKind.BasicMatmulTlaVisitor: "BasicMatmulTlaVisitor",
    GemmKind.Gemm: "KernelGemm",
    GemmKind.Group: "KernelGroupGemm",
    GemmKind.GroupedMatmulSliceMTla: "GroupedMatmulSliceMTla",
}


EVGGemmKindMap = {
    GemmKind.BasicMatmulTla: GemmKind.BasicMatmulTlaVisitor,
}


class ArchType(enum.Enum):
    A2 = enum_auto()
    A5 = enum_auto()


ArchTypeTag = {
    ArchType.A2: "Arch::AtlasA2",
    ArchType.A5: "Arch::AtlasA5",
}


ArchTypeNames = {
    ArchType.A2: "A2",
    ArchType.A5: "A5",
}


class TensorDescription:
    def __init__(self, element, layout):
        self.element = element
        self.layout = layout


class TileDescription:
    def __init__(self, L1TileShape, L0TileShape):
        self.l1_tile_shape = list(L1TileShape)
        self.l0_tile_shape = list(L0TileShape)

    @property
    def l1_m(self):
        return self.l1_tile_shape[0]

    @property
    def l1_n(self):
        return self.l1_tile_shape[1]

    @property
    def l1_k(self):
        return self.l1_tile_shape[2]

    @property
    def l0_m(self):
        return self.l0_tile_shape[0]

    @property
    def l0_n(self):
        return self.l0_tile_shape[1]

    @property
    def l0_k(self):
        return self.l0_tile_shape[2]

    def set_l1_tile(self, new_l1_tile):
        self.l1_tile_shape = list(new_l1_tile)
        self.l0_tile_shape[0] = self.l1_m
        self.l0_tile_shape[1] = self.l1_n
        # the new l1k may be less than l0k
        if self.l0_k > self.l1_k:
            self.l0_tile_shape[2] = self.l1_k

    def procedural_name(self):
        return "l1_{l1m}x{l1n}x{l1k}_l0_{l0m}x{l0n}x{l0k}".format(
            l1m=self.l1_tile_shape[0],
            l1n=self.l1_tile_shape[1],
            l1k=self.l1_tile_shape[2],
            l0m=self.l0_tile_shape[0],
            l0n=self.l0_tile_shape[1],
            l0k=self.l0_tile_shape[2],
        )

    def l1_tile_typename(self, is_tla=False):
        if is_tla:
            tile_fmt = "tla::Shape<Int<{l1m}>, Int<{l1n}>, Int<{l1k}>>"
        else:
            tile_fmt = "GemmShape<{l1m}, {l1n}, {l1k}>"
        return tile_fmt.format(
            l1m=self.l1_tile_shape[0],
            l1n=self.l1_tile_shape[1],
            l1k=self.l1_tile_shape[2],
        )

    def l0_tile_typename(self, is_tla=False):
        if is_tla:
            tile_fmt = "tla::Shape<Int<{l0m}>, Int<{l0n}>, Int<{l0k}>>"
        else:
            tile_fmt = "GemmShape<{l0m}, {l0n}, {l0k}>"
        return tile_fmt.format(
            l0m=self.l0_tile_shape[0],
            l0n=self.l0_tile_shape[1],
            l0k=self.l0_tile_shape[2],
        )


class BroadcastType(enum.Enum):
    RowBroadcast = enum_auto()
    ColBroadcast = enum_auto()
    RowColBroadcast = enum_auto()
    NoBroadcast = enum_auto()


BroadcastTag = {
    BroadcastType.RowBroadcast: "RowBroadcast",
    BroadcastType.ColBroadcast: "ColBroadcast",
    BroadcastType.RowColBroadcast: "RowAndColBroadcast",
    BroadcastType.NoBroadcast: "NoBroadcast",
}


class EpilogueOp(enum.Enum):
    # unary op
    Cast = enum_auto()
    Exp = enum_auto()
    # binary op
    Add = enum_auto()
    Adds = enum_auto()  # scalar ver
    Div = enum_auto()
    Max = enum_auto()
    Min = enum_auto()
    Mul = enum_auto()
    Muls = enum_auto()  # scalar ver
    Sub = enum_auto()

    # activation op
    Gelu = enum_auto()
    LeakyRelu = enum_auto()
    Prelu = enum_auto()
    Relu = enum_auto()
    Rsqrt = enum_auto()
    Sigmoid = enum_auto()
    Silu = enum_auto()
    Tanh = enum_auto()


EpilogueOpTag = {
    # unary op
    EpilogueOp.Cast: "Catlass::Epilogue::Fusion::Cast",
    EpilogueOp.Exp: "Catlass::Epilogue::Fusion::Exp",
    # binary op
    EpilogueOp.Add: "Catlass::Epilogue::Fusion::Add",
    EpilogueOp.Adds: "Catlass::Epilogue::Fusion::Adds",
    EpilogueOp.Div: "Catlass::Epilogue::Fusion::Div",
    EpilogueOp.Max: "Catlass::Epilogue::Fusion::Max",
    EpilogueOp.Min: "Catlass::Epilogue::Fusion::Min",
    EpilogueOp.Mul: "Catlass::Epilogue::Fusion::Mul",
    EpilogueOp.Muls: "Catlass::Epilogue::Fusion::Muls",
    EpilogueOp.Sub: "Catlass::Epilogue::Fusion::Sub",

    # activation op
    EpilogueOp.Gelu: "Catlass::Epilogue::Fusion::Gelu",
    EpilogueOp.LeakyRelu: "Catlass::Epilogue::Fusion::LeakyRelu",
    EpilogueOp.Prelu: "Catlass::Epilogue::Fusion::Prelu",
    EpilogueOp.Relu: "Catlass::Epilogue::Fusion::Relu",
    EpilogueOp.Rsqrt: "Catlass::Epilogue::Fusion::Rsqrt",
    EpilogueOp.Sigmoid: "Catlass::Epilogue::Fusion::Sigmoid",
    EpilogueOp.Silu: "Catlass::Epilogue::Fusion::Silu",
    EpilogueOp.Tanh: "Catlass::Epilogue::Fusion::Tanh",
}


EpilogueOpVectorToScalar = {
    EpilogueOp.Add: EpilogueOp.Adds,
    EpilogueOp.Mul: EpilogueOp.Muls,
}


EpilogueScalarOp = {
    EpilogueOp.Adds,
    EpilogueOp.Muls,
}


class CastType(enum.Enum):
    NONE = enum_auto()    # When there is precision loss in conversion, it means RINT mode; when there is no precision loss, it means no rounding
    RINT = enum_auto()    # round to nearest even (bankers' rounding)
    FLOOR = enum_auto()   # round towards negative infinity
    CEIL = enum_auto()    # round towards positive infinity 
    ROUND = enum_auto()   # round half away from zero
    TRUNC = enum_auto()   # round half away from zero
    ODD = enum_auto()     # Von Neumann rounding, round to nearest odd

CastTypeTag = {
    CastType.NONE: "AscendC::RoundMode::CAST_NONE",
    CastType.RINT: "AscendC::RoundMode::CAST_RINT",
    CastType.FLOOR: "AscendC::RoundMode::CAST_FLOOR",
    CastType.CEIL: "AscendC::RoundMode::CAST_CEIL",
    CastType.ROUND: "AscendC::RoundMode::CAST_ROUND",
    CastType.TRUNC: "AscendC::RoundMode::CAST_TRUNC",
    CastType.ODD: "AscendC::RoundMode::CAST_ODD",
}