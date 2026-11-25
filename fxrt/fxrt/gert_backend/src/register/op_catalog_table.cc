#include "register/op_catalog_register.h"

namespace fx_converter {

// Builtin rows join the same list register_op writes. Parentheses keep builder commas intact.

FX_CATALOG_OP((CatalogRowBuilder("matmul")
                   .Opp("MatMul")
                   .Inputs("x1,x2")
                   .OptionalInputs("bias")
                   .Attr("transpose_x1", GeAttrType::kBool)
                   .Attr("transpose_x2", GeAttrType::kBool)),
              (BehaviorBuilder().InferTransposeFromLayout()));
FX_CATALOG_OP((CatalogRowBuilder("mm")
                   .Opp("MatMul")
                   .Inputs("x1,x2")
                   .OptionalInputs("bias")
                   .Attr("transpose_x1", GeAttrType::kBool)
                   .Attr("transpose_x2", GeAttrType::kBool)),
              (BehaviorBuilder().InferTransposeFromLayout()));
// aten.addmm is registered by AddmmAclnnLowering (lowering/addmm_op_lowering.cc):
// its beta/alpha arrive as positional IR doubles and need an explicit role
// mapping, which a plain catalog row cannot express.

FX_CATALOG_OP((CatalogRowBuilder("batch_matmul")
                   .Opp("BatchMatMul")
                   .Inputs("x1,x2")
                   .Attr("adj_x1", GeAttrType::kBool)
                   .Attr("adj_x2", GeAttrType::kBool)),
              (BehaviorBuilder().InferTransposeFromLayout()));
// cat/concat lower through aclnnCat (OPP type Concat). concat_dim is attr0 for the
// prepare wrapper; N is the packed tensor count. aclnnCat accepts a negative dim.
FX_CATALOG_OP((CatalogRowBuilder("cat")
                   .Opp("Concat")
                   .Arity(0)
                   .PackedX()
                   .RequiredAttr("concat_dim", GeAttrType::kInt)
                   .DerivedAttr("N", GeAttrType::kInt, GeAttrDerived::kInputCount)),
              (BehaviorBuilder().CaptureAxis("concat_dim")));
FX_CATALOG_OP((CatalogRowBuilder("concat")
                   .Opp("Concat")
                   .Arity(0)
                   .PackedX()
                   .RequiredAttr("concat_dim", GeAttrType::kInt)
                   .DerivedAttr("N", GeAttrType::kInt, GeAttrDerived::kInputCount)),
              (BehaviorBuilder().CaptureAxis("concat_dim")));
FX_CATALOG_OP((CatalogRowBuilder("add")
                 .Opp("Add")
                 .Inputs("x1,x2")
                 .RefInput(2)
                 .Attr("alpha", GeAttrType::kFloat, 1)
                 .Attr("alpha_int", GeAttrType::kInt, 1)));
FX_CATALOG_OP((CatalogRowBuilder("neg").Opp("Neg").Arity(1).AiCore()));
FX_CATALOG_OP((CatalogRowBuilder("mul").Opp("Mul").AiCore()));
// mul_scalar mirrors sub_scalar: Collect materializes the trailing scalar as x2. A
// symbolic scalar (a Dynamo-lifted SymInt placeholder) rides the sym_axis channel and
// its materialized payload is refreshed per Execute, so x * s works under dynamic=True.
FX_CATALOG_OP((CatalogRowBuilder("mul_scalar").Opp("Mul").Arity(2).AiCore()));
// The converter factory hook supplies (dimensions, value) for both zeros and full.
// Ordered arguments retain the empty dimensions tuple of a rank-0 factory.
FX_CATALOG_OP((CatalogRowBuilder("zeros")
                   .Opp("Fill")
                   .Arity(2)
                   .AiCore()
                   .OrderedArguments()
                   .ShapeInput(0)
                   .ShapeListVector()
                   .Inputs("dims,value")
                   .InferDepends("dims")));
// Pairs with reduce_sum_output_hook (fx_converter/rewrites/output.py): the rewrite
// always keeps dims on the IR op and lowers a false keepdim to a following view, so
// keep_dims stays a const 1 here and the keepdim Bool is dropped (SkipBool).
// GE ReduceSum has only keep_dims (default false).
FX_CATALOG_OP((CatalogRowBuilder("reduce_sum")
                 .Opp("ReduceSum")
                 .OrderedArguments()
                 .ShapeInput(1)
                 .ShapeListVector()
                 .Inputs("x,axes")
                 .InferDepends("axes")
                 .Attr("keep_dims", GeAttrType::kBool, 1)),
              (BehaviorBuilder().SkipBool()));
// Scalar cmp / Sub rows declare Arity(2): Collect materializes the trailing scalar
// as the second tensor input of the GE op (Equal/GreaterEqual/NotEqual/Less/Sub are
// two-input ops).
FX_CATALOG_OP((CatalogRowBuilder("eq_scalar").Opp("Equal").Arity(2).AiCore()));
FX_CATALOG_OP((CatalogRowBuilder("ge_scalar").Opp("GreaterEqual").Arity(2).AiCore()));
FX_CATALOG_OP((CatalogRowBuilder("ge").Opp("GreaterEqual").AiCore()));
FX_CATALOG_OP((CatalogRowBuilder("ne").Opp("NotEqual").Arity(2).AiCore()));
FX_CATALOG_OP((CatalogRowBuilder("where").Opp("SelectV2").Arity(3)));
FX_CATALOG_OP((CatalogRowBuilder("expand").Opp("Expand").AiCore().ShapeInput(1).Inputs("x,shape")));
FX_CATALOG_OP((CatalogRowBuilder("softmax")
                   .Opp("SoftmaxV2")
                   .Arity(1)
                   .AiCore()
                   .Attr("axes", GeAttrType::kListInt, -1)
                   .Attr("half_to_float", GeAttrType::kBool, 0)),
              (BehaviorBuilder().CaptureAxis("axes")));
FX_CATALOG_OP((CatalogRowBuilder("gather").Opp("GatherElements").Inputs("x,index").StridedInputs()
                   .Attr("dim", GeAttrType::kInt)),
              (BehaviorBuilder().CaptureAxis("dim")));
FX_CATALOG_OP((CatalogRowBuilder("repeat")
                   .Opp("Tile")
                   .AiCore()
                   .ShapeInput(1)
                   .ShapeListVector()
                   .Inputs("x,multiples")));
// GE REG_OP(TopKV2) declares ATTR(indices_dtype, Int, DT_INT32) with ge::DT_INT32 == 3.
// Declared explicitly so the GE attr and OutputDtypes(",DT_INT32") cannot silently drift
// apart: the bins produce INT32 indices, and both fields must say so together.
FX_CATALOG_OP((CatalogRowBuilder("topk")
                   .Opp("TopKV2")
                   .AiCore()
                   .ShapeInput(1)
                   .ShapeListVector()
                   .Inputs("x,k")
                   .Outputs("values,indices")
                   .OutputDtypes(",DT_INT32")
                   .ShapeListInt32()
                   .InferDepends("k")
                   .Attr("sorted", GeAttrType::kBool, 1)
                   .Attr("dim", GeAttrType::kInt, -1)
                   .Attr("largest", GeAttrType::kBool, 1)
                   .Attr("indices_dtype", GeAttrType::kInt, 3)),
              (BehaviorBuilder().CaptureAxis("dim").CaptureTopkBools().SelectSuffix("{sorted},,{largest}")));
// Pairs with pad_arg_hook (fx_converter/rewrites/arg.py): the hook stores the pad
// list already in GE contiguous order as the "paddings" shape input.
// PadV3's optional fill value is passed to tiling/launch but omitted from its
// simplifiedKey, which contains only x, paddings and y.
FX_CATALOG_OP((CatalogRowBuilder("constant_pad_nd")
                   .Opp("PadV3")
                   .Arity(3)
                   .AiCore()
                   .ShapeInput(1)
                   .ShapeListVector()
                   .Inputs("x,paddings")
                   .OptionalInputs("constant_values")
                   .InferDepends("paddings")
                   .Attr("mode", GeAttrType::kStr, 0, "constant")
                   .Attr("paddings_contiguous", GeAttrType::kBool, 1)),
              (BehaviorBuilder().BoolAsInt8().SelectSuffix("constant,1")));
FX_CATALOG_OP((CatalogRowBuilder("full_like")
                   .Opp("Fills")
                   .Arity(1)
                   .AiCore()
                   .Inputs("x")
                   .RequiredAttr("value", GeAttrType::kFloat)),
              (BehaviorBuilder().CaptureFill()));
// Embedding is a dim-0 row lookup. GatherV2 covers shapes outside the native
// Embedding OPP's SIMT range without specializing the FX graph by row width.
FX_CATALOG_OP((CatalogRowBuilder("embedding")
                   .Opp("GatherV2")
                   .Arity(3)
                   .AiCore()
                   .Inputs("x,indices,axis")
                   .AppendDim0Axis()
                   .InferDepends("axis")
                   .Attr("batch_dims", GeAttrType::kInt, 0)
                   .Attr("is_preprocessed", GeAttrType::kBool, 0)
                   .Attr("negative_index_support", GeAttrType::kBool, 0)),
              (BehaviorBuilder().PreferHighPerformance(false).SelectSuffix(",0").SkipInt()));
FX_CATALOG_OP((CatalogRowBuilder("permute_view")
                   .Opp("Transpose")
                   .AiCore()
                   .ShapeInput(1)
                   .ShapeListVector()
                   .Inputs("x,perm")));
FX_CATALOG_OP(
    (CatalogRowBuilder("permute").Opp("Transpose").AiCore().ShapeInput(1).ShapeListVector().Inputs("x,perm")));
// Slice family (slice_view/slice/narrow_view/narrow/getitem_slice): the slice/getitem
// arg hooks in fx_converter/rewrites/arg.py emit exactly (x, offsets, size) int
// tuples. Keep Arity(3) + ShapeInput(1) in sync with them.
FX_CATALOG_OP((CatalogRowBuilder("slice_view")
                   .Opp("Slice")
                   .Arity(3)
                   .AiCore()
                   .ShapeInput(1)
                   .ShapeListVector()
                   .Inputs("x,offsets,size")),
              (BehaviorBuilder().PreferHighPerformance(false)));
FX_CATALOG_OP((CatalogRowBuilder("slice")
                   .Opp("Slice")
                   .Arity(3)
                   .AiCore()
                   .ShapeInput(1)
                   .ShapeListVector()
                   .Inputs("x,offsets,size")),
              (BehaviorBuilder().PreferHighPerformance(false)));
FX_CATALOG_OP((CatalogRowBuilder("narrow_view")
                   .Opp("Slice")
                   .Arity(3)
                   .AiCore()
                   .ShapeInput(1)
                   .ShapeListVector()
                   .Inputs("x,offsets,size")),
              (BehaviorBuilder().PreferHighPerformance(false)));
FX_CATALOG_OP((CatalogRowBuilder("narrow")
                   .Opp("Slice")
                   .Arity(3)
                   .AiCore()
                   .ShapeInput(1)
                   .ShapeListVector()
                   .Inputs("x,offsets,size")),
              (BehaviorBuilder().PreferHighPerformance(false)));
FX_CATALOG_OP((CatalogRowBuilder("getitem_slice")
                   .Opp("Slice")
                   .Arity(3)
                   .AiCore()
                   .ShapeInput(1)
                   .ShapeListVector()
                   .Inputs("x,offsets,size")),
              (BehaviorBuilder().PreferHighPerformance(false)));
// select_output_hook (fx_converter/rewrites/output.py) rewrites select into a
// length-1 slice + view; these GatherV2 rows cover graphs that still carry a select op.
FX_CATALOG_OP((CatalogRowBuilder("select_view")
                   .Opp("GatherV2")
                   .Arity(3)
                   .AiCore()
                   .ShapeInput(1)
                   .ShapeListVector()
                   .Inputs("x,indices,axis")
                   .Attr("batch_dims", GeAttrType::kInt, 0)
                   .Attr("is_preprocessed", GeAttrType::kBool, 0)
                   .Attr("negative_index_support", GeAttrType::kBool, 1)),
              (BehaviorBuilder().CaptureAxis().PreferHighPerformance(false).SelectSuffix(",1")));
FX_CATALOG_OP((CatalogRowBuilder("select")
                   .Opp("GatherV2")
                   .Arity(3)
                   .AiCore()
                   .ShapeInput(1)
                   .ShapeListVector()
                   .Inputs("x,indices,axis")
                   .Attr("batch_dims", GeAttrType::kInt, 0)
                   .Attr("is_preprocessed", GeAttrType::kBool, 0)
                   .Attr("negative_index_support", GeAttrType::kBool, 1)),
              (BehaviorBuilder().CaptureAxis().PreferHighPerformance(false).SelectSuffix(",1")));
// Same GatherV2 dim0 contract as embedding. The IR dim Int is always 0 and is skipped.
FX_CATALOG_OP((CatalogRowBuilder("gather_v2")
                   .Opp("GatherV2")
                   .Arity(3)
                   .AiCore()
                   .Inputs("x,indices,axis")
                   .AppendDim0Axis()
                   .InferDepends("axis")
                   .Attr("batch_dims", GeAttrType::kInt, 0)
                   .Attr("is_preprocessed", GeAttrType::kBool, 0)
                   .Attr("negative_index_support", GeAttrType::kBool, 0)),
              (BehaviorBuilder().PreferHighPerformance(false).SelectSuffix(",0").SkipInt()));
FX_CATALOG_OP((CatalogRowBuilder("index_tensor")
                   .Opp("GatherV2")
                   .Arity(3)
                   .AiCore()
                   .Inputs("x,indices,axis")
                   .AppendDim0Axis()
                   .InferDepends("axis")
                   .Attr("batch_dims", GeAttrType::kInt, 0)
                   .Attr("is_preprocessed", GeAttrType::kBool, 0)
                   .Attr("negative_index_support", GeAttrType::kBool, 0)),
              (BehaviorBuilder().PreferHighPerformance(false).SelectSuffix(",0").SkipInt()));
// _rewrite_as_strided_known_patterns_ (fx_converter/rewrites/as_strided.py) turns
// known as_strided layouts into permute/expand before IR build; the AsStrided
// recipe here keeps the layouts that rewrite declines.
FX_CATALOG_OP((CatalogRowBuilder("as_strided_view")
                   .Opp("Slice")
                   .Arity(3)
                   .AiCore()
                   .ShapeInput(1)
                   .ShapeListVector()
                   .Inputs("x,offsets,size")),
              (BehaviorBuilder().AsStrided().PreferHighPerformance(false)));
// add_scalar mirrors mul_scalar: Collect materializes the trailing scalar as x2, and a
// symbolic scalar's payload is refreshed per Execute (same sym_axis channel contract).
FX_CATALOG_OP((CatalogRowBuilder("add_scalar").Opp("Add").Arity(2).AiCore()));
FX_CATALOG_OP((CatalogRowBuilder("sub").Opp("Sub").AiCore().Inputs("x1,x2")));
FX_CATALOG_OP((CatalogRowBuilder("sub_scalar").Opp("Sub").Arity(2).AiCore()));
FX_CATALOG_OP((CatalogRowBuilder("eq").Opp("Equal").AiCore()));
FX_CATALOG_OP((CatalogRowBuilder("lt").Opp("Less").AiCore()));
FX_CATALOG_OP((CatalogRowBuilder("lt_scalar").Opp("Less").Arity(2).AiCore()));
FX_CATALOG_OP((CatalogRowBuilder("cast")
                   .Opp("Cast")
                   .Arity(1)
                   .AiCore()
                   .Inputs("x")
                   .DerivedAttr("dst_type", GeAttrType::kInt, GeAttrDerived::kOutputDtype, /*required=*/true)),
              (BehaviorBuilder().SkipInt()));
FX_CATALOG_OP((CatalogRowBuilder("sigmoid").Opp("Sigmoid").Arity(1).AiCore()));
FX_CATALOG_OP((CatalogRowBuilder("rsqrt").Opp("Rsqrt").Arity(1).AiCore()));
FX_CATALOG_OP((CatalogRowBuilder("bitwise_and_tensor").Opp("BitwiseAnd").AiCore().Inputs("x1,x2")));
// Preserve integer width and bool semantics through the native ACLNN adapter.
FX_CATALOG_OP((CatalogRowBuilder("bitwise_not").Opp("Invert").Inputs("x").Arity(1).StridedInputs()));
FX_CATALOG_OP((CatalogRowBuilder("masked_fill_scalar").Opp("MaskedFill").Arity(3).AiCore().Inputs("x,mask,value")));
FX_CATALOG_OP((CatalogRowBuilder("inplace_masked_fill_scalar")
                   .Opp("MaskedFill")
                   .Arity(3)
                   .AiCore()
                   .Inputs("x,mask,value")
                   .RefInput(0)));
// RmsNorm: x+gamma tensors, epsilon via CaptureFill (same source as Fills/Adds), y+rstd.
FX_CATALOG_OP((CatalogRowBuilder("rms_norm")
                   .Opp("RmsNorm")
                   .AiCore()
                   .Inputs("x,gamma")
                   .Outputs("y,rstd")
                   .Attr("epsilon", GeAttrType::kFloat)),
              (BehaviorBuilder().CaptureFill("epsilon")));
// LayerNormV4: normalized_shape rides as an INT32 shape-list input (Slice style); its bin
// keys encode one representative output slot although the launch keeps y/mean/rstd. The
// mean/rstd slots drop the trailing norm dims, so the slots do not share shapes.
FX_CATALOG_OP((CatalogRowBuilder("norm")
                   .Opp("LayerNormV4")
                   .Arity(4)
                   .AiCore()
                   .ShapeInput(1)
                   .ShapeListVector()
                   .ShapeListInt32()
                   .Inputs("x,normalized_shape,gamma,beta")
                   .Outputs("y,mean,rstd")
                   .Attr("epsilon", GeAttrType::kFloat)),
              (BehaviorBuilder().CaptureFill("epsilon").SelectSingleOutput()));
// Plain unary elementwise. Exp's base/scale/shift defaults spell y = e^x.
FX_CATALOG_OP((CatalogRowBuilder("exp")
                   .Opp("Exp")
                   .Arity(1)
                   .AiCore()
                   .Attr("base", GeAttrType::kFloat, -1)
                   .Attr("scale", GeAttrType::kFloat, 1)
                   .Attr("shift", GeAttrType::kFloat, 0)));
FX_CATALOG_OP((CatalogRowBuilder("bitwise_or_tensor").Opp("BitwiseOr").AiCore().Inputs("x1,x2")));
FX_CATALOG_OP((CatalogRowBuilder("silu").Opp("Swish").Arity(1).AiCore().Attr("scale", GeAttrType::kFloat, 1)));
FX_CATALOG_OP((CatalogRowBuilder("leaky_relu")
                   .Opp("LeakyRelu")
                   .Arity(1)
                   .AiCore()
                   .Attr("negative_slope", GeAttrType::kFloat)),
              (BehaviorBuilder().CaptureFill("negative_slope")));
// Binary elementwise / compare.
FX_CATALOG_OP((CatalogRowBuilder("div").Opp("Div").AiCore().Inputs("x1,x2")));
FX_CATALOG_OP((CatalogRowBuilder("gt").Opp("Greater").AiCore()));
FX_CATALOG_OP((CatalogRowBuilder("le").Opp("LessEqual").AiCore()));
// Scalar variants declare Arity(2): Collect materializes the trailing scalar as the
// second tensor input (same contract as eq_scalar / sub_scalar).
FX_CATALOG_OP((CatalogRowBuilder("le_scalar").Opp("LessEqual").Arity(2).AiCore()));
FX_CATALOG_OP((CatalogRowBuilder("pow_scalar").Opp("Pow").Arity(2).AiCore()));
// ReduceMean mirrors reduce_sum: the shared reduce output hook lowers a false
// keepdim to a following view, so keep_dims stays a const 1 and the keepdim
// Bool is dropped (SkipBool). The reduce IR also carries a trailing dtype Int
// (aten schema); SkipInt drops it because ReduceMean keeps the input dtype.
// GE ReduceMean attr order is keep_dims then noop_with_empty_axes. The legacy
// bin reads the axes shape-list as DT_INT32 (ShapeListInt32).
FX_CATALOG_OP((CatalogRowBuilder("reduce_mean")
                   .Opp("ReduceMean")
                   .AiCore()
                   .ShapeInput(1)
                   .ShapeListVector()
                   .Inputs("x,axes")
                   .InferDepends("axes")
                   .Attr("keep_dims", GeAttrType::kBool, 1)
                   .Attr("noop_with_empty_axes", GeAttrType::kBool, 0)
                   .ShapeListInt32()
                   .NormalizeShapeListAxes()),
              (BehaviorBuilder().SkipBool().SkipInt()));
// LogSoftmaxV2 mirrors SoftmaxV2: the single Int dim is captured into the
// ListInt axes attr. The aten _log_softmax half_to_float Bool is dropped, so
// the row is only valid for dtype-preserving calls (half_to_float=False).
FX_CATALOG_OP((CatalogRowBuilder("log_softmax")
                   .Opp("LogSoftmaxV2")
                   .Arity(1)
                   .AiCore()
                   .Attr("axes", GeAttrType::kListInt, -1)),
              (BehaviorBuilder().CaptureAxis("axes").SkipBool()));
// Ten more Path C aicore rows, picked to spread over distinct generalization axes:
// plain unary/binary elementwise, an int-attr mask, a scalar-materialized FloorMod,
// an axis-tensor-input scan, a CommReduce reduce, and a dual-output Sort.
FX_CATALOG_OP((CatalogRowBuilder("relu").Opp("Relu").Arity(1).AiCore()));
FX_CATALOG_OP((CatalogRowBuilder("square").Opp("Square").Arity(1).AiCore()));
FX_CATALOG_OP((CatalogRowBuilder("pow_tensor").Opp("Pow").AiCore().Inputs("x1,x2")));
FX_CATALOG_OP((CatalogRowBuilder("remainder_tensor_tensor").Opp("FloorMod").AiCore().Inputs("x1,x2")));
// Scalar variant declares Arity(2): Collect materializes the trailing scalar as x2
// (same contract as eq_scalar / le_scalar / pow_scalar).
FX_CATALOG_OP((CatalogRowBuilder("remainder_tensor_scalar").Opp("FloorMod").Arity(2).AiCore().Inputs("x1,x2")));
// Gelu covers torch's default approximate='none' (erf recipe); the 'tanh' recipe is a
// different op and must not map here. The legacy Gelu opp implements a polynomial
// fast recipe whose values drift from x*Phi(x), so the row targets GeluV2 whose proto
// is defined as x*Phi(x) and documents PyTorch Gelu compatibility.
FX_CATALOG_OP((CatalogRowBuilder("gelu").Opp("GeluV2").Arity(1).AiCore()));
// tril: the Int diagonal (k=0 main, negative below) is captured into the attr.
FX_CATALOG_OP((CatalogRowBuilder("tril").Opp("Tril").Arity(1).AiCore().Attr("diagonal", GeAttrType::kInt, 0)),
              (BehaviorBuilder().CaptureAxis("diagonal")));
// amax is dual-stage aclnnAmax. The ReduceMax float32/int32 bin is MIX-only and this
// device cannot launch it. The shared reduce output hook lowers a false keepdim to a
// following view, so keep_dims stays a const 1 and the keepdim Bool is dropped. Attr
// order is keep_dims then noop_with_empty_axes; axes is a host INT32 shape-list.
FX_CATALOG_OP((CatalogRowBuilder("amax")
                   .Opp("ReduceMax")
                   .ShapeInput(1)
                   .ShapeListVector()
                   .Inputs("x,axes")
                   .InferDepends("axes")
                   .Attr("keep_dims", GeAttrType::kBool, 1)
                   .Attr("noop_with_empty_axes", GeAttrType::kBool, 0)
                   .ShapeListInt32()
                   .NormalizeShapeListAxes()),
              (BehaviorBuilder().SkipBool()));
// argsort: the argsort_hook normalizes IR inputs to [x, stable(Bool), dim(Int),
// descending(Bool)]; same-kind Bool rules are consumed in declaration order. GE Sort
// bins write INT32 indices, so y2_dtype says DT_INT32 (3) and OutputDtypes forces the
// second output to DT_INT32 like TopKV2 (the backend widens it to torch's int64).
FX_CATALOG_OP((CatalogRowBuilder("argsort")
                   .Opp("Sort")
                   .Arity(1)
                   .AiCore()
                   .Outputs("y1,y2")
                   .OutputDtypes(",DT_INT32")
                   .Attr("axis", GeAttrType::kInt, -1)
                   .Attr("descending", GeAttrType::kBool, 0)
                   .Attr("stable", GeAttrType::kBool, 0)
                   .Attr("y2_dtype", GeAttrType::kInt, 3)),
              (BehaviorBuilder().CaptureAttr(IrInputKind::kBool, "stable")
                                    .CaptureAttr(IrInputKind::kBool, "descending")
                                    .CaptureAxis("axis")
                                    // Sort bin keys end in a ",{descending},{stable}" attr tail; without
                                    // the suffix the prefix fallback picks any equal-length variant.
                                    .SelectSuffix(",{descending},{stable}")));
// cumsum_ext: axis is a rank-0 tensor input on the opp, so the IR Int dim materializes
// as the second tensor input; exclusive/reverse spell torch.cumsum's defaults. Cumsum
// bins only accept an INT32/INT64 axis, so ScalarDtype pins the materialization to
// ge::DT_INT32 == 3 instead of following x's float dtype. The bin simplifiedKeys carry
// an exclusive,reverse attr tail; without SelectSuffix the prefix fallback picks any
// equal-length variant (e.g. reverse=1).
FX_CATALOG_OP((CatalogRowBuilder("cumsum_ext")
                   .Opp("Cumsum")
                   .Arity(2)
                   .AiCore()
                   .Inputs("x,axis")
                   .ScalarDtype(3)
                   .Attr("exclusive", GeAttrType::kBool, 0)
                   .Attr("reverse", GeAttrType::kBool, 0)),
              (BehaviorBuilder().SelectSuffix("0,0")));
// Second batch of ten, again spreading distinct channels: a scalar-materialized compare,
// a scalar-materialized div, an axis+index gather, a variadic packed-x Pack, an int_list
// tile, a bool-dtype binary, an int64-index argmax, a ReduceMin reduce, and two unary
// controls.
// not_equal is the scalar-form ne: the fx_backend ne_op_hook reroutes literal-scalar ne
// here while tensor-tensor ne keeps the ne row above (both land on the NotEqual opp).
FX_CATALOG_OP((CatalogRowBuilder("not_equal").Opp("NotEqual").Arity(2).AiCore()));
// div_scalar mirrors sub_scalar: Collect materializes the trailing scalar as x2.
FX_CATALOG_OP((CatalogRowBuilder("div_scalar").Opp("Div").Arity(2).AiCore()));
// index_select mirrors cumsum_ext rather than select: indices already arrives as a real
// tensor (no int_list shape input), and the exported axis Int falls through to the scalar
// slot, materialized rank-0 like the Cumsum axis. ScalarDtype pins it to ge::DT_INT32 == 3
// (GatherV2 axis is IndexNumberType) instead of following x's float dtype; InferDepends
// makes the axis payload host-readable -- the GatherV2 proto InferShape reads it via
// GetInputConstData and fails with "axis not exists" otherwise.
FX_CATALOG_OP((CatalogRowBuilder("index_select")
                   .Opp("GatherV2")
                   .Arity(3)
                   .AiCore()
                   .Inputs("x,indices,axis")
                   .ScalarDtype(3)
                   .InferDepends("axis")
                   .Attr("batch_dims", GeAttrType::kInt, 0)
                   .Attr("is_preprocessed", GeAttrType::kBool, 0)
                   .Attr("negative_index_support", GeAttrType::kBool, 1)),
              (BehaviorBuilder().PreferHighPerformance(false).SelectSuffix(",1")));
// stack is the variadic Pack on the AiCore path. The dual-stage route is out (the Pack
// opp exposes no prepare/launch), but the prebuilt .o set is complete: GE dynamic
// packed-x keys encode one representative x slot plus the input count
// (Pack/d,p/dtype,format,N/...), so the row is PackedX + Arity(0) and
// stack_flatten_arg_hook (fx_converter/rewrites/arg.py) flattens the leading tensor
// list into individual IR inputs ahead of the variadic collect. Pack's AutoTiling
// wrapper rejects axis == rank (appending a trailing dim), so stack_output_hook
// (fx_converter/rewrites/output.py) lowers that corner as rank+1 views plus ConcatD.
FX_CATALOG_OP((CatalogRowBuilder("stack")
                   .Opp("Pack")
                   .Arity(0)
                   .PackedX()
                   .AiCore()
                   .RequiredAttr("axis", GeAttrType::kInt)
                   .DerivedAttr("N", GeAttrType::kInt, GeAttrDerived::kInputCount)),
              (BehaviorBuilder().CaptureAxis("axis").NormalizeAxis(1)));
// tile twins repeat: the dims tuple arrives as the multiples shape-list input.
FX_CATALOG_OP((CatalogRowBuilder("tile")
                   .Opp("Tile")
                   .AiCore()
                   .ShapeInput(1)
                   .ShapeListVector()
                   .Inputs("x,multiples")));
// Bool-dtype binary; the bins key on the bool input ABI directly.
FX_CATALOG_OP((CatalogRowBuilder("logical_and").Opp("LogicalAnd").AiCore().Inputs("x1,x2")));
// argmax mirrors cumsum_ext: dimension is a rank-0 tensor input on the ArgMaxV2 proto
// (not an attr), so the IR Int dim materializes as the second input; ScalarDtype pins it
// to ge::DT_INT32 == 3 like the Cumsum axis. The dtype attr defaults to DT_INT64, which
// already matches torch.argmax's int64 indices. keepdim is handled by argmax_output_hook
// (ArgMaxV2 natively squeezes; a true keepdim becomes a following view), so the Bool is
// dropped here (SkipBool). InferDepends makes the dimension payload host-readable -- with
// a symbolic axis the proto InferShape re-reads it per Execute to pick the squeezed dim.
FX_CATALOG_OP((CatalogRowBuilder("argmax")
                   .Opp("ArgMaxV2")
                   .Arity(2)
                   .AiCore()
                   .Inputs("x,dimension")
                   .ScalarDtype(3)
                   .InferDepends("dimension")),
              (BehaviorBuilder().SkipBool()));
// amin mirrors amax: keep_dims stays a const 1 via the shared reduce output hook, and
// the axes shape-list reads as DT_INT32 like every legacy reduce bin.
FX_CATALOG_OP((CatalogRowBuilder("amin")
                   .Opp("ReduceMin")
                   .AiCore()
                   .ShapeInput(1)
                   .ShapeListVector()
                   .Inputs("x,axes")
                   .InferDepends("axes")
                   .Attr("keep_dims", GeAttrType::kBool, 1)
                   .Attr("noop_with_empty_axes", GeAttrType::kBool, 0)
                   .ShapeListInt32()
                   .NormalizeShapeListAxes()),
              (BehaviorBuilder().SkipBool()));
// Plain unary controls.
FX_CATALOG_OP((CatalogRowBuilder("abs").Opp("Abs").Arity(1).AiCore()));
// IsNan writes a BOOL output; no dtype forcing needed, the bin ABI says BOOL already.
FX_CATALOG_OP((CatalogRowBuilder("isnan").Opp("IsNan").Arity(1).AiCore()));

// Third batch: npu custom ops and tuple-output ops that need the per-slot InferShape
// rules. The trailing div_mode Int on the div_mod rows is the fxrt-side rounding
// discriminator (only mode 2 == floor is exported today); FloorDiv carries no attr for
// it, so it is skipped.
FX_CATALOG_OP((CatalogRowBuilder("div_mod").Opp("FloorDiv").AiCore().Inputs("x1,x2")),
              (BehaviorBuilder().SkipInt()));
// div_mod_scalar mirrors div_scalar: Collect materializes the scalar as x2; only the
// trailing div_mode Int (second kInt input) is skipped.
FX_CATALOG_OP((CatalogRowBuilder("div_mod_scalar").Opp("FloorDiv").Arity(2).AiCore()),
              (BehaviorBuilder().SkipIntAt(1)));
// SwiGlu: the Int dim is the proto attr.
FX_CATALOG_OP((CatalogRowBuilder("swiglu").Opp("SwiGlu").Arity(1).AiCore().Attr("dim", GeAttrType::kInt, -1)),
              (BehaviorBuilder().CaptureAxis("dim")));
// InterleaveRope: plain 3-tensor row, no attrs on the proto.
FX_CATALOG_OP((CatalogRowBuilder("interleave_rope")
                   .Opp("InterleaveRope")
                   .Arity(3)
                   .AiCore()
                   .Inputs("x,cos,sin")));
// AddRmsNorm: y/rstd/x slots; rstd drops the norm dims while y and x keep the input
// shape -- the per-slot InferShape rules carry each shape. epsilon arrives as a Double.
FX_CATALOG_OP((CatalogRowBuilder("add_rms_norm")
                   .Opp("AddRmsNorm")
                   .Arity(3)
                   .AiCore()
                   .Inputs("x1,x2,gamma")
                   .Outputs("y,rstd,x")
                   .Attr("epsilon", GeAttrType::kFloat)),
              (BehaviorBuilder().CaptureFill("epsilon")));
// Range: start/limit/delta arrive as python numbers and materialize as rank-0 scalars
// (the eq_scalar channel). The proto default is_closed=false matches torch.arange's
// half-open interval, but the tiling reads the attr by pointer and fails when the node
// carries no attr at all, so the row must declare it (written as the default false).
// Dynamo also exports the torch dtype enum as a trailing Int operand -- CANN Range
// takes the output dtype from the GE output desc, so it is skipped, not wired. The Int
// ordinal counts a symbolic slot too, so the dtype Int is ordinal 3 whether limit
// arrives literal (static trace) or symbolic (dynamic trace): start=0, limit=1,
// delta=2, dtype=3.
FX_CATALOG_OP((CatalogRowBuilder("arange")
                   .Opp("Range")
                   .Arity(3)
                   .AiCore()
                   .Inputs("start,limit,delta")
                   .Attr("is_closed", GeAttrType::kBool)),
              (BehaviorBuilder().SkipIntAt(3)));
// MoeGatingTopKSoftmax: k is the only proto attr; expert_idx/row_idx bins are INT32
// like the TopKV2 indices. finished rides optional with gen_placeholder mode, so the
// launch pads one placeholder slot without registering the ir input (keeps tiling on
// the no-finished kernel variant).
FX_CATALOG_OP((CatalogRowBuilder("moe_gating_top_k_softmax")
                   .Opp("MoeGatingTopKSoftmax")
                   .Arity(1)
                   .AiCore()
                   .Inputs("x")
                   .LaunchPadInputs("finished")
                   .Outputs("y,expert_idx,row_idx")
                   .OutputDtypes(",DT_INT32,DT_INT32")
                   .Attr("k", GeAttrType::kInt, 0)),
              (BehaviorBuilder().CaptureAxis("k")));
// SplitV twins: the IR view-fallback rows carry [x, size_splits, split_dim]; the CANN
// key encodes one representative output slot (dynamic-output row), hence the single
// output spec at select time. num_split equals the size_splits length.
FX_CATALOG_OP((CatalogRowBuilder("split_tensor")
                   .Opp("SplitV")
                   .Arity(3)
                   .AiCore()
                   .ShapeInput(1)
                   .ShapeListVector()
                   .ShapeListInt32()
                   .Inputs("x,size_splits,split_dim")
                   .ScalarDtype(3)),
              (BehaviorBuilder().SelectSingleOutput()));
FX_CATALOG_OP((CatalogRowBuilder("split_with_size")
                   .Opp("SplitV")
                   .Arity(3)
                   .AiCore()
                   .ShapeInput(1)
                   .ShapeListVector()
                   .ShapeListInt32()
                   .Inputs("x,size_splits,split_dim")
                   .ScalarDtype(3)),
              (BehaviorBuilder().SelectSingleOutput()));
// AscendQuantV2: zero_points (proto "offset") is absent on the symmetric path, so the
// bins key 2-in and the launch pads one placeholder slot. The proto attr order is
// sqrt_mode, round_mode, dst_type, axis -- tiling reads them by index, so all four are
// declared in that order. dst_type derives from the IR output dtype (torch.int8 ->
// DT_INT8); the trailing div_mode Bool only selects the torch-side path, not a proto
// attr. The launch-only pad keeps tiling on the no-offset kernel variant.
FX_CATALOG_OP((CatalogRowBuilder("npu_quantize")
                   .Opp("AscendQuantV2")
                   .Arity(2)
                   .AiCore()
                   .Inputs("x,scale")
                   .LaunchPadInputs("offset")
                   .Attr("sqrt_mode", GeAttrType::kBool, 0)
                   .Attr("round_mode", GeAttrType::kStr, 0, "round")
                   .DerivedAttr("dst_type", GeAttrType::kInt, GeAttrDerived::kOutputDtype)
                   .Attr("axis", GeAttrType::kInt, -1)),
              (BehaviorBuilder().CaptureAxis("axis").SkipInt().SkipBool()));
// DynamicQuant: y is INT8 while scale stays fp32. smooth_scales/group_index ride
// optional with gen_placeholder mode, so the launch pads two placeholder slots without
// registering the ir inputs. dst_type is the only proto attr (DT_INT8 default). The IR
// carries smooth_scales/group_index/dst_type Nones, the quant_mode string and the
// dst_type_max double behind x; Nones and Strings classify as kDouble on this path,
// so the behavior skips five kDouble slots and the trailing dst_type_max double.
FX_CATALOG_OP((CatalogRowBuilder("npu_dynamic_quant")
                   .Opp("DynamicQuant")
                   .Arity(1)
                   .AiCore()
                   .Inputs("x")
                   .LaunchPadInputs("smooth_scales,group_index")
                   .Outputs("y,scale")
                   .OutputDtypes("DT_INT8,")
                   .Attr("dst_type", GeAttrType::kInt, 2)),
              (BehaviorBuilder()
                   .ClassifyAt(IrInputKind::kDouble, InputRole::kSkip, 0)
                   .ClassifyAt(IrInputKind::kDouble, InputRole::kSkip, 1)
                   .ClassifyAt(IrInputKind::kDouble, InputRole::kSkip, 2)
                   .ClassifyAt(IrInputKind::kDouble, InputRole::kSkip, 3)
                   .ClassifyAt(IrInputKind::kDouble, InputRole::kSkip, 4)));
// AddRmsNormQuant: y1/y2 are the INT8 quantized pair, x keeps the input dtype.
// scales2 / zero_points stay at their optional absence. Tiling reads the attrs by
// proto index (axis, epsilon, div_mode, dst_type), so all four are declared in that
// order. OrderedArguments keeps the four None slots (unwired optional
// scales/zero_points) and the div_mode Bool in the IR -- the normalized bag would
// drop them and shift epsilon into kDouble ordinal 0, where the first skip rule
// silently ate it (the attr default rode; non-default epsilon never took effect).
// Nones classify as kDouble on this path, so the behavior burns four kDouble skips
// before the real epsilon capture and one more for the trailing dst_type None.
FX_CATALOG_OP((CatalogRowBuilder("add_rms_norm_quant")
                   .Opp("AddRmsNormQuant")
                   .Arity(4)
                   .AiCore()
                   .Inputs("x1,x2,gamma,scales1")
                   .OrderedArguments()
                   // The proto carries four trailing optionals (beta included) with
                   // optionalInputMode=gen_placeholder. They are padded as launch io
                   // placeholders only -- registering them as ir inputs makes tiling
                   // treat them as present and pick a kernel variant that DMAs from
                   // the null placeholder addresses.
                   .LaunchPadInputs("scales2,zero_points1,zero_points2,beta")
                   .Outputs("y1,y2,x")
                   .OutputDtypes("DT_INT8,DT_INT8,")
                   .Attr("axis", GeAttrType::kInt, -1)
                   .Attr("epsilon", GeAttrType::kFloat, 0, "1e-06")
                   .Attr("div_mode", GeAttrType::kBool, 1)
                   .Attr("dst_type", GeAttrType::kInt, 2)),
              (BehaviorBuilder()
                   .CaptureAttr(IrInputKind::kInt, "axis")
                   .CaptureAttr(IrInputKind::kBool, "div_mode")
                   .ClassifyAt(IrInputKind::kDouble, InputRole::kSkip, 0)
                   .ClassifyAt(IrInputKind::kDouble, InputRole::kSkip, 1)
                   .ClassifyAt(IrInputKind::kDouble, InputRole::kSkip, 2)
                   .ClassifyAt(IrInputKind::kDouble, InputRole::kSkip, 3)
                   .CaptureAttr(IrInputKind::kDouble, "epsilon")
                   .ClassifyAt(IrInputKind::kDouble, InputRole::kSkip, 5)));
// CrossEntropyLoss: the fx_backend pre-flatten hook already normalized the input
// layout to [x, target, weight, reduction, ignore_index, label_smoothing,
// lse_square_scale, return_zloss]. weight stays unconnected on the default path and
// rides as a gen_placeholder pad; the string reduction is captured through the
// kDouble bucket (IrInputKind has no kStr) so it selects the right kernel variant.
// Attrs follow the proto order exactly (tiling reads runtime_attrs by index).
// OrderedArguments keeps the None weight slot / reduction string / zloss bool in the
// IR (the normalized bag would drop them); HasInputRole probes every rule of a kind,
// so capture and ordinal-skip rules mix freely in declaration order.
FX_CATALOG_OP((CatalogRowBuilder("cross_entropy_loss")
                   .Opp("CrossEntropyLoss")
                   .Arity(2)
                   .AiCore()
                   .Inputs("input,target")
                   .LaunchPadInputs("weight")
                   .OrderedArguments()
                   .Outputs("loss,log_prob,zloss,lse_for_zloss")
                   .Attr("reduction", GeAttrType::kStr, 0, "mean")
                   .Attr("ignore_index", GeAttrType::kInt, -100)
                   .Attr("label_smoothing", GeAttrType::kFloat, 0.0)
                   .Attr("lse_square_scale_for_zloss", GeAttrType::kFloat, 0.0)
                   .Attr("return_zloss", GeAttrType::kBool, 0)),
              (BehaviorBuilder()
                   .CaptureAxis("ignore_index")
                   .CaptureAttr(IrInputKind::kDouble, "reduction")
                   .CaptureAttr(IrInputKind::kDouble, "label_smoothing")
                   .ClassifyAt(IrInputKind::kDouble, InputRole::kSkip, 0)
                   .ClassifyAt(IrInputKind::kDouble, InputRole::kSkip, 3)
                   .CaptureAttr(IrInputKind::kBool, "return_zloss")));
// DequantSwigluQuant: only the mandatory x input is wired on this path; the optional
// scales/bias stay absent. The GE proto attrs are all required and tiling reads them
// by proto index, so the declaration order below must match the proto registration
// order (the 16-slot torch schema attr tail). Every knob is pinned to the aclnn V1
// default the eager path takes: quant_mode="static", dst_type=1 (int8 y),
// round_mode="rint", activate_dim=-1, swiglu_mode=0; the float defaults ride the
// text channel because GeAttr::literal is int64. The fxrt op_hook normalizes the
// call to the 9-slot V1 layout, so the IR carries one Bool (activate_left) plus the
// quant_mode String the scan classifies as kDouble -- the behavior captures the
// bool and drops the string.
FX_CATALOG_OP((CatalogRowBuilder("dequant_swiglu_quant")
                   .Opp("DequantSwigluQuant")
                   .Arity(1)
                   .AiCore()
                   .Inputs("x")
                   // The proto registers six optional inputs under gen_placeholder
                   // mode (weight/activation scales, bias, quant scale/offset and
                   // group_index); the launch pads one io slot per name without
                   // registering the ir inputs, so tiling stays on the no-optional
                   // kernel variant and every later output slot keeps its proto index.
                   .LaunchPadInputs("weight_scale,activation_scale,bias,quant_scale,quant_offset,group_index")
                   .Outputs("y,scale")
                   .OutputDtypes("DT_INT8,")
                   .Attr("activate_left", GeAttrType::kBool, 0)
                   .Attr("quant_mode", GeAttrType::kStr, 0, "static")
                   .Attr("dst_type", GeAttrType::kInt, 1)
                   .Attr("round_mode", GeAttrType::kStr, 0, "rint")
                   .Attr("activate_dim", GeAttrType::kInt, -1)
                   .Attr("swiglu_mode", GeAttrType::kInt, 0)
                   .Attr("clamp_limit", GeAttrType::kFloat, 0, "7.0")
                   .Attr("glu_alpha", GeAttrType::kFloat, 0, "1.702")
                   .Attr("glu_bias", GeAttrType::kFloat, 0, "1.0")),
              (BehaviorBuilder()
                   .CaptureAttr(IrInputKind::kBool, "activate_left")
                   .Classify(IrInputKind::kDouble, InputRole::kSkip)));
// masked_fill_tensor / inplace_masked_fill_tensor: MaskedFill already takes value as a
// tensor (the scalar rows materialize a python number into that same slot). The two
// rows share one proto so a value-dtype or broadcast fix covers both.
FX_CATALOG_OP((CatalogRowBuilder("masked_fill_tensor").Opp("MaskedFill").Arity(3).AiCore().Inputs("x,mask,value")));
FX_CATALOG_OP((CatalogRowBuilder("inplace_masked_fill_tensor")
                   .Opp("MaskedFill")
                   .Arity(3)
                   .AiCore()
                   .Inputs("x,mask,value")
                   .RefInput(0)));
// inplace_fill_scalar is Fills, the same REQUIRED_ATTR value path as full_like.
FX_CATALOG_OP((CatalogRowBuilder("inplace_fill_scalar")
                   .Opp("Fills")
                   .Arity(1)
                   .StridedInputs()
                   .OrderedArguments()
                   .Inputs("x")
                   .RefInput(0)
                   .RequiredAttr("value", GeAttrType::kFloat)),
              (BehaviorBuilder().CaptureFill()));
// inplace_copy is Assign. non_blocking is an undeclared bool and is ignored. Attrs follow
// the proto order (validate_shape default true, use_locking default false).
FX_CATALOG_OP((CatalogRowBuilder("inplace_copy")
                   .Opp("Assign")
                   .Arity(2)
                   .StridedInputs()
                   .Inputs("ref,value")
                   .Outputs("ref")
                   .RefInput(0)
                   .Attr("validate_shape", GeAttrType::kBool, 1)
                   .Attr("use_locking", GeAttrType::kBool, 0)));
// scatter_nd_update_ writes var in place. Rows the indices do not cover stay in that storage.
FX_CATALOG_OP((CatalogRowBuilder("scatter_nd_update_")
                   .Opp("ScatterNdUpdate")
                   .Arity(3)
                   .AiCore()
                   .Inputs("var,indices,updates")
                   .Outputs("var")
                   .RefInput(0)
                   .Attr("use_locking", GeAttrType::kBool, 0)));
// apply_rotary_pos_emb writes query and key in place. The arg hook supplies layout as an
// Int; rotary_mode stays the proto default. Both outputs reuse the matching inputs.
FX_CATALOG_OP((CatalogRowBuilder("apply_rotary_pos_emb")
                   .Opp("ApplyRotaryPosEmb")
                   .Arity(4)
                   .AiCore()
                   .Inputs("query,key,cos,sin")
                   .Outputs("query,key")
                   .Ref(0, 0)
                   .Ref(1, 1)
                   .Attr("layout", GeAttrType::kInt, 1)
                   .Attr("rotary_mode", GeAttrType::kStr, 0, "half")),
              (BehaviorBuilder().CaptureAttr(IrInputKind::kInt, "layout")));
// reshape_and_cache writes key_cache and value_cache. Input order is
// key, value, key_cache, value_cache, slot_mapping. Unmapped slots stay in those caches.
// The IR node is void; the output hook supplies the two cache tensors so this row lowers.
FX_CATALOG_OP((CatalogRowBuilder("reshape_and_cache")
                   .Opp("ReshapeAndCache")
                   .Arity(5)
                   .AiCore()
                   .Inputs("keyIn,valueIn,keyCacheIn,valueCacheIn,slotMapping")
                   .Outputs("keyCacheOut,valueCacheOut")
                   .Ref(0, 2)
                   .Ref(1, 3)));
// reshape_and_cache_siso writes key_cache only. Input order is key, key_cache,
// slot_mapping. The IR node is void; the output hook publishes key_cache.
FX_CATALOG_OP((CatalogRowBuilder("reshape_and_cache_siso")
                   .Opp("ReshapeAndCacheNdSiso")
                   .Arity(3)
                   .AiCore()
                   .Inputs("keyIn,keyCacheIn,slotMapping")
                   .Outputs("keyCacheOut")
                   .Ref(0, 1)));
// MoeReRouting: 910B keys are 2 inputs + 4 outputs. per_token_scales is optional and
// absent on the shipped keys, so it is a launch pad (gen_placeholder) rather than a
// select input. Attrs follow the proto order.
FX_CATALOG_OP((CatalogRowBuilder("moe_re_routing")
                   .Opp("MoeReRouting")
                   .Arity(2)
                   .AiCore()
                   .Inputs("tokens,expert_token_num_per_rank")
                   .LaunchPadInputs("per_token_scales")
                   .Outputs("permute_tokens,permute_per_token_scales,permute_token_idx,expert_token_num")
                   .Attr("expert_token_num_type", GeAttrType::kInt, 1)
                   .Attr("idx_type", GeAttrType::kInt, 0)),
              (BehaviorBuilder()
                   .CaptureAttr(IrInputKind::kInt, "expert_token_num_type")
                   .CaptureAttr(IrInputKind::kInt, "idx_type")));
// MoeTokenUnpermute: 910B keys are permuted_tokens + sorted_indices + one output.
// probs is optional and omitted from every shipped key, so it is a launch pad. The
// with-probs call has no binary on this pack.
FX_CATALOG_OP((CatalogRowBuilder("moe_token_unpermute")
                   .Opp("MoeTokenUnpermute")
                   .Arity(2)
                   .AiCore()
                   .Inputs("permuted_tokens,sorted_indices")
                   .LaunchPadInputs("probs")
                   .Outputs("unpermuted_tokens")
                   .Attr("padded_mode", GeAttrType::kBool, 0)
                   .Attr("restore_shape", GeAttrType::kListInt, 0)),
              (BehaviorBuilder().CaptureAttr(IrInputKind::kBool, "padded_mode").SkipInt()));

// Elementwise ops whose 910B binary matches the torch contract (one tensor in, one out,
// or two tensors in). Activations with a baked alpha (Elu/Selu/Softplus) are not listed:
// the binary has no scale input, and a graph rewrite would only fake that input.
FX_CATALOG_OP((CatalogRowBuilder("tanh").Opp("Tanh").Arity(1).AiCore()));
FX_CATALOG_OP((CatalogRowBuilder("sqrt").Opp("Sqrt").Arity(1).AiCore()));
FX_CATALOG_OP((CatalogRowBuilder("sin").Opp("Sin").Arity(1).AiCore()));
FX_CATALOG_OP((CatalogRowBuilder("cos").Opp("Cos").Arity(1).AiCore()));
FX_CATALOG_OP((CatalogRowBuilder("reciprocal").Opp("Reciprocal").Arity(1).AiCore()));
FX_CATALOG_OP((CatalogRowBuilder("floor").Opp("Floor").Arity(1).AiCore()));
FX_CATALOG_OP((CatalogRowBuilder("ceil").Opp("Ceil").Arity(1).AiCore()));
FX_CATALOG_OP((CatalogRowBuilder("erf").Opp("Erf").Arity(1).AiCore()));
FX_CATALOG_OP((CatalogRowBuilder("sign").Opp("Sign").Arity(1).AiCore()));
FX_CATALOG_OP((CatalogRowBuilder("trunc").Opp("Trunc").Arity(1).AiCore()));
// Round stores decimals in compileInfo._var_attrs. The same .o covers every value;
// a graph Int overrides the default 0, and an omitted decimals stays 0.
FX_CATALOG_OP((CatalogRowBuilder("round").Opp("Round").Arity(1).AiCore().Attr("decimals", GeAttrType::kInt, 0)),
              (BehaviorBuilder().CaptureAxis("decimals")));
FX_CATALOG_OP((CatalogRowBuilder("log1p").Opp("Log1p").Arity(1).AiCore()));
FX_CATALOG_OP((CatalogRowBuilder("expm1").Opp("Expm1").Arity(1).AiCore()));
FX_CATALOG_OP((CatalogRowBuilder("atan").Opp("Atan").Arity(1).AiCore()));
FX_CATALOG_OP((CatalogRowBuilder("erfc").Opp("Erfc").Arity(1).AiCore()));
FX_CATALOG_OP((CatalogRowBuilder("mish").Opp("Mish").Arity(1).AiCore()));
// HardSigmoid stores alpha and beta in _var_attrs. torch.nn.functional.hardsigmoid is
// clamp(x / 6 + 1/2, 0, 1), so the defaults are 1/6 and 1/2. Other alphas stay a gap.
FX_CATALOG_OP((CatalogRowBuilder("hardsigmoid")
                   .Opp("HardSigmoid")
                   .Arity(1)
                   .AiCore()
                   .Attr("alpha", GeAttrType::kFloat, 0, "0.16666667")
                   .Attr("beta", GeAttrType::kFloat, 0, "0.5")));
FX_CATALOG_OP((CatalogRowBuilder("hardswish").Opp("HardSwish").Arity(1).AiCore()));
FX_CATALOG_OP((CatalogRowBuilder("maximum").Opp("Maximum").AiCore().Inputs("x1,x2")));
FX_CATALOG_OP((CatalogRowBuilder("minimum").Opp("Minimum").AiCore().Inputs("x1,x2")));
// LogicalOr bins are bool-only. A non-bool input has no binary and must not be rewritten.
FX_CATALOG_OP((CatalogRowBuilder("logical_or").Opp("LogicalOr").AiCore().Inputs("x1,x2")));
// Bool outputs, same contract as isnan: the bin ABI already says BOOL.
FX_CATALOG_OP((CatalogRowBuilder("isinf").Opp("IsInf").Arity(1).AiCore()));
FX_CATALOG_OP((CatalogRowBuilder("isfinite").Opp("IsFinite").Arity(1).AiCore()));
FX_CATALOG_OP((CatalogRowBuilder("triu").Opp("Triu").Arity(1).AiCore().Attr("diagonal", GeAttrType::kInt, 0)),
              (BehaviorBuilder().CaptureAxis("diagonal")));
// FlashAttentionScore via aclnn. Output order is softmax_max, softmax_sum,
// softmax_out, attention_out. The converter hook rebuilds the IR tuple in that
// order. Optional tensors stay off this row (the python hook demotes them).
FX_CATALOG_OP((CatalogRowBuilder("npu_fusion_attention")
                   .Opp("FlashAttentionScore")
                   .Arity(3)
                   .Outputs("softmax_max,softmax_sum,softmax_out,attention_out")
                   .OrderedArguments()
                   .Attr("scale_value", GeAttrType::kFloat, 0, "1")
                   .Attr("keep_prob", GeAttrType::kFloat, 0, "1")
                   .Attr("pre_tockens", GeAttrType::kInt, 2147483647)
                   .Attr("next_tockens", GeAttrType::kInt, 2147483647)
                   .RequiredAttr("head_num", GeAttrType::kInt)
                   .RequiredAttr("input_layout", GeAttrType::kStr, 0, "BSH")
                   .Attr("inner_precise", GeAttrType::kInt, 0)
                   .Attr("sparse_mode", GeAttrType::kInt, 0)
                   .Attr("pse_type", GeAttrType::kInt, 1)
                   .Attr("seed", GeAttrType::kInt, 0)
                   .Attr("offset", GeAttrType::kInt, 0)
                   .Attr("out_dtype", GeAttrType::kInt, 0)
                   .Attr("softmax_out_layout", GeAttrType::kStr, 0, "")),
              (BehaviorBuilder()
                   .CaptureAttr(IrInputKind::kInt, "head_num")
                   // Two kDouble slots. A catch-all (-1) would bind both to the first
                   // name, so the scale double overwrites the layout string.
                   .ClassifyAt(IrInputKind::kDouble, InputRole::kNamedAttr, 0, "input_layout")
                   .ClassifyAt(IrInputKind::kDouble, InputRole::kNamedAttr, 1, "scale_value")));

}  // namespace fx_converter
