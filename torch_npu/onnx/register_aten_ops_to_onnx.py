import torch
import torch.onnx.symbolic_registry as sym_registry
from torch.onnx import symbolic_helper
from torch.onnx.symbolic_opset9 import sub, mul, add, pow, sqrt, reciprocal


@symbolic_helper.parse_args("v", "is", "v", "v", "f")
def native_layer_norm(
    g,
    input,
    normalized_shape,
    weight,
    bias,
    eps):

    axes = [-i for i in range(len(normalized_shape), 0, -1)]

    two_cst = symbolic_helper._generate_wrapped_number(g, 2.0)
    eps_cst = symbolic_helper._generate_wrapped_number(g, eps)

    mean = g.op("ReduceMean", input, axes_i=axes)
    numerator = sub(g, input, mean)

    # Cast it to eps dtype to avoid precision loss
    is_type_half = numerator.type().scalarType() == "Half"
    if is_type_half:
        numerator = g.op(
            "Cast", numerator, to_i=symbolic_helper.cast_pytorch_to_onnx["Double"])

    # variance = e((x - e(x))^2), and (x - e(x)) is the numerator in the layer_norm formula
    variance = g.op("ReduceMean", pow(g, numerator, two_cst), axes_i=axes)
    denominator = sqrt(g, g.op("Add", variance, eps_cst))
    normalized = g.op("Div", numerator, denominator)

    # Cast back to input type as eps related ops are all done
    input_dtype_i = symbolic_helper.cast_pytorch_to_onnx[input.type().scalarType()]
    if is_type_half:
        normalized = g.op(
            "Cast", normalized, to_i=input_dtype_i
        )
    if not (weight is None or symbolic_helper._is_none(weight)):
        normalized = mul(g, normalized, weight)
    if not (bias is None or symbolic_helper._is_none(bias)):
        normalized = add(g, normalized, bias)

    if is_type_half:
        denominator = g.op(
            "Cast", denominator, to_i=input_dtype_i
        )
        rdenominator = g.op("Reciprocal", denominator)
    else:
        rdenominator = reciprocal(g, denominator)

    return normalized, mean, rdenominator


def register_aten_ops_symbolic():
    sym_registry.register_op("native_layer_norm", op=native_layer_norm, domain="", version=11)
