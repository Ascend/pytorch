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

    # variance = e((x - e(x))^2), and (x - e(x)) is the numerator in the layer_norm formula
    variance = g.op("ReduceMean", pow(g, numerator, two_cst), axes_i=axes)
    denominator = sqrt(g, g.op("Add", variance, eps_cst))
    normalized = g.op("Div", numerator, denominator)

    if not (weight is None or symbolic_helper._is_none(weight)):
        normalized = mul(g, normalized, weight)
    if not (bias is None or symbolic_helper._is_none(bias)):
        normalized = add(g, normalized, bias)

    # rdenominator := 1 / sqrt(variance + eps)
    rdenominator = reciprocal(g, denominator)

    return normalized, mean, rdenominator


def register_aten_ops_symbolic():
    sym_registry.register_op("native_layer_norm", op=native_layer_norm, domain="", version=11)
    sym_registry.register_op("native_layer_norm", op=native_layer_norm, domain="", version=12)
    sym_registry.register_op("native_layer_norm", op=native_layer_norm, domain="", version=13)
    sym_registry.register_op("native_layer_norm", op=native_layer_norm, domain="", version=14)
    sym_registry.register_op("native_layer_norm", op=native_layer_norm, domain="", version=15)
