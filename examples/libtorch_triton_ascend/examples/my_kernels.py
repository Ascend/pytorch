import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr: tl.tensor,
               y_ptr: tl.tensor,
               output_ptr,
               n_elements,
               BLOCK_SIZE: tl.constexpr,
               ):
    """向量加法 kernel（参考 test_libtorch_ffmh）"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def layer_norm_kernel(
    output_ptr,
    input_ptr,
    weight_ptr,
    bias_ptr,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    stride,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused LayerNorm kernel: output = (x - mean) / sqrt(var + eps) * weight + bias

    每个 program 处理一行数据，计算该行的均值和方差后归一化。
    支持 2D 输入 (n_rows x n_cols)，weight/bias 为长度 n_cols 的向量。
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # 加载一行数据
    x = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=0.0)

    # 计算均值
    x_mean = tl.sum(x, axis=0) / n_cols

    # 计算方差
    x_centered = x - x_mean
    x_var = tl.sum(x_centered * x_centered, axis=0) / n_cols

    # 归一化
    rstd = 1.0 / tl.sqrt(x_var + eps)
    x_norm = x_centered * rstd

    # 仿射变换: weight * x_norm + bias
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)
    output = weight * x_norm + bias

    tl.store(output_ptr + row_start + col_offsets, output, mask=mask)
