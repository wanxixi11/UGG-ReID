from fvcore.nn.jit_handles import elementwise_flop_counter
from functools import partial

import math


def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    assert not with_complex
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops


def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try:
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)


def selective_scan_flop_jit(inputs, outputs, flops_fn=flops_selective_scan_fn):
    print_jit_input_names(inputs)
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False)
    return flops


def flops_scaled_dot_product_attention_fn(query, key, value, scale=None, dropout_p=0.0):
    """
    Calculate the FLOPs for scaled dot-product attention mechanism.

    Args:
        query: Tensor of shape (batch, num_heads, L, dim)
        key: Tensor of shape (batch, num_heads, S, dim)
        value: Tensor of shape (batch, num_heads, S, dim)
        scale: Optional scaling factor (if None, default to 1/sqrt(dim))
        dropout_p: Dropout probability (if greater than 0, dropout is applied)

    Returns:
        total_flops: Total FLOPs for the scaled dot-product attention.
    """
    # Get dimensions
    batch, num_heads, L, dim = query.type().sizes()
    _, _, S, _ = key.type().sizes()  # key and value have the same batch, num_heads, and dim as query

    # 1. FLOPs for dot product QK^T (batch, num_heads, L, S)
    # Matrix multiplication: Q (batch, num_heads, L, dim) * K^T (batch, num_heads, dim, S)
    # Each element in the output requires dim multiplications and dim-1 additions.
    flops_qk = batch * num_heads * L * S * dim  # Multiplications and additions

    # 2. FLOPs for scaling by sqrt(dim): Scalar multiplication for each element in QK^T
    flops_scale = batch * num_heads * L * S  # Just scalar multiplications

    # 3. FLOPs for adding attention bias (e.g., for masking)
    # Simple element-wise addition
    flops_bias = batch * num_heads * L * S

    # 4. FLOPs for Softmax
    # Softmax involves exponentiation, sum, and division for normalization
    # Assuming 5 operations per element (2 for exponentiation, 3 for normalization)
    flops_softmax = batch * num_heads * L * S

    # 5. Dropout FLOPs
    # Dropout does not directly contribute to FLOPs, as it's only a mask operation.
    # We can safely ignore it in FLOPs calculation.

    # 6. FLOPs for matrix multiplication Attention * Value (batch, num_heads, L, dim)
    # Matrix multiplication: A (batch, num_heads, L, S) * V (batch, num_heads, S, dim)
    flops_av = batch * num_heads * L * S * dim  # Multiplications

    # Total FLOPs
    total_flops = flops_qk + flops_scale + flops_bias + flops_softmax + flops_av

    return total_flops


def scaled_dot_product_attention(inputs, outputs, flops_fn=flops_scaled_dot_product_attention_fn):
    query, key, value = inputs[0], inputs[1], inputs[2]
    # Assuming query, key, and value are the inputs for the attention mechanism
    flops = flops_fn(query, key, value)

    # Your logic for scaled dot product attention should go here.
    # For now, we'll just print the FLOPs.
    # print(f"FLOPs for scaled dot product attention: {flops}")

    return flops


def MambaInnerFn_jit(inputs, outputs):
    """
    conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
    x, z = xz.chunk(2, dim=1)
    conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias,None, True)
    x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
    delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l = L)
    B = x_dbl[:, delta_rank:delta_rank + d_state]
    B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
    C = x_dbl[:, -d_state:]
    C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
    out, scan_intermediates, out_z = selective_scan_cuda.fwd(
        conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
    )
    F.linear(rearrange(out_z, "b d l -> b l d"), out_proj_weight, out_proj_bias)
    """
    xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight, out_proj_weight, A, D, delta_bias = inputs[:]
    Batch, _, L = xz.type().sizes()
    CWidth = conv1d_weight.type().sizes()[-1]
    H = A.type().sizes()[-1]  # 16
    Dim, R = delta_proj_weight.type().sizes()
    assert tuple(xz.type().sizes()) == (Batch, 2 * Dim, L)
    assert tuple(conv1d_weight.type().sizes()) == (Dim, 1, CWidth)
    assert tuple(x_proj_weight.type().sizes()) == (R + H + H, Dim)
    assert tuple(A.type().sizes()) == (Dim, H)

    with_Z = True
    with_D = False
    if "D" in inputs[7].debugName():
        assert tuple(inputs[7].type().sizes()) == (Dim,)
        with_D = True

    flops = 0
    flops += Batch * (Dim * L) * CWidth  # causal_conv1d_cuda.causal_conv1d_fwd
    flops += Batch * (Dim * L) * (R + H + H)  # x_dbl = F.linear(...
    flops += Batch * (Dim * R) * (L)  # delta_proj_weight @ x_dbl[:, :delta_rank]

    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * Batch * L * Dim * H
    if with_D:
        flops += Batch * Dim * L
    if with_Z:
        flops += Batch * Dim * L

    out_weight_shape = out_proj_weight.type().sizes()
    assert out_weight_shape[1] == Dim
    flops += Batch * Dim * L * out_weight_shape[0]

    return flops


def embedding_jit(inputs, outputs):
    return 0


def give_supported_ops():
    return {
        "aten::silu": elementwise_flop_counter(0, 1),
        "aten::gelu": elementwise_flop_counter(0, 1),
        "aten::neg": elementwise_flop_counter(0, 1),
        "aten::exp": elementwise_flop_counter(0, 1),
        "aten::flip": elementwise_flop_counter(0, 1),
        "aten::mul": elementwise_flop_counter(0, 1),
        "aten::div": elementwise_flop_counter(0, 1),
        "aten::softmax": elementwise_flop_counter(0, 2),
        "aten::sigmoid": elementwise_flop_counter(0, 1),
        "aten::add": elementwise_flop_counter(0, 1),
        "aten::add_": elementwise_flop_counter(0, 1),
        "aten::radd": elementwise_flop_counter(0, 1),
        "aten::sub": elementwise_flop_counter(0, 1),
        "aten::sub_": elementwise_flop_counter(0, 1),
        "aten::rsub": elementwise_flop_counter(0, 1),
        "aten::mul_": elementwise_flop_counter(0, 1),
        "aten::rmul": elementwise_flop_counter(0, 1),
        "aten::div_": elementwise_flop_counter(0, 1),
        "aten::rdiv": elementwise_flop_counter(0, 1),
        "aten::cumsum": elementwise_flop_counter(0, 1),
        "aten::ne": elementwise_flop_counter(0, 1),
        "aten::silu_": elementwise_flop_counter(0, 1),
        "aten::dropout_": elementwise_flop_counter(0, 1),
        "aten::log_softmax": elementwise_flop_counter(0, 2),
        "aten::argmax": elementwise_flop_counter(0, 1),
        "aten::one_hot": elementwise_flop_counter(0, 1),
        "aten::flatten": elementwise_flop_counter(0, 0),
        "aten::unflatten": elementwise_flop_counter(0, 0),
        "aten::mean": elementwise_flop_counter(1, 0),
        "aten::sum": elementwise_flop_counter(1, 0),
        "aten::abs": elementwise_flop_counter(0, 1),
        "aten::tanh": elementwise_flop_counter(0, 1),
        "aten::relu": elementwise_flop_counter(0, 1),
        "aten::where": elementwise_flop_counter(0, 1),
        "aten::le": elementwise_flop_counter(0, 1),
        "aten::topk": elementwise_flop_counter(1, 1),
        "aten::sort": elementwise_flop_counter(1, 1),
        "aten::argsort": elementwise_flop_counter(1, 1),
        "aten::scatter": elementwise_flop_counter(1, 1),
        "aten::gather": elementwise_flop_counter(1, 1),
        "aten::adaptive_max_pool2d": elementwise_flop_counter(1, 0),
        "aten::adaptive_max_pool1d": elementwise_flop_counter(1, 0),
        "aten::adaptive_avg_pool2d": elementwise_flop_counter(1, 0),
        "aten::avg_pool1d": elementwise_flop_counter(1, 0),
        'aten::adaptive_avg_pool1d': elementwise_flop_counter(1, 0),
        "aten::repeat": elementwise_flop_counter(0, 1),
        "aten::pow": elementwise_flop_counter(0, 1),
        "prim::PythonOp.SelectiveScanMamba": partial(selective_scan_flop_jit, flops_fn=flops_selective_scan_fn),
        "aten::scaled_dot_product_attention": partial(scaled_dot_product_attention,
                                                      flops_fn=flops_scaled_dot_product_attention_fn),
        "prim::PythonOp.MambaInnerFn": MambaInnerFn_jit,
        # "aten::embedding": embedding_jit,
    }
