import math
from typing import List, Optional, Union

import torch
import triton
import triton.language as tl
from torch.autograd.function import FunctionCtx
from torch.cuda.amp import custom_fwd


def attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    sm_scale: float,
    is_causal: bool,
    attention_mask: Union[torch.Tensor, None],
) -> torch.Tensor:
    """
    Reference implementation for attention
    @param q: Query matrix size (batch, heads, m_size, BLOCK_DHEAD)
    @param k: Key matrix size (batch, heads, n_size, BLOCK_DHEAD)
    @param v: Value matrix size (batch, heads, n_size, BLOCK_DHEAD)
    @param output: Output matrix size (batch, heads, m_size, BLOCK_DHEAD)
    @param sm_scale: SM (softmax) scaling factor applied on Q•K^T just before the softmax
    @param is_causal: Whether to apply causal attention
    @param attention_mask: Attention mask broadcastable to (batch, heads, m_size, n_size). Warning the mask
    isn't a binary mask like the one you use normally. This mask is directly added to QxK.
    @return:
    """
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale

    if attention_mask is not None:
        p += attention_mask
    if is_causal:
        m_size = q.size(2)
        n_size = k.size(2)
        M = torch.tril(torch.ones((m_size, n_size), device="cuda"))
        p = torch.where(M == 0, float("-inf"), p)
    p = torch.nn.functional.softmax(p, dim=-1)
    ref_out = torch.matmul(p.to(v.dtype), v, out=output)
    return ref_out


def closest_power_of_2(n: int, min_range: int = 16, max_range: int = 128) -> List[int]:
    """return the closests power of 2 for n, in 16-128 range"""
    n = max(min(n, max_range), min_range)
    min_range = math.floor(math.log2(n - 1))
    max_range = math.ceil(math.log2(n + 1))
    ranges = [2**i for i in range(min_range, max_range + 1)]
    return ranges


def prune(configs, named_args):
    """remove block shapes unlikely to provide optimal speedup"""
    pruned_configs = []
    sizes_m = closest_power_of_2(named_args["m_size"])
    sizes_n = closest_power_of_2(named_args["n_size"])
    is_causal = named_args["IS_CAUSAL"]
    for c in configs:
        if is_causal and c.kwargs["BLOCK_M_SIZE"] != c.kwargs["BLOCK_N_SIZE"]:
            continue
        if c.kwargs["BLOCK_M_SIZE"] in sizes_m and c.kwargs["BLOCK_N_SIZE"] in sizes_n:
            pruned_configs.append(c)

    assert len(pruned_configs) > 0
    return pruned_configs


@triton.jit
def _fwd_kernel(
    head_size,
    m_size,
    n_size,
    q_ptr,
    k_ptr,
    v_ptr,
    attention_mask_ptr,
    output_ptr,
    q_batch_stride,
    q_head_stride,
    q_m_stride,
    q_k_stride,
    k_batch_stride,
    k_head_stride,
    k_n_stride,  # k
    k_k_stride,  # n, transposed
    v_batch_stride,
    v_head_stride,
    v_k_stride,
    v_n_stride,
    output_batch_stride,
    output_head_stride,
    output_row_stride,
    output_col_stride,
    attention_mask_batch_stride,
    attention_mask_head_stride,
    attention_mask_m_stride,
    attention_mask_n_stride,
    min_clamp_value,
    attention_mask_batch_size,
    attention_mask_head_size,
    attention_mask_m_size,
    attention_mask_n_size,
    BLOCK_DHEAD_SIZE: tl.constexpr,
    BLOCK_M_SIZE: tl.constexpr,
    BLOCK_N_SIZE: tl.constexpr,
):
    block_m_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    current_batch_idx = head_idx // head_size
    current_head_idx = head_idx % head_size

    m_range_offs = tl.arange(0, BLOCK_M_SIZE)  # first block on M dimension
    n_range_offs = tl.arange(0, BLOCK_N_SIZE)  # first block on N dimension
    dhead_range_offs = tl.arange(0, BLOCK_DHEAD_SIZE)  # full head

    m_offs = block_m_idx * BLOCK_M_SIZE + m_range_offs  # rows offsets on M axis

    q_offs = (
        current_batch_idx * q_batch_stride
        + current_head_idx * q_head_stride
        + (m_offs[:, None] * q_m_stride + dhead_range_offs[None, :] * q_k_stride)
    )

    k_offs = (
        current_batch_idx * k_batch_stride
        + current_head_idx * k_head_stride
        + (n_range_offs[:, None] * k_n_stride + dhead_range_offs[None, :] * k_k_stride)
    )

    v_offs = (
        current_batch_idx * v_batch_stride
        + current_head_idx * v_head_stride
        + (n_range_offs[:, None] * v_k_stride + dhead_range_offs[None, :] * v_n_stride)
    )

    output_offs = (
        current_batch_idx * output_batch_stride
        + current_head_idx * output_head_stride
        + (
            m_offs[:, None] * output_row_stride
            + dhead_range_offs[None, :] * output_col_stride
        )
    )

    q_ptrs = q_ptr + q_offs
    k_ptrs = k_ptr + k_offs
    v_ptrs = v_ptr + v_offs
    output_ptrs = output_ptr + output_offs

    l_i = tl.zeros((BLOCK_M_SIZE,), dtype=tl.float32) - float("inf")
    d_i = tl.zeros((BLOCK_M_SIZE,), dtype=tl.float32)

    acc = tl.zeros((BLOCK_M_SIZE, BLOCK_DHEAD_SIZE), dtype=tl.float32)
    q = tl.load(q_ptrs)
    block_n_end = n_size

    attention_mask_batch_idx = (current_batch_idx,)
    if attention_mask_batch_size == 1:
        attention_mask_batch_idx = 0

    attention_mask_head_idx = current_head_idx
    if attention_mask_head_size == 1:
        attention_mask_head_idx = 0

    attention_mask_off = (
        attention_mask_batch_idx * attention_mask_batch_stride
        + attention_mask_head_idx * attention_mask_head_stride
    )

    for block_n_start_idx in range(0, block_n_end, BLOCK_N_SIZE):
        block_n_offs = block_n_start_idx + n_range_offs
        k = tl.load(k_ptrs + block_n_start_idx * k_n_stride)
        qk = tl.zeros((BLOCK_M_SIZE, BLOCK_N_SIZE), dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        attention_mask_offs = (
            attention_mask_off + block_n_offs * attention_mask_n_stride
        )
        attention_mask_offs = (
            attention_mask_offs[None, :] + m_offs[:, None] * attention_mask_m_stride
        )
        attention_mask = tl.load(
            attention_mask_ptr + attention_mask_offs,
            eviction_policy="evict_first",
        )
        attention_mask = tl.where(
            attention_mask == float("-inf"), min_clamp_value, attention_mask
        )
        qk += attention_mask
        l_j = tl.max(qk, 1)
        numerators = tl.exp(qk - l_j[:, None])
        d_j = tl.sum(numerators, 1)
        l_new = tl.maximum(l_i, l_j)
        alpha = tl.exp(l_i - l_new)
        beta = tl.exp(l_j - l_new)
        d_new = alpha * d_i + beta * d_j
        p_scale = beta / d_new
        qk_softmax = numerators * p_scale[:, None]
        acc_scale = d_i / d_new * alpha
        acc = acc * acc_scale[:, None]
        v = tl.load(v_ptrs + block_n_start_idx * v_k_stride)
        qk_softmax = qk_softmax.to(q_ptr.dtype.element_ty)
        acc += tl.dot(qk_softmax, v)
        d_i = d_new
        l_i = l_new

    tl.store(output_ptrs, acc)


class TileAttention(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx: FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        output: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        batch, head_size, m_size, dhead = q.size()
        n_size = k.size(2)

        grid = lambda args: (
            triton.cdiv(m_size, args["BLOCK_M_SIZE"]),
            batch * head_size,
        )

        _fwd_kernel[grid](
            head_size,  # heads
            m_size,  # m_size
            n_size,  # n_size
            q,  # Q
            k,  # K
            v,  # V
            attention_mask,  # attention_mask
            output,  # output
            *q.stride(),  # (batch, heads, m_size, size_k)
            *k.stride(),  # (batch, heads, n_size, size_k)
            *v.stride(),  # (batch, heads, size_k, n_size)
            *output.stride(),  # (batch, heads, m_size, n_size)
            *attention_mask.stride(),  # (batch, heads, m_size, size_k)
            torch.finfo(attention_mask.dtype).min,  # min_clamp_value
            *attention_mask.size(),  # (batch, heads, m_size, size_k)
            dhead,  # BLOCK_DHEAD
            128,  # BLOCK_M_SIZE
            128,  # BLOCK_N_SIZE
            num_warps=4 if k.size(3) <= 64 else 8,
            num_stages=2,
        )
        return output


def attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
):
    return TileAttention.apply(q, k, v, output, attention_mask)


b = 1
h = 32
s = 128
d = 32

# b h s d
q = torch.rand([b, h, s, d]).cuda().half()
k = torch.rand([b, h, s, d]).cuda().half()
v = torch.rand([b, h, s, d]).cuda().half()
o = torch.rand([b, h, s, d]).cuda().half()
o1 = torch.rand([b, h, s, d]).cuda().half()
# b h s s
m = torch.ones([b, h, s, s]).cuda().half()
# fa
attention_forward(q, k, v, o, m)
attention_reference(q, k, v, o1, 1.0, False, m)

assert torch.allclose(o, o1, rtol=1e-3, atol=1e-3)
