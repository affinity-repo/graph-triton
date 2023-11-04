from ctypes import c_void_p, c_long
import torch
import math
import random
from torch import empty_strided, as_strided, device
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
async_compile = AsyncCompile()

import triton
import triton.language as tl
from torch._inductor.triton_ops.autotune import grid
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


@triton.jit
def triton__0(
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
    BLOCK_V_DHEAD_SIZE: tl.constexpr,
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
    vhead_range_offs = tl.arange(0, BLOCK_V_DHEAD_SIZE)  # full head

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
        + (n_range_offs[:, None] * v_k_stride + vhead_range_offs[None, :] * v_n_stride)
    )

    output_offs = (
        current_batch_idx * output_batch_stride
        + current_head_idx * output_head_stride
        + (
            m_offs[:, None] * output_row_stride
            + vhead_range_offs[None, :] * output_col_stride
        )
    )

    q_ptrs = q_ptr + q_offs
    k_ptrs = k_ptr + k_offs
    v_ptrs = v_ptr + v_offs
    output_ptrs = output_ptr + output_offs

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

    acco = tl.zeros((BLOCK_M_SIZE, BLOCK_V_DHEAD_SIZE), dtype=tl.float32)
    r_wo_clamp = tl.zeros(((BLOCK_M_SIZE,)), dtype=tl.float32)
    r = tl.zeros(((BLOCK_M_SIZE,)), dtype=tl.float32)

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
        v = tl.load(v_ptrs + block_n_start_idx * v_k_stride)

        tmp0 = qk
        tmp1 = attention_mask
        tmp4 = r
        tmp9 = acco
        tmp8 = v

        tmp2 = tmp0 * tmp1
        tmp3 = tl.abs(tmp2)
        r_wo_clamp += tl.sum(tmp3, axis=1)
        tmp5 = tl.where(r_wo_clamp > 1, r_wo_clamp , 1)
        tmp9 = tmp9 * tmp4 / tmp5
        tmp6 = tmp0 / tmp5
        tmp7 = tmp6.to(tl.float16)
        acco += tl.dot(tmp7, tmp8)
        r = tmp5

    tl.store(output_ptrs, acco)


async_compile.wait(globals())
del async_compile


def welder_config():
    from loguru import logger

    logger.info("Extracting config from welder config")

    batch, head_size, m_size, dhead, n_size, vhead, grid_1, grid_0 = (4, 8, 2048, 128, 2048, 256, 64, 128,)

    grid = (grid_0, grid_1)

    return (grid, batch, head_size, m_size, n_size, dhead, vhead)


def call(args):
    (arg0_1, arg0_2, arg0_3, arg0_4, buf0, the_welder_configs) = args
    args.clear()
    grid, batch, head_size, m_size, n_size, dhead, vhead = the_welder_configs
    triton__0[grid](
        head_size,
        m_size,
        n_size,
        arg0_1,
        arg0_2,
        arg0_3,
        arg0_4,
        buf0,
        *arg0_1.stride(),
        *arg0_2.stride(),
        *arg0_3.stride(),
        *buf0.stride(),
        *arg0_4.stride(),
        0.0,
        *arg0_4.size(),
        dhead,
        vhead,
        64,
        64,
        num_warps=8,
        num_stages=2,
    )
    return (buf0,)


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    import welder

    arg0_1 = rand_strided(
        (4, 8, 2048, 128),
        (2097152, 27144, 128, 1),
        device="cuda:0",
        dtype=torch.float16,
    )

    arg0_2 = rand_strided(
        (4, 8, 2048, 128),
        (2097152, 27144, 128, 1),
        device="cuda:0",
        dtype=torch.float16,
    )

    arg0_3 = rand_strided(
        (4, 8, 2048, 128),
        (4194304, 524288, 256, 1),
        device="cuda:0",
        dtype=torch.float16,
    )

    arg0_4 = rand_strided(
        (4, 8, 2048, 2048),
        (33554432, 4194304, 2048, 1),
        device="cuda:0",
        dtype=torch.float16,
    )

    buf0 = empty_strided(
        (4, 8, 2048, 256),
        (4194304, 524288, 256, 1),
        device="cuda",
        dtype=torch.float16,
    )

    the_welder_config = welder_config()
    welder.print_performance(
        lambda: call([arg0_1, arg0_2, arg0_3, arg0_4, buf0, the_welder_config])
    )
