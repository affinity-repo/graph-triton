import jinja2


fa_template = """
@triton.jit
def triton_(
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
        qk = qk * attention_mask
        r_wo_clamp += tl.sum(tl.abs(qk), axis=1)
        qk = qk.to(q_ptr.dtype.element_ty)
        v = tl.load(v_ptrs + block_n_start_idx * v_k_stride)
        acco += tl.dot(qk, v)
    r_new = tl.where(r_wo_clamp > 1, r_wo_clamp, 1)
    acco = acco / r_new
    tl.store(output_ptrs, acco)
"""
