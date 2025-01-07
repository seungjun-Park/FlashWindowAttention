import triton
import triton.language as tl
import torch


# (batch, img_h, img_w, head * head_dim)
@triton.jit
def _window_fwd_kernel(
    Q,
    K,
    V,
    bias,
    O,
    O_lse,
    scale_qk: tl.constexpr,
    img_h: tl.constexpr,
    img_w: tl.constexpr,
    head: tl.constexpr,
    head_dim: tl.constexpr,
    head_chunk: tl.constexpr,
    chunk_dim: tl.constexpr,
    patch_h: tl.constexpr,
    patch_w: tl.constexpr,
    shift_h: tl.constexpr,
    shift_w: tl.constexpr,
    dtype: tl.constexpr
):
    batch_id = tl.program_id(0) // head
    head_id = tl.program_id(0) % head
    block_h_id = tl.program_id(1)

    channels = head * head_dim
    stride_img_h = channels * img_w
    stride_batch = stride_img_h * img_h

    off_batch = batch_id * stride_batch
    off_h = block_h_id * patch_h + shift_h
    off_w = shift_w
    off_head = head_id * head_dim
    if shift_h > 0:
        off_h -= patch_h
    if shift_w > 0:
        off_w -= patch_w

    # load bias
    if bias is not None:
        # (head, patch_h * patch_w, patch_h * patch_w)
        patch = patch_h * patch_w
        Bias_ptr = tl.make_block_ptr(
            base=bias + head_id * patch * patch,
            shape=(patch, patch),
            strides=(patch, 1),
            offsets=(0, 0),
            block_shape=(patch_h * patch_w, patch_h * patch_w),  # TODO: (patch, patch) doesn't work
            order=(1, 0),
        )
        bias_data = tl.load(Bias_ptr)

    mask_h = (off_h + tl.arange(0, patch_h * patch_w) // patch_w) < img_h
    for off_w_loop in range(off_w, img_w, patch_w):
        # compute attn matrix
        qk = tl.zeros((patch_h * patch_w, patch_h * patch_w), dtype=tl.float32)
        Q_ptr = tl.make_block_ptr(
            base=Q + off_batch,
            shape=(img_h, img_w, channels),
            strides=(stride_img_h, channels, 1),
            offsets=(off_h, off_w_loop, off_head),
            block_shape=(patch_h, patch_w, chunk_dim),
            order=(2, 1, 0),
        )
        K_ptr = tl.make_block_ptr(
            base=K + off_batch,
            shape=(img_h, img_w, channels),
            strides=(stride_img_h, channels, 1),
            offsets=(off_h, off_w_loop, off_head),
            block_shape=(patch_h, patch_w, chunk_dim),
            order=(2, 1, 0),
        )
        for _ in range(head_chunk):
            # load data
            q_data = tl.load(Q_ptr, boundary_check=(0, 1), padding_option="zero")
            k_data = tl.load(K_ptr, boundary_check=(0, 1), padding_option="zero")
            q_data = tl.reshape(q_data, (patch_h * patch_w, chunk_dim))
            k_data = tl.reshape(k_data, (patch_h * patch_w, chunk_dim))
            # dot of bf16 -> fp32
            qk = tl.dot(q_data, k_data.trans(1, 0), qk)
            Q_ptr = tl.advance(Q_ptr, (0, 0, chunk_dim))
            K_ptr = tl.advance(K_ptr, (0, 0, chunk_dim))

        qk *= scale_qk
        # apply bias and boundary mask
        if bias is not None:
            qk += bias_data
        mask_w = (off_w_loop + tl.arange(0, patch_h * patch_w) % patch_w) < img_w
        qk += tl.where(mask_h & mask_w, 0, -float("inf"))

        # softmax
        qk -= tl.max(qk, axis=1, keep_dims=True)
        qk = tl.math.exp(qk)
        p_sum = tl.sum(qk, axis=1, keep_dims=True)
        qk /= p_sum

        # save log_sum_exp
        p_sum = tl.reshape(p_sum, (patch_h, patch_w))
        Lse_ptr = tl.make_block_ptr(
            base=O_lse + tl.program_id(0) * img_h * img_w,
            shape=(img_h, img_w),
            strides=(img_w, 1),
            offsets=(off_h, off_w_loop),
            block_shape=(patch_h, patch_w),
            order=(1, 0),
        )
        tl.store(Lse_ptr, tl.math.log(p_sum).cast(dtype), boundary_check=(0, 1))

        # save output
        V_ptr = tl.make_block_ptr(
            base=V + off_batch,
            shape=(img_h, img_w, channels),
            strides=(stride_img_h, channels, 1),
            offsets=(off_h, off_w_loop, off_head),
            block_shape=(patch_h, patch_w, chunk_dim),
            order=(2, 1, 0),
        )
        O_ptr = tl.make_block_ptr(
            base=O + off_batch,
            shape=(img_h, img_w, channels),
            strides=(stride_img_h, channels, 1),
            offsets=(off_h, off_w_loop, off_head),
            block_shape=(patch_h, patch_w, chunk_dim),
            order=(2, 1, 0),
        )

        for _ in range(head_chunk):
            v_data = tl.load(V_ptr, boundary_check=(0, 1), padding_option="zero")
            v_data = tl.reshape(v_data, (patch_h * patch_w, chunk_dim))
            o_data = tl.dot(qk.cast(v_data.dtype), v_data)
            o_data = tl.reshape(o_data, (patch_h, patch_w, chunk_dim))
            tl.store(O_ptr, o_data.cast(v_data.dtype), boundary_check=(0, 1, 2))

            V_ptr = tl.advance(V_ptr, (0, 0, chunk_dim))
            O_ptr = tl.advance(O_ptr, (0, 0, chunk_dim))
