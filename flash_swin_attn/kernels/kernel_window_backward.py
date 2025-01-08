import triton
import triton.language as tl
import torch


# (batch, img_h, img_w, head * head_dim)
@triton.jit
def _window_bwd_kernel(
    Q,
    K,
    V,
    bias,
    d_O,
    d_Q,
    d_K,
    d_V,
    d_bias,
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
        attn = tl.zeros((patch_h * patch_w, patch_h * patch_w), dtype=tl.float32)
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
            attn = tl.dot(q_data, k_data.trans(1, 0), attn)
            Q_ptr = tl.advance(Q_ptr, (0, 0, chunk_dim))
            K_ptr = tl.advance(K_ptr, (0, 0, chunk_dim))

        attn *= scale_qk
        # apply bias and boundary mask
        if bias is not None:
            attn += bias_data
        mask_w = (off_w_loop + tl.arange(0, patch_h * patch_w) % patch_w) < img_w
        attn += tl.where(mask_h & mask_w, 0, -float("inf"))

        # softmax
        attn -= tl.max(attn, axis=1, keep_dims=True)
        attn = tl.math.exp(attn)
        p_sum = tl.sum(attn, axis=1, keep_dims=True)
        attn /= p_sum
        
        # compute d_v, d_attn
        d_attn = tl.zeros((patch_h * patch_w, patch_h * patch_w), dtype=tl.float32)
        d_O_ptr = tl.make_block_ptr(
            base=d_O + off_batch,
            shape=(img_h, img_w, channels),
            strides=(stride_img_h, channels, 1),
            offsets=(off_h, off_w_loop, off_head),
            block_shape=(patch_h, patch_w, chunk_dim),
            order=(2, 1, 0),
        )
        V_ptr = tl.make_block_ptr(
            base=V + off_batch,
            shape=(img_h, img_w, channels),
            strides=(stride_img_h, channels, 1),
            offsets=(off_h, off_w_loop, off_head),
            block_shape=(patch_h, patch_w, chunk_dim),
            order=(2, 1, 0),
        )
        d_V_ptr = tl.make_block_ptr(
            base=d_V + off_batch,
            shape=(img_h, img_w, channels),
            strides=(stride_img_h, channels, 1),
            offsets=(off_h, off_w_loop, off_head),
            block_shape=(patch_h, patch_w, chunk_dim),
            order=(2, 1, 0),
        )
        for _ in range(head_chunk):
            # load data
            d_o_data = tl.load(d_O_ptr, boundary_check=(0, 1), padding_option="zero")
            v_data = tl.load(V_ptr, boundary_check=(0, 1), padding_option="zero")
            d_o_data = tl.reshape(d_o_data, (patch_h * patch_w, chunk_dim))
            v_data = tl.reshape(v_data, (patch_h * patch_w, chunk_dim))

            # accumulate
            d_attn = tl.dot(d_o_data, v_data.trans(1, 0), d_attn)

            # store d_v
            d_v_data = tl.dot(attn.trans(1, 0), d_o_data).cast(v_data.dtype)
            d_v_data = tl.reshape(d_v_data, (patch_h, patch_w, chunk_dim))
            tl.store(d_V_ptr, d_v_data, boundary_check=(0, 1))

            d_O_ptr = tl.advance(d_O_ptr, (0, 0, chunk_dim))
            V_ptr = tl.advance(V_ptr, (0, 0, chunk_dim))
            d_V_ptr = tl.advance(d_V_ptr, (0, 0, chunk_dim))

        d_attn_sum = tl.sum(d_attn * d_attn, axis=1, keep_dims=True)
        d_attn = attn * (d_attn - d_attn_sum)

        # compute d_bias
        if bias is not None:
            patch = patch_h * patch_w
            d_Bias_ptr = tl.make_block_ptr(
                base=bias + head_id * patch * patch,
                shape=(patch, patch),
                strides=(patch, 1),
                offsets=(0, 0),
                block_shape=(patch_h * patch_w, patch_h * patch_w),  # TODO: (patch, patch) doesn't work
                order=(1, 0),
            )
            tl.atomic_add(d_Bias_ptr, d_attn)
        
        # compute d_q, d_k
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
        d_Q_ptr = tl.make_block_ptr(
            base=d_Q + off_batch,
            shape=(img_h, img_w, channels),
            strides=(stride_img_h, channels, 1),
            offsets=(off_h, off_w_loop, off_head),
            block_shape=(patch_h, patch_w, chunk_dim),
            order=(2, 1, 0),
        )
        d_K_ptr = tl.make_block_ptr(
            base=d_K + off_batch,
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
            
            d_q_data = tl.dot(d_attn, k_data).cast(q_data.dtype)
            d_q_data = tl.reshape(d_q_data, (patch_h, patch_w, chunk_dim))
            tl.store(d_Q_ptr, d_q_data, boundary_check=(0, 1))

            d_k_data = tl.dot(d_attn.trans(1, 0), q_data).cast(k_data.dtype)
            d_k_data = tl.reshape(d_k_data, (patch_h, patch_w, chunk_dim))
            tl.store(d_K_ptr, d_k_data, boundary_check=(0, 1))

            Q_ptr = tl.advance(Q_ptr, (0, 0, chunk_dim))
            K_ptr = tl.advance(K_ptr, (0, 0, chunk_dim))
            d_Q_ptr = tl.advance(d_Q_ptr, (0, 0, chunk_dim))
            d_K_ptr = tl.advance(d_K_ptr, (0, 0, chunk_dim))
    

if __name__ == '__main__':
    batch, img_h, img_w, head, head_dim = 1, 16, 16, 4, 64
    head_chunk = 1
    chunk_dim = head_dim // head_chunk
    patch_h, patch_w = 4, 4
    shift_h, shift_w = 0, 0

    dtype = torch.bfloat16
    q = torch.randn(batch, img_h, img_w, head * head_dim).cuda().to(dtype)
    k = torch.randn(batch, img_h, img_w, head * head_dim).cuda().to(dtype)
    v = torch.randn(batch, img_h, img_w, head * head_dim).cuda().to(dtype)
    bias = None
    d_o = torch.randn(batch, img_h, img_w, head * head_dim).cuda().to(dtype)
    d_q = torch.empty_like(q).cuda().to(dtype)
    d_k = torch.empty_like(k).cuda().to(dtype)
    d_v = torch.empty_like(v).cuda().to(dtype)

    grid = (batch * head, img_h // patch_h, 1)
    _window_bwd_kernel[grid](
        q,
        k,
        v,
        bias,
        d_o,
        d_q,
        d_k,
        d_v,
        None,
        1.0,
        img_h,
        img_w,
        head,
        head_dim,
        head_chunk,
        chunk_dim,
        patch_h,
        patch_w,
        shift_h,
        shift_w,
    )

