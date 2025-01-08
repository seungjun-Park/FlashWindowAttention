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
    scale_qk: tl.constexpr,
    img_h: tl.constexpr,
    img_w: tl.constexpr,
    head: tl.constexpr,
    head_dim: tl.constexpr,
    head_chunk: tl.constexpr,
    chunk_dim: tl.constexpr,
    patch: tl.constexpr,
    patch_pad: tl.constexpr,
    shift: tl.constexpr,
):
    batch_id = tl.program_id(0) // head
    head_id = tl.program_id(0) % head
    block_h_id = tl.program_id(1)

    channels = head * head_dim
    stride_img_h = channels * img_w
    stride_batch = stride_img_h * img_h

    off_batch = batch_id * stride_batch
    off_h = block_h_id * patch + shift
    off_w = shift
    off_head = head_id * head_dim
    if shift> 0:
        off_h -= patch
        off_w -= patch

    # load bias
    if bias is not None:
        # (head, patch * patch, patch * patch)
        patch_2 = patch_pad * patch_pad
        Bias_ptr = tl.make_block_ptr(
            base=bias + head_id * patch_2 * patch_2,
            shape=(patch_2, patch_2),
            strides=(patch_2, 1),
            offsets=(0, 0),
            block_shape=(patch_pad * patch_pad, patch_pad * patch_pad),
            order=(1, 0),
        )
        bias_data = tl.load(Bias_ptr)

    mask_h = (tl.arange(0, patch_pad) < patch) & (off_h + tl.arange(0, patch_pad) < img_h)

    for off_w_loop in range(off_w, img_w, patch):
        # compute attn matrix
        attn = tl.zeros((patch_pad * patch_pad, patch_pad * patch_pad), dtype=tl.float32)
        Q_ptr = tl.make_block_ptr(
            base=Q + off_batch,
            shape=(img_h, img_w, channels),
            strides=(stride_img_h, channels, 1),
            offsets=(off_h, off_w_loop, off_head),
            block_shape=(patch_pad, patch_pad, chunk_dim),
            order=(2, 1, 0),
        )
        K_ptr = tl.make_block_ptr(
            base=K + off_batch,
            shape=(img_h, img_w, channels),
            strides=(stride_img_h, channels, 1),
            offsets=(off_h, off_w_loop, off_head),
            block_shape=(patch_pad, patch_pad, chunk_dim),
            order=(2, 1, 0),
        )
        for _ in range(head_chunk):
            # load data
            # TODO: fix boundary check
            q_data = tl.load(Q_ptr, boundary_check=(0, 1, 2), padding_option="zero")
            k_data = tl.load(K_ptr, boundary_check=(0, 1, 2), padding_option="zero")
            q_data = tl.reshape(q_data, (patch_pad * patch_pad, chunk_dim))
            k_data = tl.reshape(k_data, (patch_pad * patch_pad, chunk_dim))
            # dot of bf16 -> fp32
            attn = tl.dot(q_data, k_data.trans(1, 0), attn)
            Q_ptr = tl.advance(Q_ptr, (0, 0, chunk_dim))
            K_ptr = tl.advance(K_ptr, (0, 0, chunk_dim))

        attn *= scale_qk
        # apply bias and boundary mask
        if bias is not None:
            attn += bias_data
        
        # TODO:
        mask_w = (tl.arange(0, patch_pad) < patch) & (off_w_loop + tl.arange(0, patch_pad) < img_h)
        mask_attn = mask_h[:, None] & mask_w[None, :]
        mask_attn = mask_attn.reshape(1, patch_pad * patch_pad)
        attn += tl.where(mask_attn, 0, -float("inf"))

        # softmax
        attn -= tl.max(attn, axis=1, keep_dims=True)
        attn = tl.math.exp(attn)
        p_sum = tl.sum(attn, axis=1, keep_dims=True)
        attn /= p_sum
        attn = attn.cast(Q.dtype.element_ty)

        
        # save output
        V_ptr = tl.make_block_ptr(
            base=V + off_batch,
            shape=(img_h, img_w, channels),
            strides=(stride_img_h, channels, 1),
            offsets=(off_h, off_w_loop, off_head),
            block_shape=(patch_pad, patch_pad, chunk_dim),
            order=(2, 1, 0),
        )
        index = (
            off_batch + off_h * stride_img_h + off_w_loop * channels + off_head + 
            tl.arange(0, patch_pad)[:, None, None] * stride_img_h + 
            tl.arange(0, patch_pad)[None, :, None] * channels + 
            tl.arange(0, chunk_dim)[None, None, :]
        )
        O_ptr = O + index
        for _ in range(head_chunk):
            v_data = tl.load(V_ptr, boundary_check=(0, 1, 2), padding_option="zero")
            v_data = tl.reshape(v_data, (patch_pad * patch_pad, chunk_dim))
            o_data = tl.dot(attn, v_data)
            o_data = tl.reshape(o_data, (patch_pad, patch_pad, chunk_dim))
            o_data = o_data.cast(Q.dtype.element_ty)

            tl.store(O_ptr, o_data, mask=mask_attn.reshape(patch_pad, patch_pad, 1))
            V_ptr = tl.advance(V_ptr, (0, 0, chunk_dim))
            O_ptr += chunk_dim
