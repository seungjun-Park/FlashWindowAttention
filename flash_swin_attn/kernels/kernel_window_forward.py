import triton
import triton.language as tl


# (batch, head, img_h, img_w, head_dim)
@triton.jit
def _window_fwd_kernel(
    Q,
    K,
    V,
    bias,
    O,
    O_lse,
    scale_qk,
    head: tl.constexpr,
    img_h: tl.constexpr,
    img_w: tl.constexpr,
    head_dim: tl.constexpr,
    patch_h: tl.constexpr,
    patch_w: tl.constexpr,
    num_patch_w: tl.constexpr,
    shift_h: tl.constexpr,
    shift_w: tl.constexpr,
):
    batch_id = tl.program_id(0) // head
    head_id = tl.program_id(0) % head
    block_h_id = tl.program_id(1)
    block_w_id = tl.program_id(2)

    stride_img_w = head_dim
    stride_img_h = img_w * stride_img_w
    stride_head = img_h * stride_img_h
    stride_batch = head * stride_head

    off_batch_head = batch_id * stride_batch + head_id * stride_head
    off_h = block_h_id * patch_h + shift_h
    off_w = block_w_id * num_patch_w * patch_w + shift_w

    # get block ptr
    Q_block_ptr = tl.make_block_ptr(
        base=Q + off_batch_head,
        shape=(img_h, img_w, head_dim),
        strides=(stride_img_h, stride_img_w, 1),
        offsets=(off_h, off_w, 0),
        block_shape=(patch_h, patch_w, head_dim),
        order=(2, 1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + off_batch_head,
        shape=(img_h, img_w, head_dim),
        strides=(stride_img_h, stride_img_w, 1),
        offsets=(off_h, off_w, 0),
        block_shape=(patch_h, patch_w, head_dim),
        order=(2, 1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + off_batch_head,
        shape=(img_h, img_w, head_dim),
        strides=(stride_img_h, stride_img_w, 1),
        offsets=(off_h, off_w, 0),
        block_shape=(patch_h, patch_w, head_dim),
        order=(2, 1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=O + off_batch_head,
        shape=(img_h, img_w, head_dim),
        strides=(stride_img_h, stride_img_w, 1),
        offsets=(off_h, off_w, 0),
        block_shape=(patch_h, patch_w, head_dim),
        order=(2, 1, 0),
    )
    Lse_block_ptr = tl.make_block_ptr(
        base=O_lse + tl.program_id(0) * img_h * img_w,
        shape=(img_h, img_w),
        strides=(img_w, 1),
        offsets=(off_h, off_w),
        block_shape=(patch_h, patch_w),
        order=(1, 0),
    )
    if bias is not None:
        # (head, patch_h * patch_w, patch_h * patch_w)
        patch = patch_h * patch_w
        Bias_block_ptr = tl.make_block_ptr(
            base=bias + head_id * patch * patch,
            shape=(patch, patch),
            strides=(patch, 1),
            offsets=(0, 0),
            block_shape=(patch_h * patch_w, patch_h * patch_w),  # TODO: (patch, patch) doesn't work
            order=(1, 0),
        )
        bias_data = tl.load(Bias_block_ptr)

    mask_h = (off_h + tl.arange(0, patch_h * patch_w) // patch_w) < img_h
    for _ in range(num_patch_w):
        off_w_loop = off_w + patch_w * _
        if off_w_loop < img_w:
            # load data
            q_data = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
            k_data = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
            v_data = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
            q_data = tl.reshape(q_data, (patch_h * patch_w, head_dim))
            k_data = tl.reshape(k_data, (patch_h * patch_w, head_dim))
            v_data = tl.reshape(v_data, (patch_h * patch_w, head_dim))

            # attention
            qk = tl.dot(q_data, k_data.trans(1, 0)) * scale_qk
            if bias is not None:
                qk += bias_data
            # mask
            mask_w = (off_w_loop + tl.arange(0, patch_h * patch_w) % patch_w) < img_w

            qk += tl.where(mask_h & mask_w, 0, -float("inf"))

            qk -= tl.max(qk, axis=1, keep_dims=True)
            qk = tl.math.exp(qk)
            p_sum = tl.sum(qk, axis=1, keep_dims=True)
            o_data = tl.dot(qk, v_data)
            o_data /= p_sum

            o_data = tl.reshape(o_data, (patch_h, patch_w, head_dim))
            p_sum = tl.reshape(p_sum, (patch_h, patch_w))

            tl.store(O_block_ptr, o_data, boundary_check=(0, 1))
            tl.store(Lse_block_ptr, tl.math.log(p_sum), boundary_check=(0, 1))

            # move ptr
            Q_block_ptr = tl.advance(Q_block_ptr, (0, patch_w, 0))
            K_block_ptr = tl.advance(K_block_ptr, (0, patch_w, 0))
            V_block_ptr = tl.advance(V_block_ptr, (0, patch_w, 0))
            O_block_ptr = tl.advance(O_block_ptr, (0, patch_w, 0))
            Lse_block_ptr = tl.advance(Lse_block_ptr, (0, patch_w))


