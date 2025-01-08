from .kernels import _window_fwd_kernel
import torch
import triton


MAX_HEAD_DIM = 128


def flash_swin_attn_fwd_func(Q, K, V, bias, scale_qk, head, patch_h, patch_w, shift_h, shift_w):
    batch, img_h, img_w, c = Q.size()
    head_dim = c // head
    head_chunk = 1 if head_dim <= MAX_HEAD_DIM else head_dim // MAX_HEAD_DIM
    O = torch.empty_like(Q)

    # deal with shift
    delta_h = 0 if shift_h == 0 else patch_h
    grid = (batch * head, triton.cdiv(img_h + delta_h, patch_h), 1)
    _window_fwd_kernel[grid](
        Q,
        K,
        V,
        bias,
        O,
        scale_qk,
        img_h,
        img_w,
        head,
        head_dim,
        head_chunk,
        head_dim // head_chunk,
        patch_h,
        patch_w,
        shift_h,
        shift_w,
    )

    return O
