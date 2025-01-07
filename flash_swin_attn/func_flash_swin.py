from .kernels import _window_fwd_kernel
import torch
import triton


def flash_swin_attn_fwd_func(Q, K, V, bias, scale_qk, head, patch_h, patch_w, shift_h, shift_w, head_chunk=1):
    batch, img_h, img_w, c = Q.size()
    head_dim = c // head
    O = torch.empty_like(Q)
    O_lse = torch.empty((batch, head, img_h, img_w), device=Q.device, dtype=torch.float32)

    # deal with shift
    delta_h = 0 if shift_h == 0 else patch_h
    grid = (batch * head, triton.cdiv(img_h + delta_h, patch_h), 1)
    _window_fwd_kernel[grid](
        Q,
        K,
        V,
        bias,
        O,
        O_lse,
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

    return O, O_lse.to(O.dtype)
