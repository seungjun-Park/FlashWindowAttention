from .kernels import _window_fwd_kernel
import torch
import triton


def flash_swin_attn_fwd_func(Q, K, V, bias, scale_qk, patch_h, patch_w, shift_h, shift_w):
    batch, head, img_h, img_w, head_dim = Q.size()
    O = torch.empty_like(Q)
    O_lse = torch.empty((batch, head, img_h, img_w), device=Q.device, dtype=Q.dtype)
    num_patch_w = 4 # TODO: auto-tune this
    grid = (batch * head, triton.cdiv(img_h, patch_h), triton.cdiv(img_w, patch_w * num_patch_w))
    _window_fwd_kernel[grid](
            Q,
            K,
            V,
            bias,
            O,
            O_lse,
            scale_qk,
            head,
            img_h,
            img_w,
            head_dim,
            patch_h,
            patch_w,
            num_patch_w,
            shift_h,
            shift_w,
        )