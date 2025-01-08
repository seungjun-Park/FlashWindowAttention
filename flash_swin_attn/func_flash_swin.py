from .kernels import _window_fwd_kernel, _window_bwd_kernel
import torch
import triton


MAX_HEAD_DIM = 128


def ceil_pow2(x):
    n = x.bit_length()
    if x == (1 << n):
        return x
    else:
        return 1 << (n + 1)    


def flash_swin_attn_fwd_func(Q, K, V, bias, scale_qk, head, patch, shift):
    batch, img_h, img_w, c = Q.size()
    head_dim = c // head
    head_chunk = 1 if head_dim <= MAX_HEAD_DIM else head_dim // MAX_HEAD_DIM
    patch_pad = ceil_pow2(patch)
    O = torch.empty_like(Q)

    # deal with shift
    delta_h = 0 if shift == 0 else patch
    grid = (batch * head, triton.cdiv(img_h + delta_h, patch), 1)
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
        patch,
        patch_pad,
        shift,
    )

    return O


def flash_swin_attn_bwd_func(Q, K, V, d_O, bias, scale_qk, head, patch_h, patch_w, shift_h, shift_w):
    batch, img_h, img_w, c = Q.size()
    head_dim = c // head
    head_chunk = 1 if head_dim <= MAX_HEAD_DIM else head_dim // MAX_HEAD_DIM

    d_Q = torch.empty_like(Q)
    d_K = torch.empty_like(K)
    d_V = torch.empty_like(V)
    d_Bias = torch.zeros_like(bias) if bias is not None else None
    
    # deal with shift
    delta_h = 0 if shift_h == 0 else patch_h
    grid = (batch * head, triton.cdiv(img_h + delta_h, patch_h), 1)
    _window_bwd_kernel[grid](
            Q,
            K,
            V,
            bias,
            d_O,
            d_Q,
            d_K,
            d_V,
            d_Bias,
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
    
    return d_Q, d_K, d_V, d_Bias
