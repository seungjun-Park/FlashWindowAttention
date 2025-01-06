import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def mha_core(q, k, v, bias, mask, scale_qk):
    # (B, heads, N, C)
    B, heads, N, head_dim = q.size()
    q = q * scale_qk
    attn = (q @ k.transpose(-2, -1))
    if bias is not None:
        attn = attn + bias.unsqueeze(0)
    if mask is not None:
        nW = mask.shape[0]
        attn = attn.view(B // nW, nW, heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, heads, N, N)
        attn = F.softmax(attn, -1)
    else:
        attn = F.softmax(attn, -1)    
    return attn @ v


def swin_attention_func(x, bias, mask, scale_qk, f_qkv, heads, window_size, shift_size):
    B, H, W, C = x.size()
    # cyclic shift
    if shift_size > 0:
        shifted_x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
        x_windows = window_partition(shifted_x, window_size)  # nW*B, window_size, window_size, C
    else:
        shifted_x = x
        x_windows = window_partition(shifted_x, window_size)  # nW*B, window_size, window_size, C

    x_windows = x_windows.view(-1, window_size * window_size, C)  # nW*B, window_size*window_size, C
    B_, N = x_windows.size(0), x_windows.size(1)

    # W-MSA/SW-MSA
    qkv = f_qkv(x_windows).reshape(B_, N, 3, heads, C // heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)\
    attn_windows = mha_core(q, k, v, bias, mask, scale_qk)

    # merge windows
    attn_windows = attn_windows.transpose(1, 2).reshape(B_, N, C)
    attn_windows = attn_windows.view(-1, window_size, window_size, C)

    # reverse cyclic shift
    if shift_size > 0:
        shifted_x = window_reverse(attn_windows, window_size, H, W)  # B H' W' C
        x = torch.roll(shifted_x, shifts=(shift_size, shift_size), dims=(1, 2))
    else:
        shifted_x = window_reverse(attn_windows, window_size, H, W)  # B H' W' C
        x = shifted_x

    return x


if __name__ == '__main__':
    q = torch.rand(512, 8, 16, 32).cuda().to(torch.bfloat16)
    k = torch.rand(512, 8, 16, 32).cuda().to(torch.bfloat16)
    v = torch.rand(512, 8, 16, 32).cuda().to(torch.bfloat16)

    from flash_attn import flash_attn_func 
    o1 = mha_core(q, k, v, None, None, 1.0)
    o2 = flash_attn_func(q, k, v, softmax_scale=1.0)
    err = (o1 - o2).abs()
    print(o1.abs().mean(), o2.abs().mean())
    print(err.max(), err.mean())
    print(o2.size(), o1.size())