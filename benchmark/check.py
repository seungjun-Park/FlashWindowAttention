from flash_swin_attn import flash_swin_attn_fwd_func, flash_swin_attn_bwd_func, mha_core

import torch
import einops
from torch import nn
import itertools


@torch.inference_mode
def compare(x, y):
    diff = (x - y).abs()
    err_max = diff.max().item()
    err_mean = diff.mean().item()
    print(f"{err_max:.5f} {err_mean:.5f}")

    return err_mean < 1e-2


@torch.inference_mode
def forward_core(batch, head, seq, head_dim, bias=None):
    q = torch.randn(batch, head, seq, head_dim).cuda()
    k = torch.randn(batch, head, seq, head_dim).cuda()
    v = torch.randn(batch, head, seq, head_dim).cuda()
    if bias is not None:
        bias = torch.randn(head, seq, seq).cuda()
    o1 = flash_swin_attn_fwd_func(q, k, v, bias, 1.0)
    o2 = mha_core(q, k, v, bias, None, 1.0)

    return compare(o1, o2)


def backward_core(batch, head, seq, head_dim, bias=None):
    q = torch.randn(batch, head, seq, head_dim).cuda()
    k = torch.randn(batch, head, seq, head_dim).cuda()
    v = torch.randn(batch, head, seq, head_dim).cuda()
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    if bias is not None:
        bias = torch.randn(head, seq, seq).cuda()
        bias.requires_grad_()
    d_o = torch.randn(batch, head, seq, head_dim).cuda()

    o2 = mha_core(q, k, v, bias, None, 1.0)
    o2.backward(d_o)

    with torch.no_grad():
        d_q, d_k, d_v, d_b = flash_swin_attn_bwd_func(q, k, v, bias, d_o, 1.0)

    is_d_q = compare(d_q, q.grad)
    is_d_k = compare(d_k, k.grad)
    is_d_v = compare(d_v, v.grad)
    if bias is not None:
        is_d_b = compare(d_b, bias.grad)
    else:
        is_d_b = True

    return is_d_q and is_d_k and is_d_v and is_d_b


if __name__ == '__main__':
    batch = [1, 32]
    head = [1, 8]
    seq = [9, 16, 49, 64]
    head_dim = [16, 128]

    for _batch, _head, _seq, _head_dim in itertools.product(batch, head, seq, head_dim):
        print(f"batch={_batch}, head={_head}, seq={_seq}, head={_head}, head_dim={_head_dim}")
        assert forward_core(_batch, _head, _seq, _head_dim)
        assert forward_core(_batch, _head, _seq, _head_dim, True)
        assert backward_core(_batch, _head, _seq, _head_dim)
        assert backward_core(_batch, _head, _seq, _head_dim, True)
