from flash_swin_attn import (
    flash_swin_attn_fwd_func,
    flash_swin_attn_bwd_func,
    flash_swin_attn_func,
    mha_core,
)
from flash_attn import flash_attn_func

import torch
import einops
from torch import nn
import itertools
import time
from utils import BenchMethod, measure_speed_memory


@torch.inference_mode
def forward(batch, head, seq, head_dim, bias=None, bf16=False, method=BenchMethod.SWIN):
    q = torch.randn(batch, head, seq, head_dim).cuda()
    k = torch.randn(batch, head, seq, head_dim).cuda()
    v = torch.randn(batch, head, seq, head_dim).cuda()
    if method == BenchMethod.FLASH or bf16:
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)
    if bias is not None:
        bias = torch.randn(head, seq, seq).cuda().to(q.dtype)

    if method == BenchMethod.SWIN:
        t, memory = measure_speed_memory(mha_core, q, k, v, bias, None, 1.0)
    elif method == BenchMethod.FLASH:
        t, memory = measure_speed_memory(flash_attn_func, q, k, v)
    elif method == BenchMethod.FLASH_SWIN:
        t, memory = measure_speed_memory(flash_swin_attn_func, q, k, v, bias, 1.0)

    return t, memory


def backward(batch, head, seq, head_dim, bias=None, bf16=False, method=BenchMethod.SWIN):
    q = torch.randn(batch, head, seq, head_dim).cuda()
    k = torch.randn(batch, head, seq, head_dim).cuda()
    v = torch.randn(batch, head, seq, head_dim).cuda()
    d_o = torch.randn(batch, head, seq, head_dim).cuda()
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    if method == BenchMethod.FLASH or bf16:
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)
        d_o = d_o.to(torch.bfloat16)
    if bias is not None:
        bias = torch.randn(head, seq, seq).cuda().to(q.dtype)
        bias.requires_grad_()

    if method == BenchMethod.SWIN:

        def _f(q, v, k, bias, d_o):
            o = mha_core(q, k, v, bias, None, 1.0)
            o.backward(d_o)

        t, memory = measure_speed_memory(_f, q, v, k, bias, d_o)
    elif method == BenchMethod.FLASH:

        def _f(q, k, v, d_o):
            o = flash_attn_func(q, k, v)
            o.backward(d_o)

        t, memory = measure_speed_memory(_f, q, k, v, d_o)
    elif method == BenchMethod.FLASH_SWIN:

        def _f(q, k, v, bias, d_o):
            o = flash_swin_attn_func(q, k, v, bias, 1.0)
            o.backward(d_o)

        t, memory = measure_speed_memory(_f, q, k, v, bias, d_o)

    return t, memory


if __name__ == '__main__':
    head = 4
    batch = [64, 256, 1024, 4096]
    head_dim = [16, 64, 256]
    seq = [16, 64]

    print("Compare swin and flash_swin with float32")
    for _batch, _seq, _head_dim in itertools.product(batch, seq, head_dim):
        print(f"batch={_batch}, seq={_seq}, head_dim={_head_dim}")
        t, memory = forward(_batch, head, _seq, _head_dim, True, False, BenchMethod.SWIN)
        print(f"SWIN: {t:.4f} {memory:.4f}")
        t, memory = forward(_batch, head, _seq, _head_dim, True, False, BenchMethod.FLASH_SWIN)
        print(f"FLASH_SWIN: {t:.4f} {memory:.4f}")

        t, memory = backward(_batch, head, _seq, _head_dim, True, False, BenchMethod.SWIN)
        print(f"SWIN: {t:.4f} {memory:.4f}")
        t, memory = backward(_batch, head, _seq, _head_dim, True, False, BenchMethod.FLASH_SWIN)
        print(f"FLASH_SWIN: {t:.4f} {memory:.4f}")

    batch = [64, 256, 1024]
    head_dim = [16, 64, 256]
    seq = [16, 64]

    print("Compare swin and flash_swin with bfloat16")
    for _batch, _seq, _head_dim in itertools.product(batch, seq, head_dim):
        print(f"batch={_batch}, seq={_seq}, head_dim={_head_dim}")
        t, memory = forward(_batch, head, _seq, _head_dim, None, True, BenchMethod.FLASH)
        print(f"FLASH: {t:.4f} {memory:.4f}")
        t, memory = forward(_batch, head, _seq, _head_dim, None, True, BenchMethod.FLASH_SWIN)
        print(f"FLASH_SWIN: {t:.4f} {memory:.4f}")

        t, memory = backward(_batch, head, _seq, _head_dim, None, True, BenchMethod.FLASH)
        print(f"FLASH: {t:.4f} {memory:.4f}")
        t, memory = backward(_batch, head, _seq, _head_dim, None, True, BenchMethod.FLASH_SWIN)
        print(f"FLASH_SWIN: {t:.4f} {memory:.4f}")
