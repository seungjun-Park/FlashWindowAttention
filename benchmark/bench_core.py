from flash_swin_attn import flash_swin_attn_fwd_func, swin_attention_func, mha_core
import torch
import einops
from torch import nn
import itertools
from flash_attn import flash_attn_func
import time
from utils import BenchMethod, measure_speed_memory


@torch.inference_mode
def forward(batch, h, w, head, head_dim, patch, bias=None, bf16=False, method=BenchMethod.SWIN):
    if method == BenchMethod.SWIN or method == BenchMethod.FLASH:
        n_patch = (h // patch) * (w // patch)
        q = torch.randn(batch * n_patch, head, patch ** 2, head_dim).cuda()
        k = torch.randn(batch * n_patch, head, patch ** 2, head_dim).cuda()
        v = torch.randn(batch * n_patch, head, patch ** 2, head_dim).cuda()
    elif method == BenchMethod.FLASH_SWIN:
        q = torch.randn(batch, h, w, head * head_dim).cuda()
        k = torch.randn(batch, h, w, head * head_dim).cuda()
        v = torch.randn(batch, h, w, head * head_dim).cuda()
    if method == BenchMethod.FLASH or bf16:
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)

    if method == BenchMethod.SWIN:
        t, memory = measure_speed_memory(mha_core, q, k, v, bias, None, 1.0)
    elif method == BenchMethod.FLASH:
        t, memory = measure_speed_memory(flash_attn_func, q, k, v)
    elif method == BenchMethod.FLASH_SWIN:
        t, memory = measure_speed_memory(flash_swin_attn_fwd_func, q, k, v, bias, 1.0, head, patch, patch, 0, 0)

    return t, memory


if __name__ == '__main__':
    batch, h, w, head, head_dim, patch = 8, 128, 128, 4, 128, 8
    bias = None
    bf16 = True

    for method in BenchMethod:
        t, memory = forward(batch, h, w, head, head_dim, patch, bias, bf16, method)
        print(f"{method.name}: {t:.5f} {memory:.5f}")