from flash_swin_attn import flash_swin_attn_fwd_func, swin_attention_func, mha_core
import torch
import einops
from torch import nn
import itertools
from flash_attn import flash_attn_func
import time
from utils import BenchMethod


def measure_speed_memory(f, *args, **kwargs):
    # warm up
    f(*args, **kwargs)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.time()
    for i in range(100):
        f(*args, **kwargs)
    torch.cuda.synchronize()
    t = time.time() - start
    memory = torch.cuda.max_memory_allocated() / 1000000

    return t, memory


@torch.inference_mode
def forward(batch, h, w, head, head_dim, patch, bias=None, bf16=False, method=BenchMethod.SWIN):
    if method == BenchMethod.SWIN or method == BenchMethod.FLASH:
        n_patch = (h // patch) * (w // patch)
        q = torch.randn(batch * n_patch, head, patch ** 2, head_dim).cuda()
        k = torch.randn(batch * n_patch, head, patch ** 2, head_dim).cuda()
        v = torch.randn(batch * n_patch, head, patch ** 2, head_dim).cuda()
    else:
        q = torch.randn(batch, h, w, head * head_dim).cuda()
        k = torch.randn(batch, h, w, head * head_dim).cuda()
        v = torch.randn(batch, h, w, head * head_dim).cuda()
    if method == BenchMethod.FLASH or bf16:
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)

    if method == BenchMethod.SWIN:
        f = mha_core
        t, memory = measure_speed_memory(mha_core, q, k, v, bias, None, 1.0)
    elif method == BenchMethod.FLASH:
        f = flash_attn_func
        t, memory = measure_speed_memory(flash_attn_func, q, k, v)
    elif method == BenchMethod.FLASH_SWIN:
        t, memory = measure_speed_memory(flash_swin_attn_fwd_func, q, k, v, bias, 1.0, head, patch, patch, 0, 0)

    return t, memory


if __name__ == '__main__':
    batch, h, w, head, head_dim, patch = 8, 64, 64, 4, 16, 32
    bias = None
    bf16 = False

    for method in BenchMethod:
        if method == BenchMethod.FLASH_SWIN:
            continue
        t, memory = forward(batch, h, w, head, head_dim, patch, bias, bf16, method)
        print(f"{method.name}: {t:.5f} {memory:.5f}")