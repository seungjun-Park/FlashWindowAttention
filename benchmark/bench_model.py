from flash_swin_attn import SwinTransformer
import torch
from torch import nn
import torch.nn.functional as F
import itertools
from utils import BenchMethod, measure_speed_memory


@torch.inference_mode
def forward(batch, method=BenchMethod.SWIN):
    x = torch.randn(batch, 3, 224, 224).cuda()
    if method == BenchMethod.SWIN:
        m = SwinTransformer(is_flash=False).cuda()
        t, memory = measure_speed_memory(m.forward, x)
    elif method == BenchMethod.FLASH_SWIN:
        m = SwinTransformer(is_flash=True).cuda()
        t, memory = measure_speed_memory(m.forward, x)

    return t, memory


def backward(batch, method=BenchMethod.SWIN):
    x = torch.randn(batch, 3, 224, 224).cuda()
    y = torch.randint(0, 1000, size=(batch,)).cuda()
    if method == BenchMethod.SWIN:
        m = SwinTransformer(is_flash=False).cuda()
        opt = torch.optim.Adam(m.parameters(), lr=1e-4)

        def _f(m, opt, x, y):
            opt.zero_grad()
            o = m.forward(x)
            loss = F.cross_entropy(o, y)
            loss.backward()
            opt.step()

        t, memory = measure_speed_memory(_f, m, opt, x, y)
    elif method == BenchMethod.FLASH_SWIN:
        m = SwinTransformer(is_flash=True).cuda()
        opt = torch.optim.Adam(m.parameters(), lr=1e-4)

        def _f(m, opt, x, y):
            opt.zero_grad()
            o = m.forward(x)
            loss = F.cross_entropy(o, y)
            loss.backward()
            opt.step()

        t, memory = measure_speed_memory(_f, m, opt, x, y)

    return t, memory


if __name__ == '__main__':
    batch = [1, 4, 16, 64, 256]

    for _batch in batch:
        print(f"batch={_batch}")
        t, memory = forward(_batch, BenchMethod.SWIN)
        print(f"SWIN: {t:.4f} {memory:.4f}")
        t, memory = forward(_batch, BenchMethod.FLASH_SWIN)
        print(f"FLASH_SWIN: {t:.4f} {memory:.4f}")
        t, memory = backward(_batch, BenchMethod.SWIN)
        print(f"SWIN: {t:.4f} {memory:.4f}")
        t, memory = backward(_batch, BenchMethod.FLASH_SWIN)
        print(f"FLASH_SWIN: {t:.4f} {memory:.4f}")
