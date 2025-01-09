from flash_swin_attn import WindowAttention
import torch
from torch import nn
import itertools
from utils import BenchMethod, measure_speed_memory


@torch.inference_mode
def forward(batch, head, window, head_dim, method=BenchMethod.SWIN):
    seq = window * window
    x = torch.randn(batch, seq, head * head_dim).cuda()
    if method == BenchMethod.SWIN:
        m = WindowAttention(head_dim * head, (window, window), head, is_flash=False).cuda()
        t, memory = measure_speed_memory(m.forward, x)
    elif method == BenchMethod.FLASH_SWIN:
        m = WindowAttention(head_dim * head, (window, window), head, is_flash=True).cuda()
        t, memory = measure_speed_memory(m.forward, x)

    return t, memory


def backward(batch, head, window, head_dim, method=BenchMethod.SWIN):
    seq = window * window
    x = torch.randn(batch, seq, head * head_dim).cuda()
    d_o = torch.randn(batch, seq, head * head_dim).cuda()
    if method == BenchMethod.SWIN:
        m = WindowAttention(head_dim * head, (window, window), head, is_flash=False).cuda()

        def _f(m, x, o):
            o = m.forward(x)
            o.backward(d_o)

        t, memory = measure_speed_memory(_f, m, x, d_o)
    elif method == BenchMethod.FLASH_SWIN:
        m = WindowAttention(head_dim * head, (window, window), head, is_flash=True).cuda()

        def _f(m, x, o):
            o = m.forward(x)
            o.backward(d_o)

        t, memory = measure_speed_memory(_f, m, x, d_o)

    return t, memory


if __name__ == '__main__':
    head = 4
    batch = [64, 256, 1024, 4096]
    head_dim = [16, 64, 256]
    window = [4, 7, 8]

    for _batch, _window, _head_dim in itertools.product(batch, window, head_dim):
        print(f"batch={_batch}, window={_window}, head_dim={_head_dim}")
        t, memory = forward(_batch, head, _window, _head_dim, BenchMethod.SWIN)
        print(f"SWIN: {t:.4f} {memory:.4f}")
        t, memory = forward(_batch, head, _window, _head_dim, BenchMethod.FLASH_SWIN)
        print(f"FLASH_SWIN: {t:.4f} {memory:.4f}")
        t, memory = backward(_batch, head, _window, _head_dim, BenchMethod.SWIN)
        print(f"SWIN: {t:.4f} {memory:.4f}")
        t, memory = backward(_batch, head, _window, _head_dim, BenchMethod.FLASH_SWIN)
        print(f"FLASH_SWIN: {t:.4f} {memory:.4f}")
