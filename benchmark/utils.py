from enum import Enum
import torch
import time


class BenchMethod(Enum):
    SWIN = 1
    FLASH_SWIN = 2
    FLASH = 3


def measure_speed_memory(f, *args, **kwargs):
    # warm up
    f(*args, **kwargs)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.time()
    for i in range(100):
        f(*args, **kwargs)
    torch.cuda.synchronize()
    t = time.time() - start
    memory = torch.cuda.max_memory_allocated() / ((2**20) * 1000)

    return t, memory