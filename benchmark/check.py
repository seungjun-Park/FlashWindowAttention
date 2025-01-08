from flash_swin_attn import flash_swin_attn_fwd_func, swin_attention_func, mha_core
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

    return err_mean < 1e-3


@torch.inference_mode
def forward_core(batch, h, w, head, head_dim, patch, bias=None):
    q = torch.randn(batch, h, w, head * head_dim).cuda()
    k = torch.randn(batch, h, w, head * head_dim).cuda()
    v = torch.randn(batch, h, w, head * head_dim).cuda()
    bias = torch.randn(head, patch ** 2, patch ** 2).cuda()
    
    o1 = flash_swin_attn_fwd_func(q, k, v, bias, 1.0, head, patch, patch, 0, 0)

    q_ = einops.rearrange(q, 'b (h p) (w q) (head c) -> (b h w) head (p q) c', p=patch, q=patch, head=head)
    k_ = einops.rearrange(k, 'b (h p) (w q) (head c) -> (b h w) head (p q) c', p=patch, q=patch, head=head)
    v_ = einops.rearrange(v, 'b (h p) (w q) (head c) -> (b h w) head (p q) c', p=patch, q=patch, head=head)
    o2 = mha_core(q_, k_, v_, bias, None, 1.0)
    o2 = einops.rearrange(
        o2, '(b h w) head (p q) c -> b (h p) (w q) (head c)', h=h // patch, w=w // patch, p=patch, q=patch
    )

    return compare(o1, o2)


@torch.inference_mode
def forward_attn(batch, h, w, head, head_dim, patch, shift, bias=None):
    x = torch.randn(batch, h, w, head * head_dim).cuda()
    f_qkv = nn.Linear(head * head_dim, 3 * head * head_dim, bias=False).cuda()

    q, k, v = f_qkv(x).chunk(3, dim=-1)
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    o1 = flash_swin_attn_fwd_func(q, k, v, bias, 1.0, head, patch, patch, shift, shift)
    o2 = swin_attention_func(x, bias, None, 1.0, f_qkv, head, patch, shift)

    if shift == 0:
        return compare(o1, o2)
    else:
        end = patch - shift
        return compare(o1[:, shift:-end, shift:-end], o2[:, shift:-end, shift:-end])


if __name__ == '__main__':
    batch = [1, 2, 4, 8]
    h = [16, 32]
    w = [16, 32]
    head = [1, 2, 4]
    head_dim = [32, 64]
    patch = [4, 8]
    shift = [0, 1]
    for b, h, w, head, head_dim, patch, shift in itertools.product(batch, h, w, head, head_dim, patch, shift):
        print(f"batch={b}, h={h}, w={w}, head={head}, head_dim={head_dim}, patch={patch}, shift={shift}")
        assert forward_core(b, h, w, head, head_dim, patch)
        assert forward_attn(b, h, w, head, head_dim, patch, shift)