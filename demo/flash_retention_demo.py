import torch
import welder


def retention_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, m: torch.Tensor
):
    qk = q @ k.transpose(-1, -2)
    qkm = qk * m
    r = qkm.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1)
    s = qkm / r
    o = s @ v
    return o


class FlashRetentionBody(torch.nn.Module):
    def forward(
        self,
        qk: torch.Tensor,
        m: torch.Tensor,
        acco: torch.Tensor,
        r: torch.Tensor,
        r_wo_clamp: torch.Tensor,
        j: int,
    ):
        qkm = qk * m
        r_wo_clamp += qkm.detach().abs().sum(dim=-1, keepdim=True)
        r_new = max(r_wo_clamp, 1)
        if j > 0:
            acco = acco * r / r_new
        acc = qkm / r_new
        return acco, acc, r_new


block_m = 64
block_n = 64

qk = torch.rand([block_m, block_n]).cuda().half()
r_wo_clamp = torch.rand([block_m, 1]).cuda().half()
r = torch.rand([block_m, 1]).cuda().half()
m = torch.ones([block_m, block_n]).cuda().half()
acco = torch.rand([block_m, block_n]).cuda().half()
j = 1
fr_body_demo = FlashRetentionBody()
welder.graph_compile(fr_body_demo, *[qk, m, acco, r, r_wo_clamp, j])
