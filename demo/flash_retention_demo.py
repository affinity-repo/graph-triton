import torch
import welder
from welder import block_m, block_n
from loguru import logger


class FlashRetentionBody(torch.nn.Module):
    def forward(
        self,
        qk: torch.Tensor,
        m: torch.Tensor,
        acco: torch.Tensor,
        r: torch.Tensor,
        r_wo_clamp: torch.Tensor,
    ):
        qkm = qk * m
        r_wo_clamp += qkm.detach().abs().sum(dim=-1, keepdim=True)
        r_new = max(r_wo_clamp, 1)
        # if j > 0:
        acco = acco * r / r_new
        acc = qkm / r_new
        return acco, acc, r_new


qk = torch.rand([block_m, block_n]).cuda().half()
r_wo_clamp = torch.rand([block_m, 1]).cuda().half()
r = torch.rand([block_m, 1]).cuda().half()
m = torch.ones([block_m, block_n]).cuda().half()
acco = torch.rand([block_m, block_n]).cuda().half()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="FlashRetention Triton Kernel Generator"
    )
    parser.add_argument("--fa_body", "-f", type=str, default="FlashRetentionBody")
    parser.add_argument(
        "--welder_config",
        "-n",
        type=str,
        default="demo/flash_retention_welder_config.json",
    )
    args = parser.parse_args()
    logger.info(f"{args}")

    fr_body_demo = FlashRetentionBody()
    welder.graph_compile_fa(
        fr_body_demo,
        args.welder_config,
        *[qk, m, acco, r, r_wo_clamp],
    )
