import torch
import welder


class GraphFuseDemo(torch.nn.Module):
    def forward(
        self,
        a0: torch.tensor,
        a1: torch.tensor,
        e0: torch.tensor,
    ):
        # A
        b0 = a0 * a1
        # B
        b_out0 = b0 * 2.0
        # C
        c_out0 = b_out0 - 1.42
        # D
        d_out0 = torch.sigmoid(b_out0)
        # E
        e_out0 = torch.relu(e0)
        return c_out0, d_out0, e_out0


demo = GraphFuseDemo()
a0 = torch.randn(2, 2, 64, 64).cuda()
a1 = torch.randn(2, 2, 1, 1).cuda()
e0 = torch.randn(2048, 8).cuda()

welder.graph_compile(demo, *[a0, a1, e0])
