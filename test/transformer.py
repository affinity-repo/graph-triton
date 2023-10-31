import torch
import welder
from transformers import BertTokenizer, BertModel


class AttentionLayer(torch.nn.Module):
    def forward(self, Q: torch.tensor, K: torch.tensor, V: torch.tensor):
        # P = Q @ K.t()
        # S = torch.nn.functional.softmax(P)
        # R = S @ V
        R = Q + K + V
        R1 = K + V
        return R, R1


attn = AttentionLayer()
attn_compiled = torch.compile(attn, backend="welder")
_input = torch.randn(64, 64).cuda()
_half_input = torch.randn(1, 64).cuda()
_output = attn_compiled(_input, _input, _input)
