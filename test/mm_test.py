import welder
import torch

a = torch.randn(1024, 1024).cuda()
b = torch.randn(1024, 1024).cuda()


c = welder.mm.tuned_mm(a, b)

print(c)
