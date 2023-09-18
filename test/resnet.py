import torch

import torch._dynamo as torchdynamo
import torch._inductor as torchinductor

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True).cuda()
opt_model = torch.compile(model, backend="inductor")
_input = torch.randn(1,3,64,64).cuda()
out = opt_model(_input)