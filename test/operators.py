import welder
import torch

a = torch.randn(1024, 1024).cuda()
b = torch.randn(1024, 1024).cuda()


class Model(torch.nn.Module):
    def forward(self, a, b):
        return torch.matmul(a, b)


model = Model()

model = torch.compile(
    model, backend="welder"
)  # This is the only line of code that we changed

output = model(a, b)
