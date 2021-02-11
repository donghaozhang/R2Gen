import torch
a = torch.arange(4)
b = torch.reshape(a, (2, 1, 2))
print(b.size())