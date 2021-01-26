import torch
import torch.nn as nn
m = nn.Linear(20, 30)
input = torch.randn(4, 20, 128, 20)
print('the size of input', input.size())
output = m(input)
print('the size of output', output.size())