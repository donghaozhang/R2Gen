import torch
import torch.nn as nn
a = torch.randn(2, 4, 3)
print('a', a)
b = torch.sum(a, 1)
print('b', b)
c = torch.sum(a, -1)
print('c', c.size())
d = torch.sum(a, -2)
print('d', d.size())
A = True
B = False
if A and B:
	print('ffff')
else:
	print('aaaa')