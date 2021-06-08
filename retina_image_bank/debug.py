import torch
# (3, 224, 224) => (16, 1, 3, 224, 224)
# ([40, 2, 3, 224, 224])
bz = 16 # batch isze
a = torch.randn(3, 224, 224)
print('a size', a.size())
a = a.unsqueeze(0)
print('a size after refinement', a.size())
a = a.unsqueeze(0)
print('a size after second refinement', a.size())
# b = a.repeat(3, 1, 1, 1)
b = a.repeat(bz, 1, 1, 1, 1)
print('b shape', b.size())
