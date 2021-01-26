# Add two lists using map and lambda   
# numbers1 = [1, 2, 3] 
# numbers2 = [4, 5, 6] 
# result = map(lambda x, y: x + y, numbers1, numbers2) 
# print(list(result))

# str1 = "geek"
# print(id(str1))
import torch
import torch.nn as nn
m = nn.ELU()
input = torch.randn(2)
print(input)
output = m(input)
print(output)