import torch

n = 5
x = torch.ones(n, n, dtype=torch.bool)
print(x)

print(torch.triu(x))

print(torch.triu(x, diagonal=1))

print(torch.triu(x, diagonal=-1))
