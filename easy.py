
import torch



x = torch.tensor([2,3,4])
y = x[1:2]

print(y)

x = x * 10
print(y)