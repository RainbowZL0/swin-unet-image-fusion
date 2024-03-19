import torch
from torch import nn

if __name__ == '__main__':
    a = torch.tensor(data=[1, 2, 3, 4, 5, 6],
                     requires_grad=True,
                     dtype=torch.float)
    b = torch.sum(a*3)
    b.backward()
    print(a.grad)
    nn.Parameter()
