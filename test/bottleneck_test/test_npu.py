# Owner(s): ["module: unknown"]

import torch
import torch.nn as nn
import torch_npu


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(20, 20)

    def forward(self, input_):
        out = self.linear(input_[:, 10:30])
        return out.sum()


def main():
    data = torch.randn(10, 50).npu()
    model = Model().npu()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    for i in range(10):
        optimizer.zero_grad()
        loss = model(data)
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    main()
