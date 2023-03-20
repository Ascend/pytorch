# Copyright (c) 2023 Huawei Technologies Co., Ltd
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import torch_npu

torch_npu.npu.set_device(0)

JIT_COMPILE_EXTENSION = True
if JIT_COMPILE_EXTENSION:
    from torch_npu.utils.cpp_extension import load
    relu_bisheng = load(name="relu_bisheng", sources=["./relu_bisheng.cpp"], verbose=1)
else:
    # need run before: python setup.py install
    import relu_bisheng

DEVICE = "npu"


class ReLUBiShengFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1):
        ctx.save_for_backward(input1)
        output = relu_bisheng.forward(input1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, *_ = ctx.saved_tensors
        grad_input = relu_bisheng.backward(grad_output, output)
        return grad_input


class BiShengReLU(torch.nn.Module):
    def forward(self, x):
        return ReLUBiShengFunction.apply(x)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            BiShengReLU(),
            nn.Linear(512, 512),
            BiShengReLU(),
            nn.Linear(512, 10))

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train_nn(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_nn(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main():
    torch.manual_seed(0)
    training_data = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=ToTensor())

    test_data = datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=ToTensor())

    nn_model = NeuralNetwork().to(DEVICE)
    print(nn_model)

    nn_loss_fn = nn.CrossEntropyLoss()
    nn_optimizer = torch.optim.SGD(nn_model.parameters(), lr=1e-3)
    batch_size = 64
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_nn(train_dataloader, nn_model, nn_loss_fn, nn_optimizer)
        test_nn(test_dataloader, nn_model, nn_loss_fn)
    print("Done!")


if __name__ == "__main__":
    main()
