import torch


class ConvModel(torch.nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()
        self.conv = torch.nn.Conv2d(3, 2, 3)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


def trace_model():
    conv_model = ConvModel()
    inputs = torch.rand(4, 3, 4, 4)
    traced_model = torch.jit.trace(conv_model, inputs)
    torch.jit.save(traced_model, "./conv_trace_model.pt")
    print("Generated TorchSctipt model.")


if __name__ == "__main__":
    trace_model()
