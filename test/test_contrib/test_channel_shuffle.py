import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.contrib.module import ChannelShuffle


class TestChannelShuffle(TestCase):
    def cpu_channel_shuffle(self, x, groups, split_shuffle):

        # split_shuffle cpu仅支持False场景
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups
        x.requires_grad_(True)
        # reshape
        x = x.view(batchsize, groups, channels_per_group, height, width)

        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(batchsize, -1, height, width)
        output = x.view(batchsize, -1, height, width)
        return output.detach().numpy()

    def npu_channel_shuffle(self, x, groups, split_shuffle):
        model = ChannelShuffle(groups, split_shuffle=split_shuffle)
        x = x.npu()
        model = model.npu()
        output = model(x, x)

        return output.detach().cpu().numpy()

    def npu_channel_shuffle_backward(self, x, groups, split_shuffle):

        model = ChannelShuffle(4, split_shuffle=split_shuffle)
        x = x.npu()
        x.requires_grad_(True)
        model = model.npu()
        output = model(x, x)

        loss = sum([i.sum() for i in output]) if split_shuffle else output.sum()
        loss.backward()

        return output[0], output[1]

    def test_channel_shuffle_1_False(self):
        split_shuffle = False
        x = torch.randn(2, 2, 3, 3)
        conv = torch.nn.Conv2d(2, 2, 1)
        x1 = conv(x)
        cpu_out = self.cpu_channel_shuffle(x1, groups=2, split_shuffle=False)
        x1 = x1.npu()
        npu_out = self.npu_channel_shuffle(x1, groups=2, split_shuffle=False)

        self.assertRtolEqual(cpu_out, npu_out)

    def test_npu_channel_shuffle_2_True(self):
        # There is no benchmarking data when split_shuffle=True,
        x = torch.randn(2, 2, 3, 3)
        conv = torch.nn.Conv2d(2, 2, 1)
        x1 = conv(x)
        x1 = x1.npu()
        npu_output1, npu_output2 = self.npu_channel_shuffle_backward(x1, groups=4, split_shuffle=True)

        expedt_cpu_output1 = torch.tensor([[[[0.0385, -0.3217, -0.0174],
                                             [0.1337, -0.1197, -0.0415],
                                             [0.0843, 0.1638, -0.0149]],

                                            [[0.0385, -0.3217, -0.0174],
                                             [0.1337, -0.1197, -0.0415],
                                             [0.0843, 0.1638, -0.0149]]],


                                           [[[-0.0203, -0.3950, -0.1230],
                                             [0.2059, 0.0822, 0.6951],
                                               [-0.0773, 0.0535, -0.0462]],

                                            [[-0.0203, -0.3950, -0.1230],
                                               [0.2059, 0.0822, 0.6951],
                                               [-0.0773, 0.0535, -0.0462]]]], dtype=torch.float32)

        expedt_cpu_output2 = torch.tensor([[[[0.5454, -0.0463, 0.4660],
                                             [0.7197, 0.2986, 0.4197],
                                             [0.6225, 0.7925, 0.4614]],

                                            [[0.5454, -0.0463, 0.4660],
                                             [0.7197, 0.2986, 0.4197],
                                             [0.6225, 0.7925, 0.4614]]],


                                           [[[0.4537, -0.1535, 0.3048],
                                             [0.8306, 0.6178, 1.7047],
                                               [0.3617, 0.5625, 0.4009]],

                                            [[0.4537, -0.1535, 0.3048],
                                               [0.8306, 0.6178, 1.7047],
                                               [0.3617, 0.5625, 0.4009]]]], dtype=torch.float32)
        self.assertRtolEqual(expedt_cpu_output1.numpy(), npu_output1.detach().cpu().numpy())
        self.assertRtolEqual(expedt_cpu_output2.numpy(), npu_output2.detach().cpu().numpy())


if __name__ == "__main__":
    run_tests()
