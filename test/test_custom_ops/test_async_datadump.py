import os
import shutil
import time

import numpy
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

# Runtime2.0 unsupport OutfeedEnqueueOpV2
os.environ['RUNTIME_V2_BLACKLIST'] = 'OutfeedEnqueueOpV2'


class TestAsyncDatadump(TestCase):

    def test_matmul_forward(self):
        path = './datadump_output_test_matmul_forward/'
        remove_test_tmp_dir(path)
        # datadump
        input0 = torch.tensor([[1, 1, 1], [2, 2, 2]], dtype=torch.float, requires_grad=True).npu()
        input1 = torch.tensor([[1, 2], [1, 2], [1, 2]], dtype=torch.float).npu()
        input0 = torch_npu.npu.dump_enable(input0, path=path)
        output0 = torch.matmul(input0, input1)
        output0 = torch_npu.npu.dump_disable(output0)
        loss = output0.sum()
        loss.backward()
        # load and check
        time.sleep(1)
        self.assertEqual(len(os.listdir(path)), 3)
        tensorDict = load(path, 'MatMul')
        for index in tensorDict:
            loadInput0 = tensorDict[index]['input0']
            loadInput1 = tensorDict[index]['input1']
            loadOutput0 = tensorDict[index]['output0']
            self.assertEqual(loadOutput0, torch.matmul(loadInput0, loadInput1))
        remove_test_tmp_dir(path)

    def test_matmul_backward(self):
        path = './datadump_output_test_matmul_backward/'
        remove_test_tmp_dir(path)
        # datadump
        input0 = torch.tensor([[1, 1, 1], [2, 2, 2]], dtype=torch.float, requires_grad=True).npu()
        input1 = torch.tensor([[1, 2], [1, 2], [1, 2]], dtype=torch.float).npu()
        input0 = torch_npu.npu.dump_enable(input0, dump_backward=True, path=path)
        output0 = torch.matmul(input0, input1)
        output0 = torch_npu.npu.dump_disable(output0)
        loss = output0.sum()
        loss.backward()
        # load and check
        time.sleep(1)
        self.assertEqual(len(os.listdir(path)), 8)
        tensorDict = load(path, 'MatMul')
        for index in tensorDict:
            loadInput0 = tensorDict[index]['input0']
            loadInput1 = tensorDict[index]['input1']
            loadOutput0 = tensorDict[index]['output0']
            self.assertEqual(loadOutput0, torch.matmul(loadInput0, loadInput1))
        remove_test_tmp_dir(path)

    def test_matmul_ops_filter(self):
        path = './datadump_output_test_matmul_ops_filter/'
        remove_test_tmp_dir(path)
        # datadump
        input0 = torch.tensor([[1, 1, 1], [2, 2, 2]], dtype=torch.float, requires_grad=True).npu()
        input1 = torch.tensor([[1, 2], [1, 2], [1, 2]], dtype=torch.float).npu()
        input0 = torch_npu.npu.dump_enable(input0, dump_backward=True, ops=['MatMul'], path=path)
        output0 = torch.matmul(input0, input1)
        output0 = torch_npu.npu.dump_disable(output0)
        loss = output0.sum()
        loss.backward()
        # load and check
        time.sleep(1)
        self.assertEqual(len(os.listdir(path)), 6)
        remove_test_tmp_dir(path)

    def test_mutil_ops_filter(self):
        path = './datadump_output_test_mutil_ops_filter/'
        remove_test_tmp_dir(path)
        # datadump
        input0 = torch.tensor([[1, 1, 1], [2, 2, 2]], dtype=torch.float, requires_grad=True).npu()
        input1 = torch.tensor([[1, 2], [1, 2], [1, 2]], dtype=torch.float).npu()
        input0 = torch_npu.npu.dump_enable(input0, dump_backward=True, ops=['MatMul', 'Add'], path=path)
        output0 = torch.matmul(input0, input1)
        output0 = output0 + 1
        output0 = output0 * 2
        output0 = torch_npu.npu.dump_disable(output0)
        loss = output0.sum()
        loss.backward()
        # load and check
        time.sleep(1)
        self.assertEqual(len(os.listdir(path)), 8)
        remove_test_tmp_dir(path)

    def test_format(self):
        path = './datadump_output_test_format/'
        remove_test_tmp_dir(path)
        # datadump
        input0 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float).npu()
        input0 = torch_npu.npu.dump_enable(input0, path=path)
        output0 = torch_npu.npu_format_cast(input0, 3)
        torch_npu.npu.dump_disable(output0)
        # load and check
        time.sleep(1)
        self.assertEqual(len(os.listdir(path)), 2)
        tensorDict = load(path, 'Identity')
        for index in tensorDict:
            loadInput0 = tensorDict[index]['input0']
            loadOutput0 = tensorDict[index]['output0']
            self.assertEqual(loadOutput0, loadInput0)
        remove_test_tmp_dir(path)

    def test_contiguous(self):
        path = './datadump_output_test_contiguous/'
        remove_test_tmp_dir(path)
        # datadump
        input0 = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float).npu()
        output0 = torch.tensor([[-1, -2], [-3, -4], [-5, -6]], dtype=torch.float).npu()
        output0 = torch.as_strided(output0, (2, 2), (1, 2), 2)
        output0 = torch_npu.npu.dump_enable(output0, path=path)
        output0[1:2, :].copy_(input0[1:2, :])
        torch_npu.npu.dump_disable(output0)
        # load and check
        time.sleep(1)
        self.assertEqual(len(os.listdir(path)), 3)
        tensorDict = load(path, 'ViewCopy')
        for index in tensorDict:
            loadInput0 = tensorDict[index]['input0']
            loadInput1 = tensorDict[index]['input1']
            loadOutput0 = tensorDict[index]['output0']
            self.assertEqual(loadOutput0, loadInput0)
            self.assertEqual(loadInput1, loadInput1)
        remove_test_tmp_dir(path)

    def test_conv2d_with_format(self):
        path = './datadump_output_test_conv2d_with_format/'
        remove_test_tmp_dir(path)
        # datadump
        conv = torch.nn.Conv2d(1, 1, 2).npu()
        input0 = torch.randn(1, 1, 3, 3).npu()
        input0 = torch_npu.npu_format_cast(input0, 3)
        input0 = torch_npu.npu.dump_enable(input0, path=path)
        output0 = conv(input0)
        torch_npu.npu.dump_disable(output0)
        # load and check
        time.sleep(1)
        self.assertEqual(len(os.listdir(path)), 4)
        tensorDict = load(path, 'Conv2D')
        for index in tensorDict:
            loadInput0 = tensorDict[index]['input0']
            loadInput1 = tensorDict[index]['input1']
            loadInput2 = tensorDict[index]['input2']
            loadOutput0 = tensorDict[index]['output0']
            self.assertEqual(input0.cpu(), loadInput0)
            self.assertEqual(conv.weight.cpu(), loadInput1)
            self.assertEqual(conv.bias.cpu(), loadInput2)
            self.assertEqual(output0.cpu(), loadOutput0)
        remove_test_tmp_dir(path)


def load(path, opName):
    result = {}
    fileList = os.listdir(path)
    for fileName in fileList:
        if not '_' + opName + '_' in fileName:
            continue
        # fileName Ex: 0_MatMul_input0_shape[2,3]_stride[3,1]_offset[0]_format[2].npy
        infos = fileName.split('_')
        index = int(infos[0])
        tmpTensor = torch.tensor(numpy.load(path + fileName))
        tmpTensor = torch.as_strided(tmpTensor,
                                     tuple(map(int, infos[3][6:-1].split(','))),
                                     tuple(map(int, infos[4][7:-1].split(','))),
                                     int(infos[5][7:-1]))
        if index in result:
            result[index][infos[2]] = tmpTensor
        else:
            result[index] = {infos[2]: tmpTensor}
    print("load:", result)
    return result


def remove_test_tmp_dir(path):
    if os.path.exists(path) and os.path.isdir(path):
        print("Remove temporary dir:", path)
        shutil.rmtree(path)


if __name__ == "__main__":
    run_tests()
