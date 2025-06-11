# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import torch
import torch.nn as nn
import torch_npu
import torch_npu._inductor


class Test_issue70():

    def op_forward(self, x):
        return x.mean(-1)


    def test_issue70(self):
        test = Test_issue70()
        compiled_net = torch.compile(test.op_forward, backend="inductor")

        input = torch.randn((1, 1, 7168)).npu()

        output = test.op_forward(input)
        output1 = compiled_net(input)
        torch.testing.assert_allclose(output, output1, rtol=1e-03, atol=1e-03)
        print("valid ok")


if __name__ == "__main__":
    test = Test_issue70()
    test.test_issue70()
