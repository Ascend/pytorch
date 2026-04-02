"""
Add validation cases for torch.nn.models APIs on NPU:
1. test/test_modules.py from PyTorch community lacks sufficient API validations, so this file is added.
2. This file validates torch.nn.Module, torch.nn.Module.state_dict, torch.nn.ModuleDict, torch.nn.ModuleList (extendable).
"""

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests, TestCase
import torch_npu

device = torch.device("npu:0")


class TestNPUModuleAPIs(TestCase):
    def test_module_device_consistency(self):
        """验证Module设备迁移后参数/缓冲区设备一致"""
        m = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.BatchNorm1d(20)).to(device)
        
        for p in m.parameters():
            self.assertEqual(p.device, device)
        for b in m.buffers():
            self.assertEqual(b.device, device)

    def test_module_state_dict(self):
        """验证state_dict保存/加载后状态一致"""
        base = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.BatchNorm1d(20))
        m = nn.Module()
        m.dict = nn.ModuleDict({"linear": nn.Linear(10, 20), "base": base})
        m.list = nn.ModuleList([nn.Linear(20, 30), base])
        m.to(device)
        
        sd = m.state_dict()
        torch.npu.synchronize()
        
        for v in sd.values():
            self.assertEqual(v.device, device)
        
        base2 = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.BatchNorm1d(20))
        m2 = nn.Module()
        m2.dict = nn.ModuleDict({"linear": nn.Linear(10, 20), "base": base2})
        m2.list = nn.ModuleList([nn.Linear(20, 30), base2])
        m2.load_state_dict(sd)
        m2.to(device)
        torch.npu.synchronize()
        
        for (n1, p1), (n2, p2) in zip(m.named_parameters(), m2.named_parameters()):
            self.assertTrue(torch.allclose(p1, p2))

    def test_moduledict_operations(self):
        """验证ModuleDict增删/索引/遍历"""
        m = nn.ModuleDict({"a": nn.Linear(10, 20).to(device)})
        
        self.assertIn("a", m)
        m["b"] = nn.Linear(20, 30).to(device)
        self.assertEqual(m["b"].weight.device, device)
        del m["b"]
        self.assertNotIn("b", m)
        
        for sub in m.values():
            for p in sub.parameters():
                self.assertEqual(p.device, device)

    def test_modulelist_operations(self):
        """验证ModuleList索引/新增/删除/遍历"""
        m = nn.ModuleList([nn.Linear(20, 30).to(device), nn.BatchNorm1d(30).to(device)])
        
        self.assertEqual(m[0].weight.device, device)
        self.assertEqual(m[1].running_mean.device, device)
        
        m.append(nn.Linear(30, 40).to(device))
        m.insert(0, nn.Linear(10, 20).to(device))
        m.pop(-1)  # 修复：指定索引
        m.pop(0)   # 修复：指定索引
        
        self.assertEqual(len(m), 2)
        for sub in m:
            for p in sub.parameters():
                self.assertEqual(p.device, device)


if __name__ == "__main__":
    run_tests()