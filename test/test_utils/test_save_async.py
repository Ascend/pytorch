import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.utils.path_manager import PathManager



class TestAsyncSave(TestCase):
    test_save_path = os.path.join(
        os.path.realpath(os.path.dirname(__file__)), "test_save_async")

    @classmethod
    def setUpClass(cls):
        PathManager.make_dir_safety(TestAsyncSave.test_save_path)

    @classmethod
    def tearDownClass(cls):
        time.sleep(5)
        PathManager.remove_path_safety(TestAsyncSave.test_save_path)

        
    def test_async_save_tensor(self):
        save_tensor = torch.zeros(1024, dtype=torch.float32).npu()
        save_path = os.path.join(TestAsyncSave.test_save_path, "save_tensor.pt")
        torch_npu.utils.save_async(save_tensor, save_path)
    
    def test_async_save_model(self):
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
            nn.ReLU()
        )
        model = model.npu()

        criterion = nn.CrossEntropyLoss()
        optimerizer = optim.SGD(model.parameters(), lr=0.01)
        for epoch in range(5):
            for step in range(5):
                input_data = torch.ones(6400, 100).npu()
                labels = torch.randint(0, 5, (6400,)).npu()

                outputs = model(input_data)
                loss = criterion(outputs, labels)

                optimerizer.zero_grad()
                loss.backward()

                optimerizer.step()
            
            save_sync_path = os.path.join(TestAsyncSave.test_save_path, f"model_sync_{epoch}_{step}.path")
            save_async_path = os.path.join(TestAsyncSave.test_save_path, f"model_async_{epoch}_{step}.path")

            torch.save(model, save_sync_path)
            torch_npu.utils.save_async(model, save_async_path, model=model)

            torch.npu.synchronize()
    
    def test_async_save_checkpoint(self):
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
            nn.ReLU()
        )
        model = model.npu()

        criterion = nn.CrossEntropyLoss()
        optimerizer = optim.SGD(model.parameters(), lr=0.01)
        for epoch in range(5):
            for step in range(5):
                input_data = torch.ones(6400, 100).npu()
                labels = torch.randint(0, 5, (6400,)).npu()

                outputs = model(input_data)
                loss = criterion(outputs, labels)

                optimerizer.zero_grad()
                loss.backward()

                optimerizer.step()
            checkpoint = {
                "model" : model.state_dict(),
                "optimizer" : optimerizer.state_dict(),
                "epoch" : epoch,
                "step" : step
            }

            save_path = os.path.join(TestAsyncSave.test_save_path, f"checkpoint_{epoch}_{step}.path")

            torch_npu.utils.save_async(checkpoint, save_path, model=model)
    

if __name__ == '__main__':
    torch.npu.set_device(0)
    run_tests()
