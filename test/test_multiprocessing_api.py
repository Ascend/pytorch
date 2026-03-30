"""
Add validation cases for torch.multiprocessing APIs on NPU:
1. test/test_multiprocessing.py from PyTorch community lacks sufficient API validations, so this file is added.
2. This file validates:
    torch.multiprocessing.Array,
    torch.multiprocessing.Value,
    torch.multiprocessing.Manager,
    torch.multiprocessing.Pipe,
    torch.multiprocessing.get_start_method,
    torch.multiprocessing.set_start_method,
    torch.multiprocessing.reductions.init_reductions,
    torch.multiprocessing.reductions.reduce_tensor
    (extendable).
"""

import os
import sys
import time
import unittest
import torch
import torch.multiprocessing as mp
from torch.testing._internal.common_utils import TestCase, run_tests
import torch_npu

def _worker(queue):
    # Get the current startup method in the child process and put it into the queue
    queue.put(mp.get_start_method())


class TestMultiprocessingAPIs(TestCase):
    def setUp(self):
        # Save the original start method to restore after test
        self.original_start_method = mp.get_start_method()

    def tearDown(self):
        # Restore the original start method
        if self.original_start_method is not None:
            mp.set_start_method(self.original_start_method, force=True)

    def test_get_set_start_method(self):
        """Test get_start_method and set_start_method APIs"""
        # Test default start method
        default_method = mp.get_start_method(allow_none=True)
        self.assertIn(default_method, [None, 'spawn', 'fork', 'forkserver'])

        # Test setting start method
        for method in ['spawn', 'fork', 'forkserver']:
            # Some platforms may not support certain start methods

            mp.set_start_method(method, force=True)
            current_method = mp.get_start_method()
            self.assertEqual(current_method, method)
            
            # Verify that the child process uses the correct start method
            queue = mp.SimpleQueue()
            process = mp.Process(target=_worker, args=(queue,))
            process.start()
            process.join()
            
            # Get the start method from the child process
            self.assertFalse(queue.empty(), "Queue should contain the start method")
            child_method = queue.get()
            self.assertEqual(child_method, method, 
                                f"Child process should use {method} start method")
            
            # Verify that the main process context has not changed
            self.assertEqual(mp.get_start_method(), method)
                

    def test_value(self):
        """Test Value API"""
        # Test Value with basic types
        def func(v):
            v.value += 1

        v = mp.Value('i', 0)
        self.assertEqual(v.value, 0)

        p = mp.Process(target=func, args=(v,))
        p.start()
        p.join()

        self.assertEqual(v.value, 1)

        # Test Value with other types
        v_float = mp.Value('f', 0.5)
        self.assertAlmostEqual(v_float.value, 0.5)

        v_double = mp.Value('d', 0.123456789)
        self.assertAlmostEqual(v_double.value, 0.123456789)

    def test_array(self):
        """Test Array API"""
        # Test Array with basic types
        def func(arr):
            for i, _ in enumerate(arr):
                arr[i] += 1

        arr = mp.Array('i', [0, 1, 2, 3, 4])
        self.assertEqual(list(arr), [0, 1, 2, 3, 4])

        p = mp.Process(target=func, args=(arr,))
        p.start()
        p.join()

        self.assertEqual(list(arr), [1, 2, 3, 4, 5])

        # Test Array with other types
        arr_float = mp.Array('f', [0.0, 1.0, 2.0])
        self.assertEqual(list(arr_float), [0.0, 1.0, 2.0])

    def test_pipe(self):
        """Test Pipe API"""
        # Create pipe
        parent_conn, child_conn = mp.Pipe()

        def send_data(conn):
            conn.send('Hello from child')
            conn.send(42)
            conn.send([1, 2, 3])
            conn.close()

        # Start child process to send data
        p = mp.Process(target=send_data, args=(child_conn,))
        p.start()

        # Receive data from parent process
        self.assertEqual(parent_conn.recv(), 'Hello from child')
        self.assertEqual(parent_conn.recv(), 42)
        self.assertEqual(parent_conn.recv(), [1, 2, 3])

        p.join()

        # Test bidirectional communication
        def send_receive(conn):
            msg = conn.recv()
            conn.send(msg + ' received')

        parent_conn, child_conn = mp.Pipe()
        p = mp.Process(target=send_receive, args=(child_conn,))
        p.start()

        parent_conn.send('Test message')
        self.assertEqual(parent_conn.recv(), 'Test message received')
        p.join()

    def test_manager(self):
        """Test Manager API"""
        # Test shared list
        def modify_list(lst):
            lst.append(4)
            lst[0] = 100

        with mp.Manager() as manager:
            shared_list = manager.list([1, 2, 3])
            self.assertEqual(list(shared_list), [1, 2, 3])

            p = mp.Process(target=modify_list, args=(shared_list,))
            p.start()
            p.join()

            self.assertEqual(list(shared_list), [100, 2, 3, 4])

            # Test shared dictionary
            shared_dict = manager.dict({'a': 1, 'b': 2})
            self.assertEqual(dict(shared_dict), {'a': 1, 'b': 2})

            # Test shared value
            shared_value = manager.Value('i', 0)
            self.assertEqual(shared_value.value, 0)

            # Test shared namespace
            shared_namespace = manager.Namespace()
            shared_namespace.x = 1
            shared_namespace.y = 'test'
            self.assertEqual(shared_namespace.x, 1)
            self.assertEqual(shared_namespace.y, 'test')

    def test_reductions(self):
        """Test torch.multiprocessing.reductions.init_reductions and reduce_tensor APIs"""
        # Test init_reductions - verify it doesn't raise any exception
        mp.reductions.init_reductions()
        
        # Test reduce_tensor directly for CPU tensor
        # Create a simple CPU tensor
        tensor = torch.tensor([1, 2, 3, 4])
        reduced = mp.reductions.reduce_tensor(tensor)
        
        # Verify the reduced form is a tuple with expected structure
        self.assertIsInstance(reduced, tuple)
        self.assertEqual(len(reduced), 2)
        
        # Try to reconstruct the tensor
        constructor, args = reduced
        reconstructed = constructor(*args)
        
        # Verify reconstruction worked
        self.assertTrue(torch.equal(tensor, reconstructed))
        self.assertEqual(tensor.device, reconstructed.device)
        self.assertEqual(tensor.dtype, reconstructed.dtype)
        
        # Test with NPU tensor if available
        if torch.npu.is_available():
            # Create a simple NPU tensor
            npu_tensor = torch.tensor([1, 2, 3, 4], device='npu:0')
            
            # Test reduce_tensor for NPU tensor
            reduced_npu = mp.reductions.reduce_tensor(npu_tensor)
            self.assertIsInstance(reduced_npu, tuple)
            self.assertEqual(len(reduced_npu), 2)
            
            # Verify reconstruction for NPU tensor
            constructor_npu, args_npu = reduced_npu
            reconstructed_npu = constructor_npu(*args_npu)
            self.assertTrue(torch.equal(npu_tensor.cpu(), reconstructed_npu.cpu()))
    
    def test_reductions_invalid_input(self):
        """Test reduction APIs with invalid inputs"""
        # Test reduce_tensor with invalid input
        with self.assertRaises(Exception):
            mp.reductions.reduce_tensor(None)
        
        with self.assertRaises(Exception):
            mp.reductions.reduce_tensor("not a tensor")
        
        # Test init_reductions multiple times (should be safe)
        mp.reductions.init_reductions()
        mp.reductions.init_reductions()


if __name__ == '__main__':
    run_tests()