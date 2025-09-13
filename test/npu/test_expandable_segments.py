import os
import gc
import unittest

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch.testing._internal.common_utils import TestCase, run_tests, TEST_PRIVATEUSE1

os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"


class Test_expandable_segments(TestCase):
    def test_empty_virt_addr_cache(self):
        gc.collect()
        torch_npu.npu.empty_cache()
        prev = 0

        x = torch.empty((7500, 1024, 1024), device="npu")
        del x
        last_r = torch_npu.npu.memory_reserved()

        torch_npu.npu.empty_virt_addr_cache()
        new_r = torch_npu.npu.memory_reserved()
        self.assertEqual(new_r, prev)
        self.assertEqual(torch_npu.npu.max_memory_reserved(), last_r)

        # test re-alloc after empty virtual address
        try:
            y = torch.empty((7500, 1024, 1024), device="npu")
            self.assertGreater(torch_npu.npu.memory_allocated(), prev)
        finally:
            if y is not None:
                del y
                self.assertEqual(torch_npu.npu.memory_allocated(), prev)
                torch_npu.npu.empty_virt_addr_cache()
                # empty unmapped physical handles with empty_cache()
                torch_npu.npu.empty_cache()
                self.assertEqual(torch_npu.npu.memory_reserved(), prev)

    @unittest.skipIf(TEST_PRIVATEUSE1, "NPU not available for graph capture")
    def test_set_segment_state_to_checkpoint_when_expandable_segments(self):
        def tensor_metadata(x):
            return {
                "nbytes": x.untyped_storage().nbytes(),
                "data_ptr": x.untyped_storage().data_ptr(),
                "size": x.shape,
                "stride": x.stride(),
                "dtype": x.dtype,
                "device": x.device,
                "storage_offset": x.storage_offset(), }

        def reconstruct_from_tensor_metadata(metadata):
            s = torch._C._construct_storage_from_data_pointer(
                metadata["data_ptr"], metadata["device"], metadata["nbytes"])
            t = torch.empty([0], device=metadata["device"], dtype=metadata["dtype"])
            t.set_(source=s, storage_offset=metadata["storage_offset"],
                   size=metadata["size"], stride=metadata["stride"], )
            return t

        def cudagraphify(fn, inputs, pool, stream):
            torch.npu.synchronize()
            gc.collect()
            torch.npu.empty_cache()

            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream, pool=pool):
                static_outputs = fn(*inputs)
            return graph, static_outputs

        def foo(x, idx):
            r1 = x.expand([1, 2097152 // 8]).sqrt()
            r2 = x.expand([idx, 2097152]).clone()
            return r1, r2

        # init
        pool_id = torch.npu.graph_pool_handle()
        com_stream = torch.npu.Stream()
        com_device = torch_npu.npu.current_device()
        inp = torch.tensor([7]).npu()

        # start capture graph1
        graph1, outputs1 = cudagraphify(foo, [inp, 1], pool=pool_id, stream=com_stream)
        graph1_state = torch_npu._C._npu_getCheckpointState(com_device, pool_id)
        output1_metadata = [tensor_metadata(t) for t in outputs1]
        outputs1 = None

        # start capture graph2
        graph2, outputs2 = cudagraphify(foo, [inp, 2], pool=pool_id, stream=com_stream)
        graph2_state = torch_npu._C._npu_getCheckpointState(com_device, pool_id)
        graph2.replay()
        outputs2 = None

        # replay graph1
        graph1.replay()
        reconstructed_tensors1 = [reconstruct_from_tensor_metadata(metadata) for metadata in output1_metadata]
        output1_new_storage = [output.untyped_storage()._cdata for output in reconstructed_tensors1]
        torch_npu._C._npu_setCheckpointPoolState(com_device, graph1_state, [], output1_new_storage)


if __name__ == '__main__':
    run_tests()
