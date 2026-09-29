# Copyright (c) 2026 Huawei Technologies Co., Ltd
# All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NPU coverage for Tensor.col_indices on sparse tensors.

This dedicated module keeps sparse-index API validation independent from
the broader view-ops test suite and can be extended for related APIs.
"""

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import TestCase, run_tests

import torch_npu.testing  # noqa: F401


class TestTensorColIndices(TestCase):
    def test_crow_col_indices(self, device):
        dense = torch.ones(4, device=device)
        with self.assertRaisesRegex(RuntimeError, "col_indices"):
            dense.col_indices()

        # Cover CSR/BSR, empty/non-empty, and int32/int64 indices.
        for layout in (torch.sparse_csr, torch.sparse_bsr):
            for is_empty in (False, True):
                for index_dtype in (torch.int32, torch.int64):
                    if layout == torch.sparse_csr:
                        crow = torch.tensor(
                            (0, 0, 0) if is_empty else (0, 1, 2),
                            dtype=index_dtype,
                            device=device,
                        )
                        col = torch.tensor(
                            () if is_empty else (1, 0),
                            dtype=index_dtype,
                            device=device,
                        )
                        values = (
                            torch.empty(
                                (0,),
                                dtype=torch.float32,
                                device=device,
                            )
                            if is_empty
                            else torch.tensor(
                                (1.0, 2.0),
                                dtype=torch.float32,
                                device=device,
                            )
                        )
                        t = torch.sparse_csr_tensor(
                            crow,
                            col,
                            values,
                            size=(2, 2),
                            device=device,
                            check_invariants=False,
                        )
                    else:
                        crow = torch.tensor(
                            (0, 0, 0) if is_empty else (0, 1, 2),
                            dtype=index_dtype,
                            device=device,
                        )
                        col = torch.tensor(
                            () if is_empty else (1, 0),
                            dtype=index_dtype,
                            device=device,
                        )
                        values = (
                            torch.empty(
                                (0, 2, 2),
                                dtype=torch.float32,
                                device=device,
                            )
                            if is_empty
                            else torch.arange(
                                1,
                                9,
                                dtype=torch.float32,
                                device=device,
                            ).reshape(2, 2, 2)
                        )
                        t = torch.sparse_bsr_tensor(
                            crow,
                            col,
                            values,
                            size=(4, 4),
                            device=device,
                            check_invariants=False,
                        )

                    # This is the test. If crow_indices is not a view op it'll
                    # trigger an internal assert due to use count greater than 1
                    # in debug build.
                    t.crow_indices()
                    result = t.col_indices()

                    self.assertEqual(result, col)
                    self.assertEqual(result.shape, col.shape)
                    self.assertEqual(result.dtype, col.dtype)
                    self.assertEqual(result.device, col.device)
                    self.assertEqual(
                        result.data_ptr(),
                        t.col_indices().data_ptr(),
                    )


instantiate_device_type_tests(
    TestTensorColIndices,
    globals(),
    only_for="privateuse1",
)


if __name__ == "__main__":
    run_tests()
