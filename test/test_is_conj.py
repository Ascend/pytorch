# Copyright (c) 2026 Huawei Technologies Co., Ltd
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Test cases for torch.is_conj API.

This module tests the torch.is_conj function which returns True if the
input tensor is a conjugated tensor (i.e. its conjugate bit is set).

Test coverage:
- Basic functionality: conj() on complex tensor sets conjugate bit
- Non-conjugated tensor: is_conj() returns False
- conj_physical() does NOT set conjugate bit
- Real tensor conj() does NOT set conjugate bit (conjugate of real is itself)
- Different dtypes (float32, float64, complex64, complex128, int32)
- Different tensor shapes (1D, 2D, 3D, scalar, empty)
- Contiguous and non-contiguous tensors
- Chained operations: conj().conj() cancels out
- view/slice/transpose after conj preserves conjugate bit
- Method equivalence: torch.is_conj(t) == t.is_conj()
"""

import torch
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
)

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestIsConj(TestCase):
    """Test cases for torch.is_conj."""

    def test_is_conj_false_for_normal_tensor(self):
        """is_conj returns False for a normal tensor without conjugate bit."""
        t = torch.tensor([1.0, 2.0, 3.0]).to(device_type)
        self.assertFalse(torch.is_conj(t))

    def test_is_conj_true_after_conj_complex(self):
        """is_conj returns True after calling .conj() on a complex tensor."""
        t = torch.tensor([1 + 2j, 3 + 4j]).to(device_type)
        self.assertTrue(torch.is_conj(t.conj()))

    def test_is_conj_false_after_conj_physical(self):
        """is_conj returns False after conj_physical (physical conjugation)."""
        t = torch.tensor([1 + 2j, 3 + 4j]).to(device_type)
        self.assertFalse(torch.is_conj(t.conj_physical()))

    def test_is_conj_false_real_tensor_after_conj(self):
        """is_conj returns False after conj() on real tensors (no conjugate bit set)."""
        t = torch.tensor([1.0, 2.0, 3.0]).to(device_type)
        self.assertFalse(torch.is_conj(t.conj()))

    def test_is_conj_double_conj_cancels(self):
        """is_conj returns False after double conj (conj().conj())."""
        t = torch.tensor([1 + 2j, 3 + 4j]).to(device_type)
        self.assertFalse(torch.is_conj(t.conj().conj()))

    def test_is_conj_complex64(self):
        """is_conj works with complex64 dtype."""
        t = torch.tensor([1 + 2j, 3 + 4j], dtype=torch.complex64).to(device_type)
        self.assertFalse(torch.is_conj(t))
        self.assertTrue(torch.is_conj(t.conj()))

    def test_is_conj_complex128(self):
        """is_conj works with complex128 dtype."""
        t = torch.tensor([1 + 2j, 3 + 4j], dtype=torch.complex128).to(device_type)
        self.assertFalse(torch.is_conj(t))
        self.assertTrue(torch.is_conj(t.conj()))

    def test_is_conj_float32(self):
        """is_conj works with float32 dtype - real tensor conj does not set bit."""
        t = torch.randn(5, dtype=torch.float32).to(device_type)
        self.assertFalse(torch.is_conj(t))
        self.assertFalse(torch.is_conj(t.conj()))

    def test_is_conj_float64(self):
        """is_conj works with float64 dtype - real tensor conj does not set bit."""
        t = torch.randn(5, dtype=torch.float64).to(device_type)
        self.assertFalse(torch.is_conj(t))
        self.assertFalse(torch.is_conj(t.conj()))

    def test_is_conj_int_tensor(self):
        """is_conj works with integer dtype - real tensor conj does not set bit."""
        t = torch.tensor([1, 2, 3], dtype=torch.int32).to(device_type)
        self.assertFalse(torch.is_conj(t))
        self.assertFalse(torch.is_conj(t.conj()))

    def test_is_conj_scalar_tensor(self):
        """is_conj works with scalar (0-dim) complex tensor."""
        t = torch.tensor(3.14 + 2.0j, dtype=torch.complex64).to(device_type)
        self.assertFalse(torch.is_conj(t))
        self.assertTrue(torch.is_conj(t.conj()))

    def test_is_conj_empty_tensor(self):
        """is_conj works with empty complex tensor."""
        t = torch.tensor([], dtype=torch.complex64).to(device_type)
        self.assertFalse(torch.is_conj(t))
        self.assertTrue(torch.is_conj(t.conj()))

    def test_is_conj_2d_tensor(self):
        """is_conj works with 2D complex tensor."""
        t = torch.randn(3, 4, dtype=torch.complex64).to(device_type)
        self.assertFalse(torch.is_conj(t))
        self.assertTrue(torch.is_conj(t.conj()))

    def test_is_conj_3d_tensor(self):
        """is_conj works with 3D complex tensor."""
        t = torch.randn(2, 3, 4, dtype=torch.complex64).to(device_type)
        self.assertFalse(torch.is_conj(t))
        self.assertTrue(torch.is_conj(t.conj()))

    def test_is_conj_non_contiguous(self):
        """is_conj works with non-contiguous complex tensor."""
        t = torch.randn(4, 4, dtype=torch.complex64).to(device_type)
        t_non_contig = t[::2, ::2]
        self.assertFalse(t_non_contig.is_contiguous())
        self.assertFalse(torch.is_conj(t_non_contig))
        self.assertTrue(torch.is_conj(t_non_contig.conj()))

    def test_is_conj_view_preserves_conj_bit(self):
        """is_conj: view after conj preserves the conjugate bit."""
        t = torch.randn(4, 4, dtype=torch.complex64).to(device_type)
        tc = t.conj()
        tv = tc.view(2, 8)
        self.assertTrue(torch.is_conj(tv))

    def test_is_conj_slice_preserves_conj_bit(self):
        """is_conj: slicing after conj preserves the conjugate bit."""
        t = torch.randn(4, 4, dtype=torch.complex64).to(device_type)
        tc = t.conj()
        ts = tc[1:3, :]
        self.assertTrue(torch.is_conj(ts))

    def test_is_conj_method_equivalence(self):
        """is_conj: torch.is_conj(t) is equivalent to t.is_conj()."""
        t = torch.tensor([1 + 2j, 3 + 4j]).to(device_type)
        self.assertEqual(torch.is_conj(t.conj()), t.conj().is_conj())
        self.assertEqual(torch.is_conj(t), t.is_conj())

    def test_is_conj_transpose_preserves_conj_bit(self):
        """is_conj: transpose after conj preserves the conjugate bit."""
        t = torch.randn(3, 4, dtype=torch.complex64).to(device_type)
        tc = t.conj()
        tt = tc.t()
        self.assertTrue(torch.is_conj(tt))

if __name__ == "__main__":
    run_tests()
