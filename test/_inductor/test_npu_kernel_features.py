import sympy
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu._inductor.codegen.npu_kernel_features import NumelList


class TestNumeList(TestCase):
    def test_numels(self):
        numel_list = NumelList([2, 3, 4])
        self.assertEqual(numel_list.numels(), 24)
    
    def test_equality(self):
        numel_list1 = NumelList([2, 3, 4])
        numel_list2 = NumelList([2, 3, 4])
        self.assertTrue(numel_list1 == numel_list2)

        self.assertTrue(numel_list1 == 24)
        self.assertFalse(numel_list1 == 25)

    def test_less_than(self):
        numel_list1 = NumelList([2, 3, 4])
        numel_list2 = NumelList([3, 4, 5])
        self.assertTrue(numel_list1 < numel_list2)

        self.assertTrue(numel_list1 < 25)
        self.assertFalse(numel_list1 < 24)

    def test_greater_than(self):
        numel_list1 = NumelList([2, 3, 5])
        numel_list2 = NumelList([2, 3, 4])
        self.assertTrue(numel_list1 > numel_list2)

    def test_less_than_or_equal(self):
        numel_list1 = NumelList([2, 3, 4])
        numel_list2 = NumelList([3, 4, 5])
        self.assertTrue(numel_list1 <= numel_list2)

        self.assertTrue(numel_list1 <= 25)
        self.assertTrue(numel_list1 <= 24)
        self.assertFalse(numel_list1 <= 23)

    def test_greater_than_or_equal(self):
        numel_list1 = NumelList([2, 3, 5])
        numel_list2 = NumelList([2, 3, 4])
        self.assertTrue(numel_list1 >= numel_list2)
    
    def test_modulo(self):
        numel_list = NumelList([2, 3, 4])
        self.assertEqual(numel_list % 5, 4)
    
    def test_division(self):
        numel_list = NumelList([2, 3, 4])
        self.assertEqual(numel_list / 2, 12.0)
        self.assertEqual(numel_list // 2, 12)

    def test_multiplication(self):
        numel_list = NumelList([2, 3, 4])
        self.assertEqual(numel_list * 2, 48)
        self.assertEqual(2 * numel_list, 48)

    def test_addition(self):
        numel_list = NumelList([2, 3, 4])
        self.assertEqual(numel_list + 2, 26)
        self.assertEqual(2 + numel_list, 26)               
    
    def test_hash(self):
        # 测试相同内容的hash值相同
        numel_list1 = NumelList([2, 3, 4])
        numel_list2 = NumelList([2, 3, 4])
        self.assertEqual(hash(numel_list1), hash(numel_list2))

        # 测试不同内容的hash值不同
        numel_list3 = NumelList([2, 3, 5])
        self.assertNotEqual(hash(numel_list1), hash(numel_list3))

        # 测试NumelList对象的hash值与整数的hash值不同
        self.assertNotEqual(hash(numel_list1), hash(24))

if __name__ == "__main__":
    run_tests()