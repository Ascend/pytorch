import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestAPIDocs(TestCase):

    def test_ops_doc(self):
        no_doc_api = [func for func in torch_npu.__all__ if getattr(torch_npu, func).__doc__ is None]
        self.assertFalse(no_doc_api, f"Some APIs lack the '__doc__' attribute, The list of APIs is {str(no_doc_api)}")


if __name__ == "__main__":
    run_tests()
