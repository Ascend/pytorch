from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.utils._error_code import _except_handler


class ErrorCode(TestCase):

    def test_judge_is_npu_exception(self):
        error_msg = "[ERROR] 2023-08-10-15:23:38 (PID:1234, Device:0, RankID:0) ERR107020"
        _except_handler.exception = error_msg
        try:
            result = _except_handler._is_exception(_except_handler.npu_exception)
        finally:
            _except_handler.exception = None
        self.assertTrue(result)


if __name__ == '__main__':
    run_tests()
