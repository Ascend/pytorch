from unittest import mock

from torch_npu.profiler._flops_registry import (
    _default_npu_flop_registry,
    _npu_flop_registry,
    get_flop_func,
    get_npu_flop_targets,
    register_npu_flop,
)
from torch_npu.testing.testcase import run_tests, TestCase


class TestFlopsRegistry(TestCase):
    def tearDown(self):
        for op_name in (
            "test_default_override",
            "test_external_conflict",
            "test_external_formula_only",
        ):
            _default_npu_flop_registry.pop(op_name, None)
            _npu_flop_registry.pop(op_name, None)

    def test_external_registration_overrides_default_registration(self):
        @register_npu_flop(target="torch:mm", op_name="test_default_override")
        def external_flops():
            return 2

        @register_npu_flop(
            target="torch:bmm", op_name="test_default_override", is_default=True
        )
        def default_flops():
            return 1

        self.assertIs(external_flops, get_flop_func("test_default_override"))
        self.assertEqual("torch:mm", get_npu_flop_targets()["test_default_override"])

    def test_duplicate_external_registration_logs_error_and_uses_later_one(self):
        @register_npu_flop(op_name="test_external_conflict")
        def first_flops():
            return 1

        with mock.patch(
            "torch_npu.profiler._flops_registry.logger.error"
        ) as mock_error:

            @register_npu_flop(op_name="test_external_conflict")
            def second_flops():
                return 2

        mock_error.assert_called_once()
        self.assertIs(second_flops, get_flop_func("test_external_conflict"))

    def test_external_formula_uses_default_target_when_omitted(self):
        @register_npu_flop(
            target="torch:mm", op_name="test_external_formula_only", is_default=True
        )
        def default_flops():
            return 1

        @register_npu_flop(op_name="test_external_formula_only")
        def external_flops():
            return 2

        self.assertIs(external_flops, get_flop_func("test_external_formula_only"))
        self.assertEqual("torch:mm", get_npu_flop_targets()["test_external_formula_only"])


if __name__ == "__main__":
    run_tests()
