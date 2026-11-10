# Owner(s): ["module: dynamo"]

import torch
from torch.testing._internal.common_utils import TestCase, run_tests


_SEED_RULE_PAIRS = (
    ("torch.npu.manual_seed", "torch.cuda.manual_seed"),
    ("torch.npu.manual_seed_all", "torch.cuda.manual_seed_all"),
    ("torch.npu.random.manual_seed", "torch.cuda.random.manual_seed"),
    ("torch.npu.random.manual_seed_all", "torch.cuda.random.manual_seed_all"),
    ("torch.npu.random.seed", "torch.cuda.random.seed"),
    ("torch.npu.random.seed_all", "torch.cuda.random.seed_all"),
)


def _rule_class(qualified_name):
    found = [
        rule_map[qualified_name]
        for rule_map in torch._dynamo.trace_rules.torch_name_rule_map
        if qualified_name in rule_map
    ]
    if len(found) != 1:
        raise AssertionError(f"{qualified_name} is registered {len(found)} times: {found}")
    return found[0]


class TestNpuSeedTraceRules(TestCase):
    def test_seed_rule_matches_cuda_after_transfer_to_npu(self):
        import torch_npu  # noqa: F401
        import torch._dynamo  # noqa: F401
        from torch_npu.contrib import transfer_to_npu  # noqa: F401

        trace_rules = torch._dynamo.trace_rules
        trace_rules.get_torch_obj_rule_map.cache_clear()
        trace_rules.get_torch_obj_rule_map()

        for npu_name, cuda_name in _SEED_RULE_PAIRS:
            self.assertIs(_rule_class(npu_name), _rule_class(cuda_name))


if __name__ == "__main__":
    run_tests()
