import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu
import torch_npu._inductor
from torch_npu._inductor.fx_passes.utils.fx_pass_level import FxPassLevel
from torch_npu._inductor.fx_passes.utils.fx_pass_level import PassType


class TestFxPassLevel(TestUtils):

    def test_comparison_between_members(self):
        """枚举成员之间的大小比较"""
        assert FxPassLevel.LEVEL1 < FxPassLevel.LEVEL2
        assert FxPassLevel.LEVEL1 <= FxPassLevel.LEVEL2
        assert FxPassLevel.LEVEL2 > FxPassLevel.LEVEL1
        assert FxPassLevel.LEVEL2 >= FxPassLevel.LEVEL1
        assert FxPassLevel.LEVEL2 == FxPassLevel.LEVEL2
        assert FxPassLevel.LEVEL1 != FxPassLevel.LEVEL3

        # 严格顺序
        assert FxPassLevel.LEVEL1 < FxPassLevel.LEVEL2 < FxPassLevel.LEVEL3


    def test_sorting_and_min_max(self):
        """支持排序、min、max 等操作"""
        levels = [FxPassLevel.LEVEL3, FxPassLevel.LEVEL1, FxPassLevel.LEVEL2]
        sorted_levels = sorted(levels)
        assert sorted_levels == [FxPassLevel.LEVEL1, FxPassLevel.LEVEL2, FxPassLevel.LEVEL3]

        assert min(levels) == FxPassLevel.LEVEL1
        assert max(levels) == FxPassLevel.LEVEL3


    def test_hash_consistency(self):
        """哈希值一致性"""
        assert hash(FxPassLevel.LEVEL1) == hash(1)
        assert hash(FxPassLevel.LEVEL2) == hash(2)
        assert hash(FxPassLevel.LEVEL3) == hash(3)

        # 相同值哈希相同
        assert hash(FxPassLevel.LEVEL1) == hash(FxPassLevel.LEVEL1)

        # 不同值哈希不同
        assert hash(FxPassLevel.LEVEL1) != hash(FxPassLevel.LEVEL2)

        # 可用于集合
        s = {FxPassLevel.LEVEL1, FxPassLevel.LEVEL2}
        assert FxPassLevel.LEVEL1 in s
        assert FxPassLevel.LEVEL3 not in s


class TestPassType(TestUtils):

    def test_comparison_between_members(self):
        """枚举成员之间的大小比较"""
        assert PassType.PRE < PassType.POST
        assert PassType.PRE <= PassType.POST
        assert PassType.POST > PassType.PRE
        assert PassType.POST >= PassType.PRE
        assert PassType.PRE == PassType.PRE
        assert PassType.PRE != PassType.POST


    def test_sorting_and_min_max(self):
        """支持排序、min、max"""
        types = [PassType.POST, PassType.PRE]
        sorted_types = sorted(types)
        assert sorted_types == [PassType.PRE, PassType.POST]

        assert min(types) == PassType.PRE
        assert max(types) == PassType.POST


    def test_hash_consistency(self):
        """哈希值一致性"""
        assert hash(PassType.PRE) == hash(1)
        assert hash(PassType.POST) == hash(2)

        s = {PassType.PRE, PassType.POST}
        assert PassType.PRE in s
        assert PassType.PRE in s  # 重复添加不影响


instantiate_parametrized_tests(TestFxPassLevel)
instantiate_parametrized_tests(TestPassType)


if __name__ == "__main__":
    run_tests()