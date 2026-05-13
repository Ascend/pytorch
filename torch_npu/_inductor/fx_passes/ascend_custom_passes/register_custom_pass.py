import os
from typing import Callable

from torch.utils._ordered_set import OrderedSet

from ...config import log
from ..utils.fx_pass_level import FxPassLevel, PassType


ASCEND_CUSTOME_PASS_REGISTER = {
    pass_type: {level: [] for level in FxPassLevel} for pass_type in PassType
}


# 默认关闭的 pass 列表：新增/调试中的 pass 暂时默认关闭，
# 待功能/精度验证完成后，将对应名字从此列表中移除即可启用。
# 用户可通过环境变量 SHUT_DOWN_FX_PASS_LIST 追加更多需要关闭的 pass。
_DEFAULT_SHUT_DOWN_FX_PASSES = (
    "cat_to_view_pass",
    "repeat_to_expand_pass",
    "fold_iota_arithmetic_pass",
    "broadcast_const_mask_compress",
    "masked_add_compose_pass",
    "bool_cast_mul_to_where_pass",
    "sign_diff_hamming_fuse_pass",
    "batch_embedding_fusion_pass",
)


def _get_shut_down_pass_set():
    """汇总当前生效的 pass 关闭集合：默认关闭列表 ∪ 环境变量 SHUT_DOWN_FX_PASS_LIST 指定的 pass。"""
    shut_down_str = os.environ.get("SHUT_DOWN_FX_PASS_LIST", "")
    env_list = [p.strip() for p in shut_down_str.split(",") if p.strip()]
    return OrderedSet(_DEFAULT_SHUT_DOWN_FX_PASSES) | OrderedSet(env_list)


def register_custom_pass(
    pass_type: int = PassType.PRE, fx_pass_level: int = FxPassLevel.LEVEL1
):
    def decorator(fn: Callable) -> Callable:
        # 默认 pass 不会出现重名的情况
        # 合并「内置默认关闭列表」与环境变量 SHUT_DOWN_FX_PASS_LIST
        shut_down_set = _get_shut_down_pass_set()
        # 如果函数名在关闭列表中，则跳过注册（关闭该 pass）
        if "all" in shut_down_set:
            log.debug("Ignoring all registration in graph optimizer.")
            return fn
        elif fn.__name__ in shut_down_set:
            log.debug("Ignoring registration of %s", fn.__name__)
            return fn  # 返回原函数，不注册
        else:
            # 否则正常注册
            log.debug(
                "Registering function %s from module %s with pass_type=%s, fx_pass_level=%s",
                fn.__name__,
                fn.__module__,
                pass_type,
                fx_pass_level,
            )
            ASCEND_CUSTOME_PASS_REGISTER[pass_type][fx_pass_level].append(fn)
            return fn

    return decorator
