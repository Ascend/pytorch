import os
from typing import Callable

from torch.utils._ordered_set import OrderedSet

from ...config import enable_fused_matmul_relu, log
from ..utils.fx_pass_level import FxPassLevel, PassType


ASCEND_CUSTOME_PASS_REGISTER = {
    pass_type: {level: [] for level in FxPassLevel} for pass_type in PassType
}

# 内置默认关闭列表：需由各自的 config 开关显式打开的 pass。关闭时该 pass 不注册，
# 因此推理与训练两条路径都不会执行它。
DEFAULT_SHUT_DOWN_PASSES = (
    () if enable_fused_matmul_relu else ("fused_matmul_relu_pass",)
)


def _get_shut_down_pass_set():
    """汇总当前生效的 pass 关闭集合：默认关闭列表 ∪ 环境变量 SHUT_DOWN_FX_PASS_LIST 指定的 pass。"""
    shut_down_str = os.environ.get("SHUT_DOWN_FX_PASS_LIST", "")
    env_list = [p.strip() for p in shut_down_str.split(",") if p.strip()]
    return OrderedSet(DEFAULT_SHUT_DOWN_PASSES + tuple(env_list))


def register_custom_pass(
    pass_type: int = PassType.PRE,
    fx_pass_level: int = FxPassLevel.LEVEL1,
    ignore_inference_check: bool = False,
):
    """注册自定义 fx pass。

    ignore_inference_check：该 pass 只做等价替换、训练图下同样成立，driver 不必用
    is_inference_check 把它挡掉。标记记在函数上而不是 driver 里的名字列表，pass
    重命名时不会静默失效。
    """

    def decorator(fn: Callable) -> Callable:
        fn.ignore_inference_check = ignore_inference_check
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
