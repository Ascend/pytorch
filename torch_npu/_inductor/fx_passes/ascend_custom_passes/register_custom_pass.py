import os
from typing import Callable
from ..utils.fx_pass_level import FxPassLevel, PassType
from ...config import log


ASCEND_CUSTOME_PASS_REGISTER = {
    pass_type: {level: [] for level in FxPassLevel}
    for pass_type in PassType
}


def register_custom_pass(pass_type: int = PassType.PRE, fx_pass_level: int = FxPassLevel.LEVEL1):
    def decorator(fn: Callable) -> Callable:
        # 默认 pass 不会出现重名的情况
        # 获取环境变量 SHUT_DOWN_FX_PASS_LIST，默认空字符串（表示未设置）
        shut_down_str = os.environ.get('SHUT_DOWN_FX_PASS_LIST', '')
        # 解析为列表，忽略空项
        shut_down_list = [p.strip() for p in shut_down_str.split(',') if p.strip()]
        # 如果函数名在关闭列表中，则跳过注册（关闭该 pass）
        if "all" in shut_down_list:
            log.debug(f"Ingnoring all registration in graph optimizer.")
            return fn
        elif fn.__name__ in shut_down_list:
            log.debug(f"Ingnoring registration of {fn.__name__}")
            return fn  # 返回原函数，不注册
        else:
            # 否则正常注册
            log.debug(f"Registering function {fn.__name__} from module {fn.__module__} with pass_type={pass_type}, fx_pass_level={fx_pass_level}")
            ASCEND_CUSTOME_PASS_REGISTER[pass_type][fx_pass_level].append(fn)
            return fn
    return decorator