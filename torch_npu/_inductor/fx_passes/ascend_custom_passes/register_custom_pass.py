from typing import Callable
from ..utils.fx_pass_level import FxPassLevel, PassType
from ...config import log


ASCEND_CUSTOME_PASS_REGISTER = {
    pass_type: {level: [] for level in FxPassLevel}
    for pass_type in PassType
}


def register_custom_pass(pass_type: int = PassType.PRE, fx_pass_level: int = FxPassLevel.LEVEL1):
    def decorator(fn: Callable) -> Callable:

        log.debug(f"Registering function {fn.__name__} from module {fn.__module__} with pass_type={pass_type}, fx_pass_level={fx_pass_level}")
        ASCEND_CUSTOME_PASS_REGISTER[pass_type][fx_pass_level].append(fn)
        return fn
    return decorator