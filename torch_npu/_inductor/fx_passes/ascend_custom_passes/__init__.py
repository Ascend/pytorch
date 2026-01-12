import pkgutil
import importlib
from ..utils.check_mode import is_inference_check
from .register_custom_pass import ASCEND_CUSTOME_PASS_REGISTER
from ..utils.fx_pass_level import FxPassLevel, PassType
from ...config import log


for _, module_name, _ in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{module_name}")


def run_register_pre_custom_passes(gm):
    log.debug(f"before pre_grad graph optimizer pass, graph is: {gm}")
    if is_inference_check():
        for level in sorted(FxPassLevel):
            for fn in ASCEND_CUSTOME_PASS_REGISTER[PassType.PRE][level]:
                fn(gm)

        log.debug(f"after pre_grad graph optimizer pass, graph is: {gm}")
        

def run_register_post_custom_passes(gm):
    log.debug(f"before post_grad graph optimizer pass, graph is: {gm}")
    if is_inference_check():
        for level in sorted(FxPassLevel):
            for fn in ASCEND_CUSTOME_PASS_REGISTER[PassType.POST][level]:
                fn(gm)

        log.debug(f"after post_grad graph optimizer pass, graph is: {gm}")