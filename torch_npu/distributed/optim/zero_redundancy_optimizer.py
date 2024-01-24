import logging as logger

from torch.distributed.optim import (
    _FunctionalAdadelta,
    _FunctionalAdagrad,
    _FunctionalAdam,
    _FunctionalAdamax,
    _FunctionalAdamW,
    _FunctionalRMSprop,
    _FunctionalRprop,
    _FunctionalSGD
)

functional_optim_map = {
    "Adagrad": _FunctionalAdagrad,
    "Adam": _FunctionalAdam,
    "AdamW": _FunctionalAdamW,
    "SGD": _FunctionalSGD,
    "Adadelta": _FunctionalAdadelta,
    "RMSprop": _FunctionalRMSprop,
    "Rprop": _FunctionalRprop,
    "Adamax": _FunctionalAdamax,
}


def _get_optimizer_constructor(self, optimizer_class):
    """
    [patch] One of the functions exists in ZERO_REDUNDANCY_OPTIMIZER.
    """
    functional_optims = functional_optim_map.values()
    optimizer_class_name = optimizer_class.__name__
    if not self._overlap_with_ddp:
        if optimizer_class in functional_optims:
            raise ValueError(
                f"Passing in a functional optimizer {optimizer_class_name} "
                "when `overlap_with_ddp=False`"
            )
        else:
            return optimizer_class
    else:
        if optimizer_class in functional_optims:
            return optimizer_class
        elif optimizer_class_name in functional_optim_map:
            optim_constructor = functional_optim_map[optimizer_class_name]
            logger.info(
                "Using the functional optimizer %s "
                "instead of %s since "
                "`overlap_with_ddp=True`",
                optim_constructor, optimizer_class_name
            )
            return optim_constructor
        else:
            raise ValueError(
                "Using `ddp_with_overlap=True` requires using a "
                "functional optimizer, but there is no supported functional "
                f"optimizer equivalent for {optimizer_class_name}"
            )
