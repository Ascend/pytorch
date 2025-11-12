import sys
import logging
import re

import torch
import torch.cuda._sanitizer as csan
from torch.utils._python_dispatch import TorchDispatchMode
import torch_npu


logger = logging.getLogger(__name__)

# Note that this is only factories that take Tensor as input as they are
# the ones we care about.
FACTORY_FUNCTION_REGEX = re.compile("(new_.*|.*_like)")


class NPUSanitizerDispatchMode(TorchDispatchMode):

    def __init__(self, event_handler: csan.EventHandler):
        super().__init__()
        self.event_handler = event_handler
        self.args_handler = None
        self.npu_adjust_autograd = [
            "adaptive_avg_pool2d", "batch_norm",
            "log_softmax", "nll_loss", "to"
        ]

    def enable_autograd(self, aten_api):
        if aten_api in self.npu_adjust_autograd:
            torch._C._dispatch_tls_set_dispatch_key_excluded(torch._C.DispatchKey.AutogradFunctionality, False)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        is_factory = bool(FACTORY_FUNCTION_REGEX.match(func._schema.name))

        self.args_handler = csan.ArgumentHandler()
        aten_api = func.__name__.split(".")[0]
        self.enable_autograd(aten_api)
        self.parse_inputs(func._schema, args, kwargs, is_factory=is_factory)
        # execute operator
        outputs = func(*args, **kwargs)

        self.parse_outputs(func._schema, outputs, is_factory=is_factory)

        npu_stream = 0
        try:
            npu_stream = torch_npu.npu.current_stream().npu_stream
        except RuntimeError as err:
            logger.info(
                "Failed to get current stream, ignore this kernel launch record. error info is: %s",
                err
            )
            return outputs
        self.check_errors(func, npu_stream)

        return outputs

    def parse_inputs(self, schema, args, kwargs, is_factory=False):
        self.args_handler.parse_inputs(schema, args, kwargs, is_factory=is_factory)

    def parse_outputs(self, schema, outputs, is_factory=False):
        self.args_handler.parse_outputs(schema, outputs, is_factory=is_factory)

    def check_errors(self, func, npu_stream):
        errors = self.event_handler._handle_kernel_launch(
            npu_stream,
            self.args_handler.dataptrs_read - self.args_handler.dataptrs_written,
            self.args_handler.dataptrs_written,
            self.args_handler.outputs,
            func._schema,
            self.args_handler.tensor_aliases
        )
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            raise csan.CUDASanitizerErrors(errors)


def apply_sanitizer_patch():
    torch.Tensor.is_cuda = torch.Tensor.is_npu
