import sys
import logging

import torch
import torch.cuda._sanitizer as csan
from torch.utils._python_dispatch import TorchDispatchMode
import torch_npu


logger = logging.getLogger(__name__)


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

        self.args_handler = csan.ArgumentHandler()
        aten_api = func.__name__.split(".")[0]
        self.enable_autograd(aten_api)
        self.parse_inputs(func._schema, args, kwargs)
        # execute operator
        outputs = func(*args, **kwargs)

        self.parse_outputs(outputs)

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

    def parse_inputs(self, schema, args, kwargs):
        self.args_handler.parse_inputs(schema, args, kwargs)

    def parse_outputs(self, outputs):
        self.args_handler.parse_outputs(outputs)

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
