import sys
import logging
import re
import functools
import textwrap
import traceback
import inspect
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import torch
import torch.cuda._sanitizer as csan
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode
import torch_npu


logger = logging.getLogger(__name__)

# Note that this is only factories that take Tensor as input as they are
# the ones we care about.
FACTORY_FUNCTION_REGEX = re.compile("(new_.*|.*_like)")


@dataclass
class CrossStreamUsage:
    """Records when a tensor is used on a stream different from its allocation stream."""
    usage_stream: csan.StreamId
    seq_num: csan.SeqNum
    operator: str
    stack_trace: traceback.StackSummary


@dataclass
class NPUTensorInfo:
    """Tracks tensor allocation and cross-stream usage for record_stream detection.

    Maintained independently from the parent EventHandler's TensorInfo, which tracks
    read/write accesses for data race detection.
    """
    allocation_stream: Optional[csan.StreamId] = None
    allocation_stack_trace: Optional[traceback.StackSummary] = None
    recorded_streams: Set[csan.StreamId] = field(default_factory=set)
    cross_stream_usages: Dict[csan.StreamId, CrossStreamUsage] = field(default_factory=dict)


class MissingRecordStreamError(csan.SynchronizationError):
    """Tensor was used across streams without record_stream or proper sync.

    Detected at tensor deallocation or via flush_record_stream_warnings().
    Per PyTorch docs, record_stream is NOT required if creation_stream has been
    synchronized to wait for the usage_stream before deallocation:
      - creation_stream.wait_stream(usage_stream)
      - creation_stream.wait_event(event_on_usage_stream)
      - torch.npu.synchronize() (device-level sync covers all directions)
    """

    def __init__(
        self,
        data_ptr: csan.DataPtr,
        allocation_stack_trace: Optional[traceback.StackSummary],
        allocation_stream: csan.StreamId,
        usage_stream: csan.StreamId,
        usage: CrossStreamUsage,
        recorded_streams: set[csan.StreamId],
    ):
        self.data_ptr = data_ptr
        self.allocation_stack_trace = allocation_stack_trace
        self.allocation_stream = allocation_stream
        self.usage_stream = usage_stream
        self.usage = usage
        self.recorded_streams = recorded_streams

    def __repr__(self):
        return (
            f"MissingRecordStreamError(data_ptr={self.data_ptr}, "
            f"alloc_stream={self.allocation_stream}, "
            f"usage_stream={self.usage_stream}, "
            f"operator='{self.usage.operator}')"
        )

    def __str__(self):
        result = textwrap.dedent(
            f"""\
            ============================
            NPUSanitizer: missing record_stream detected!
            Tensor (data ptr: {self.data_ptr}) allocated on stream {self.allocation_stream}
            was used on stream {self.usage_stream} without record_stream or
            creation_stream.wait_stream(usage_stream).

            This may cause use-after-free if the caching allocator reuses memory
            on the allocation stream before the usage stream finishes.

            Fix with ONE of:
              A) tensor.record_stream(stream) — tell allocator about the usage
              B) creation_stream.wait_stream(usage_stream) before deallocation

            Cross-stream usage during kernel:
              {self.usage.operator}
            """
        )
        result += f"With stack trace:\n{''.join(self.usage.stack_trace.format())}\n"
        if self.recorded_streams:
            result += f"Streams recorded via record_stream: {self.recorded_streams}\n"
        else:
            result += "No streams were recorded via record_stream.\n"
        if self.allocation_stack_trace:
            result += (
                "Tensor was allocated with stack trace:\n"
                f"{''.join(self.allocation_stack_trace.format())}"
            )
        return result


class NPURecordStreamHandler(csan.EventHandler):
    """EventHandler with deferred record_stream detection.

    Record_stream checks are deferred to deallocation time (or manual flush via
    flush_record_stream_warnings), because what matters for memory safety is whether
    creation_stream has synced with usage_stream BEFORE the tensor's memory is
    reused — not at the time of the cross-stream kernel launch.

    This avoids false positives when the user plans to sync after the kernel launch
    but before tensor deallocation, which is the typical usage pattern.
    """

    def __init__(self) -> None:
        super().__init__()
        self._npu_tensors: dict[csan.DataPtr, NPUTensorInfo] = {}
        self.record_stream_errors: list[MissingRecordStreamError] = []

    def _handle_memory_allocation(self, data_ptr: csan.DataPtr) -> None:
        super()._handle_memory_allocation(data_ptr)
        alloc_trace = None
        try:
            alloc_trace = self.tensors_accessed.get_allocation_stack_trace(data_ptr)
        except KeyError:
            pass
        current_stream: Optional[csan.StreamId] = None
        try:
            current_stream = int(torch_npu.npu.current_stream().npu_stream)
        except RuntimeError:
            pass
        self._npu_tensors[data_ptr] = NPUTensorInfo(
            allocation_stream=current_stream,
            allocation_stack_trace=alloc_trace,
        )

    def _handle_memory_deallocation(self, data_ptr: csan.DataPtr) -> None:
        if data_ptr in self._npu_tensors:
            for error in self._get_record_stream_errors(data_ptr):
                print(error, file=sys.stderr)
                self.record_stream_errors.append(error)
            del self._npu_tensors[data_ptr]
        super()._handle_memory_deallocation(data_ptr)

    def _handle_kernel_launch(
        self,
        stream: csan.StreamId,
        read_only: set[csan.DataPtr],
        read_write: set[csan.DataPtr],
        outputs: set[csan.DataPtr],
        operator: str,
        tensor_aliases: dict[int, list[str]],
        storage_dataptrs_accessed: Optional[Set[csan.DataPtr]] = None
    ) -> List[csan.SynchronizationError]:
        errors = super()._handle_kernel_launch(
            stream, read_only, read_write, outputs, operator, tensor_aliases
        )
        # Use storage-level data_ptrs (matches the allocation callback and
        # the C++ recordStream trace), falling back to tensor data_ptrs
        # only when not provided by the dispatch mode.
        accessed = (
            storage_dataptrs_accessed
            if storage_dataptrs_accessed is not None
            else (read_only | read_write)
        )
        self._record_cross_stream_usage(stream, accessed, operator)
        return errors

    def _record_cross_stream_usage(
        self, stream: csan.StreamId, all_accessed: set[csan.DataPtr], operator: str
    ) -> None:
        """Record that tensors are being accessed on a non-allocation stream.

        Stack trace is captured lazily (only once per kernel launch) to avoid
        redundant walks when multiple tensors are accessed in the same kernel.
        Records the current seq_num so that sync checks can verify the sync
        happened AFTER this usage, not just from stream creation inheritance.
        """
        stack_trace = None
        current_seq = self.seq_num
        for data_ptr in all_accessed:
            info = self._npu_tensors.get(data_ptr)
            if info is None or info.allocation_stream is None:
                continue
            if info.allocation_stream == stream:
                continue
            existing = info.cross_stream_usages.get(stream)
            if existing is not None:
                existing.seq_num = current_seq
                existing.operator = operator
                if stack_trace is None:
                    stack_trace = traceback.StackSummary.extract(
                        traceback.walk_stack(inspect.currentframe()),
                        lookup_lines=False,
                    )
                    stack_trace.reverse()
                existing.stack_trace = stack_trace
                continue
            if stack_trace is None:
                stack_trace = traceback.StackSummary.extract(
                    traceback.walk_stack(inspect.currentframe()),
                    lookup_lines=False,
                )
                stack_trace.reverse()
            info.cross_stream_usages[stream] = CrossStreamUsage(
                usage_stream=stream,
                seq_num=current_seq,
                operator=operator,
                stack_trace=stack_trace,
            )

    def _get_record_stream_errors(self, data_ptr: csan.DataPtr) -> List[MissingRecordStreamError]:
        info = self._npu_tensors.get(data_ptr)
        if info is None or info.allocation_stream is None:
            return []
        errors = []
        for usage_stream, usage in info.cross_stream_usages.items():
            if usage_stream in info.recorded_streams:
                continue
            if self._is_creation_stream_synced_to_usage(
                info.allocation_stream, usage_stream, usage.seq_num
            ):
                continue
            errors.append(MissingRecordStreamError(
                data_ptr=data_ptr,
                allocation_stack_trace=info.allocation_stack_trace,
                allocation_stream=info.allocation_stream,
                usage_stream=usage_stream,
                usage=usage,
                recorded_streams=info.recorded_streams.copy(),
            ))
        return errors

    def flush_record_stream_warnings(self) -> List[MissingRecordStreamError]:
        """Check all tracked tensors for missing record_stream and print errors.

        Call after all stream operations are complete (including any synchronization)
        to detect tensors used cross-stream without record_stream or
        creation-to-usage synchronization.

        Errors are printed to stderr and appended to self.record_stream_errors.
        """
        errors = []
        for data_ptr in list(self._npu_tensors):
            new_errors = self._get_record_stream_errors(data_ptr)
            for error in new_errors:
                print(error, file=sys.stderr)
            errors.extend(new_errors)
        self.record_stream_errors.extend(errors)
        return errors

    def _is_creation_stream_synced_to_usage(
        self,
        creation_stream: csan.StreamId,
        usage_stream: csan.StreamId,
        usage_seq_num: csan.SeqNum = 0,
    ) -> bool:
        """Check if creation stream has synced with usage stream's operations.

        Compares against usage_seq_num to distinguish real synchronization from
        stream creation inheritance. Stream creation sets initial sync state to 0,
        but actual kernel seq_nums start at 1, so comparing >= usage_seq_num
        ensures we detect real sync operations rather than inherited initial state.
        """
        try:
            creation_state = self.syncs.current_sync_states.get(creation_stream, {})
            return creation_state.get(usage_stream, -1) >= usage_seq_num
        except (AttributeError, KeyError):
            return False

    def _handle_record_stream(self, data_ptr: csan.DataPtr, stream: csan.StreamId) -> None:
        """Track a record_stream call for memory safety checking."""
        if data_ptr not in self._npu_tensors:
            self._npu_tensors[data_ptr] = NPUTensorInfo()
        self._npu_tensors[data_ptr].recorded_streams.add(stream)

    def _handle_erase_stream(
        self, data_ptr: csan.DataPtr, stream: csan.StreamId
    ) -> None:
        """Track eraseStream after a communication work no longer owns a stream."""
        info = self._npu_tensors.get(data_ptr)
        if info is None:
            return
        info.recorded_streams.discard(stream)


class NPUArgumentHandler:
    def __init__(self):
        self.dataptrs_read: set[csan.DataPtr] = set()
        self.dataptrs_written: set[csan.DataPtr] = set()
        self.tensor_aliases: dict[int, list[str]] = {}
        self.outputs: set[csan.DataPtr] = set()
        self.storage_dataptrs_accessed: set[csan.DataPtr] = set()

    def _handle_argument(
        self,
        value,
        is_write: bool,
        metadata_only: bool,
        name: Optional[str] = None,
        is_output: bool = False,
    ) -> None:
        if not isinstance(value, torch.Tensor) or not value.is_npu:
            return

        # View / metadata_only tensor arguments do not represent real data access.
        # Do not record read/write/storage access for them.
        if metadata_only:
            return

        data_ptr = value.data_ptr() if value.data_ptr() else id(value)
        if is_write:
            self.dataptrs_written.add(data_ptr)
        else:
            self.dataptrs_read.add(data_ptr)

        self.tensor_aliases.setdefault(data_ptr, [])
        if name is not None:
            self.tensor_aliases[data_ptr].append(name)
        if is_output:
            self.outputs.add(data_ptr)

        # Also collect the storage start for record_stream tracking.
        try:
            storage = value.untyped_storage()
            if storage is not None:
                storage_ptr = storage.data_ptr()
                if storage_ptr:
                    self.storage_dataptrs_accessed.add(storage_ptr)
        except (RuntimeError, AttributeError):
            pass

    def parse_inputs(self, schema, args, kwargs, *, is_factory: bool = False) -> None:
        from torch.cuda._sanitizer import zip_arguments
        for argument, value in zip_arguments(schema, args, kwargs):
            is_write = argument.alias_info is not None and argument.alias_info.is_write
            metadata_only = is_factory or (
                argument.alias_info is not None and not argument.alias_info.is_write
            )
            pytree.tree_map_(
                functools.partial(
                    self._handle_argument,
                    is_write=is_write,
                    name=argument.name,
                    metadata_only=metadata_only,
                ),
                value,
            )

    def parse_outputs(self, schema, outputs, *, is_factory: bool = False) -> None:
        from torch.cuda._sanitizer import zip_arguments
        for res, value in zip(schema.returns, (outputs,)):
            metadata_only = res.alias_info is not None and not res.alias_info.is_write
            pytree.tree_map_(
                functools.partial(
                    self._handle_argument,
                    is_write=True,
                    metadata_only=metadata_only,
                    is_output=True,
                ),
                value,
            )


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

        func_name = func.__name__ if hasattr(func, '__name__') else str(func)
        if "record_stream" in func_name:
            return self._handle_record_stream_op(func, args, kwargs)

        is_factory = bool(FACTORY_FUNCTION_REGEX.match(func._schema.name))

        self.args_handler = NPUArgumentHandler()
        aten_api = func.__name__.split(".")[0]
        self.enable_autograd(aten_api)
        self.parse_inputs(func._schema, args, kwargs, is_factory=is_factory)
        # execute operator
        outputs = func(*args, **kwargs)

        self.parse_outputs(func._schema, outputs, is_factory=is_factory)
        if (
            not self.args_handler.dataptrs_read
            and not self.args_handler.dataptrs_written
            and not self.args_handler.outputs
            and not self.args_handler.storage_dataptrs_accessed
        ):
            return outputs

        npu_stream = 0
        try:
            npu_stream = int(torch_npu.npu.current_stream().npu_stream)
        except RuntimeError as err:
            logger.info(
                "Failed to get current stream, ignore this kernel launch record. error info is: %s",
                err
            )
            return outputs
        self.check_errors(func, npu_stream)

        return outputs

    def _handle_record_stream_op(self, func, args, kwargs):
        """Short-circuit record_stream so it isn't treated as a regular kernel launch.

        Tracking is done in C++ via the NPURecordStreamCallbacks trace, which is fired
        from NpuCachingAllocator::recordStream / NPUPluggableAllocator::recordStream —
        the chokepoint that all entry points (aten op, NPUGuardImpl, HCCL/LCCL, RPC,
        pluggable allocator) funnel through. Doing tracking here would only cover the
        aten-op path and would use tensor.data_ptr() (with view offset), which would
        not match the storage data_ptr that the allocation callback uses.
        """
        return func(*args, **kwargs)

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
            str(func._schema),
            self.args_handler.tensor_aliases,
            storage_dataptrs_accessed=self.args_handler.storage_dataptrs_accessed,
        )
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            raise csan.CUDASanitizerErrors(errors)
