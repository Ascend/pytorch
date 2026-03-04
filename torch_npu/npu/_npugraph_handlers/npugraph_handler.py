"""NPU Graph Operator Handler -- base class, global registry, and utilities.

This module contains the core building blocks of the NPU Graph operator
handler framework:

- :class:`NpuGraphOpHandler`          -- abstract base class for handlers.
- :data:`_NPU_GRAPH_OP_HANDLERS`      -- global registry (dict).
- :func:`register_npu_graph_handler`  -- class-decorator for registration.

Design Contracts
----------------
1. **Stateless Handler** -- All hook methods are ``@classmethod``; the
   registry stores *class objects*, never instances.
2. **Function Replacement Consistency** -- When ``prepare_capture``
   substitutes ``func`` with ``actual_func``, the handler registered for
   ``actual_func.__name__`` **must** implement a compatible
   ``update_args`` (typically guaranteed by a shared intermediate base).

See Also
--------
``torch_npu/npu/graphs.py`` for the template-method skeleton that
consumes this registry.
"""

import logging
from copy import deepcopy

import torch
from torch_npu._C import _weak_ref_tensor as TensorWeakRef

logger = logging.getLogger(__name__)


class NpuGraphOpHandler:
    r"""Base class for NPU Graph operator handlers.

    Subclasses override ``@classmethod`` hooks to customize capture / update
    behavior for specific operators, while the framework keeps
    stream / event / task-group orchestration in a common template.

    **Stateless by design** -- All hook methods are ``@classmethod`` (first
    parameter is ``cls``, not ``self``).  There is no instance; the global
    registry stores **class objects** directly.  This structurally prevents
    storing mutable per-invocation state.  Class-level constants (e.g.
    ``_OP_ARG_SPECS``) are accessible via ``cls``.

    Users should inherit this class and override the needed hooks:

    .. code-block:: python

        @register_npu_graph_handler(["my_op", "my_op.default"])
        class MyHandler(NpuGraphOpHandler):
            @classmethod
            def update_args(cls, record, update_input):
                if "batch" in update_input and len(record.args) >= 3:
                    record.args[2] = update_input["batch"]
    """

    @classmethod
    def prepare_capture(cls, func, args, kwargs):
        r"""Prepare operator call before graph-task recording.

        This hook runs **before** ``graph_task_group_begin`` and can be used
        for operator-specific preprocessing such as workspace allocation,
        output pre-allocation, or switching from ``.default`` to ``.out``
        overloads.

        .. note:: **Function Replacement Contract**

            If ``actual_func`` differs from ``func``, ensure the handler
            registered for ``actual_func.__name__`` implements a compatible
            ``update_args``.  The recommended approach is to share a common
            base class that defines ``update_args`` (see ``_IFAv1Base`` /
            ``_IFAv2Base``).

        Args:
            func (OpOverload): Original operator callable.
            args (tuple): Original arguments.
            kwargs (dict): Original keyword arguments.

        Returns:
            tuple[Callable, tuple, dict]: ``(actual_func, args, kwargs)`` to
            execute during recording.
        """
        return func, args, kwargs

    @classmethod
    def postprocess_result(cls, result, kwargs):
        r"""Post-process operator return value after recording.

        Called after ``graph_task_group_end`` and dispatch-record creation.

        Args:
            result: Raw return value from ``actual_func(*args, **kwargs)``.
            kwargs (dict): Current keyword arguments (may contain ``"out"``).

        Returns:
            Final value returned to the Python caller.
        """
        return result

    @classmethod
    def update_args(cls, dispatch_record, update_input):
        r"""Apply operator-specific indexed-arg updates.

        Framework-level kwargs updates are handled by the dispatch skeleton.
        Override this hook only when update values must be applied to
        arguments by index.

        Args:
            dispatch_record (_GraphDispatchRecord): Recorded operator call.
                Args can be modified via ``dispatch_record.args[i]``.
            update_input (dict): User-provided update payload.
        """
        pass

    @classmethod
    def record_wrap_kwarg(cls, key, value, tensor_param_names):
        r"""Convert a kwarg value into record-time storage representation.

        Called only during the **capture** phase (creating the dispatch
        record).  The purpose of ``TensorWeakRef`` conversion is to avoid the
        Python-side record holding strong references to NPU tensors, letting
        the C++ graph runtime manage tensor memory lifetimes.

        .. note::

            The **update** phase uses direct assignment for kwargs
            (``record.kwargs[key] = update_input[key]``), consistent with the
            original implementation and with how ``update_args`` handles
            arguments.  Update is a short-lived "assign -> replay"
            flow where weak-ref conversion is unnecessary.

        Logic (consistent with original, with list/tuple generalisation):

        - ``None`` -> ``None`` (fast path)
        - ``list`` / ``tuple`` -> element-wise: NPU Tensor -> ``TensorWeakRef``,
          else -> ``deepcopy`` (replaces old hardcoded ``if key == "out"``
          that assumed exactly 2 Tensors).  Only NPU tensors are wrapped;
          CPU tensors use ``deepcopy`` because ``TensorWeakRef`` from
          torch_npu._C is only valid for NPU tensors.
        - Single value where ``key`` in ``tensor_param_names`` and value is an
          NPU Tensor -> ``TensorWeakRef``
        - Everything else -> ``deepcopy``

        Args:
            key (str): Kwarg name.
            value: Kwarg value.
            tensor_param_names (list[str]): Tensor-typed kwarg names parsed
                from operator schema.

        Returns:
            Stored value for the dispatch record.
        """
        if value is None:
            return None

        def _is_npu_tensor(t):
            return torch.is_tensor(t) and "npu" in str(t.device)

        if isinstance(value, (list, tuple)):
            wrapped = [
                TensorWeakRef(t) if _is_npu_tensor(t) else deepcopy(t)
                for t in value
            ]
            return type(value)(wrapped)

        if key in tensor_param_names and _is_npu_tensor(value):
            return TensorWeakRef(value)

        return deepcopy(value)


# ---------------------------------------------------------------------------
#  Global Registry
# ---------------------------------------------------------------------------

_NPU_GRAPH_OP_HANDLERS = {}


def register_npu_graph_handler(op_names):
    r"""Register an operator handler via class decorator.

    The decorated class itself (not an instance) is stored in the global
    registry.  All hook methods must be ``@classmethod``.

    Args:
        op_names (str or list[str]): Operator names resolved from
            ``func.__name__`` in ``__torch_dispatch__`` (for example,
            ``"my_op"``, ``"my_op.default"``, ``"my_op.out"``).

    Returns:
        A class decorator.

    Example::

        @register_npu_graph_handler(["my_op", "my_op.default"])
        class MyHandler(NpuGraphOpHandler):
            ...
    """
    def decorator(cls):
        names = op_names if isinstance(op_names, (list, tuple)) else [op_names]
        for name in names:
            if name in _NPU_GRAPH_OP_HANDLERS:
                existing = _NPU_GRAPH_OP_HANDLERS[name].__name__
                logger.warning(
                    f"NpuGraphOpHandler for '{name}' is being overridden: "
                    f"{existing} -> {cls.__name__}"
                )
            _NPU_GRAPH_OP_HANDLERS[name] = cls   # store class, not instance
        return cls
    return decorator
