from __future__ import annotations

import threading
from typing import Any, Dict, List

import torch


class GraphResourcePool:
    """Per-device graph resource registry with opaque integer keys.

    Each graph resource (``NPUGraphNode`` in tree mode, ``torch.npu.NPUGraph``
    in simple mode) is registered with an auto-incrementing integer key.  The
    key is a pure lookup handle — it carries no semantic information.

    Two kinds of resources are supported, corresponding to the two graph
    compilation modes:

    * tree mode — ``NPUGraphNode`` (has ``release()`` → block pointers)
    * simple mode — ``torch.npu.NPUGraph`` (has ``reset()``)

    **Key lifecycle**::

        # Lower layer (graph capture, inside src_fn):
        key = pool.register(node)              # → opaque int

        # Upper layer (shape_handling, after src_fn returns):
        keys = pool.consume_recent_keys()       # → List[int]

        # Background (gear eviction):
        pool.remove_by_keys([42, 43])           # idempotent

    Thread safety
    -------------
    All mutations to ``_entries`` and ``_pending_by_thread`` are serialised
    under ``self._lock``.  The heavy work — ``torch.npu.synchronize`` and
    ``resource.release() / reset()`` — is performed **outside** the lock.

    ``register`` and ``consume_recent_keys`` form a per-thread
    producer-consumer pair: ``register`` appends keys to the calling thread's
    pending list, ``consume_recent_keys`` drains it.  This guarantees that
    concurrent inferences on different threads never mix their keys.

    ``remove_by_keys`` is idempotent: a key that has already been removed
    (by a prior gear eviction) simply results in ``_entries.pop(key, None)``
    returning ``None``, which is skipped.
    """

    _pools: Dict[int, "GraphResourcePool"] = {}

    # ---- pool lifecycle -------------------------------------------------

    @classmethod
    def get_pool(cls, device_index: int) -> "GraphResourcePool":
        if device_index not in cls._pools:
            cls._pools[device_index] = cls(device_index)
        return cls._pools[device_index]

    @classmethod
    def reset_all(cls) -> None:
        cls._pools.clear()

    def __init__(self, device_index: int) -> None:
        self.device_index = device_index
        self._lock = threading.Lock()
        self._next_key = 0
        # key → graph resource (1:1 mapping)
        self._entries: Dict[int, Any] = {}
        # thread-id → keys registered since last consume (producer→consumer)
        self._pending_by_thread: Dict[int, List[int]] = {}
        # thread-id → activation refcount (0 ≈ not active)
        self._active_threads: Dict[int, int] = {}

    # ---- activation (per-thread, refcounted) ----------------------------

    def activate(self) -> None:
        """Allow :meth:`register` on the calling thread.

        Safe to call multiple times — each call must be balanced by a
        corresponding :meth:`deactivate`.
        """
        tid = threading.get_ident()
        with self._lock:
            self._active_threads[tid] = self._active_threads.get(tid, 0) + 1

    def deactivate(self) -> None:
        """Revoke one :meth:`activate` on the calling thread.

        When the refcount drops to zero, subsequent :meth:`register` calls
        on this thread become no-ops.
        """
        tid = threading.get_ident()
        with self._lock:
            v = self._active_threads.get(tid, 1) - 1
            if v <= 0:
                self._active_threads.pop(tid, None)
            else:
                self._active_threads[tid] = v

    def is_active(self) -> bool:
        """Return ``True`` if the calling thread has been activated."""
        tid = threading.get_ident()
        with self._lock:
            return tid in self._active_threads

    # ---- lower-layer API (graph capture) --------------------------------

    def register(self, resource: Any) -> int:
        """Register a graph resource and return an opaque integer key.

        Called during graph capture from ``record_function`` (tree mode) or
        ``npugraphify_impl`` (simple mode).  Callers should guard the call
        with :meth:`is_active` when registration is conditional on the
        adaptive-gear feature being enabled.
        """
        tid = threading.get_ident()
        with self._lock:
            key = self._next_key
            self._next_key += 1
            self._entries[key] = resource
            self._pending_by_thread.setdefault(tid, []).append(key)
        return key

    # ---- upper-layer API (after src_fn returns) -------------------------

    def consume_recent_keys(self) -> List[int]:
        """Return and clear all keys registered on this thread since the
        last call to ``consume_recent_keys``.

        Called in ``new_fn`` immediately after ``src_fn`` returns, so the
        returned keys correspond exactly to the graph resources captured
        during that ``src_fn`` invocation.
        """
        tid = threading.get_ident()
        with self._lock:
            return self._pending_by_thread.pop(tid, [])

    # ---- cleanup API (gear eviction) ------------------------------------

    def remove_by_keys(self, keys: List[int]) -> None:
        """Release graph resources identified by *keys*.

        Idempotent — keys that have already been removed are silently
        skipped.  Called by ``GearUpdateWorker`` on the background thread
        after gear eviction.
        """
        to_cleanup: List[Any] = []
        with self._lock:
            for key in keys:
                resource = self._entries.pop(key, None)
                if resource is not None:
                    to_cleanup.append(resource)
        for resource in to_cleanup:
            torch.npu.synchronize()
            if hasattr(resource, "release"):
                resource.release()
            else:
                resource.reset()

    # ---- debugging ------------------------------------------------------

    @property
    def entry_count(self) -> int:
        """Number of currently registered graph resources (for tests)."""
        with self._lock:
            return len(self._entries)
