"""Thread-affine execution domains for isolated NPU graph trees.

One domain owns one native ``TreeManagerContainer`` and therefore its own
``NPUGraphTreeManager`` with a private memory pool.  Two domains give the two
dual-stream lanes two fully independent native tree states; no manager or node
class is subclassed or edited.

The domain has no batch/gear/stream policy.  Activation is the opt-in
boundary: while active, container lookup and global reset on the graph tree
core are served through the external wrappers installed by ``hooks.py``.
Context switching must stay outside the Dynamo-traced region and cover the
entire invocation, including graph breaks.
"""

import contextlib
import contextvars
import threading
import weakref


_current_domain = contextvars.ContextVar("npu_dual_stream_graph_domain", default=None)

# Weak registry: an abandoned domain must not be kept alive by this module.
_domains = weakref.WeakSet()


def current_domain():
    """Return the active :class:`GraphExecutionDomain`, or None."""
    return _current_domain.get()


def active_domain_container(device_index):
    """Return the active domain's container, or None for the default TLS path."""
    domain = current_domain()
    return None if domain is None else domain.container_for(device_index)


def reset_graph_execution_domains():
    """Reset every live domain at a safe point.

    All domains are checked idle before any of them is mutated; global reset is
    rejected while another thread still owns a live active domain.
    """
    domains = list(_domains)
    for domain in domains:
        domain.check_idle()
    for domain in domains:
        domain.reset()


class GraphExecutionDomain:
    """One lane's isolated execution domain.

    Lifetime contract:

    - ``activate()`` is thread-affine, rejects nesting, and installs the
      external graph-tree/Inductor wrappers once for the process.
    - The native container (and, on first use, the native manager plus its
      private pool) is created lazily on first container lookup.
    - ``close()`` / ``reset()`` tear the domain down as a whole via the native
      ``manager.shutdown()``; individual graph nodes are never released here.
    - ``epoch`` increases on every teardown; per-domain caches are dropped so
      stale artifacts can never be reached again.
    """

    def __init__(self, device_index):
        self.device_index = device_index
        self.epoch = 0
        self._thread = threading.current_thread()
        self._active = False
        self._closed = False
        self._container = None
        self._callables = {}
        self._pending = []
        _domains.add(self)

    # -- invariant checks ---------------------------------------------------

    def check_owner(self):
        if threading.current_thread() is not self._thread:
            raise RuntimeError("NPU graph execution domains are thread-affine")
        if self._closed:
            raise RuntimeError("NPU graph execution domain is closed")

    def check_active(self):
        self.check_owner()
        if not self._active or current_domain() is not self:
            raise RuntimeError("NPU graph execution domain must be active")

    def check_idle(self):
        self.check_owner()
        if self._active:
            raise RuntimeError("Cannot reset or close an active NPU graph execution domain")

    # -- activation ---------------------------------------------------------

    @contextlib.contextmanager
    def activate(self):
        self.check_idle()
        if current_domain() is not None:
            raise RuntimeError("Nested NPU graph execution domains are not supported")
        # Baseline Inductor setup must finish first so a later ordinary
        # compilation cannot overwrite the experimental binding.
        from torch_npu._experimental.dual_stream_graph import hooks
        from torch_npu.npu import _graph_tree

        hooks.install_domain_compile_hooks()
        hooks.install_graph_tree_domain_hooks(_graph_tree)
        token = _current_domain.set(self)
        self._active = True
        try:
            yield self
        finally:
            self._active = False
            _current_domain.reset(token)

    # -- native container / per-domain caches -------------------------------

    def container_for(self, device_index):
        self.check_active()
        if device_index != self.device_index:
            raise RuntimeError("An NPU graph execution domain cannot span devices")
        if self._container is None:
            from torch_npu.npu._graph_tree import TreeManagerContainer

            self._container = TreeManagerContainer(device_index)
        return self._container

    @property
    def native_container(self):
        """Read-only observation of the native container (None before first use)."""
        return self._container

    def get_or_create_callable(self, artifact, factory):
        """Return this domain's lazy wrapper for one compiled artifact.

        ``factory`` must build a fresh original lazy callable; reusing a
        callable already bound to another domain's manager is not allowed.
        """
        self.check_active()
        wrapper = self._callables.get(artifact)
        if wrapper is None:
            wrapper = factory()
            self._callables[artifact] = wrapper
        return wrapper

    # -- in-flight reference keeping ----------------------------------------

    def retain_until(self, event, references):
        """Keep ``references`` alive until ``event`` completes on the device."""
        self.check_owner()
        self._pending = [(done, refs) for done, refs in self._pending if not done.query()]
        self._pending.append((event, references))

    # -- teardown -----------------------------------------------------------

    def _teardown(self):
        # Also covers a partially submitted call whose completion event never
        # got recorded.  Python references are dropped only after the device
        # has actually finished.
        if self._container is not None or self._pending:
            import torch

            torch.npu.synchronize(self.device_index)
        container = self._container
        manager = container.tree_manager if container is not None else None
        if manager is not None:
            manager.shutdown()
        self._callables.clear()
        self._container = None
        self._pending.clear()
        self.epoch += 1

    def reset(self):
        self.check_idle()
        self._teardown()

    def close(self):
        if self._closed:
            return
        self.check_idle()
        self._teardown()
        self._closed = True
        _domains.discard(self)

    def __del__(self):
        # Best-effort safety net for an abandoned domain; explicit close()
        # remains the deterministic lifecycle API.
        try:
            if not self._closed:
                self._teardown()
        except Exception:
            pass
