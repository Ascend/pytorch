# Owner(s): ["module: dynamo"]
import functools
import gc
import unittest
import weakref
import torch
import torch._dynamo.test_case
from torch._dynamo.graph_bytecode_inputs import (
    CURRENT_STREAM_INDEX,
    index_to_external_object_weakref,
    reset_user_object_tracking,
)
import torch_npu

requires_npu = functools.partial(unittest.skipIf, not torch.npu.is_available(), "requires npu")


class StreamintoDynamoTests(torch._dynamo.test_case.TestCase):

    @requires_npu()
    def test_stream(self):
        def model_1(x):
            a = x * x
            s = torch.npu.Stream()
            s.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(s):
                b = x + a
            return b
        inp = torch.randn(2, 8).npu()
        m = torch.compile(model_1, backend="aot_eager", fullgraph=True)
        output = m(inp)
        output1 = model_1(inp)
        torch.allclose(output, output1)

    def _assert_weakref_callback_fires(self, factory):
        """Backend Stream/Event tp_dealloc overrides must call
        PyObject_ClearWeakRefs so that weakrefs to destroyed instances
        are properly cleared.  Without it, the weakref's wr_object is
        left dangling and any later access (e.g. the dynamo external-
        object registry being cleared at interpreter finalization)
        hits a use-after-free.

        The callback firing is the only reliable Python-level signal —
        `weakref.ref(s)() is None` can return True even with the bug
        present, because other CPython paths may null wr_object
        without invoking callbacks (Objects/weakrefobject.c).
        """
        called = []
        obj = factory()
        # ``_weakref_keepalive`` must outlive ``del obj`` so the callback
        # has a chance to fire; the binding is load-bearing.
        _weakref_keepalive = weakref.ref(obj, lambda _ref: called.append(True))
        del obj
        gc.collect()
        self.assertEqual(called, [True])
        del _weakref_keepalive

    @requires_npu()
    def test_npu_stream_event_weakref_callback(self):
        self._assert_weakref_callback_fires(torch.npu.Stream)
        self._assert_weakref_callback_fires(torch.npu.Event)

    @requires_npu()
    def test_dynamo_registry_no_dangling_weakref(self):
        """Natural repro of the original UAF pattern.

        ``torch.compile(fn, backend="eager")`` with ``fn`` referencing
        ``torch.npu.current_stream()`` causes dynamo to register a
        weakref to the captured stream in
        ``index_to_external_object_weakref`` (via
        ``store_user_object_weakrefs``, which does NOT pin via
        ``keep_alive``).  As soon as ``fn`` returns, the captured wrapper
        has no strong references and is freed via ``THNPStream_dealloc``.

        At that point the patched tp_dealloc must call
        ``PyObject_ClearWeakRefs``, otherwise the registry now holds a
        weakref whose ``wr_object`` is a dangling pointer to freed
        memory.  Later, when ``_PyModule_ClearDict`` tears down the
        registry at interpreter finalization, dereferencing that
        dangling pointer to clear the weakref hits a use-after-free.
        """
        # Start from a known-clean registry so the assertion below is
        # a statement about what happened in *this* test, not residue
        # from earlier tests in the same process.
        reset_user_object_tracking()

        def fn(x):
            return torch.npu.current_stream()

        x = torch.zeros(1, device="npu")
        compiled = torch.compile(fn, backend="eager")
        compiled(x)
        del compiled
        gc.collect()

        # Tripwire: torch.compile of a function referencing
        # current_stream() must still register under
        # CURRENT_STREAM_INDEX.  If this fails, the production code
        # path that originally surfaced the UAF has moved and this
        # regression test no longer covers it.
        self.assertIn(
            CURRENT_STREAM_INDEX,
            index_to_external_object_weakref,
            "torch.compile of a function referencing current_stream() must "
            "register a weakref under CURRENT_STREAM_INDEX",
        )

        # The captured stream wrapper was freed when fn returned.  The
        # patched tp_dealloc must have cleared the registry's weakref;
        # dereferencing it now must return None rather than a dangling
        # pointer into freed memory.
        self.assertIsNone(
            index_to_external_object_weakref[CURRENT_STREAM_INDEX](),
            "dynamo registry holds a weakref to a Stream wrapper that has "
            "been freed; tp_dealloc must call PyObject_ClearWeakRefs to "
            "clear it, otherwise the registry retains a dangling pointer",
        )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
