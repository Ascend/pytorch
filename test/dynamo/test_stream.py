# Owner(s): ["module: dynamo"]
import functools
import gc
import unittest
import weakref
import torch
import torch._dynamo.test_case
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


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
