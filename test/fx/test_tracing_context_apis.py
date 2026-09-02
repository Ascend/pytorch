# Owner(s): ["module: fx"]
"""
Add validation cases for torch._guards.TracingContext API.

API Introduction:
torch._guards.TracingContext is a context object used by PyTorch's compiler
and tracing infrastructure. It holds the current FakeTensorMode and guards
context during tracing. It can be accessed via TracingContext.try_get()
(returns None when not in tracing) or TracingContext.get() (raises RuntimeError
when not in tracing).
"""
from torch._guards import TracingContext, tracing
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.testing._internal.common_utils import TestCase, run_tests


class TestTracingContext(TestCase):

    def test_try_get_returns_none_outside_context(self):
        result = TracingContext.try_get()
        self.assertIsNone(result)

    def test_try_get_returns_context_inside_tracing(self):
        ctx = TracingContext(fake_mode=None)
        with tracing(ctx):
            result = TracingContext.try_get()
            self.assertIs(result, ctx)

    def test_try_get_returns_none_after_context_exits(self):
        ctx = TracingContext(fake_mode=None)
        with tracing(ctx):
            pass
        result = TracingContext.try_get()
        self.assertIsNone(result)

    def test_get_raises_outside_context(self):
        with self.assertRaises(RuntimeError):
            TracingContext.get()

    def test_get_returns_context_inside_tracing(self):
        ctx = TracingContext(fake_mode=None)
        with tracing(ctx):
            result = TracingContext.get()
            self.assertIs(result, ctx)

    def test_init_with_fake_mode(self):
        with FakeTensorMode() as fake_mode:
            ctx = TracingContext(fake_mode=fake_mode)
            self.assertIs(ctx.fake_mode, fake_mode)

    def test_init_without_fake_mode(self):
        ctx = TracingContext(fake_mode=None)
        self.assertIsNone(ctx.fake_mode)

    def test_guards_context_initialized(self):
        ctx = TracingContext(fake_mode=None)
        self.assertIsNotNone(ctx.guards_context)


if __name__ == '__main__':
    run_tests()
