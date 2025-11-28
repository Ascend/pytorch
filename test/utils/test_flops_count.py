import warnings
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.utils.flops_count import _FlopsCounter, FlopsCounter


class TestFlopsCount(TestCase):

    def test_flops_counter_deprecation(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            counter = FlopsCounter()
            self.assertTrue(len(w) > 0)
            self.assertTrue(issubclass(w[0].category, FutureWarning))
            self.assertIn("will be deprecated", str(w[0].message))

    def test_get_flops_method(self):
        counter = _FlopsCounter()
        result = counter.get_flops()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], (int, float))
        self.assertIsInstance(result[1], (int, float))

    def test_pause_resume_methods(self):
        counter = _FlopsCounter()

        try:
            counter.pause()
            counter.resume()
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"pause() or resume() raised an exception: {e}")

    def test_start_stop_methods(self):
        counter = _FlopsCounter()

        try:
            counter.start()
            counter.stop()
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"start() or stop() raised an exception: {e}")


if __name__ == '__main__':
    run_tests()