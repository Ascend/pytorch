# Owner(s): ["module: dynamo"]
"""
Test that the deprecated ``torch._dynamo.config.inline_inbuilt_nn_modules``
config option raises a ``FutureWarning`` on access.

Per https://github.com/pytorch/pytorch/pull/178205, inline_inbuilt_nn_modules
is always True now and setting/unsetting it should trigger a deprecation warning.
"""

import warnings
import unittest

import torch


class TestDeprecatedInlineInbuiltNnModules(unittest.TestCase):
    """Verify that accessing ``inline_inbuilt_nn_modules`` triggers a deprecation warning.

    The config entry is declared with ``deprecated=True`` and a
    ``deprecation_message`` in ``torch._dynamo.config``.  Both reads and writes
    go through the ``ConfigModule.__getattr__`` / ``__setattr__`` machinery,
    which calls ``_warn_if_deprecated`` once per config name and emits a
    ``FutureWarning``.
    """

    @staticmethod
    def _reset_deprecation_warned():
        """Reset the ``_deprecation_warned`` flag on the config entry.

        The config infrastructure only emits the FutureWarning *once* per
        config name (guarded by ``_deprecation_warned``), so we reset it
        before each test that needs to observe a fresh warning.
        """
        entry = torch._dynamo.config._config.get("inline_inbuilt_nn_modules")
        if entry is not None:
            entry._deprecation_warned = False

    def setUp(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self._saved_inline_inbuilt = torch._dynamo.config.inline_inbuilt_nn_modules
        self._reset_deprecation_warned()

    def tearDown(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            torch._dynamo.config.inline_inbuilt_nn_modules = self._saved_inline_inbuilt

    def _check_warning(self, warn_records):
        """Assert the warning list is non-empty and every warning is a FutureWarning."""
        self.assertGreater(len(warn_records), 0)
        for r in warn_records:
            self.assertTrue(
                issubclass(r.category, FutureWarning),
                f"Expected FutureWarning, got {r.category}",
            )
            msg = str(r.message)
            self.assertIn("inline_inbuilt_nn_modules", msg)
            self.assertIn("deprecated", msg.lower())

    def test_read_triggers_future_warning(self):
        """Reading ``inline_inbuilt_nn_modules`` should emit a FutureWarning once."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = torch._dynamo.config.inline_inbuilt_nn_modules

        self._check_warning(w)

    def test_write_triggers_future_warning(self):
        """Writing (setting) ``inline_inbuilt_nn_modules`` should emit a FutureWarning once."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            torch._dynamo.config.inline_inbuilt_nn_modules = False

        self._check_warning(w)

    def test_restore_triggers_future_warning(self):
        """Restoring (setting to its original value) should also emit a FutureWarning."""
        # Read the current value in a separate context (suppress its warning).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            original = torch._dynamo.config.inline_inbuilt_nn_modules

        # reset the flag so the restore write is observed
        self._reset_deprecation_warned()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            torch._dynamo.config.inline_inbuilt_nn_modules = original

        self._check_warning(w)

    def test_warning_only_fired_once(self):
        """The FutureWarning should only fire the *first* time the config is read."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = torch._dynamo.config.inline_inbuilt_nn_modules  # first read → warning
            _ = torch._dynamo.config.inline_inbuilt_nn_modules  # second read → no warning

        self.assertEqual(len(w), 1, "Warning should only be emitted once")

    def test_warning_message_contains_deprecation_explanation(self):
        """The warning message should mention that inline_inbuilt_nn_modules is always True."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = torch._dynamo.config.inline_inbuilt_nn_modules

        self.assertGreater(len(w), 0)
        message = str(w[0].message)
        # The actual message from config.py:
        #   "torch._dynamo.config.inline_inbuilt_nn_modules is deprecated and
        #    does not do anything, inline_inbuilt_nn_modules is always True."
        self.assertIn('torch._dynamo.config.inline_inbuilt_nn_modules is deprecated', message)

    def test_value_defaults_to_true(self):
        """The config value should be True by default."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            val = torch._dynamo.config.inline_inbuilt_nn_modules

        self.assertTrue(val, "inline_inbuilt_nn_modules should default to True")

    def test_write_allowed_but_deprecated(self):
        """Setting the config to a different value is deprecated but still honoured.

        The config entry is *not* hard-coded to True; it can be set to False
        (with a deprecation warning).  The deprecation is a statement that
        the flag no longer has any effect on compilation behaviour, not that
        the accessor returns a constant.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            torch._dynamo.config.inline_inbuilt_nn_modules = False
            val = torch._dynamo.config.inline_inbuilt_nn_modules

        self.assertFalse(
            val,
            "The value written should be readable back; the deprecation "
            "only warns, it does not reject the write",
        )


if __name__ == "__main__":
    unittest.main()
