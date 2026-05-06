"""
Add validation cases for torch.distributed.elastic.agent.server.health_check_server APIs.

1. PyTorch community tests do not cover HealthCheckServer APIs, so this file is added.
2. This file validates : 
torch.distributed.elastic.agent.server.health_check_server.HealthCheckServer
torch.distributed.elastic.agent.server.health_check_server.HealthCheckServer.start
torch.distributed.elastic.agent.server.health_check_server.HealthCheckServer.stop
torch.distributed.elastic.agent.server.health_check_server.create_healthcheck_server
(extendable)
"""

import logging
from unittest.mock import MagicMock

from torch.distributed.elastic.agent.server.health_check_server import (
    HealthCheckServer,
    create_healthcheck_server,
)
from torch_npu.testing.testcase import TestCase, run_tests

LOGGER_NAME = "torch.distributed.elastic.agent.server.health_check_server"


class TestHealthCheckServer(TestCase):
    """Unit tests for HealthCheckServer and factory method create_healthcheck_server."""

    def test_init(self):
        """Test HealthCheckServer initialization, verify attributes are assigned correctly
        and callback is not called prematurely.
        """
        alive_callback = MagicMock(return_value=123)
        server = HealthCheckServer(alive_callback, 0, 30)

        self.assertIs(server._alive_callback, alive_callback)
        self.assertEqual(server._port, 0)
        self.assertEqual(server._timeout, 30)
        alive_callback.assert_not_called()

    def test_create_healthcheck_server(self):
        """Test the factory function create_healthcheck_server, returns a valid HealthCheckServer instance."""
        alive_callback = MagicMock(return_value=123)
        server = create_healthcheck_server(alive_callback, 0, 45)

        self.assertIsInstance(server, HealthCheckServer)
        self.assertIs(server._alive_callback, alive_callback)
        self.assertEqual(server._port, 0)
        self.assertEqual(server._timeout, 45)
        alive_callback.assert_not_called()

    def test_start_logs_warning(self):
        """Test start() outputs expected WARNING log (current noop stub behavior)."""
        server = HealthCheckServer(lambda: 0, 0, 30)

        with self.assertLogs(LOGGER_NAME, level=logging.WARNING) as captured:
            server.start()

        self.assertEqual(len(captured.records), 1)
        self.assertEqual(
            captured.records[0].getMessage(),
            "No health check server started",
        )

    def test_stop_logs_info(self):
        """Test stop() outputs expected INFO log (current noop stub behavior)."""
        server = HealthCheckServer(lambda: 0, 0, 30)

        with self.assertLogs(LOGGER_NAME, level=logging.INFO) as captured:
            server.stop()

        self.assertEqual(len(captured.records), 1)
        self.assertEqual(
            captured.records[0].getMessage(),
            "Stopping noop health check server.",
        )

    def test_start_stop_lifecycle_safe(self):
        """Verify start -> stop full lifecycle runs safely without exceptions."""
        server = HealthCheckServer(lambda: True, 0, 5)
        server.start()
        server.stop()
        # Passes if no exception is thrown, no additional assertions needed

    def test_stop_idempotent(self):
        """Verify stop() is idempotent: repeated calls do not throw exceptions."""
        server = HealthCheckServer(lambda: True, 0, 5)
        server.start()
        server.stop()
        server.stop()
        server.stop()
        # Passes if no exception is thrown, no additional assertions needed


if __name__ == "__main__":
    run_tests()