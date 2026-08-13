# Copyright (c) 2026 Huawei Technologies Co., Ltd
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Add validation cases for torch.distributed.elastic.metrics APIs:

1. PyTorch community lacks sufficient API validations, so this file is added.
2. This file validates the following apis:
torch.distributed.elastic.metrics.api.ConsoleMetricHandler
torch.distributed.elastic.metrics.api.NullMetricHandler
torch.distributed.elastic.metrics.configure
torch.distributed.elastic.metrics.put_metric
(extendable)
"""

import unittest.mock as mock

import torch.distributed.elastic.metrics.api as metrics_api
from torch.testing._internal.common_utils import TestCase, run_tests

from torch.distributed.elastic.metrics import configure, put_metric
from torch.distributed.elastic.metrics.api import (
    ConsoleMetricHandler,
    MetricData,
    MetricHandler,
    NullMetricHandler,
)


class ElasticMetricsApiTest(TestCase):

    def setUp(self):
        super().setUp()
        self._original_default_metrics_handler = metrics_api._default_metrics_handler
        self._original_metrics_map = metrics_api._metrics_map.copy()
        self.addCleanup(self._restore_metrics_state)
        metrics_api._metrics_map.clear()

    def _restore_metrics_state(self):
        metrics_api._default_metrics_handler = self._original_default_metrics_handler
        metrics_api._metrics_map.clear()
        metrics_api._metrics_map.update(self._original_metrics_map)

    def test_put_metric(self):
        default_handler = mock.Mock(spec=MetricHandler)
        custom_handler = mock.Mock(spec=MetricHandler)
        configure(default_handler)
        configure(custom_handler, group="custom_group")

        put_metric("default_metric", 3)
        put_metric("", -1, "custom_group")
        put_metric("zero_metric", 0, "custom_group")

        self.assertEqual(default_handler.emit.call_count, 1)
        self.assertEqual(custom_handler.emit.call_count, 2)

        default_record = default_handler.emit.call_args.args[0]
        self.assertIsInstance(default_record, MetricData)
        self.assertEqual(default_record.group_name, "torchelastic")
        self.assertEqual(default_record.name, "default_metric")
        self.assertEqual(default_record.value, 3)
        self.assertGreater(default_record.timestamp, 0)

        custom_records = [
            call.args[0] for call in custom_handler.emit.call_args_list
        ]
        self.assertEqual(
            [
                (record.group_name, record.name, record.value)
                for record in custom_records
            ],
            [
                ("custom_group", "", -1),
                ("custom_group", "zero_metric", 0),
            ],
        )
        self.assertTrue(all(record.timestamp > 0 for record in custom_records))

    def test_console_metric_handler_emit(self):
        handler = ConsoleMetricHandler()
        metric_data = MetricData(
            timestamp=1,
            group_name="test_group",
            name="test_metric",
            value=1,
        )

        with mock.patch("builtins.print") as print_mock:
            ret = handler.emit(metric_data)

        self.assertIsNone(ret)
        self.assertIsInstance(handler, MetricHandler)
        self.assertEqual(1, print_mock.call_count)

        output = print_mock.call_args[0][0]
        self.assertIn("1", output)
        self.assertIn("test_group", output)
        self.assertIn("test_metric", output)

    def test_null_metric_handler_emit(self):
        handler = NullMetricHandler()
        metric_data = MetricData(
            timestamp=1,
            group_name="test_group",
            name="test_metric",
            value=1,
        )

        with mock.patch("builtins.print") as print_mock:
            ret = handler.emit(metric_data)

        self.assertIsNone(ret)
        self.assertIsInstance(handler, MetricHandler)
        self.assertEqual(0, print_mock.call_count)

    def test_configure_default_console_metric_handler(self):
        handler = ConsoleMetricHandler()

        with mock.patch("builtins.print") as print_mock:
            configure(handler)
            stream = metrics_api.getStream("default_group")
            stream.add_value("default_metric", 1)

        self.assertEqual(1, print_mock.call_count)

        output = print_mock.call_args[0][0]
        self.assertIn("default_group", output)
        self.assertIn("default_metric", output)
        self.assertIn("1", output)

    def test_configure_default_null_metric_handler(self):
        handler = NullMetricHandler()

        with mock.patch("builtins.print") as print_mock:
            configure(handler)
            stream = metrics_api.getStream("default_group")
            ret = stream.add_value("default_metric", 1)

        self.assertIsNone(ret)
        self.assertEqual(0, print_mock.call_count)

    def test_configure_group_specific_console_metric_handler(self):
        default_handler = NullMetricHandler()
        group_handler = ConsoleMetricHandler()

        with mock.patch("builtins.print") as print_mock:
            configure(default_handler)
            configure(group_handler, group="console_group")

            group_stream = metrics_api.getStream("console_group")
            default_stream = metrics_api.getStream("default_group")

            group_stream.add_value("group_metric", 2)
            default_stream.add_value("default_metric", 3)

        self.assertEqual(1, print_mock.call_count)

        output = print_mock.call_args[0][0]
        self.assertIn("console_group", output)
        self.assertIn("group_metric", output)
        self.assertIn("2", output)
        self.assertNotIn("default_metric", output)

    def test_configure_group_specific_null_metric_handler(self):
        default_handler = ConsoleMetricHandler()
        group_handler = NullMetricHandler()

        with mock.patch("builtins.print") as print_mock:
            configure(default_handler)
            configure(group_handler, group="null_group")

            null_stream = metrics_api.getStream("null_group")
            default_stream = metrics_api.getStream("default_group")

            null_stream.add_value("null_metric", 1)
            default_stream.add_value("default_metric", 2)

        self.assertEqual(1, print_mock.call_count)

        output = print_mock.call_args[0][0]
        self.assertIn("default_group", output)
        self.assertIn("default_metric", output)
        self.assertIn("2", output)
        self.assertNotIn("null_metric", output)


if __name__ == "__main__":
    run_tests()
