"""
Add validation cases for torch.distributed.elastic.events.record API:

1. PyTorch community lacks direct test cases for torch.distributed.elastic.events.record
   in the standard test suite, so this file is added.

2. This file validates the following APIs:
   torch.distributed.elastic.events.record (This is a pure Python-level event logging API with no hardware dependency.)
   torch.distributed.elastic.events.api.EventMetadataValue (Type alias for metadata values)
   (extendable)
"""

import json
import time
from typing import Union, Optional, get_args, get_origin
from unittest.mock import patch, MagicMock

import torch
from torch.distributed.elastic.events import record, get_logging_handler
from torch.distributed.elastic.events.api import Event, EventSource, EventMetadataValue
from torch.testing._internal.common_utils import TestCase, run_tests


class TestEventsRecord(TestCase):
    """Test torch.distributed.elastic.events.record method."""

    DESTINATION_NULL = "null"
    DESTINATION_CONSOLE = "console"
    TEST_EVENT_PREFIX = "test_event_"

    def tearDown(self):
        """Clean up resources after each test case to ensure test isolation."""
        from torch.distributed.elastic.events import _events_loggers
        _events_loggers.clear()

        if hasattr(get_logging_handler, "cache_clear"):
            get_logging_handler.cache_clear()

        patch.stopall()
        super().tearDown()

    def test_record_null_destination(self):
        """Verify that record does not raise an exception when using the default null destination."""
        event = Event(
            name=f"{self.TEST_EVENT_PREFIX}null",
            source=EventSource.WORKER,
            metadata={"key": "value"}
        )
        record(event)

    def test_record_console_destination(self):
        """Verify that record does not raise an exception when using the console destination."""
        event = Event(
            name=f"{self.TEST_EVENT_PREFIX}console",
            source=EventSource.AGENT,
            metadata={"stage": "init"}
        )
        record(event, destination=self.DESTINATION_CONSOLE)

    def test_record_with_timestamp(self):
        """Verify that record can correctly log an event with a custom or auto-generated timestamp."""
        custom_timestamp = int(time.time() * 1000)
        event = Event(
            name=f"{self.TEST_EVENT_PREFIX}timestamp_custom",
            source=EventSource.WORKER,
            timestamp=custom_timestamp,
            metadata={"ts": custom_timestamp}
        )
        record(event, destination=self.DESTINATION_NULL)
        self.assertEqual(event.timestamp, custom_timestamp)
        self.assertIsInstance(event.timestamp, int)
        self.assertGreaterEqual(event.timestamp, 0)

        event_with_auto_timestamp = Event(
            name=f"{self.TEST_EVENT_PREFIX}timestamp_auto",
            source=EventSource.WORKER
        )
        record(event_with_auto_timestamp)
        self.assertIsNotNone(event_with_auto_timestamp.timestamp)
        self.assertIsInstance(event_with_auto_timestamp.timestamp, int)
        self.assertGreaterEqual(event_with_auto_timestamp.timestamp, 0)

    @patch("torch.distributed.elastic.events._get_or_create_logger")
    def test_record_with_various_metadata_types(self, mock_get_logger):
        """Verify that record correctly passes various metadata types to the underlying logger."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        metadata = {
            "str_val": "hello",
            "int_val": 42,
            "float_val": 3.14,
            "bool_val": True,
            "none_val": None,
            "list_val": [1, 2, 3],
            "nested_dict": {"sub_key": "sub_val"}
        }
        event = Event(
            name=f"{self.TEST_EVENT_PREFIX}metadata_types",
            source=EventSource.AGENT,
            metadata=metadata
        )
        record(event, destination=self.DESTINATION_CONSOLE)

        # Verify the internal call chain
        mock_get_logger.assert_called_once_with(self.DESTINATION_CONSOLE)
        mock_logger.info.assert_called_once()

        # Verify that the serialized metadata is correctly preserved
        logged_raw = mock_logger.info.call_args[0][0]
        logged_data = json.loads(logged_raw)
        self.assertEqual(logged_data["metadata"], metadata)

    def test_record_multiple_events(self):
        """Verify that consecutive calls to record do not interfere with each other."""
        event1 = Event(
            name=f"{self.TEST_EVENT_PREFIX}multiple_1",
            source=EventSource.WORKER,
            metadata={"seq": 1}
        )
        event2 = Event(
            name=f"{self.TEST_EVENT_PREFIX}multiple_2",
            source=EventSource.WORKER,
            metadata={"seq": 2}
        )

        record(event1)
        record(event2)

        self.assertIn("seq", event1.metadata)
        self.assertEqual(event1.metadata["seq"], 1)
        self.assertIn("seq", event2.metadata)
        self.assertEqual(event2.metadata["seq"], 2)

    @patch("torch.distributed.elastic.events._get_or_create_logger")
    def test_record_calls_get_or_create_logger(self, mock_get_logger):
        """
        Verify that record internally calls _get_or_create_logger to obtain the logger,
        and then calls .info() on that logger with the serialized event.
        """
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        event = Event(
            name=f"{self.TEST_EVENT_PREFIX}mock",
            source=EventSource.WORKER
        )
        record(event, destination=self.DESTINATION_CONSOLE)

        mock_get_logger.assert_called_once_with(self.DESTINATION_CONSOLE)
        mock_logger.info.assert_called_once_with(event.serialize())

    def test_record_event_name_empty(self):
        """Verify that record does not raise an exception when the event name is an empty string."""
        event = Event(name="", source=EventSource.WORKER)
        record(event)

    def test_record_event_name_special_chars(self):
        """Verify event names with special characters (spaces, symbols, unicode) are handled gracefully."""
        special_names = ["test event", "test@event", "测试事件", "a" * 256]
        for name in special_names:
            with self.subTest(name=name):
                event = Event(name=name, source=EventSource.WORKER)
                record(event, destination=self.DESTINATION_NULL)

    def test_record_all_event_sources(self):
        """Verify record compatibility with all EventSource enum values."""
        for source in EventSource:
            with self.subTest(source=source):
                event = Event(
                    name=f"{self.TEST_EVENT_PREFIX}source_{source.name.lower()}",
                    source=source,
                    metadata={"source_type": source.name}
                )
                record(event)
                self.assertEqual(event.source, source)
                self.assertIn("source_type", event.metadata)
                self.assertEqual(event.metadata["source_type"], source.name)


class TestEventMetadataValue(TestCase):
    """Test torch.distributed.elastic.events.api.EventMetadataValue type alias."""

    def test_event_metadata_value_is_defined(self):
        """Verify that EventMetadataValue is exported and defined."""
        self.assertIsNotNone(EventMetadataValue)
        self.assertIs(get_origin(EventMetadataValue), Union)

    def test_event_metadata_value_type_structure(self):
        """
        Verify that EventMetadataValue is a Union of str, int, float, bool, None.
        Expected: Optional[Union[str, int, float, bool]]
        """
        origin = get_origin(EventMetadataValue)
        args = get_args(EventMetadataValue)

        self.assertIs(origin, Union)
        self.assertIn(str, args)
        self.assertIn(int, args)
        self.assertIn(float, args)
        self.assertIn(bool, args)
        self.assertIn(type(None), args)
        self.assertEqual(len(args), 5)

    def test_event_metadata_value_legal_primitives(self):
        """Verify that all legal primitive values are accepted by Event metadata."""
        test_cases = [
            ("string_val", "hello"),
            ("int_val", 42),
            ("int_val_neg", -7),
            ("float_val", 3.14159),
            ("float_val_zero", 0.0),
            ("bool_true", True),
            ("bool_false", False),
            ("none_val", None),
        ]

        for key, value in test_cases:
            with self.subTest(key=key, value=value):
                event = Event(
                    name=f"test_metadata_{key}",
                    source=EventSource.WORKER,
                    metadata={key: value}
                )
                self.assertIn(key, event.metadata)
                self.assertEqual(event.metadata[key], value)
                record(event, destination=TestEventsRecord.DESTINATION_NULL)

    def test_event_metadata_value_none_explicit(self):
        """Verify that None is explicitly allowed as a metadata value."""
        event = Event(
            name="test_metadata_none",
            source=EventSource.AGENT,
            metadata={"explicit_none": None}
        )
        self.assertIsNone(event.metadata["explicit_none"])
        record(event, destination=TestEventsRecord.DESTINATION_NULL)

    def test_event_metadata_value_mixed_dict(self):
        """Verify that a metadata dict containing all legal types can be constructed and recorded."""
        metadata = {
            "epoch": 10,
            "loss": 0.1234,
            "model_name": "resnet50",
            "is_training": True,
            "checkpoint_path": None,
        }
        event = Event(
            name="test_metadata_mixed",
            source=EventSource.WORKER,
            metadata=metadata
        )
        self.assertEqual(len(event.metadata), 5)
        record(event, destination=TestEventsRecord.DESTINATION_NULL)

    def test_event_metadata_value_serialization_roundtrip(self):
        """
        Verify that metadata values survive the Event.serialize() roundtrip
        and remain as their original Python types.
        """
        metadata = {
            "lr": 0.01,
            "step": 100,
            "tag": "train",
            "enabled": False,
            "optional": None,
        }
        event = Event(
            name="test_serialization",
            source=EventSource.AGENT,
            metadata=metadata
        )
        serialized = event.serialize()

        self.assertIsInstance(serialized, str)
        data = json.loads(serialized)
        self.assertIn("metadata", data)

        serialized_metadata = data["metadata"]
        self.assertEqual(serialized_metadata["lr"], 0.01)
        self.assertEqual(serialized_metadata["step"], 100)
        self.assertEqual(serialized_metadata["tag"], "train")
        self.assertEqual(serialized_metadata["enabled"], False)
        self.assertIsNone(serialized_metadata["optional"])

    def test_event_serialize_empty_metadata(self):
        """Verify that empty metadata serializes correctly without data loss."""
        event = Event(
            name="test_empty_metadata",
            source=EventSource.WORKER,
            metadata={}
        )
        data = json.loads(event.serialize())
        self.assertEqual(data["metadata"], {})

    def test_event_serialize_unsupported_metadata_type(self):
        """
        Verify that unsupported metadata types (e.g. set, bytes) raise TypeError
        during serialization / record, since Event itself does not validate at construction time.
        """
        invalid_metadata_cases = [
            ("set_val", {1, 2, 3}),
            ("bytes_val", b"test"),
            ("object_val", MagicMock()),
        ]

        for key, value in invalid_metadata_cases:
            with self.subTest(key=key):
                event = Event(
                    name=f"test_invalid_{key}",
                    source=EventSource.WORKER,
                    metadata={key: value}
                )
                # Event construction succeeds, but serialize/record should fail
                with self.assertRaises(TypeError):
                    event.serialize()


if __name__ == "__main__":
    run_tests()