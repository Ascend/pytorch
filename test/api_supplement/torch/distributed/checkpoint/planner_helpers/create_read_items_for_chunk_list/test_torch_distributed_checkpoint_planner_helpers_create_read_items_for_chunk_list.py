"""
Test purpose: Verify create_read_items_for_chunk_list behavior.
API name: torch.distributed.checkpoint.planner_helpers.create_read_items_for_chunk_list
API signature: create_read_items_for_chunk_list(fqn, checkpoint_md, local_chunks)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Chunk intersections and empty requests | Check cross-chunk offsets and an empty read request. |

Execution: Single-process unittest.
Scope and prerequisites:
- The one-dimensional fixture isolates overlap arithmetic. Filesystem readers are outside this
  helper.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from distributed_test_support import SupplementTestCase, run_tests

sys.path.pop(0)


class TestCreateReadItemsForChunkList(SupplementTestCase):
    def test_create_read_items_for_chunk_list_intersecting_chunks(self):
        import torch
        from torch.distributed.checkpoint.metadata import (
            ChunkStorageMetadata,
            TensorProperties,
            TensorStorageMetadata,
        )
        from torch.distributed.checkpoint.planner_helpers import (
            create_read_items_for_chunk_list,
        )

        def chunk(start, length):
            return ChunkStorageMetadata(torch.Size([start]), torch.Size([length]))

        metadata = TensorStorageMetadata(
            TensorProperties(), torch.Size([8]), [chunk(0, 4), chunk(4, 4)]
        )
        self.assertEqual(create_read_items_for_chunk_list("weight", metadata, []), [])
        items = create_read_items_for_chunk_list("weight", metadata, [chunk(2, 4)])
        self.assertEqual(
            [
                (
                    item.storage_index.fqn,
                    tuple(item.storage_index.offset),
                    tuple(item.storage_offsets),
                    tuple(item.dest_offsets),
                    tuple(item.lengths),
                )
                for item in items
            ],
            [("weight", (0,), (2,), (0,), (2,)), ("weight", (4,), (0,), (2,), (2,))],
        )


if __name__ == "__main__":
    run_tests()
