"""
Test purpose: Verify AsyncSaveResponse behavior.
API name: torch.distributed.checkpoint.state_dict_saver.AsyncSaveResponse
API signature: AsyncSaveResponse(staging_completion, upload_completion)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Future completion and errors | Complete staging before upload and check upload error propagation. |

Execution: Single-process unittest.
Scope and prerequisites:
- The two futures are manually completed to isolate the response object from storage and transport.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from distributed_test_support import SupplementTestCase, run_tests

sys.path.pop(0)


class TestAsyncSaveResponse(SupplementTestCase):
    def test_async_save_response_independent_completion(self):
        from concurrent.futures import Future

        from torch.distributed.checkpoint.state_dict_saver import AsyncSaveResponse

        staging, upload = Future(), Future()
        response = AsyncSaveResponse(staging_completion=staging, upload_completion=upload)
        staging.set_result(None)
        self.assertIsNone(response.staging_completion.result(timeout=1))
        self.assertFalse(response.upload_completion.done())
        upload.set_exception(OSError("upload failed"))
        with self.assertRaisesRegex(OSError, "upload failed"):
            response.upload_completion.result(timeout=1)


if __name__ == "__main__":
    run_tests()
