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
Add validation cases for torch.utils.model_zoo APIs on NPU:
1. PyTorch community lacks sufficient and direct API validations for some APIs, so this file is added.
2. This file validates torch.utils.model_zoo.load_url (extendable).
"""

import hashlib
import os
import tempfile
from unittest.mock import patch

import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils.model_zoo import load_url


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestModelZooAPIs(TestCase):

    def test_load_url_cached_checkpoint(self):
        with tempfile.TemporaryDirectory() as model_dir:
            file_name = "weights.pth"
            # A CPU checkpoint is required to verify map_location remaps it to NPU.
            checkpoint = {"weight": torch.arange(6, device="cpu").reshape(2, 3)}
            torch.save(checkpoint, os.path.join(model_dir, file_name))

            loaded = load_url(
                "https://invalid.example/unused.pth",
                model_dir=model_dir,
                map_location=device_type,
                progress=False,
                file_name=file_name,
                weights_only=True,
            )

            self.assertEqual(checkpoint["weight"].device.type, "cpu")
            self.assertEqual(loaded["weight"].device.type, device_type)
            self.assertEqual(loaded["weight"].cpu(), checkpoint["weight"])

    def test_load_url_download_defaults(self):
        checkpoint = {"weight": torch.ones(2, 2, device=device_type)}

        def save_checkpoint(url, dst, hash_prefix, progress):
            self.assertEqual(url, "https://example.com/model.pth")
            self.assertIsNone(hash_prefix)
            self.assertTrue(progress)
            torch.save(checkpoint, dst)

        with tempfile.TemporaryDirectory() as hub_dir:
            with patch("torch.hub.get_dir", return_value=hub_dir):
                with patch("torch.hub.download_url_to_file", side_effect=save_checkpoint):
                    loaded = load_url("https://example.com/model.pth")

            self.assertEqual(loaded["weight"], checkpoint["weight"])
            self.assertTrue(os.path.exists(os.path.join(hub_dir, "checkpoints", "model.pth")))

    def test_load_url_check_hash(self):
        checkpoint = {"weight": torch.ones(1, device=device_type)}
        serialized = tempfile.NamedTemporaryFile(delete=False)
        serialized.close()
        torch.save(checkpoint, serialized.name)
        with open(serialized.name, "rb") as checkpoint_file:
            hash_prefix = hashlib.sha256(checkpoint_file.read()).hexdigest()[:8]
        os.unlink(serialized.name)
        file_name = f"model-{hash_prefix}.pth"

        def save_checkpoint(url, dst, actual_hash_prefix, progress):
            self.assertEqual(actual_hash_prefix, hash_prefix)
            self.assertFalse(progress)
            torch.save(checkpoint, dst)

        with tempfile.TemporaryDirectory() as model_dir:
            with patch("torch.hub.download_url_to_file", side_effect=save_checkpoint):
                loaded = load_url(
                    f"https://example.com/{file_name}",
                    model_dir=model_dir,
                    progress=False,
                    check_hash=True,
                    weights_only=True,
                )

            self.assertEqual(loaded["weight"], checkpoint["weight"])


if __name__ == "__main__":
    run_tests()
