# Copyright (c) 2023 Huawei Technologies Co., Ltd
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

import os
import sys
import traceback

from torchgen.model import DispatchKey

def rename_privateuse1_dispatch_key():
    # rename DispatcherKey about PrivateUse1
    custom_backend = "NPU"
    def PrivateUse1Str(self):
        return self.name.replace("PrivateUse1", custom_backend)

    @staticmethod
    def parse(value: str) -> "DispatchKey":
        for k, v in DispatchKey.__members__.items():
            if k == value.replace(custom_backend, "PrivateUse1"):
                return v
        raise AssertionError(f"unknown dispatch key {value}")

    DispatchKey.__str__ = PrivateUse1Str
    DispatchKey.parse = parse

def get_torchgen_dir():
    # get path of torchgen, then get tags.yaml and native_functions.yaml
    try:
        import torchgen
        return os.path.dirname(os.path.abspath(torchgen.__file__))
    except Exception:
        _, _, exc_traceback = sys.exc_info()
        frame_summary = traceback.extract_tb(exc_traceback)[-1]
        return os.path.dirname(frame_summary.filename)
