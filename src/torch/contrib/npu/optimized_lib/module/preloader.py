# Copyright (c) 2020, Huawei Technologies.All rights reserved.
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

import torch

class PreLoader(object):
    def __init__(self, loader, device):
        self.device = device
        self.loader = iter(loader)
        self.stream = torch.npu.Stream()
        self.preload()

    def __len__(self):
        return len(self.loader)

    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_data = None
            return

        with torch.npu.stream(self.stream):
            for d in self.next_data:
                d['image_preprocess'] = d['image_preprocess'].to(self.device, non_blocking=True)
                if "instances" in d:
                    d['instances'] = d['instances'].to(self.device, non_blocking=True)

    def next(self):
        torch.npu.current_stream().wait_stream(self.stream)
        data = self.next_data
        self.preload()
        return data
