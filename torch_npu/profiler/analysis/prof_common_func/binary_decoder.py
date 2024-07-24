# Copyright (c) 2023, Huawei Technologies.
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

class BinaryDecoder:

    @classmethod
    def decode(cls, all_bytes: bytes, class_bean: any, struct_size: int) -> list:
        result_data = []
        all_bytes_len = len(all_bytes)
        start_index = 0
        while start_index + struct_size <= all_bytes_len:
            end_index = start_index + struct_size
            result_data.append(class_bean(all_bytes[start_index: end_index]))
            start_index = end_index
        return result_data
