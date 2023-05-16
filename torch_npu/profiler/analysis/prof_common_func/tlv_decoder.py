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
import collections
import struct
from warnings import warn

from ..prof_common_func.constant import Constant


class TLVDecoder:
    T_LEN = 2
    L_LEN = 4

    @classmethod
    def decode(cls, all_bytes: bytes, class_bean: any, constant_struct_size: int) -> list:
        result_data = []
        records = cls.tlv_list_decode(all_bytes)
        for record in records:
            if constant_struct_size > len(record):
                warn("The collected data has been lost")
                continue
            constant_bytes = record[0: constant_struct_size]
            tlv_fields = cls.tlv_list_decode(record[constant_struct_size:], is_field=True)
            tlv_fields[Constant.CONSTANT_BYTES] = constant_bytes
            result_data.append(class_bean(tlv_fields))
        return result_data

    @classmethod
    def tlv_list_decode(cls, tlv_bytes: bytes, is_field: bool = False) -> dict:
        result_data = {} if is_field else []
        index = 0
        all_bytes_len = len(tlv_bytes)
        while index < all_bytes_len:
            if index + cls.T_LEN > all_bytes_len:
                warn("The collected data has been lost")
                break
            type_id = struct.unpack("<H", tlv_bytes[index: index + cls.T_LEN])[0]
            index += cls.T_LEN
            if index + cls.L_LEN > all_bytes_len:
                warn("The collected data has been lost")
                break
            value_len = struct.unpack("<I", tlv_bytes[index: index + cls.L_LEN])[0]
            index += cls.L_LEN
            if index + value_len > all_bytes_len:
                warn("The collected data has been lost")
                break
            value = tlv_bytes[index: index + value_len]
            index += value_len
            if is_field:
                result_data[type_id] = bytes.decode(value)
            else:
                result_data.append(value)
        return result_data
