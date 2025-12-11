import struct
from warnings import warn

from ._constant import Constant

__all__ = []


class TLVDecoder:
    T_LEN = 2
    L_LEN = 4
    undecodable_msg_count = 0

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
        if cls.undecodable_msg_count > 0:
            warn_msg = ("The collected data can't be decode by bytes.decode, "
                        "find {} items of undecoded tlv data").format(cls.undecodable_msg_count)
            warn(warn_msg)
            cls.undecodable_msg_count = 0
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
                try:
                    result_data[type_id] = bytes.decode(value)
                except UnicodeDecodeError:
                    cls.undecodable_msg_count += 1
                    result_data[type_id] = 'N/A'
            else:
                result_data.append(value)
        return result_data
