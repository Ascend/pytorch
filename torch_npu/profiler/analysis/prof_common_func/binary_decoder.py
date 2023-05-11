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
