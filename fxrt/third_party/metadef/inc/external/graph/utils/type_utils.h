/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_GRAPH_UTILS_TYPE_UTILS_H_
#define INC_GRAPH_UTILS_TYPE_UTILS_H_

#include <string>
#include "graph/types.h"
#include "graph/ascend_string.h"

namespace ge {
class TypeUtils {
 public:
  // todo: add ATTRIBUTED_DEPRECATED for std::string interface
  static std::string DataTypeToSerialString(const DataType data_type);
  static DataType SerialStringToDataType(const std::string& str);
  static std::string FormatToSerialString(const Format format);
  static Format SerialStringToFormat(const std::string& str);
  static Format DataFormatToFormat(const std::string& str);
  static bool GetDataTypeLength(const ge::DataType data_type, uint32_t& length);

  static AscendString DataTypeToAscendString(const DataType& data_type);
  static DataType AscendStringToDataType(const AscendString& str);
  static AscendString FormatToAscendString(const Format& format);
  static Format AscendStringToFormat(const AscendString& str);
  static Format DataFormatToFormat(const AscendString& str);
};
} // namespace ge
#endif // INC_GRAPH_UTILS_TYPE_UTILS_H_
