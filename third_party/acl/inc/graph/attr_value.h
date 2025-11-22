/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_EXTERNAL_GRAPH_ATTR_VALUE_H_
#define INC_EXTERNAL_GRAPH_ATTR_VALUE_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "./ge_error_codes.h"
#include "ascend_string.h"
#include "tensor.h"
#include "types.h"

using std::make_shared;
using std::map;
using std::pair;
using std::string;
using std::to_string;
using std::unique_ptr;
using std::vector;

namespace ge {
class AttrValueImpl;
/*lint -e148*/
class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY AttrValue {
 public:
  using INT = int64_t;
  using FLOAT = float;
  using STR = std::string;

  AttrValue();
  ~AttrValue() = default;

  // GetValue, not list type
  template <typename T, typename DT>
  graphStatus GetValue(DT &val) const {
    T valGet;
    const auto status = GetValue(valGet);
    if (status != GRAPH_SUCCESS) {
      return status;
    }
    val = DT(valGet);
    return GRAPH_SUCCESS;
  }

  template <typename T, typename DT>
  static T CreateFrom(DT &&val) {
    return val;
  }

  graphStatus GetValue(AscendString &val);
  graphStatus SetAttrValue(const int64_t &attr_value) const;
  graphStatus GetAttrValue(int64_t &attr_value) const;
  graphStatus SetAttrValue(const float32_t &attr_value) const;
  graphStatus GetAttrValue(float32_t &attr_value) const;
  graphStatus SetAttrValue(const AscendString &attr_value) const;
  graphStatus GetAttrValue(AscendString &attr_value) const;
  graphStatus SetAttrValue(const bool &attr_value) const;
  graphStatus GetAttrValue(bool &attr_value) const;
  graphStatus SetAttrValue(const Tensor &attr_value) const;
  graphStatus GetAttrValue(Tensor &attr_value) const;
  graphStatus SetAttrValue(const std::vector<int64_t> &attr_value) const;
  graphStatus GetAttrValue(std::vector<int64_t> &attr_value) const;
  graphStatus SetAttrValue(const std::vector<float32_t> &attr_value) const;
  graphStatus GetAttrValue(std::vector<float32_t> &attr_value) const;
  graphStatus SetAttrValue(const std::vector<AscendString> &attr_values) const;
  graphStatus GetAttrValue(std::vector<AscendString> &attr_values) const;
  graphStatus SetAttrValue(const std::vector<bool> &attr_value) const;
  graphStatus GetAttrValue(std::vector<bool> &attr_value) const;
  graphStatus SetAttrValue(const std::vector<Tensor> &attr_value) const;
  graphStatus GetAttrValue(std::vector<Tensor> &attr_value) const;
  graphStatus SetAttrValue(const std::vector<std::vector<int64_t>> &attr_value) const;
  graphStatus GetAttrValue(std::vector<std::vector<int64_t>> &attr_value) const;
  graphStatus SetAttrValue(const std::vector<ge::DataType> &attr_value) const;
  graphStatus GetAttrValue(std::vector<ge::DataType> &attr_value) const;
  graphStatus SetAttrValue(const ge::DataType &attr_value) const;
  graphStatus GetAttrValue(ge::DataType &attr_value) const;

  std::shared_ptr<AttrValueImpl> impl;

 private:
  friend class AttrValueImpl;
#define VALUE_SET_GET_DEC(DT) graphStatus GetValue(DT &val) const;
  VALUE_SET_GET_DEC(AttrValue::STR)
  VALUE_SET_GET_DEC(AttrValue::INT)
  VALUE_SET_GET_DEC(AttrValue::FLOAT)
#undef VALUE_SET_GET_DEC
};
/*lint +e148*/
}  // namespace ge
#endif  // INC_EXTERNAL_GRAPH_ATTR_VALUE_H_
