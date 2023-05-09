#ifndef INC_EXTERNAL_GRAPH_ATTR_VALUE_H_
#define INC_EXTERNAL_GRAPH_ATTR_VALUE_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "./ge_error_codes.h"
#include "ascend_string.h"

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
    auto status = GetValue(valGet);
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

  std::shared_ptr<AttrValueImpl> impl;

 private:
#define VALUE_SET_GET_DEC(DT) graphStatus GetValue(DT &val) const;
  VALUE_SET_GET_DEC(AttrValue::STR)
  VALUE_SET_GET_DEC(AttrValue::INT)
  VALUE_SET_GET_DEC(AttrValue::FLOAT)
#undef VALUE_SET_GET_DEC
};
/*lint +e148*/
}  // namespace ge
#endif  // INC_EXTERNAL_GRAPH_ATTR_VALUE_H_
