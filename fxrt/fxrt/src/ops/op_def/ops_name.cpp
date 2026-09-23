#include "ops/op_def/ops_name.h"

#include <unordered_map>

#include "common/logger.h"

namespace fxrt {
namespace ops {
#define OP(O) {#O, Op_##O},
std::unordered_map<std::string_view, Op> _opNames{
#include "ops/op_def/ops.list"
};
#undef OP

Op MatchOp(const char* op) {
  if (_opNames.count(op) == 0) {
    RT_GLOG(ERROR) << "Not found op with name '" << op << "'";
    for (auto it = _opNames.cbegin(); it != _opNames.cend(); ++it) {
      RT_GLOG(ERROR) << "\t#" << it->first << ", " << it->second;
    }
    exit(EXIT_FAILURE);
  }
  return _opNames[op];
}

#define OP(O) #O,
const char* _opStr[] = {
#include "ops/op_def/ops.list"
    "End",
};
#undef OP

const char* ToStr(const Op op) {
  return _opStr[op];
}
} // namespace ops
} // namespace fxrt
