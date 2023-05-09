#include "graph/graph.h"
#include "graph/ge_error_codes.h"

namespace ge {
graphStatus aclgrphGenerateForOp(const AscendString &op_type, const std::vector<TensorDesc> &inputs,
                                 const std::vector<TensorDesc> &outputs, Graph &graph);
}
