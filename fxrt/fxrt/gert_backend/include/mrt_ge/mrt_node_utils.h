#pragma once

#include <cstdint>
#include <vector>

#include "graph/types.h"
#include "ir/common/dtype.h"
#include "ir/graph.h"
#include "ir/tensor/tensor.h"

namespace fx_converter {

// Concrete dims from an MRT tensor; non-positive dims become -1 (dynamic).
// Returns {-1,-1} if the tensor is null or has no usable shape.
std::vector<int64_t> ShapeOfTensor(const fxrt::ir::TensorPtr& tensor);
// ShapeOfTensor(node->output) when the output is a Tensor; otherwise {-1,-1}.
std::vector<int64_t> ShapeOfNode(const fxrt::ir::NodePtr& node);

// Map MRT DataType -> GE DataType. Unknown / unsupported -> DT_FLOAT with a warning path
// left to the caller via *mapped (false means fallback was used).
ge::DataType MrtToGeDataType(fxrt::ir::DataType dtype, bool* mapped = nullptr);

// GE dtype of a tensor; falls back to DT_FLOAT if missing / unmapped.
ge::DataType DtypeOfTensor(const fxrt::ir::TensorPtr& tensor);
// DtypeOfTensor(node->output) when the output is a Tensor; otherwise DT_FLOAT.
ge::DataType DtypeOfNode(const fxrt::ir::NodePtr& node);

// Tensor slots of node->output. A Tensor is one slot; a Tuple is one slot per
// Tensor element. Missing / non-tensor elements are nullptr.
std::vector<fxrt::ir::TensorPtr> TensorsOfNode(const fxrt::ir::NodePtr& node);

// Check if a tensor node is a model feed: non-empty / symbolic shape, or a rank-0
// tensor that a compute node consumes as a real input (not a Dynamo-lifted scalar
// whose only consumer is the aten.item fold). The graph gives the consumer view.
bool IsModelFeedTensor(const fxrt::ir::Graph& graph, const fxrt::ir::NodePtr& node);

} // namespace fx_converter
