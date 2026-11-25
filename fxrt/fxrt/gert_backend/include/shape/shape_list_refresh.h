#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "exe_graph/execute_graph_builder.h"
#include "exe_graph/runtime/tensor.h"

namespace fx_converter {

bool RefreshShapeListInputs(
    const BuiltExecuteGraph& built,
    gert::Tensor** inputs,
    size_t input_num,
    const std::vector<std::pair<std::string, int64_t>>& scalar_syms);

} // namespace fx_converter
