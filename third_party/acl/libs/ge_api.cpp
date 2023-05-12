#include "ge/ge_api.h"

namespace ge {
Status Session::RunGraphWithStreamAsync(
    uint32_t graph_id,
    void* stream,
    const std::vector<Tensor>& inputs,
    std::vector<Tensor>& outputs) {

    sessionId_ = -1;
  return ge::SUCCESS;
}

Session::Session(const std::map<AscendString, AscendString>& options) {}

Session::~Session() {}

Status Session::AddGraph(uint32_t graphId, const Graph& graph) {
  return ge::SUCCESS;
}

Status GEInitialize(const std::map<AscendString, AscendString>& options) {
  return ge::SUCCESS;
}

Status GEFinalize() {
  return ge::SUCCESS;
}
} // namespace ge