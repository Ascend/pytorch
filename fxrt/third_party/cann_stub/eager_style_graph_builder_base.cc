// Link-only stubs. Never use this library at runtime.
#include "../cann/include/ge/es_c_graph_builder.h"
#include "../cann/include/ge/es_c_tensor_holder.h"

extern "C" void fxrt_cann_stub_marker() {
  __builtin_trap();
}

// Only completes the private PImpl for link-only constructor/destructor stubs.
struct EsCGraphBuilder::EsCGraphBuilderImpl {};
auto EsCTensorHolder::GetProducer() -> ge::GNode & {
  __builtin_trap();
}
EsCGraphBuilder::EsCGraphBuilder(const char *, ge::GraphKind) {
  __builtin_trap();
}
EsCGraphBuilder::~EsCGraphBuilder() {
  __builtin_trap();
}
auto EsCGraphBuilder::GetTensorHolderFromNode(const ge::GNode &, int32_t) -> EsCTensorHolder * {
  __builtin_trap();
}
auto EsCGraphBuilder::AddGraphInput(int32_t, const ge::char_t *, const ge::char_t *, C_DataType, C_Format, const int64_t *, int64_t) -> EsCTensorHolder * {
  __builtin_trap();
}
auto EsCGraphBuilder::SetGraphOutput(EsCTensorHolder *, int64_t) -> ge::Status {
  __builtin_trap();
}
auto EsCGraphBuilder::GetGraph() -> ge::Graph * {
  __builtin_trap();
}
auto EsCGraphBuilder::BuildGraphAndReset() -> std::unique_ptr<ge::Graph> {
  __builtin_trap();
}
