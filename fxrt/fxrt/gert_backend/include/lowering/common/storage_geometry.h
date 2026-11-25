#pragma once

#include "lowering/op_lowering.h"

namespace fx_converter {

// Wire values match fxrt.view_offset.ViewOffsetSemantics. Only the converter
// entry interprets provenance; consumers see absolute geometry on an anchor.
enum class ViewOffsetSemantics : int64_t {
  kAbsolute = 0,
  kImmediateRelative = 1,
  kFeedRelative = 2,
  kInherit = 3,
};

// The address/capacity and offset must travel together. `offset_expr` replaces
// `offset`; runtime_offset is an additional per-Execute feed-origin term.
// Allocation ownership is carried by anchor, not used to guess a storage base.
struct StorageGeometry {
  TensorRef anchor;
  int64_t offset = 0;
  fxrt::ir::SymbolicExprPtr offset_expr;
  TensorEndpoint runtime_offset;
};

bool ReadViewOffsetSemantics(const fxrt::ir::NodePtr& node, ViewOffsetSemantics* semantics);

bool ComposeViewStorageGeometry(
    const TensorRef& source,
    ViewOffsetSemantics semantics,
    int64_t offset,
    const fxrt::ir::SymbolicExprPtr& offset_expr,
    StorageGeometry* result);

// Resolve against actual lowered storage. Relative/inherited views keep the
// immediate anchor. Absolute views may select a larger header only when it
// has the same address; never walk through an allocation or a rebased address.
bool ResolveViewStorageGeometry(
    const fxrt::ir::NodePtr& node,
    const LoweringContext& ctx,
    ViewOffsetSemantics semantics,
    int64_t offset,
    const fxrt::ir::SymbolicExprPtr& offset_expr,
    StorageGeometry* result);

// Address adaptation consumes the complete offset exactly once. Keep the
// borrowed allocation alive while making the new address the local origin.
void ConsumeStorageOffset(const TensorRef& source, TensorRef* rebased);

} // namespace fx_converter
