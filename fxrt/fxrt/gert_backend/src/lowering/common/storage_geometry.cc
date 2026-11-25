#include "lowering/common/storage_geometry.h"

#include <limits>

#include "lowering/common/view_utils.h"

namespace fx_converter {
namespace {
bool BoolAttr(const fxrt::ir::NodePtr &node, const char *name) {
  const auto it = node->attrs.find(name);
  return it != node->attrs.end() && it->second != nullptr && it->second->IsBool() && it->second->ToBool();
}
}  // namespace

bool ReadViewOffsetSemantics(const fxrt::ir::NodePtr &node, ViewOffsetSemantics *semantics) {
  if (node == nullptr || semantics == nullptr) return false;
  const auto it = node->attrs.find("fxrt_as_strided_offset_semantics");
  if (it != node->attrs.end()) {
    if (it->second == nullptr || !it->second->IsInt()) return false;
    const auto value = it->second->ToInt();
    if (value < static_cast<int64_t>(ViewOffsetSemantics::kAbsolute) ||
        value > static_cast<int64_t>(ViewOffsetSemantics::kInherit)) return false;
    *semantics = static_cast<ViewOffsetSemantics>(value);
    return true;
  }
  // Compatibility for IR callers that predate the single semantics attribute.
  const bool relative = BoolAttr(node, "fxrt_as_strided_immediate_relative_offset");
  const bool feed_relative = BoolAttr(node, "fxrt_compiled_kernel_relative_offset");
  if (relative && feed_relative) return false;
  *semantics = relative ? ViewOffsetSemantics::kImmediateRelative
                       : feed_relative ? ViewOffsetSemantics::kFeedRelative : ViewOffsetSemantics::kAbsolute;
  return true;
}

bool ComposeViewStorageGeometry(const TensorRef &source, ViewOffsetSemantics semantics,
                                int64_t offset, const fxrt::ir::SymbolicExprPtr &offset_expr,
                                StorageGeometry *result) {
  if (result == nullptr) return false;
  StorageGeometry geometry;
  geometry.anchor = source;
  geometry.offset = offset;
  geometry.offset_expr = offset_expr;
  if (semantics == ViewOffsetSemantics::kInherit) {
    geometry.offset = source.storage_offset;
    geometry.offset_expr = source.storage_offset_expr;
  } else if (semantics == ViewOffsetSemantics::kImmediateRelative) {
    if (source.storage_offset_expr != nullptr || offset_expr != nullptr) {
      const auto parent = source.storage_offset_expr != nullptr
                              ? source.storage_offset_expr
                              : fxrt::ir::MakeIntrusive<fxrt::ir::SymbolicConst>(source.storage_offset);
      const auto delta = offset_expr != nullptr ? offset_expr
                          : fxrt::ir::MakeIntrusive<fxrt::ir::SymbolicConst>(offset);
      geometry.offset_expr = parent + delta;
      geometry.offset = 0;
    } else {
      // A signed increment may move backwards within the same storage. Check
      // the composed absolute term and signed overflow, not the delta alone.
      if (source.storage_offset < 0 || offset < -source.storage_offset ||
          (offset > 0 && offset > std::numeric_limits<int64_t>::max() - source.storage_offset)) return false;
      geometry.offset += source.storage_offset;
    }
  }
  if (semantics != ViewOffsetSemantics::kAbsolute) geometry.runtime_offset = source.runtime_storage_offset;
  if (geometry.offset_expr == nullptr && geometry.offset < 0) return false;
  *result = std::move(geometry);
  return true;
}

bool ResolveViewStorageGeometry(const fxrt::ir::NodePtr &node, const LoweringContext &ctx,
                                ViewOffsetSemantics semantics, int64_t offset,
                                const fxrt::ir::SymbolicExprPtr &offset_expr, StorageGeometry *result) {
  TensorRef immediate;
  if (!ResolveInputTensor(node, ctx, &immediate, nullptr) ||
      !ComposeViewStorageGeometry(immediate, semantics, offset, offset_expr, result)) return false;
  if (semantics == ViewOffsetSemantics::kAbsolute || semantics == ViewOffsetSemantics::kFeedRelative) {
    TensorRef anchor;
    if (!ResolveAsStridedBaseTensor(node, ctx, &anchor, nullptr, nullptr)) return false;
    result->anchor = std::move(anchor);
    // The immediate input decides inheritance. Walking to an earlier feed
    // must not resurrect an origin terminated by a public absolute view.
  }
  return true;
}

void ConsumeStorageOffset(const TensorRef &source, TensorRef *rebased) {
  rebased->storage_offset = 0;
  rebased->storage_offset_expr = nullptr;
  rebased->runtime_storage_offset = {};
  rebased->allocation = source.allocation.Valid() ? source.allocation : source.unpacked.addr;
}

}  // namespace fx_converter
