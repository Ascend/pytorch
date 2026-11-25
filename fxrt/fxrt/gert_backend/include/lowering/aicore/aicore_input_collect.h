#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "lowering/aicore/aicore_op_work.h"
#include "lowering/catalog_attrs.h"
#include "lowering/op_lowering.h"
#include "register/op_catalog.h"

namespace fx_converter {

// GE/TBE AsStrided proto is x, size, stride, storage_offset. Launch always passes the
// offset pointer (including 0); 910B simplifiedKey omits that optional slot.
constexpr const char kAsStridedOppType[] = "AsStrided";
constexpr const char kAsStridedInputNames[] = "x,size,stride,storage_offset";
constexpr size_t kAsStridedLaunchInputCount = 4;
constexpr size_t kAsStridedSelectInputCount = 3;

// Non-contiguous as_strided claims Path C (prefix → Slice, else GE AsStrided);
// contiguous layouts stay metadata views for ViewAlias.
bool IsAsStridedPathC(const fxrt::ir::NodePtr& node);

bool CollectAiCoreInputs(LoweringContext& ctx, AiCoreOpWork& work, std::string* err);

} // namespace fx_converter
