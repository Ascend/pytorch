#pragma once

#include <memory>

namespace ge {
class Allocator;
class GeSession;
} // namespace ge

// C++-only side of the allocator bridge. Kept out of fx_allocator_bridge.h because that
// header is also read by the ABI=1 torch module, and a std::shared_ptr in its declarations
// would put a std:: type across the ABI boundary -- exactly what the C-only surface exists
// to prevent. Nothing in here may be included from the torch side.

namespace fx_converter {

// GE builds its own gert::Allocators from the ge::Allocator registered on the stream
// (GraphManager::GetExecuteGraphAllocators for OM2, GetUserAllocatorOrFixedBaseAllocator
// at run time), so FXRT does not construct or hand out an allocator bundle itself.

// Returns the process-wide allocator built from the registered ABI=1 callbacks.
// The public ES execution API registers this allocator against the invocation stream.
std::shared_ptr<ge::Allocator> FxTorchPoolDeviceAllocator();
bool FxRegisterExternalAllocatorForStream(ge::GeSession* session, const void* stream);
bool FxUnregisterExternalAllocatorForStream(ge::GeSession* session, const void* stream);

// Registers GE's device allocator with torch_npu's caching allocator, once per process.
// Implemented in the library's only ABI=1 translation unit (torch_pool_allocator.cc),
// which is where the torch_npu-side callbacks live; everything else here is ABI=0.
//
// Called before ES graph compilation so GE uses torch_npu's caching allocator when
// configured. Honours FXRT_GE_USE_TORCH_MEMPOOL=0.
void FxRegisterTorchPoolAllocator();

} // namespace fx_converter
