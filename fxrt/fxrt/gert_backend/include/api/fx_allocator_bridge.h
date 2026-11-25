#pragma once

#include <cstddef>

// ABI=0 <-> ABI=1 boundary for the GE device allocator.
//
// fx_converter is compiled with _GLIBCXX_USE_CXX11_ABI=0 (GE host libs are ABI=0) while
// torch_npu is ABI=1. Only plain C function pointers and PODs may cross that boundary --
// no std::function, no std::string, no c10::DataPtr. These hooks are the whole surface.
//
// torch_npu does not publish these; the ABI=1 translation unit inside this library
// (src/model_v2/torch_pool_allocator.cc) calls NPUCachingAllocator directly and registers
// the callbacks through FxSetExternalAllocator. torch_npu therefore needs no knowledge of
// this bridge at all.
//
// The typedefs deliberately match ge::AllocFunc / ge::FreeFunc / ge::GetAddrFromBlockFunc
// (framework/memory/allocator_desc.h) so the descriptor converts without a shim.

namespace fx_converter {

using FxExternalAllocFn = void* (*)(void* obj, size_t size);
using FxExternalFreeFn = void (*)(void* obj, void* block);
using FxExternalGetAddrFn = void* (*)(void* block);

struct FxExternalAllocatorDesc {
  FxExternalAllocFn alloc_fn;
  FxExternalFreeFn free_fn;
  // Required, not optional: ExternalAllocator::Malloc calls
  // ge::AllocatorDesc::get_addr_from_block_func unconditionally
  // (runtime/v2/kernel/memory/external_allocator.cc:24), so leaving it null crashes on
  // the first allocation rather than falling back. FxSetExternalAllocator fills in an
  // identity function when this field is null.
  FxExternalGetAddrFn get_addr_fn;
  // Passed back to every callback untouched. Must be a POD (usually nullptr): it stays
  // inside this ABI=0 library and is never interpreted there.
  void* obj;
};

// Registers the device allocator. Must be called before the first runner Init -- i.e.
// before convert() builds a ModelV2 or OM2 executor. Idempotent: a later call replaces
// the previous descriptor, so a second session in the same process is harmless.
// A null desc (or a null alloc_fn/free_fn) is ignored, which keeps the default GE pool.
extern "C" __attribute__((visibility("default"))) void FxSetExternalAllocator(const FxExternalAllocatorDesc* desc);

// Returns true once a usable descriptor has been registered.
extern "C" __attribute__((visibility("default"))) bool FxHasExternalAllocator();

// Copies the registered descriptor into *out and returns true; false (leaving *out
// untouched) when nothing usable was registered. Lets a consumer hand the callbacks to a
// library that must build the allocator object itself, where the
// bundle has to live in libgert.so so its typeinfo matches the kernel that casts it.
extern "C" __attribute__((visibility("default"))) bool FxGetExternalAllocatorDesc(FxExternalAllocatorDesc* out);

// There is no bundle accessor: gert::Allocators is a GE-internal type, and GE builds it
// itself from the AllocatorPtr registered on the stream. The allocator handed to GE is a
// subclass of the public ge::Allocator (ge/ge_allocator.h), so an in-graph fallback would
// only duplicate GE's own wrapping.

} // namespace fx_converter
