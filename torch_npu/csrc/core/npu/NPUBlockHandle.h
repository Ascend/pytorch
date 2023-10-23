#pragma once
#include "torch_npu/csrc/core/npu/NPUMacros.h"

namespace c10_npu {
namespace NPUCachingAllocator {
// for torch2.X graph mode(non-raw malloc)
// We only expose a Block data of hgandle to the user for storing the application.
// The user does not need to perceive the actual data structure,
// but can query and release the data through handle.

/// @ingroup torch_npu
/// @brief Malloc Block from DeviceCachingAllocator
/// @param [in] size: size used for memory malloc
/// @param [in] stream: stream used for memory malloc
/// @return void*: block handle to the memory block
C10_NPU_API void* MallocBlock(size_t size, void *stream, int device = -1);

/// @ingroup torch_npu
/// @brief Free Block according to handle
/// @param [in] handle: the block handle to free
/// @return void
C10_NPU_API void FreeBlock(void *handle);

/// @ingroup torch_npu
/// @brief Get device memory address according to block handle
/// @param [in] handle: the block handle to query address
/// @return void*: the device memory address managed by block
C10_NPU_API void* GetBlockPtr(const void *handle);

/// @ingroup torch_npu
/// @brief Get device memory size according to handle
/// @param [in] handle: the block handle to query size
/// @return size: the device memory size managed by block
C10_NPU_API size_t GetBlockSize(const void *handle);
} // namespace NPUCachingAllocator
} // namespace c10_npu
