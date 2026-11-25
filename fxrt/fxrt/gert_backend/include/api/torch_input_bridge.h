#pragma once

#include <cstddef>
#include <cstdint>

namespace fx_converter {

// Caller-owned dimension buffers. Only PODs cross the Torch ABI=1 / GE ABI=0
// boundary. The collector copies metadata and never retains Python/Torch objects.
struct TorchInputDescriptor {
  int64_t* shape = nullptr;
  int64_t* stride = nullptr;
  int64_t* storage_shape = nullptr;
  size_t capacity = 0;
  size_t rank = 0;
  size_t storage_rank = 0;
  int64_t storage_offset = 0;
  int32_t format = 2;
  void* address = nullptr;
};

// py_sequence is a borrowed PyObject*. Requires the GIL. No exception crosses
// the ABI boundary; failure text is copied into the caller's error buffer.
extern "C" bool FxcCollectTensorInputs(
    void* py_sequence,
    TorchInputDescriptor* descriptors,
    size_t count,
    char* error,
    size_t error_capacity);

} // namespace fx_converter
