#ifndef BUILD_LIBTORCH

#include "torch_npu/csrc/inductor/static_launcher/runtime.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

#include <ATen/ATen.h>
#include <torch/csrc/Exceptions.h>

#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/interface/AclInterface.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace torch_npu::inductor {
namespace {

size_t AlignOffset(size_t offset, size_t alignment) {
  TORCH_INTERNAL_ASSERT(alignment != 0);
  return (offset + alignment - 1) / alignment * alignment;
}

void WriteBytes(std::vector<uint8_t>& buffer, size_t offset, const void* data, size_t size) {
  TORCH_INTERNAL_ASSERT(offset + size <= buffer.size());
  std::memcpy(buffer.data() + offset, data, size);
}

StaticNpuArgKind ParseArgKind(const std::string& kind) {
  if (kind == "tensor") {
    return StaticNpuArgKind::Tensor;
  }
  if (kind == "i8") {
    return StaticNpuArgKind::I8;
  }
  if (kind == "i16") {
    return StaticNpuArgKind::I16;
  }
  if (kind == "i32") {
    return StaticNpuArgKind::I32;
  }
  if (kind == "i64") {
    return StaticNpuArgKind::I64;
  }
  if (kind == "u8") {
    return StaticNpuArgKind::U8;
  }
  if (kind == "u16") {
    return StaticNpuArgKind::U16;
  }
  if (kind == "u32") {
    return StaticNpuArgKind::U32;
  }
  if (kind == "u64") {
    return StaticNpuArgKind::U64;
  }
  if (kind == "f32") {
    return StaticNpuArgKind::F32;
  }
  if (kind == "f64") {
    return StaticNpuArgKind::F64;
  }
  if (kind == "bool") {
    return StaticNpuArgKind::Bool;
  }
  TORCH_CHECK(false, "Unsupported NPU static launcher argument kind: ", kind);
  return StaticNpuArgKind::Tensor;
}

// Keep sizes and alignments synchronized with the generated packed structs in:
//   Triton-Ascend 3.2.2: third_party/ascend/backend/driver.py
//                        generate_npu_wrapper_src() / _ty_to_cpp()
//   Triton-Ascend 3.6.0: third_party/ascend/backend/driver.py
//                        make_launcher() / ty_to_cpp()
// The generated launchers use sizeof(C++ type) for field size, align pointers
// and 64-bit scalars to 8 bytes, and align every smaller scalar to 4 bytes.
size_t ArgSize(StaticNpuArgKind kind) {
  switch (kind) {
    case StaticNpuArgKind::Tensor:
      return sizeof(std::uintptr_t);
    case StaticNpuArgKind::I8:
      return sizeof(int8_t);
    case StaticNpuArgKind::I16:
      return sizeof(int16_t);
    case StaticNpuArgKind::I32:
    case StaticNpuArgKind::Bool:
      return sizeof(int32_t);
    case StaticNpuArgKind::I64:
      return sizeof(int64_t);
    case StaticNpuArgKind::U8:
      return sizeof(uint8_t);
    case StaticNpuArgKind::U16:
      return sizeof(uint16_t);
    case StaticNpuArgKind::U32:
      return sizeof(uint32_t);
    case StaticNpuArgKind::U64:
      return sizeof(uint64_t);
    case StaticNpuArgKind::F32:
      return sizeof(float);
    case StaticNpuArgKind::F64:
      return sizeof(double);
  }
  TORCH_INTERNAL_ASSERT(false);
  return 0;
}

size_t ArgAlignment(StaticNpuArgKind kind) {
  switch (kind) {
    case StaticNpuArgKind::Tensor:
    case StaticNpuArgKind::I64:
    case StaticNpuArgKind::U64:
    case StaticNpuArgKind::F64:
      return 8;
    default:
      // Triton-Ascend's generated packed struct explicitly aligns every
      // non-64-bit scalar field to four bytes, including i8/i16.
      return 4;
  }
}

template <typename T>
void WriteScalar(std::vector<uint8_t>& packed, const StaticNpuArgLayout& layout, py::handle value) {
  T scalar = py::cast<T>(value);
  WriteBytes(packed, layout.offset, &scalar, sizeof(scalar));
}

void WriteArgument(std::vector<uint8_t>& packed, const StaticNpuArgLayout& layout, py::handle value) {
  switch (layout.kind) {
    case StaticNpuArgKind::Tensor: {
      void* pointer = nullptr;
      if (value.is_none()) {
        pointer = nullptr;
      } else if (PyLong_Check(value.ptr())) {
        pointer = PyLong_AsVoidPtr(value.ptr());
        TORCH_CHECK(!PyErr_Occurred(), "NPU static launcher pointer conversion failed");
      } else {
        pointer = py::cast<at::Tensor>(value).data_ptr();
      }
      const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(pointer);
      WriteBytes(packed, layout.offset, &address, sizeof(address));
      return;
    }
    case StaticNpuArgKind::I8:
      return WriteScalar<int8_t>(packed, layout, value);
    case StaticNpuArgKind::I16:
      return WriteScalar<int16_t>(packed, layout, value);
    case StaticNpuArgKind::I32:
      return WriteScalar<int32_t>(packed, layout, value);
    case StaticNpuArgKind::I64:
      return WriteScalar<int64_t>(packed, layout, value);
    case StaticNpuArgKind::U8:
      return WriteScalar<uint8_t>(packed, layout, value);
    case StaticNpuArgKind::U16:
      return WriteScalar<uint16_t>(packed, layout, value);
    case StaticNpuArgKind::U32:
      return WriteScalar<uint32_t>(packed, layout, value);
    case StaticNpuArgKind::U64:
      return WriteScalar<uint64_t>(packed, layout, value);
    case StaticNpuArgKind::F32:
      return WriteScalar<float>(packed, layout, value);
    case StaticNpuArgKind::F64:
      return WriteScalar<double>(packed, layout, value);
    case StaticNpuArgKind::Bool: {
      int32_t scalar = py::cast<bool>(value) ? 1 : 0;
      WriteBytes(packed, layout.offset, &scalar, sizeof(scalar));
      return;
    }
  }
  TORCH_INTERNAL_ASSERT(false);
}

uint32_t ValidateGrid(uint32_t grid0, uint32_t grid1, uint32_t grid2) {
  const uint32_t grid[] = {grid0, grid1, grid2};
  uint64_t blocks = 1;
  for (size_t index = 0; index < 3; ++index) {
    TORCH_CHECK(grid[index] > 0, "NPU static launcher grid must be positive");
    TORCH_CHECK(
        grid[index] <= static_cast<uint32_t>(std::numeric_limits<int32_t>::max()),
        "NPU static launcher grid dimension exceeds int32 max");
    blocks *= grid[index];
    TORCH_CHECK(blocks <= std::numeric_limits<uint32_t>::max(), "NPU static launcher grid product exceeds uint32 max");
  }
  return static_cast<uint32_t>(blocks);
}

} // namespace

bool StaticNpuKernel::IsSupported() {
  return c10_npu::acl::IsExistAclrtLaunchKernelWithHostArgs();
}

std::shared_ptr<StaticNpuKernel> StaticNpuKernel::Load(
    const py::bytes& binary,
    const std::string& kernelName,
    int device,
    const std::vector<std::string>& argKinds,
    const std::string& mixMode,
    bool enableSimt,
    uint64_t sharedMemDynamicSize,
    bool isPureSimt,
    bool targetSupportFfts,
    uint64_t trailingPointerCount) {
  TORCH_CHECK(
      IsSupported(),
      "CANN runtime lacks APIs required by the NPU static launcher, "
      "including aclrtLaunchKernelWithHostArgs");
  TORCH_CHECK(!kernelName.empty(), "NPU static launcher kernel name is empty");
  TORCH_CHECK(mixMode == "aiv" || mixMode == "aic", "Unsupported mix mode: ", mixMode);
  TORCH_CHECK(
      sharedMemDynamicSize <= std::numeric_limits<uint32_t>::max(), "shared_mem_dynamic_size exceeds uint32 max");
  TORCH_CHECK(!isPureSimt || enableSimt, "is_pure_simt requires enable_simt");
  TORCH_CHECK(
      trailingPointerCount <= 3, "Unsupported NPU static launcher trailing pointer count: ", trailingPointerCount);

  std::string binaryData = binary;
  TORCH_CHECK(!binaryData.empty(), "NPU static launcher binary is empty");

  auto kernel = std::shared_ptr<StaticNpuKernel>(new StaticNpuKernel());
  kernel->kernelName_ = kernelName;
  kernel->device_ = device;
  kernel->enableSimt_ = enableSimt;
  kernel->sharedMemDynamicSize_ = static_cast<uint32_t>(sharedMemDynamicSize);
  kernel->isPureSimt_ = isPureSimt;
  kernel->targetSupportFfts_ = targetSupportFfts;
  kernel->trailingPointerCount_ = static_cast<size_t>(trailingPointerCount);
  kernel->BuildPackedLayout(argKinds);

  c10_npu::NPUGuard deviceGuard(device);
  TORCH_CHECK(
      aclrtGetCurrentContext(&kernel->context_) == ACL_SUCCESS,
      "aclrtGetCurrentContext failed while loading NPU static kernel");

  uint32_t magic = mixMode == "aiv" ? ACL_RT_BINARY_MAGIC_ELF_VECTOR_CORE : ACL_RT_BINARY_MAGIC_ELF_AICORE;
  aclrtBinaryLoadOption options[] = {
      {.type = ACL_RT_BINARY_LOAD_OPT_LAZY_LOAD, .value = {.isLazyLoad = 0}},
      {.type = ACL_RT_BINARY_LOAD_OPT_MAGIC, .value = {.magic = magic}},
  };
  aclrtBinaryLoadOptions loadOptions = {
      .options = options,
      .numOpt = sizeof(options) / sizeof(options[0]),
  };
  aclError result =
      c10_npu::acl::AclrtBinaryLoadFromData(binaryData.data(), binaryData.size(), &loadOptions, &kernel->binaryHandle_);
  TORCH_CHECK(result == ACL_SUCCESS, "aclrtBinaryLoadFromData failed for ", kernelName, ": ", static_cast<int>(result));

  result = c10_npu::acl::AclrtBinaryGetFunction(kernel->binaryHandle_, kernelName.c_str(), &kernel->functionHandle_);
  if (result != ACL_SUCCESS) {
    c10_npu::acl::AclrtBinaryUnLoad(kernel->binaryHandle_);
    kernel->binaryHandle_ = nullptr;
    TORCH_CHECK(false, "aclrtBinaryGetFunction failed for ", kernelName, ": ", static_cast<int>(result));
  }

  if (targetSupportFfts) {
    result = c10_npu::acl::AclrtGetHardwareSyncAddr(&kernel->fftsAddress_);
    if (result != ACL_SUCCESS || kernel->fftsAddress_ == nullptr) {
      c10_npu::acl::AclrtBinaryUnLoad(kernel->binaryHandle_);
      kernel->binaryHandle_ = nullptr;
      kernel->functionHandle_ = nullptr;
      TORCH_CHECK(false, "aclrtGetHardwareSyncAddr failed for ", kernelName, ": ", static_cast<int>(result));
    }
  }
  return kernel;
}

StaticNpuKernel::~StaticNpuKernel() {
  try {
    Close();
  } catch (...) {
  }
}

void StaticNpuKernel::BuildPackedLayout(const std::vector<std::string>& argKinds) {
  // Host-argument order mirrored from the packed `args` struct and the
  // `reserve_slot` path in the Triton-Ascend driver.py functions referenced
  // above. Optional pointer slots are zero-filled when the feature is unused.
  //
  // 3.2.2:
  //   [ffts?] [sync_lock*, workspace* unless pure SIMT]
  //   [runtime args] [gridX:i32, gridY:i32, gridZ:i32]
  //   [DTData* if device print is enabled; static launch rejects this mode]
  // 3.6.0:
  //   [ffts?] [sync_lock*, workspace* unless pure SIMT] [runtime args]
  //   [gridX:i32, gridY:i32, gridZ:i32]
  //   [global_scratch*, profile_scratch* if pure SIMT] [DTData*]
  //
  // Field size is defined by ArgSize(); field alignment by ArgAlignment();
  // hidden pointers use sizeof/alignof(void*), grid fields use int32_t, and
  // the final buffer is padded to the largest field alignment.
  size_t offset = 0;
  size_t packedAlignment = 4;
  if (targetSupportFfts_) {
    packedAlignment = std::max(packedAlignment, alignof(void*));
    offset = AlignOffset(offset, alignof(void*));
    fftsOffset_ = offset;
    offset += sizeof(void*);
  }
  if (!isPureSimt_) {
    packedAlignment = std::max(packedAlignment, alignof(void*));
    for (int index = 0; index < 2; ++index) {
      offset = AlignOffset(offset, alignof(void*));
      offset += sizeof(void*);
    }
  }
  argLayouts_.reserve(argKinds.size());
  for (const auto& argKind : argKinds) {
    const auto kind = ParseArgKind(argKind);
    const size_t alignment = ArgAlignment(kind);
    packedAlignment = std::max(packedAlignment, alignment);
    offset = AlignOffset(offset, alignment);
    argLayouts_.push_back({kind, offset});
    offset += ArgSize(kind);
  }
  for (size_t index = 0; index < 3; ++index) {
    offset = AlignOffset(offset, 4);
    gridOffsets_[index] = offset;
    offset += sizeof(int32_t);
  }
  for (size_t index = 0; index < trailingPointerCount_; ++index) {
    packedAlignment = std::max(packedAlignment, alignof(void*));
    offset = AlignOffset(offset, alignof(void*));
    offset += sizeof(void*);
  }
  packedArgsSize_ = AlignOffset(offset, packedAlignment);
}

void StaticNpuKernel::Launch(
    uint32_t grid0,
    uint32_t grid1,
    uint32_t grid2,
    uint64_t streamValue,
    const py::sequence& args) {
  const auto argCount = static_cast<size_t>(py::len(args));
  TORCH_CHECK(
      argCount == argLayouts_.size(),
      "NPU static launcher argument count mismatch: ",
      argCount,
      " vs ",
      argLayouts_.size());
  auto stream = reinterpret_cast<aclrtStream>(streamValue);
  TORCH_CHECK(stream != nullptr, "NPU static launcher stream is null");
  const uint32_t blockCount = ValidateGrid(grid0, grid1, grid2);

  std::vector<uint8_t> packed(packedArgsSize_, 0);
  if (targetSupportFfts_) {
    WriteBytes(packed, fftsOffset_, &fftsAddress_, sizeof(fftsAddress_));
  }
  for (size_t index = 0; index < argCount; ++index) {
    WriteArgument(packed, argLayouts_[index], args[index]);
  }
  const int32_t signedGrid[] = {
      static_cast<int32_t>(grid0),
      static_cast<int32_t>(grid1),
      static_cast<int32_t>(grid2),
  };
  for (size_t index = 0; index < 3; ++index) {
    WriteBytes(packed, gridOffsets_[index], &signedGrid[index], sizeof(signedGrid[index]));
  }

  std::shared_ptr<StaticNpuKernel> owner;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    TORCH_CHECK(!closed_, "NPU static launcher kernel is closed");
    owner = shared_from_this();
  }
  auto launchCall = [owner, packed = std::move(packed), blockCount, stream]() mutable {
    aclrtLaunchKernelAttr launchAttr = {};
    aclrtLaunchKernelCfg launchConfig = {};
    aclrtLaunchKernelCfg* launchConfigPtr = nullptr;
    if (owner->enableSimt_) {
      launchAttr.id = ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE;
      launchAttr.value.dynUBufSize = owner->sharedMemDynamicSize_;
      launchConfig.attrs = &launchAttr;
      launchConfig.numAttrs = 1;
      launchConfigPtr = &launchConfig;
    }
    return static_cast<int>(c10_npu::acl::AclrtLaunchKernelWithHostArgs(
        owner->functionHandle_, blockCount, stream, launchConfigPtr, packed.data(), packed.size(), nullptr, 0));
  };
  at_npu::native::OpCommand::RunOpApiV2(kernelName_, launchCall);
}

void StaticNpuKernel::Close() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (closed_) {
    return;
  }
  if (binaryHandle_ == nullptr) {
    closed_ = true;
    return;
  }

  // CANN 9.0.0 and 9.2.0-beta.2 can execute stale kernel code after unloading
  // and reusing its device address: https://gitcode.com/cann/runtime/issues/1021
  // Keep binary registrations/device code alive until runtime teardown, at the
  // cost of retained memory. Re-enable unloading after the CANN fix is verified.
  constexpr bool enableBinaryUnload = false;
  if (!enableBinaryUnload) {
    binaryHandle_ = nullptr;
    functionHandle_ = nullptr;
    closed_ = true;
    return;
  }

  c10_npu::NPUGuard deviceGuard(device_);
  aclrtContext previousContext = nullptr;
  aclError contextResult = aclrtGetCurrentContext(&previousContext);
  TORCH_CHECK(contextResult == ACL_SUCCESS, "aclrtGetCurrentContext failed while unloading NPU static kernel");
  const bool contextSwitched = context_ != nullptr && previousContext != context_;
  if (contextSwitched) {
    TORCH_CHECK(
        aclrtSetCurrentContext(context_) == ACL_SUCCESS,
        "aclrtSetCurrentContext failed while unloading NPU static kernel");
  }
  // Match the community static launcher lifecycle: benchmark synchronization
  // happens before autotune losers are released, and the selected launcher's
  // owner keeps its binary loaded. A device-wide synchronization here would
  // add an unnecessary barrier to every explicit cleanup.
  aclError unloadResult = c10_npu::acl::AclrtBinaryUnLoad(binaryHandle_);
  aclError restoreResult = ACL_SUCCESS;
  if (contextSwitched) {
    restoreResult = aclrtSetCurrentContext(previousContext);
  }
  if (unloadResult == ACL_SUCCESS) {
    binaryHandle_ = nullptr;
    functionHandle_ = nullptr;
    closed_ = true;
  }
  TORCH_CHECK(
      unloadResult == ACL_SUCCESS, "aclrtBinaryUnLoad failed for ", kernelName_, ": ", static_cast<int>(unloadResult));
  TORCH_CHECK(
      restoreResult == ACL_SUCCESS,
      "Failed to restore NPU context after unloading static kernel: ",
      static_cast<int>(restoreResult));
}

} // namespace torch_npu::inductor

#endif
