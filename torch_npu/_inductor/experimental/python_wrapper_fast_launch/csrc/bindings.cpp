#ifndef BUILD_LIBTORCH

#include "torch_npu/_inductor/experimental/python_wrapper_fast_launch/csrc/bindings.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <ATen/ATen.h>
#include <acl/acl_rt.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/pybind.h>

#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/csrc/inductor/mlir/hacl_rt.h>

namespace py = pybind11;

namespace {

enum class FastLaunchArgKind {
  Tensor,
  I32,
  I64,
  U32,
  U64,
  F32,
  F64,
  Bool,
};

struct FastLaunchArgLayout {
  FastLaunchArgKind kind = FastLaunchArgKind::Tensor;
  size_t offset = 0;
};

struct FastLaunchPlan {
  std::string kernelName;
  py::object kernelStubOwner;
  void* kernelStub = nullptr;
  std::vector<FastLaunchArgKind> argKinds;
  std::vector<FastLaunchArgLayout> argLayouts;
  size_t fftsOffset = 0;
  size_t gridOffsets[3] = {0, 0, 0};
  size_t packedArgsSize = 0;
  bool enableSimt = false;
  uint64_t sharedMemDynamicSize = 0;
  bool isPureSimt = false;
  bool targetSupportFfts = false;
  void* fftsAddress = nullptr;
};

size_t AlignOffset(size_t offset, size_t alignment) {
  TORCH_CHECK(alignment != 0, "alignment must be non-zero");
  return (offset + alignment - 1) / alignment * alignment;
}

void WriteBytesAt(
    std::vector<uint8_t>& buffer,
    size_t offset,
    const void* data,
    size_t size) {
  std::memcpy(buffer.data() + offset, data, size);
}

void WritePointerAt(
    std::vector<uint8_t>& buffer,
    size_t offset,
    void* pointer) {
  WriteBytesAt(buffer, offset, &pointer, sizeof(void*));
}

template <typename T>
void WriteScalarAt(
    std::vector<uint8_t>& buffer,
    size_t offset,
    py::handle arg) {
  T value = py::cast<T>(arg);
  WriteBytesAt(buffer, offset, &value, sizeof(T));
}

FastLaunchArgKind ParseArgKind(const std::string& kind) {
  if (kind == "tensor") {
    return FastLaunchArgKind::Tensor;
  }
  if (kind == "i32") {
    return FastLaunchArgKind::I32;
  }
  if (kind == "i64") {
    return FastLaunchArgKind::I64;
  }
  if (kind == "u32") {
    return FastLaunchArgKind::U32;
  }
  if (kind == "u64") {
    return FastLaunchArgKind::U64;
  }
  if (kind == "f32") {
    return FastLaunchArgKind::F32;
  }
  if (kind == "f64") {
    return FastLaunchArgKind::F64;
  }
  if (kind == "bool") {
    return FastLaunchArgKind::Bool;
  }
  TORCH_CHECK(false, "unsupported fast launch arg kind: ", kind);
  return FastLaunchArgKind::Tensor;
}

std::vector<FastLaunchArgKind> ParseArgKinds(
    const std::vector<std::string>& kinds) {
  std::vector<FastLaunchArgKind> parsed;
  parsed.reserve(kinds.size());
  for (const auto& kind : kinds) {
    parsed.emplace_back(ParseArgKind(kind));
  }
  return parsed;
}

size_t ArgSize(FastLaunchArgKind kind) {
  switch (kind) {
    case FastLaunchArgKind::Tensor:
      return sizeof(void*);
    case FastLaunchArgKind::I32:
      return sizeof(int32_t);
    case FastLaunchArgKind::I64:
      return sizeof(int64_t);
    case FastLaunchArgKind::U32:
      return sizeof(uint32_t);
    case FastLaunchArgKind::U64:
      return sizeof(uint64_t);
    case FastLaunchArgKind::F32:
      return sizeof(float);
    case FastLaunchArgKind::F64:
      return sizeof(double);
    case FastLaunchArgKind::Bool:
      return sizeof(int32_t);
  }
  TORCH_INTERNAL_ASSERT(false, "unsupported fast launch arg kind");
  return 0;
}

size_t ArgAlignment(FastLaunchArgKind kind) {
  switch (kind) {
    case FastLaunchArgKind::Tensor:
      return alignof(void*);
    case FastLaunchArgKind::I32:
      return alignof(int32_t);
    case FastLaunchArgKind::I64:
      return alignof(int64_t);
    case FastLaunchArgKind::U32:
      return alignof(uint32_t);
    case FastLaunchArgKind::U64:
      return alignof(uint64_t);
    case FastLaunchArgKind::F32:
      return alignof(float);
    case FastLaunchArgKind::F64:
      return alignof(double);
    case FastLaunchArgKind::Bool:
      return alignof(int32_t);
  }
  TORCH_INTERNAL_ASSERT(false, "unsupported fast launch arg kind");
  return 0;
}

void BuildPackedLayout(FastLaunchPlan& plan) {
  size_t offset = 0;
  // The generated runner uses a packed struct whose individual fields carry
  // explicit alignment.  Such a struct still has tail padding up to the
  // largest field alignment, so argsSize must be rounded up as well.
  size_t packedAlignment = alignof(int32_t);
  if (plan.targetSupportFfts) {
    packedAlignment = std::max(packedAlignment, alignof(void*));
    offset = AlignOffset(offset, alignof(void*));
    plan.fftsOffset = offset;
    offset += sizeof(void*);
  }
  // This is an ABI property, not a launch-API property.  Ascend's generated
  // runner keeps the sync-lock and workspace slots for every kernel except a
  // is_pure_simt binary, including SIMT-capable mixed-mode kernels.
  if (!plan.isPureSimt) {
    packedAlignment = std::max(packedAlignment, alignof(void*));
    for (int index = 0; index < 2; ++index) {
      offset = AlignOffset(offset, alignof(void*));
      offset += sizeof(void*);
    }
  }

  plan.argLayouts.clear();
  plan.argLayouts.reserve(plan.argKinds.size());
  for (FastLaunchArgKind kind : plan.argKinds) {
    size_t alignment = ArgAlignment(kind);
    packedAlignment = std::max(packedAlignment, alignment);
    offset = AlignOffset(offset, alignment);
    plan.argLayouts.push_back({kind, offset});
    offset += ArgSize(kind);
  }
  for (size_t index = 0; index < 3; ++index) {
    offset = AlignOffset(offset, alignof(int32_t));
    plan.gridOffsets[index] = offset;
    offset += sizeof(int32_t);
  }
  plan.packedArgsSize = AlignOffset(offset, packedAlignment);
}

void WriteArgAt(
    std::vector<uint8_t>& buffer,
    py::handle arg,
    const FastLaunchArgLayout& layout) {
  switch (layout.kind) {
    case FastLaunchArgKind::Tensor: {
      at::Tensor tensor = py::cast<at::Tensor>(arg);
      WritePointerAt(buffer, layout.offset, tensor.data_ptr());
      return;
    }
    case FastLaunchArgKind::I32:
      WriteScalarAt<int32_t>(buffer, layout.offset, arg);
      return;
    case FastLaunchArgKind::I64:
      WriteScalarAt<int64_t>(buffer, layout.offset, arg);
      return;
    case FastLaunchArgKind::U32:
      WriteScalarAt<uint32_t>(buffer, layout.offset, arg);
      return;
    case FastLaunchArgKind::U64:
      WriteScalarAt<uint64_t>(buffer, layout.offset, arg);
      return;
    case FastLaunchArgKind::F32:
      WriteScalarAt<float>(buffer, layout.offset, arg);
      return;
    case FastLaunchArgKind::F64:
      WriteScalarAt<double>(buffer, layout.offset, arg);
      return;
    case FastLaunchArgKind::Bool: {
      int32_t value = py::cast<bool>(arg) ? 1 : 0;
      WriteBytesAt(buffer, layout.offset, &value, sizeof(value));
      return;
    }
  }
  TORCH_CHECK(false, "unsupported fast launch arg kind");
}

void* ExtractPointer(py::handle object, const char* name) {
  PyObject* raw = object.ptr();
  if (PyCapsule_CheckExact(raw)) {
    const char* capsuleName = PyCapsule_GetName(raw);
    if (PyErr_Occurred()) {
      PyErr_Clear();
      capsuleName = nullptr;
    }
    void* pointer = PyCapsule_GetPointer(raw, capsuleName);
    TORCH_CHECK(pointer != nullptr, name, " PyCapsule pointer is null");
    return pointer;
  }
  if (PyLong_Check(raw)) {
    void* pointer = PyLong_AsVoidPtr(raw);
    TORCH_CHECK(!PyErr_Occurred(), name, " cannot be converted to pointer");
    TORCH_CHECK(pointer != nullptr, name, " pointer is null");
    return pointer;
  }
  if (py::hasattr(object, "value")) {
    return ExtractPointer(object.attr("value"), name);
  }
  TORCH_CHECK(false, name, " must be an integer address or PyCapsule");
}

struct PackedLaunch {
  std::vector<uint8_t> args;
  uint32_t blockNum = 0;
  rtStream_t stream = nullptr;
};

PackedLaunch PackLaunch(
    const FastLaunchPlan& plan,
    uint64_t streamValue,
    uint32_t grid0,
    uint32_t grid1,
    uint32_t grid2,
    const py::sequence& args) {
  size_t argCount = static_cast<size_t>(py::len(args));
  TORCH_CHECK(
      argCount == plan.argKinds.size(),
      "fast launch args and arg_kinds size mismatch: ",
      argCount,
      " vs ",
      plan.argKinds.size());
  rtStream_t stream = reinterpret_cast<rtStream_t>(streamValue);
  TORCH_CHECK(stream != nullptr, "fast launch stream pointer is null");

  const uint32_t grid[3] = {grid0, grid1, grid2};
  uint64_t blockNum = 1;
  for (size_t index = 0; index < 3; ++index) {
    TORCH_CHECK(grid[index] > 0, "fast launch grid dim must be positive");
    TORCH_CHECK(
        grid[index] <=
            static_cast<uint32_t>(std::numeric_limits<int32_t>::max()),
        "fast launch grid dim exceeds int32 max");
    blockNum *= grid[index];
    TORCH_CHECK(
        blockNum <= std::numeric_limits<uint16_t>::max(),
        "fast launch grid product exceeds uint16 max");
  }

  PackedLaunch packed;
  packed.blockNum = static_cast<uint32_t>(blockNum);
  packed.stream = stream;
  TORCH_INTERNAL_ASSERT(plan.argLayouts.size() == argCount);
  packed.args.resize(plan.packedArgsSize, 0);
  if (plan.targetSupportFfts) {
    WritePointerAt(packed.args, plan.fftsOffset, plan.fftsAddress);
  }
  for (size_t index = 0; index < argCount; ++index) {
    WriteArgAt(packed.args, args[index], plan.argLayouts[index]);
  }
  int32_t signedGrid[3] = {
      static_cast<int32_t>(grid0),
      static_cast<int32_t>(grid1),
      static_cast<int32_t>(grid2),
  };
  for (size_t index = 0; index < 3; ++index) {
    WriteBytesAt(
        packed.args,
        plan.gridOffsets[index],
        &signedGrid[index],
        sizeof(signedGrid[index]));
  }
  return packed;
}

void SubmitLaunch(const FastLaunchPlan& plan, PackedLaunch packed) {
  auto launchCall = [kernelStub = plan.kernelStub,
                     enableSimt = plan.enableSimt,
                     sharedMemDynamicSize = plan.sharedMemDynamicSize,
                     packed = std::move(packed)]() mutable {
    void* args = packed.args.data();
    uint32_t argsSize = static_cast<uint32_t>(packed.args.size());
    aclrtLaunchKernelAttr launchAttr = {};
    aclrtLaunchKernelCfg launchConfig = {};
    aclrtLaunchKernelCfg* launchConfigPtr = nullptr;
    if (enableSimt) {
      launchAttr.id = ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE;
      launchAttr.value.dynUBufSize =
          static_cast<uint32_t>(sharedMemDynamicSize);
      launchConfig.attrs = &launchAttr;
      launchConfig.numAttrs = 1;
      launchConfigPtr = &launchConfig;
    }
    aclError result = aclrtLaunchKernelWithHostArgs(
        reinterpret_cast<aclrtFuncHandle>(kernelStub),
        packed.blockNum,
        reinterpret_cast<aclrtStream>(packed.stream),
        launchConfigPtr,
        args,
        argsSize,
        nullptr,
        0);
    return static_cast<int>(result);
  };

  at_npu::native::OpCommand command;
  command.Name(plan.kernelName).SetCustomHandler(std::move(launchCall)).Run();
}

std::shared_ptr<FastLaunchPlan> MakeFastLaunchPlan(
    const std::string& kernelName,
    py::object kernelStub,
    const std::vector<std::string>& argKinds,
    bool enableSimt,
    uint64_t sharedMemDynamicSize,
    bool isPureSimt,
    bool targetSupportFfts) {
  TORCH_CHECK(
      sharedMemDynamicSize <= std::numeric_limits<uint32_t>::max(),
      "shared_mem_dynamic_size exceeds uint32 max");
  TORCH_CHECK(
      !isPureSimt || enableSimt, "is_pure_simt requires enable_simt");
  auto plan = std::make_shared<FastLaunchPlan>();
  plan->kernelName = kernelName;
  plan->kernelStubOwner = kernelStub;
  plan->kernelStub = ExtractPointer(kernelStub, "kernel_stub");
  plan->argKinds = ParseArgKinds(argKinds);
  plan->enableSimt = enableSimt;
  plan->sharedMemDynamicSize = sharedMemDynamicSize;
  plan->isPureSimt = isPureSimt;
  plan->targetSupportFfts = targetSupportFfts;
  if (targetSupportFfts) {
    uint64_t fftsAddress = 0;
    uint32_t fftsLength = 0;
    rtError_t result = rtGetC2cCtrlAddr(&fftsAddress, &fftsLength);
    TORCH_CHECK(
        result == RT_ERROR_NONE,
        "rtGetC2cCtrlAddr failed while creating fast launch plan: ",
        static_cast<int>(result));
    TORCH_CHECK(
        fftsAddress != 0,
        "rtGetC2cCtrlAddr returned a null fast launch FFTS address");
    plan->fftsAddress = reinterpret_cast<void*>(fftsAddress);
  }
  BuildPackedLayout(*plan);
  return plan;
}

void FastLaunchWithPlan(
    const std::shared_ptr<FastLaunchPlan>& plan,
    uint64_t stream,
    uint32_t grid0,
    uint32_t grid1,
    uint32_t grid2,
    const py::sequence& args) {
  TORCH_CHECK(plan != nullptr, "fast launch plan is null");
  SubmitLaunch(*plan, PackLaunch(*plan, stream, grid0, grid1, grid2, args));
}

} // namespace

void RegisterNPUFastLaunchBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  py::class_<FastLaunchPlan, std::shared_ptr<FastLaunchPlan>>(
      m, "_NPUInductorFastLaunchPlan", py::dynamic_attr());
  m.def(
      "_npu_inductor_make_fast_launch_plan",
      &MakeFastLaunchPlan,
      py::arg("kernel_name"),
      py::arg("kernel_stub"),
      py::arg("arg_kinds"),
      py::arg("enable_simt") = false,
      py::arg("shared_mem_dynamic_size") = 0,
      py::arg("is_pure_simt") = false,
      py::arg("target_support_ffts") = false);
  m.def(
      "_npu_inductor_fast_launch_with_plan",
      &FastLaunchWithPlan,
      py::arg("plan"),
      py::arg("stream"),
      py::arg("grid_0"),
      py::arg("grid_1"),
      py::arg("grid_2"),
      py::arg("args"));
}

#endif
