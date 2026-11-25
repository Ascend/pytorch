#include <dlfcn.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "acl/acl_rt.h"
#include "common/logger.h"
#include "exe_graph/fxrt_kernel_export.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "exe_graph/runtime/kernel_context.h"
#include "exe_graph/runtime/storage_shape.h"

namespace gert {
namespace {

constexpr size_t kMaxTilingDataSize = 64U * 1024U;

enum class TilingOutput : size_t {
  kWorkspaceSize,
  kBlockDim,
  kTilingData,
  kNum,
};

enum AutofuseAdapterValueKind : uint32_t {
  kAutofuseTensor = 0U,
  kAutofuseScalar = 1U,
};

// These structures are the versioned ABI exported by an Autofuse adapter.
struct AutofuseValue {
  uint32_t kind;
  uint32_t dtype;
  uint64_t payload;
};

struct AutofuseTilingArgMap {
  uint32_t ir_input_index;
  uint32_t adapter_index;
  uint32_t kernel_arg_index;
  uint32_t source_dtype;
  uint32_t kernel_dtype;
};

struct AutofuseLaunchArgMap {
  uint32_t ir_input_index;
  uint32_t adapter_index;
  uint32_t kernel_arg_index;
  uint32_t value_kind;
  uint32_t source_dtype;
  uint32_t kernel_dtype;
  int32_t alias_group;
};

struct AutofuseAdapterInfo {
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t flags;
  uint32_t dyn_shape_num;
  uint32_t launch_arg_num;
  uint32_t tiling_data_size;
  uint32_t workspace_num;
  uint64_t kernel_build_id;
};

using GetInfoFunc = const AutofuseAdapterInfo *(*)();
using GetTilingArgMapFunc = const AutofuseTilingArgMap *(*)(uint32_t *);
using GetLaunchArgMapFunc = const AutofuseLaunchArgMap *(*)(uint32_t *);
using InitFunc = int64_t (*)(const char *);
using FinalizeFunc = void (*)();
using TilingFunc = int64_t (*)(const int64_t *, uint32_t, void *, uint64_t *, uint32_t *);
using LaunchFunc = int64_t (*)(uint32_t, void *, const AutofuseValue *, uint32_t, void *, const void *);

struct AutofuseTilingCacheEntry {
  std::vector<int64_t> dyn_shapes;
  uint64_t workspace_size = 0U;
  uint32_t block_dim = 0U;
  std::vector<uint8_t> tiling_data;
};

struct AdapterResource {
  std::string key;
  void *handle = nullptr;
  FinalizeFunc finalize = nullptr;
  TilingFunc tiling = nullptr;
  LaunchFunc launch = nullptr;
  AutofuseAdapterInfo info{};
  std::vector<AutofuseTilingArgMap> tiling_arg_map;
  std::vector<AutofuseLaunchArgMap> launch_arg_map;
  size_t ref_count = 0U;
  // GE CacheableTiling LRU lives in PrepareCacheableTilingFwkData (Init). Autofuse
  // tiling is a custom kernel; cache on the Init-created adapter resource.
  std::vector<AutofuseTilingCacheEntry> tiling_cache;
};

constexpr size_t kAutofuseTilingCacheCap = 120U;

const AutofuseTilingCacheEntry *FindAutofuseTilingCache(AdapterResource *resource,
                                                        const std::vector<int64_t> &dyn_shapes) {
  auto &cache = resource->tiling_cache;
  for (size_t i = 0; i < cache.size(); ++i) {
    if (cache[i].dyn_shapes != dyn_shapes) {
      continue;
    }
    if (i + 1U != cache.size()) {
      AutofuseTilingCacheEntry entry = std::move(cache[i]);
      cache.erase(cache.begin() + static_cast<std::ptrdiff_t>(i));
      cache.push_back(std::move(entry));
    }
    return &cache.back();
  }
  return nullptr;
}

void StoreAutofuseTilingCache(AdapterResource *resource, const std::vector<int64_t> &dyn_shapes,
                              uint64_t workspace_size, uint32_t block_dim, const uint8_t *tiling_data,
                              size_t tiling_bytes) {
  AutofuseTilingCacheEntry entry;
  entry.dyn_shapes = dyn_shapes;
  entry.workspace_size = workspace_size;
  entry.block_dim = block_dim;
  entry.tiling_data.assign(tiling_data, tiling_data + tiling_bytes);
  auto &cache = resource->tiling_cache;
  if (cache.size() >= kAutofuseTilingCacheCap) {
    cache.erase(cache.begin());
  }
  cache.push_back(std::move(entry));
}

void LogError(const std::string &message) {
  RT_GLOG(ERROR) << "FXRT Autofuse Runtime V2: " << message;
}

template <typename T>
bool LoadSymbol(void *handle, const char *name, T *func) {
  dlerror();
  *func = reinterpret_cast<T>(dlsym(handle, name));
  const char *error = dlerror();
  if (*func == nullptr || error != nullptr) {
    LogError(std::string("failed to load adapter symbol '") + name + "': " +
             (error == nullptr ? "unknown dlsym error" : error));
    return false;
  }
  return true;
}

bool CanonicalPath(const char *path, std::string *canonical) {
  if (path == nullptr || path[0] == '\0' || canonical == nullptr) {
    LogError("received an empty adapter or kernel path");
    return false;
  }
  std::unique_ptr<char, decltype(&std::free)> resolved(realpath(path, nullptr), &std::free);
  if (resolved == nullptr) {
    LogError(std::string("cannot resolve path '") + path + "': " + std::strerror(errno));
    return false;
  }
  *canonical = resolved.get();
  return true;
}

struct AdapterApi {
  GetInfoFunc get_info = nullptr;
  GetTilingArgMapFunc get_tiling_arg_map = nullptr;
  GetLaunchArgMapFunc get_launch_arg_map = nullptr;
  InitFunc init = nullptr;
};

void CloseAdapter(AdapterResource *resource) {
  if (resource != nullptr && resource->handle != nullptr) {
    dlclose(resource->handle);
    resource->handle = nullptr;
  }
}

bool CreateAdapterResource(const std::string &canonical_adapter, const std::string &key,
                           std::unique_ptr<AdapterResource> *resource) {
  resource->reset(new (std::nothrow) AdapterResource());
  if (*resource == nullptr) {
    LogError("failed to allocate adapter resource");
    return false;
  }
  (*resource)->key = key;
  (*resource)->handle = dlopen(canonical_adapter.c_str(), RTLD_NOW | RTLD_LOCAL);
  if ((*resource)->handle != nullptr) {
    return true;
  }
  const char *error = dlerror();
  LogError(std::string("failed to open adapter '") + canonical_adapter + "': " +
           (error == nullptr ? "unknown dlopen error" : error));
  return false;
}

bool LoadAdapterApi(AdapterResource *resource, AdapterApi *api) {
  return LoadSymbol(resource->handle, "AutofuseAdapterGetInfo", &api->get_info) &&
         LoadSymbol(resource->handle, "AutofuseAdapterGetTilingArgMap", &api->get_tiling_arg_map) &&
         LoadSymbol(resource->handle, "AutofuseAdapterGetLaunchArgMap", &api->get_launch_arg_map) &&
         LoadSymbol(resource->handle, "AutofuseAdapterInit", &api->init) &&
         LoadSymbol(resource->handle, "AutofuseAdapterFinalize", &resource->finalize) &&
         LoadSymbol(resource->handle, "AutofuseAdapterTiling", &resource->tiling) &&
         LoadSymbol(resource->handle, "AutofuseAdapterLaunch", &resource->launch);
}

bool LoadAdapterMetadata(const AdapterApi &api, AdapterResource *resource) {
  const auto *info = api.get_info();
  if (info == nullptr || info->abi_version != 1U || info->struct_size < sizeof(AutofuseAdapterInfo) ||
      info->tiling_data_size > kMaxTilingDataSize || info->workspace_num != 1U) {
    LogError("adapter reports unsupported ABI, tiling-data size, or workspace count");
    return false;
  }
  resource->info = *info;

  uint32_t tiling_map_count = 0U;
  const auto *tiling_map = api.get_tiling_arg_map(&tiling_map_count);
  if (tiling_map_count != 0U && tiling_map == nullptr) {
    LogError("adapter returned a null tiling argument map");
    return false;
  }
  if (tiling_map_count != 0U) {
    resource->tiling_arg_map.assign(tiling_map, tiling_map + tiling_map_count);
  }

  uint32_t launch_map_count = 0U;
  const auto *launch_map = api.get_launch_arg_map(&launch_map_count);
  if (launch_map_count != info->launch_arg_num || (launch_map_count != 0U && launch_map == nullptr)) {
    LogError("adapter launch argument map does not match launch_arg_num");
    return false;
  }
  if (launch_map_count != 0U) {
    resource->launch_arg_map.assign(launch_map, launch_map + launch_map_count);
  }
  return true;
}

bool InitializeAdapter(const AdapterApi &api, const std::string &canonical_kernel) {
  const int64_t result = api.init(canonical_kernel.c_str());
  if (result == 0) {
    return true;
  }
  LogError("AutofuseAdapterInit failed with code " + std::to_string(result));
  return false;
}

class AdapterResourceManager {
 public:
  static AdapterResourceManager &Instance() {
    static AdapterResourceManager instance;
    return instance;
  }

  ge::graphStatus Acquire(const char *adapter_path, const char *kernel_path, AdapterResource **output) {
    if (output == nullptr) {
      LogError("Init received a null resource output");
      return ge::GRAPH_FAILED;
    }
    std::string canonical_adapter;
    std::string canonical_kernel;
    if (!CanonicalPath(adapter_path, &canonical_adapter) || !CanonicalPath(kernel_path, &canonical_kernel)) {
      return ge::GRAPH_FAILED;
    }
    const std::string key = canonical_adapter + "\n" + canonical_kernel;

    std::lock_guard<std::mutex> lock(mu_);
    const auto iter = resources_.find(key);
    if (iter != resources_.end()) {
      ++iter->second->ref_count;
      *output = iter->second.get();
      return ge::GRAPH_SUCCESS;
    }

    std::unique_ptr<AdapterResource> resource;
    if (!CreateAdapterResource(canonical_adapter, key, &resource)) {
      return ge::GRAPH_FAILED;
    }
    AdapterApi api;
    if (!LoadAdapterApi(resource.get(), &api) || !LoadAdapterMetadata(api, resource.get()) ||
        !InitializeAdapter(api, canonical_kernel)) {
      CloseAdapter(resource.get());
      return ge::GRAPH_FAILED;
    }
    resource->ref_count = 1U;
    *output = resource.get();
    resources_.emplace(key, std::move(resource));
    return ge::GRAPH_SUCCESS;
  }

  ge::graphStatus Release(AdapterResource *resource) {
    if (resource == nullptr) {
      LogError("Finalize received a null adapter resource");
      return ge::GRAPH_FAILED;
    }
    std::lock_guard<std::mutex> lock(mu_);
    const auto iter = resources_.find(resource->key);
    if (iter == resources_.end() || iter->second.get() != resource || resource->ref_count == 0U) {
      LogError("Finalize received an unknown adapter resource");
      return ge::GRAPH_FAILED;
    }
    --resource->ref_count;
    if (resource->ref_count != 0U) {
      return ge::GRAPH_SUCCESS;
    }
    resource->finalize();
    CloseAdapter(resource);
    resources_.erase(iter);
    return ge::GRAPH_SUCCESS;
  }

 private:
  std::mutex mu_;
  std::map<std::string, std::unique_ptr<AdapterResource>> resources_;
};

bool FlattenInputDims(KernelContext *context, size_t shape_start, std::vector<int64_t> *dims) {
  if (context == nullptr || dims == nullptr || shape_start > context->GetInputNum()) {
    return false;
  }
  dims->clear();
  for (size_t i = shape_start; i < context->GetInputNum(); ++i) {
    const auto *shape = context->GetInputPointer<StorageShape>(i);
    if (shape == nullptr) {
      return false;
    }
    const auto &storage_shape = shape->GetStorageShape();
    for (size_t dim = 0U; dim < storage_shape.GetDimNum(); ++dim) {
      dims->push_back(storage_shape.GetDim(dim));
    }
  }
  return true;
}

bool AllAssigned(const std::vector<bool> &assigned) {
  return std::all_of(assigned.begin(), assigned.end(), [](bool value) { return value; });
}

bool CollectTensorPayloads(KernelContext *context, size_t input_start, size_t tensor_slot_num,
                           std::vector<uint64_t> *tensor_payloads) {
  tensor_payloads->resize(tensor_slot_num);
  for (size_t i = 0U; i < tensor_slot_num; ++i) {
    const auto *tensor_data = context->GetInputPointer<GertTensorData>(input_start + i);
    if (tensor_data == nullptr) {
      LogError("Launch received a null tensor address");
      return false;
    }
    (*tensor_payloads)[i] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tensor_data->GetAddr()));
  }
  return true;
}

bool SetLaunchArgument(const AutofuseLaunchArgMap &mapping, const std::vector<uint64_t> &tensor_payloads,
                       const std::vector<int64_t> &flat_dims, size_t tensor_slot_num, AutofuseValue *arg) {
  arg->kind = mapping.value_kind;
  arg->dtype = mapping.source_dtype;
  if (mapping.value_kind == kAutofuseTensor) {
    if (mapping.ir_input_index >= tensor_payloads.size()) {
      LogError("Launch argument map refers to a missing tensor");
      return false;
    }
    arg->payload = tensor_payloads[mapping.ir_input_index];
    return true;
  }
  if (mapping.value_kind != kAutofuseScalar || mapping.ir_input_index < tensor_slot_num) {
    LogError("Launch argument map has an invalid value kind or input index");
    return false;
  }
  const size_t dim_index = mapping.ir_input_index - tensor_slot_num;
  if (dim_index >= flat_dims.size()) {
    LogError("Launch argument map refers to a missing dynamic dimension");
    return false;
  }
  arg->payload = static_cast<uint64_t>(flat_dims[dim_index]);
  return true;
}

bool BuildLaunchArguments(const AdapterResource &resource, const std::vector<uint64_t> &tensor_payloads,
                          const std::vector<int64_t> &flat_dims, size_t tensor_slot_num,
                          std::vector<AutofuseValue> *args) {
  args->resize(resource.info.launch_arg_num);
  std::vector<bool> assigned(resource.info.launch_arg_num, false);
  for (const auto &mapping : resource.launch_arg_map) {
    if (mapping.adapter_index >= args->size()) {
      LogError("Launch argument map adapter index is out of range");
      return false;
    }
    if (!SetLaunchArgument(mapping, tensor_payloads, flat_dims, tensor_slot_num, &(*args)[mapping.adapter_index])) {
      return false;
    }
    assigned[mapping.adapter_index] = true;
  }
  if (!AllAssigned(assigned)) {
    LogError("Launch argument map does not assign every adapter argument");
    return false;
  }
  return true;
}

ge::graphStatus InductorAutofuseInit(KernelContext *context) {
  if (context == nullptr || context->GetInputNum() != 2U) {
    LogError("Init received an invalid kernel context");
    return ge::GRAPH_FAILED;
  }
  auto **output = context->GetOutputPointer<AdapterResource *>(0U);
  return AdapterResourceManager::Instance().Acquire(context->GetInputValue<char *>(0U),
                                                     context->GetInputValue<char *>(1U), output);
}

ge::graphStatus InductorAutofuseFinalize(KernelContext *context) {
  if (context == nullptr || context->GetInputNum() != 1U) {
    LogError("Finalize received an invalid kernel context");
    return ge::GRAPH_FAILED;
  }
  auto *res = context->GetInputValue<AdapterResource *>(0U);
  auto ret = AdapterResourceManager::Instance().Release(res);
  return ret;
}

ge::graphStatus CreateTilingOutputs(KernelContext *context) {
  if (context == nullptr) {
    return ge::GRAPH_FAILED;
  }
  auto *workspace_output = context->GetOutput(static_cast<size_t>(TilingOutput::kWorkspaceSize));
  auto *tiling_output = context->GetOutput(static_cast<size_t>(TilingOutput::kTilingData));
  if (workspace_output == nullptr || tiling_output == nullptr) {
    LogError("failed to create tiling outputs");
    return ge::GRAPH_FAILED;
  }
  auto workspace_sizes = ContinuousVector::Create<size_t>(1U);
  auto tiling_data = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[kMaxTilingDataSize]());
  if (workspace_sizes == nullptr || tiling_data == nullptr) {
    LogError("failed to allocate tiling outputs");
    return ge::GRAPH_FAILED;
  }
  workspace_output->SetWithDefaultDeleter<uint8_t[]>(workspace_sizes.release());
  tiling_output->SetWithDefaultDeleter<uint8_t[]>(tiling_data.release());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus InductorAutofuseTiling(KernelContext *context) {
  if (context == nullptr || context->GetInputNum() < 2U) {
    LogError("Tiling received an invalid kernel context");
    return ge::GRAPH_FAILED;
  }
  auto *resource = context->GetInputValue<AdapterResource *>(0U);
  const uint32_t tensor_slot_num = context->GetInputValue<uint32_t>(1U);
  if (resource == nullptr) {
    LogError("Tiling received a null adapter resource");
    return ge::GRAPH_FAILED;
  }
  std::vector<int64_t> flat_dims;
  if (!FlattenInputDims(context, 2U, &flat_dims)) {
    LogError("Tiling could not read input shapes");
    return ge::GRAPH_FAILED;
  }
  std::vector<int64_t> dyn_shapes(resource->info.dyn_shape_num);
  std::vector<bool> assigned(resource->info.dyn_shape_num, false);
  for (const auto &mapping : resource->tiling_arg_map) {
    if (mapping.adapter_index >= dyn_shapes.size() || mapping.ir_input_index < tensor_slot_num) {
      LogError("Tiling argument map index is out of range");
      return ge::GRAPH_FAILED;
    }
    const size_t dim_index = mapping.ir_input_index - tensor_slot_num;
    if (dim_index >= flat_dims.size()) {
      LogError("Tiling argument map refers to a missing dynamic dimension");
      return ge::GRAPH_FAILED;
    }
    dyn_shapes[mapping.adapter_index] = flat_dims[dim_index];
    assigned[mapping.adapter_index] = true;
  }
  if (!AllAssigned(assigned)) {
    LogError("Tiling argument map does not assign every dynamic shape");
    return ge::GRAPH_FAILED;
  }

  auto *workspace_sizes = context->GetOutputPointer<TypedContinuousVector<size_t>>(
      static_cast<size_t>(TilingOutput::kWorkspaceSize));
  auto *block_dim = context->GetOutputPointer<uint32_t>(static_cast<size_t>(TilingOutput::kBlockDim));
  auto **tiling_data_holder =
      context->GetOutputPointer<uint8_t *>(static_cast<size_t>(TilingOutput::kTilingData));
  if (workspace_sizes == nullptr || block_dim == nullptr || tiling_data_holder == nullptr ||
      *tiling_data_holder == nullptr) {
    LogError("Tiling outputs are not initialized");
    return ge::GRAPH_FAILED;
  }
  const AutofuseTilingCacheEntry *cached = FindAutofuseTilingCache(resource, dyn_shapes);
  if (cached != nullptr && !cached->tiling_data.empty()) {
    std::memcpy(*tiling_data_holder, cached->tiling_data.data(), cached->tiling_data.size());
    *block_dim = cached->block_dim;
    workspace_sizes->SetSize(1U);
    workspace_sizes->MutableData()[0U] = cached->workspace_size;
    return ge::GRAPH_SUCCESS;
  }

  uint64_t workspace_size = 0U;
  const int64_t result = resource->tiling(dyn_shapes.data(), resource->info.dyn_shape_num, *tiling_data_holder,
                                          &workspace_size, block_dim);
  if (result != 0) {
    LogError("AutofuseAdapterTiling failed with code " + std::to_string(result));
    return ge::GRAPH_FAILED;
  }
  workspace_sizes->SetSize(1U);
  workspace_sizes->MutableData()[0U] = workspace_size;
  const size_t tiling_bytes =
      resource->info.tiling_data_size > 0U ? resource->info.tiling_data_size : kMaxTilingDataSize;
  StoreAutofuseTilingCache(resource, dyn_shapes, workspace_size, *block_dim, *tiling_data_holder, tiling_bytes);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus InductorAutofuseLaunch(KernelContext *context) {
  constexpr size_t kFixedInputNum = 7U;
  if (context == nullptr || context->GetInputNum() < kFixedInputNum) {
    LogError("Launch received an invalid kernel context");
    return ge::GRAPH_FAILED;
  }
  auto *resource = context->GetInputValue<AdapterResource *>(0U);
  const uint32_t block_dim = context->GetInputValue<uint32_t>(1U);
  auto stream = context->GetInputValue<aclrtStream>(2U);
  const auto *tiling_data = context->GetInputValue<uint8_t *>(3U);
  const auto *workspace_addrs = context->GetInputPointer<TypedContinuousVector<GertTensorData *>>(4U);
  const uint32_t input_num = context->GetInputValue<uint32_t>(5U);
  const uint32_t output_num = context->GetInputValue<uint32_t>(6U);
  if (resource == nullptr || tiling_data == nullptr || workspace_addrs == nullptr ||
      workspace_addrs->GetSize() != resource->info.workspace_num || workspace_addrs->GetData() == nullptr ||
      workspace_addrs->GetData()[0U] == nullptr) {
    LogError("Launch received invalid resource, tiling data, or workspace inputs");
    return ge::GRAPH_FAILED;
  }

  const size_t tensor_slot_num = static_cast<size_t>(input_num) + output_num;
  const size_t shape_start = kFixedInputNum + tensor_slot_num;
  if (shape_start > context->GetInputNum()) {
    LogError("Launch tensor count exceeds its kernel inputs");
    return ge::GRAPH_FAILED;
  }
  std::vector<uint64_t> tensor_payloads;
  if (!CollectTensorPayloads(context, kFixedInputNum, tensor_slot_num, &tensor_payloads)) {
    return ge::GRAPH_FAILED;
  }
  std::vector<int64_t> flat_dims;
  if (!FlattenInputDims(context, shape_start, &flat_dims)) {
    LogError("Launch could not read input shapes");
    return ge::GRAPH_FAILED;
  }

  std::vector<AutofuseValue> args;
  if (!BuildLaunchArguments(*resource, tensor_payloads, flat_dims, tensor_slot_num, &args)) {
    return ge::GRAPH_FAILED;
  }

  void *workspace = workspace_addrs->GetData()[0U]->GetAddr();
  const int64_t result = resource->launch(block_dim, stream, args.data(), resource->info.launch_arg_num,
                                          workspace, tiling_data);
  if (result != 0) {
    LogError("AutofuseAdapterLaunch failed with code " + std::to_string(result));
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

}  // namespace

FXRT_EXPORT_KERNEL_RUN(InductorAutofuseInit, InductorAutofuseInit)
FXRT_EXPORT_KERNEL_RUN(InductorAutofuseFinalize, InductorAutofuseFinalize)
FXRT_EXPORT_KERNEL_RUN(InductorAutofuseTiling, InductorAutofuseTiling)
FXRT_EXPORT_KERNEL_OUTPUTS(InductorAutofuseTiling, CreateTilingOutputs)
FXRT_EXPORT_KERNEL_RUN(InductorAutofuseLaunch, InductorAutofuseLaunch)

}  // namespace gert
