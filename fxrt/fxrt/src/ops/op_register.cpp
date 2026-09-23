#include <dirent.h>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <type_traits>

#include "common/logger.h"
#include "common/dynamic_lib_loader.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {
bool LoadOpLib(const std::string& opLibPrefix, std::stringstream* errMsg) {
  static std::unique_ptr<common::DynamicLibLoader> dynamicLibLoader = std::make_unique<common::DynamicLibLoader>();
  CHECK_IF_NULL(dynamicLibLoader);
  CHECK_IF_NULL(errMsg);
  DIR* dir = opendir(dynamicLibLoader->GetDynamicLibFilePath().c_str());
  if (dir == nullptr) {
    *errMsg << "Open Op Lib dir failed, file path:" << dynamicLibLoader->GetDynamicLibFilePath() << std::endl;
    return false;
  }
  struct dirent* entry;
  std::set<std::string> opLibs;
  while ((entry = readdir(dir)) != nullptr) {
    std::string opLibName = entry->d_name;
    if (opLibName.find(opLibPrefix) == std::string::npos) {
      continue;
    }
    if (opLibName.find_first_of(".") == std::string::npos) {
      continue;
    }
    opLibs.insert(opLibName);
  }
  for (const auto& opLibName : opLibs) {
    if (!dynamicLibLoader->LoadDynamicLib(opLibName, errMsg)) {
      RT_VLOG(VL_OPS) << "Failed to load dynamic op library: " << opLibName;
    }
  }
  (void)closedir(dir);
  return true;
}

OpFactoryBase* OpFactoryBase::GetOpFactory(const std::string_view& name) {
  auto iter = OpFactoryMap().find(name);
  if (iter == OpFactoryMap().end()) {
    return nullptr;
  }
  return iter->second.get();
}

OpFactoryBase* OpFactoryBase::CreateOpFactory(const std::string_view& name, std::unique_ptr<OpFactoryBase>&& factory) {
  if (OpFactoryMap().find(name) != OpFactoryMap().end()) {
    RT_GLOG(EXCEPTION) << name << " already has an OpFactory, please check!";
  }
  (void)OpFactoryMap().emplace(name, std::move(factory));
  return GetOpFactory(name);
}

OpFactoryBase::OpFactoryMapType& OpFactoryBase::OpFactoryMap() {
  // Functions containing static local variables should be implemented in .cpp files to prevent multiple instances of
  // the same static variable in memory within a single process, which may occur when header files are included across
  // shared libraries.
  static OpFactoryBase::OpFactoryMapType factoryMap;
  return factoryMap;
}

void OpPrototypeRegistry::Register(const std::string& opName, size_t tensorInputCount) {
  // See OpFactoryBase::OpFactoryMap(): the static map must live in a .cpp file so all
  // shared libraries share one instance within the process.
  auto& protoMap = PrototypeMap();
  auto iter = protoMap.find(opName);
  if (iter != protoMap.end()) {
    if (iter->second != tensorInputCount) {
      RT_GLOG(ERROR) << "Conflicting op prototype registered for op " << opName << ": " << iter->second << " vs "
                     << tensorInputCount << " tensor inputs";
    }
    return;
  }
  (void)protoMap.emplace(opName, tensorInputCount);
}

const size_t* OpPrototypeRegistry::GetTensorInputCount(const std::string& opName) {
  auto& protoMap = PrototypeMap();
  auto iter = protoMap.find(opName);
  if (iter == protoMap.end()) {
    return nullptr;
  }
  return &iter->second;
}

std::unordered_map<std::string, size_t>& OpPrototypeRegistry::PrototypeMap() {
  static std::unordered_map<std::string, size_t> protoMap;
  return protoMap;
}

} // namespace ops
} // namespace fxrt
