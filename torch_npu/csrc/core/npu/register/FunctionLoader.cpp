#include <cstdlib>
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/register/FunctionLoader.h"

namespace c10_npu {
namespace option {

FunctionLoader::FunctionLoader(const std::string &name, int flags)
{
    this->fileName = name + ".so";
    this->flags = flags;
}

FunctionLoader::~FunctionLoader()
{
    if (this->handle != nullptr) {
        dlclose(this->handle);
    }
}

void FunctionLoader::Set(const std::string &name)
{
    this->registry[name] = nullptr;
}

void *FunctionLoader::Get(const std::string &name)
{
    std::lock_guard<std::mutex> lock(this->mu_);

    auto itr = registry.find(name);
    if (itr == registry.end()) {
        AT_ERROR("function(", name, ") is not registered.");
        return nullptr;
    }

    if (itr->second != nullptr) {
        return itr->second;
    }

    // When LD_PRELOAD is set, prefer RTLD_DEFAULT so that symbols overridden
    // by the user's preloaded .so take precedence over libascendcl.so's own
    // implementation. Opt-in by the presence of LD_PRELOAD itself; without
    // LD_PRELOAD the behavior is identical to the original path.
    static const bool preload_enabled = []() {
        const char *env = std::getenv("LD_PRELOAD");
        bool enabled = (env != nullptr && env[0] != '\0');
        if (enabled) {
            TORCH_NPU_WARN_ONCE("LD_PRELOAD detected, FunctionLoader prefers "
                                "RTLD_DEFAULT for symbol resolution.");
        }
        return enabled;
    }();

    void *func = nullptr;
    if (preload_enabled) {
        func = dlsym(RTLD_DEFAULT, name.c_str());
    }

    if (func == nullptr) {
        if (this->handle == nullptr) {
            auto handle = dlopen(this->fileName.c_str(), this->flags);
            if (handle == nullptr) {
                AT_ERROR(dlerror());
                return nullptr;
            }
            this->handle = handle;
        }
        func = dlsym(this->handle, name.c_str());
    }

    if (func == nullptr) {
        return nullptr;
    }
    this->registry[name] = func;
    return func;
}

namespace register_function {
FunctionRegister *FunctionRegister::GetInstance()
{
    static FunctionRegister instance;
    return &instance;
}
void FunctionRegister::Register(const std::string &name, ::std::unique_ptr<FunctionLoader> &ptr)
{
    std::lock_guard<std::mutex> lock(mu_);
    registry.emplace(name, std::move(ptr));
}

void FunctionRegister::Register(const std::string &name, const std::string &funcName)
{
    auto itr = registry.find(name);
    if (itr == registry.end()) {
        AT_ERROR(name, " library should register first.");
        return;
    }
    itr->second->Set(funcName);
}

void *FunctionRegister::Get(const std::string &soName, const std::string &funcName)
{
    auto itr = registry.find(soName);
    if (itr != registry.end()) {
        return itr->second->Get(funcName);
    }
    return nullptr;
}

FunctionRegisterBuilder::FunctionRegisterBuilder(const std::string &name, ::std::unique_ptr<FunctionLoader> &ptr) noexcept
{
    FunctionRegister::GetInstance()->Register(name, ptr);
}
FunctionRegisterBuilder::FunctionRegisterBuilder(const std::string &soName, const std::string &funcName) noexcept
{
    FunctionRegister::GetInstance()->Register(soName, funcName);
}
} // namespace register_function
} // namespace option
} // namespace c10_npu
