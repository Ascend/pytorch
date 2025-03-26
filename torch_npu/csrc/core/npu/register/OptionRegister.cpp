#include <algorithm>

#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/register/OptionRegister.h"
#include "torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h"
#include "torch_npu/csrc/core/npu/npu_log.h"

namespace c10_npu {
namespace option {
OptionInterface::OptionInterface(OptionCallBack callback)
{
    this->callback = callback;
}

void OptionInterface::Set(const std::string &in)
{
    this->val = in;
    if (this->callback != nullptr) {
        if (c10_npu::NpuSysCtrl::GetInstance().GetInitFlag()) {
            ASCEND_LOGD("setoption call immediately.");
            this->callback(in);
        } else {
            ASCEND_LOGD("setoption will lazy call.");
            c10_npu::NpuSysCtrl::GetInstance().RegisterLazyFn(this->callback, in);
        }
    }
}

std::string OptionInterface::Get()
{
    return val;
}


namespace register_options {
OptionRegister *OptionRegister::GetInstance()
{
    static OptionRegister instance;
    return &instance;
}

void OptionRegister::Register(const std::string &name, ::std::unique_ptr<OptionInterface> &ptr)
{
    std::lock_guard<std::mutex> lock(mu_);
    registry.emplace(name, std::move(ptr));
}

void OptionRegister::Set(const std::string &name, const std::string &val)
{
    auto itr = registry.find(name);
    if (itr != registry.end()) {
        itr->second->Set(val);
    } else {
        AT_ERROR("invalid npu option name:", name);
    }
}

c10::optional<std::string> OptionRegister::Get(const std::string &name)
{
    auto itr = registry.find(name);
    if (itr != registry.end()) {
        return itr->second->Get();
    }
    return c10::nullopt; // default value
}

OptionInterfaceBuilder::OptionInterfaceBuilder(const std::string &name, ::std::unique_ptr<OptionInterface> &ptr,
    const std::string &type)
{
    OptionRegister::GetInstance()->Register(name, ptr);

    // init the value if env variable.
    if (type == "env") {
        std::string env_name = name;
        std::transform(env_name.begin(), env_name.end(), env_name.begin(), ::toupper);
        char *env_val = std::getenv(env_name.c_str());
        if (env_val != nullptr) {
            std::string val(env_val);
            OptionRegister::GetInstance()->Set(name, val);
        }
    }
}
} // namespace register_options

void SetOption(const std::string &key, const std::string &val)
{
    register_options::OptionRegister::GetInstance()->Set(key, val);
}

void SetOption(const std::map<std::string, std::string> &options)
{
    for (auto item : options) {
        SetOption(item.first, item.second);
    }
}

c10::optional<std::string> GetOption(const std::string &key)
{
    return register_options::OptionRegister::GetInstance()->Get(key);
}
} // namespace option
} // namespace c10_npu
