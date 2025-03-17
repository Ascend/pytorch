#include <string>
#include <vector>
#include "torch_npu/csrc/core/npu/npu_log.h"

#include "torch_npu/csrc/core/npu/register/OptionRegister.h"
#include "torch_npu/csrc/framework/utils/ForceJitCompileList.h"

using std::string;
using std::vector;

namespace at_npu {
namespace native {

ForceJitCompileList &ForceJitCompileList::GetInstance()
{
    static ForceJitCompileList jit_list;
    return jit_list;
}

void ForceJitCompileList::RegisterJitlist(const std::string &jitlist)
{
    if (jitlist.empty()) {
        return;
    }

    auto value = jitlist;
    std::string delimiter = ",";
    auto start = 0U;
    auto end = value.find(delimiter);
    std::string token;
    while (end != std::string::npos) {
        token = value.substr(start, end - start);
        if (!token.empty()) {
            jit_list_.emplace(token);
        }
        start = end + delimiter.size();
        end = value.find(delimiter, start);
    }
    // if start + end > value.size(), substring only split(start, value.size() - start)
    token = value.substr(start, end);
    if (!token.empty()) {
        jit_list_.emplace(token);
    }
    DisplayJitlist();
    return;
}

bool ForceJitCompileList::Inlist(const std::string &opName) const
{
    if (jit_list_.find(opName) != jit_list_.end()) {
        return true;
    }
    return false;
}

void ForceJitCompileList::DisplayJitlist() const
{
    if (!jit_list_.empty()) {
        for (auto &iter : jit_list_) {
            ASCEND_LOGI("check op [%s] is in jitcompile list, use Just-In-Time compile", iter.c_str());
        }
    }
    return;
}

} // namespace native
} // namespace at_npu