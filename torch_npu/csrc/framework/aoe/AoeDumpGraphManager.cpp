#include "torch_npu/csrc/framework/interface/AclOpCompileInterface.h"
#include "torch_npu/csrc/framework/aoe/AoeDumpGraphManager.h"

namespace at_npu {
namespace native {
namespace aoe {

void AoeDumpGraphManager::SetDumpGraphPath(const std::string& dump_path)
{
    autotune_graphdumppath = dump_path;
}

std::string AoeDumpGraphManager::GetDumpGraphPath() const
{
    return autotune_graphdumppath;
}

aclGraphDumpOption* AoeDumpGraphManager::CreateGraphDumpOption()
{
    AclGraphDumpOption = AclCreateGraphDumpOpt();
    return AclGraphDumpOption;
}

void AoeDumpGraphManager::DestropyGraphDumpOption()
{
    AclDestroyGraphDumpOpt(AclGraphDumpOption);
    AclGraphDumpOption = nullptr;
}

void AoeDumpGraphManager::EnableAoe()
{
    aoe_enable = true;
}

bool AoeDumpGraphManager::IsAoeEnabled() const
{
    return aoe_enable;
}

bool AoeDumpGraphManager::IsInWhitelist(const std::string &opName) const
{
    if (white_list_.find(opName) != white_list_.end()) {
        return true;
    }
    return false;
}

AoeDumpGraphManager& aoe_manager()
{
    static AoeDumpGraphManager instance;
    return instance;
}

} // namespace aoe
} // namespace native
} // namespace at_npu