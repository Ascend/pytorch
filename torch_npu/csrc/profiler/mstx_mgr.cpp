#include "torch_npu/csrc/profiler/mstx_mgr.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/framework/interface/MstxInterface.h"
#include "torch_npu/csrc/core/npu/npu_log.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/profiler/profiler_mgr.h"
#include "torch_npu/csrc/toolkit/profiler/common/utils.h"

#include <sstream>

namespace torch_npu {
namespace profiler {
MstxMgr::MstxMgr()
{
}

void MstxMgr::mark(const char* message, const aclrtStream stream)
{
    if (!isMstxEnable()) {
        return;
    }
    int id = ptRangeId_++;
    if (stream == nullptr) {
        (void)at_npu::native::MstxMarkA(message, nullptr);
        return;
    }
    auto mark_call = [msg_ptr = std::make_shared<std::string>(message), stream]() -> int {
        (void)at_npu::native::MstxMarkA(msg_ptr->c_str(), stream);
        return 0;
    };
    at_npu::native::OpCommand cmd;
    cmd.Name("mstx_mark_op");
    cmd.SetCustomHandler(mark_call);
    cmd.Run();
}

int MstxMgr::rangeStart(const char* message, const aclrtStream stream)
{
    if (!isMstxEnable()) {
        return 0;
    }
    int id = ptRangeId_++;
    if (stream == nullptr) {
        int res = at_npu::native::MstxRangeStartA(message, nullptr, id);
        return id;
    }
    {
        std::lock_guard<std::mutex> lock(mtx_);
        ptRangeIdsWithStream_.insert(id);
    }
    auto range_start_call = [msg_ptr = std::make_shared<std::string>(message), stream, id]() -> int {
        int taskId = at_npu::native::MstxRangeStartA(msg_ptr->c_str(), stream, id);
        return 0;
    };
    at_npu::native::OpCommand cmd;
    cmd.Name("mstx_range_start_op");
    cmd.SetCustomHandler(range_start_call);
    cmd.Run();
    return id;
}

void MstxMgr::rangeEnd(int ptRangeId)
{
    if (!isMstxEnable() || ptRangeId == 0) {
        return;
    }
    bool rangeIdWithStream = false;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        auto iter = ptRangeIdsWithStream_.find(ptRangeId);
        if (iter != ptRangeIdsWithStream_.end()) {
            rangeIdWithStream = true;
            ptRangeIdsWithStream_.erase(iter);
        }
    }
    if (!rangeIdWithStream) {
        at_npu::native::MstxRangeEnd(ptRangeId);
        return;
    }
    auto range_end_call = [ptRangeId]() -> int {
        at_npu::native::MstxRangeEnd(ptRangeId);
        return 0;
    };
    at_npu::native::OpCommand cmd;
    cmd.Name("mstx_range_end_op");
    cmd.SetCustomHandler(range_end_call);
    cmd.Run();
}

int MstxMgr::getRangeId()
{
    return ptRangeId_++;
}

bool MstxMgr::isProfTxEnable()
{
    return ProfilerMgr::GetInstance()->GetNpuTrace().load() && ProfilerMgr::GetInstance()->GetMsprofTx().load();
}

bool MstxMgr::isMsptiTxEnableImpl()
{
    bool ret = false;
    const char* envVal = std::getenv("LD_PRELOAD");
    if (envVal == nullptr) {
        return ret;
    }
    static const std::string soName = "libmspti.so";
    std::stringstream ss(envVal);
    std::string path;
    while (std::getline(ss, path, ':')) {
        path = torch_npu::toolkit::profiler::Utils::RealPath(path);
        if ((path.size() > soName.size()) && (path.substr(path.size() - soName.size()) == soName)) {
            ret = true;
            break;
        }
    }
    return ret;
}

bool MstxMgr::isMsptiTxEnable()
{
    static bool isEnable = isMsptiTxEnableImpl();
    return isEnable;
}

bool MstxMgr::isMstxEnable()
{
    return isProfTxEnable() || isMsptiTxEnable();
}
}
}