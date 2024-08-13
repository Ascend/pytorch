#include "torch_npu/csrc/profiler/mstx_mgr.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/framework/interface/MstxInterface.h"
#include "torch_npu/csrc/core/npu/npu_log.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/profiler/profiler_mgr.h"

namespace torch_npu {
namespace profiler {
MstxMgr::MstxMgr()
{
}

int MstxMgr::RangeStart(const char* message, const aclrtStream stream)
{
    if (!ProfilerMgr::GetInstance()->GetNpuTrace().load() || !ProfilerMgr::GetInstance()->GetMsprofTx().load()) {
        return 0;
    }
    int id = ptRangeId_++;
    if (stream == nullptr) {
        int res = at_npu::native::MstxRangeStartA(message, nullptr, id);
        return id;
    }
    auto range_start_call = [msg_ptr = std::make_shared<std::string>(message), stream, id]() -> int {
        return at_npu::native::MstxRangeStartA(msg_ptr->c_str(), stream, id);
    };
    at_npu::native::OpCommand cmd;
    cmd.Name("mstx_range_start_op");
    cmd.SetCustomHandler(range_start_call);
    cmd.Run();
    return id;
}

void MstxMgr::RangeEnd(int ptRangeId)
{
    if (!ProfilerMgr::GetInstance()->GetNpuTrace().load() || !ProfilerMgr::GetInstance()->GetMsprofTx().load()) {
        return;
    }
    if (!at_npu::native::IsRangeIdWithStream(ptRangeId)) {
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
}
}