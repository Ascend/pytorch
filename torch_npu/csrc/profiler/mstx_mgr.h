#pragma once

#include <atomic>
#include <mutex>
#include <unordered_map>
#include "torch_npu/csrc/framework/interface/MstxInterface.h"
#include "torch_npu/csrc/toolkit/profiler/common/singleton.h"
#include "torch_npu/csrc/npu/Stream.h"

namespace torch_npu {
namespace profiler {
class MstxMgr : public torch_npu::toolkit::profiler::Singleton<MstxMgr> {
friend class torch_npu::toolkit::profiler::Singleton<MstxMgr>;
public:
    int RangeStart(const char* message, const aclrtStream stream);
    void RangeEnd(int ptRangeId);

private:
    MstxMgr();
    explicit MstxMgr(const MstxMgr &obj) = delete;
    MstxMgr& operator=(const MstxMgr &obj) = delete;
    explicit MstxMgr(MstxMgr &&obj) = delete;
    MstxMgr& operator=(MstxMgr &&obj) = delete;

private:
    std::atomic<int> ptRangeId_{1};
};

}
} // namespace torch_npu