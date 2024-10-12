#pragma once

#include <atomic>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include "torch_npu/csrc/framework/interface/MstxInterface.h"
#include "torch_npu/csrc/toolkit/profiler/common/singleton.h"
#include "torch_npu/csrc/npu/Stream.h"

namespace torch_npu {
namespace profiler {
class MstxMgr : public torch_npu::toolkit::profiler::Singleton<MstxMgr> {
friend class torch_npu::toolkit::profiler::Singleton<MstxMgr>;
public:
    void mark(const char* message, const aclrtStream stream);
    int rangeStart(const char* message, const aclrtStream stream);
    void rangeEnd(int ptRangeId);
    bool isMstxEnable();
    int getRangeId();

private:
    MstxMgr();
    explicit MstxMgr(const MstxMgr &obj) = delete;
    MstxMgr& operator=(const MstxMgr &obj) = delete;
    explicit MstxMgr(MstxMgr &&obj) = delete;
    MstxMgr& operator=(MstxMgr &&obj) = delete;

    bool isProfTxEnable();
    bool isMsptiTxEnable();
    bool isMsptiTxEnableImpl();
private:
    std::atomic<int> ptRangeId_{1};
    std::unordered_set<int> ptRangeIdsWithStream_;
    std::mutex mtx_;
};

}
} // namespace torch_npu