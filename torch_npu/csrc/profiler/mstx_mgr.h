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

const std::string DOMAIN_COMMUNICATION = "communication";
const std::string DOMAIN_MSLEAKS = "msleaks";

class MstxMgr : public torch_npu::toolkit::profiler::Singleton<MstxMgr> {
friend class torch_npu::toolkit::profiler::Singleton<MstxMgr>;
public:
    void mark(const char* message, const aclrtStream stream);
    int rangeStart(const char* message, const aclrtStream stream);
    void rangeEnd(int ptRangeId);
    bool isMsleaksEnable();
    bool isMstxEnable();
    int getRangeId();
    mstxDomainHandle_t createProfDomain(const char* name);
    mstxDomainHandle_t createLeaksDomain(const char* name);
    void destroyDomain(mstxDomainHandle_t domain);
    void domainMark(mstxDomainHandle_t domain, const char* message, const aclrtStream stream);
    int domainRangeStart(mstxDomainHandle_t domain, const char* message, const aclrtStream stream);
    void domainRangeEnd(mstxDomainHandle_t domain, int ptRangeId);
    mstxMemHeapHandle_t memHeapRegister(mstxDomainHandle_t domain, mstxMemVirtualRangeDesc_t* desc);
    void memHeapUnregister(mstxDomainHandle_t domain, void* ptr);
    void memRegionsRegister(mstxDomainHandle_t domain, mstxMemVirtualRangeDesc_t* desc);
    void memRegionsUnregister(mstxDomainHandle_t domain, void* ptr);

private:
    MstxMgr();
    explicit MstxMgr(const MstxMgr &obj) = delete;
    MstxMgr& operator=(const MstxMgr &obj) = delete;
    explicit MstxMgr(MstxMgr &&obj) = delete;
    MstxMgr& operator=(MstxMgr &&obj) = delete;

    bool isMsleaksEnableImpl();
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