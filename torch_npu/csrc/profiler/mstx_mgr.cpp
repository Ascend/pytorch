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
    at_npu::native::OpCommand::RunOpApi("mstx_mark_op", mark_call);
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
    at_npu::native::OpCommand::RunOpApi("mstx_range_start_op", range_start_call);
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
    at_npu::native::OpCommand::RunOpApi("mstx_range_end_op", range_end_call);
}

int MstxMgr::getRangeId()
{
    return ptRangeId_++;
}

mstxDomainHandle_t MstxMgr::createDomain(const char* name)
{
    if (!isMsleaksEnable() && !isMstxEnable()) {
        return nullptr;
    }
    return at_npu::native::MstxDomainCreateA(name);
}

void MstxMgr::destroyDomain(mstxDomainHandle_t domain)
{
    at_npu::native::MstxDomainDestroy(domain);
}

void MstxMgr::domainMark(mstxDomainHandle_t domain, const char* message, const aclrtStream stream)
{
    if (!isMstxEnable()) {
        return;
    }
    int id = ptRangeId_++;
    if (stream == nullptr) {
        (void)at_npu::native::MstxDomainMarkA(domain, message, nullptr);
        return;
    }
    auto mark_call = [domain, msg_ptr = std::make_shared<std::string>(message), stream]() -> int {
        (void)at_npu::native::MstxDomainMarkA(domain, msg_ptr->c_str(), stream);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("mstx_domain_mark_op", mark_call);
}

int MstxMgr::domainRangeStart(mstxDomainHandle_t domain, const char* message, const aclrtStream stream)
{
    if (!isMstxEnable()) {
        return 0;
    }
    int id = ptRangeId_++;
    if (stream == nullptr) {
        int res = at_npu::native::MstxDomainRangeStartA(domain, message, nullptr, id);
        return id;
    }
    {
        std::lock_guard<std::mutex> lock(mtx_);
        ptRangeIdsWithStream_.insert(id);
    }
    auto range_start_call = [domain, msg_ptr = std::make_shared<std::string>(message), stream, id]() -> int {
        int taskId = at_npu::native::MstxDomainRangeStartA(domain, msg_ptr->c_str(), stream, id);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("mstx_domain_range_start_op", range_start_call);
    return id;
}

void MstxMgr::domainRangeEnd(mstxDomainHandle_t domain, int ptRangeId)
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
        at_npu::native::MstxDomainRangeEnd(domain, ptRangeId);
        return;
    }
    auto range_end_call = [domain, ptRangeId]() -> int {
        at_npu::native::MstxDomainRangeEnd(domain, ptRangeId);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("mstx_domain_range_end_op", range_end_call);
}

mstxMemHeapHandle_t MstxMgr::memHeapRegister(mstxDomainHandle_t domain, mstxMemVirtualRangeDesc_t* desc)
{
    if (!isMsleaksEnable() || desc==nullptr) {
        return nullptr;
    }
    mstxMemHeapDesc_t heapDesc;
    heapDesc.typeSpecificDesc = reinterpret_cast<void const *>(desc);
    return at_npu::native::MstxMemHeapRegister(domain, &heapDesc);
}

void MstxMgr::memHeapUnregister(mstxDomainHandle_t domain, void* ptr)
{
    if (!isMsleaksEnable() || ptr == nullptr) {
        return;
    }
    at_npu::native::MstxMemHeapUnregister(domain, reinterpret_cast<mstxMemHeapHandle_t>(ptr));
}

void MstxMgr::memRegionsRegister(mstxDomainHandle_t domain, mstxMemVirtualRangeDesc_t* desc)
{
    if (!isMsleaksEnable() || desc == nullptr) {
        return;
    }
    mstxMemRegionsRegisterBatch_t batch;
    batch.regionCount = 1;
    batch.regionDescArray = reinterpret_cast<const void *>(desc);
    at_npu::native::MstxMemRegionsRegister(domain, &batch);
}

void MstxMgr::memRegionsUnregister(mstxDomainHandle_t domain, void* ptr)
{
    if (!isMsleaksEnable() || ptr == nullptr) {
        return;
    }
    mstxMemRegionsUnregisterBatch_t unregisterBatch;
    unregisterBatch.refCount = 1;
    mstxMemRegionRef_t regionRef[1] = {};
    regionRef[0].refType = MSTX_MEM_REGION_REF_TYPE_POINTER;
    regionRef[0].pointer = ptr;
    unregisterBatch.refArray = regionRef;
    at_npu::native::MstxMemRegionsUnregister(domain, &unregisterBatch);
}


bool MstxMgr::isMsleaksEnable()
{
    static bool isEnable = isMsleaksEnableImpl();
    return isEnable;
}

bool MstxMgr::isMsleaksEnableImpl()
{
    bool ret = false;
    const char* envVal = std::getenv("LD_PRELOAD");
    if (envVal == nullptr) {
        return ret;
    }
    static const std::string soName = "libascend_kernel_hook.so";
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