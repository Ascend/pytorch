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
void markImpl(const char* message, const aclrtStream stream, mstxDomainHandle_t domain)
{
    if (domain == nullptr) {
        (void)at_npu::native::MstxMarkA(message, stream);
    } else {
        (void)at_npu::native::MstxDomainMarkA(domain, message, stream);
    }
}

void rangeStartImpl(const char* message, const aclrtStream stream, int ptRangeId, mstxDomainHandle_t domain)
{
    if (domain == nullptr) {
        (void)at_npu::native::MstxRangeStartA(message, stream, ptRangeId);
    } else {
        (void)at_npu::native::MstxDomainRangeStartA(domain, message, stream, ptRangeId);
    }
}

void rangeEndImpl(int ptRangeId, mstxDomainHandle_t domain)
{
    if (domain == nullptr) {
        at_npu::native::MstxRangeEnd(ptRangeId);
    } else {
        at_npu::native::MstxDomainRangeEnd(domain, ptRangeId);
    }
}

MstxMgr::MstxMgr()
{
}

void MstxMgr::mark(const char* message, const aclrtStream stream, const char* domain)
{
    if (!isMstxEnable()) {
        return;
    }
    std::string domainStr(domain);
    if (!isMstxTxDomainEnable(domainStr)) {
        return;
    }
    mstxDomainHandle_t domainHandle = createProfDomain(domainStr);
    if (stream == nullptr) {
        markImpl(message, nullptr, domainHandle);
        return;
    }
    auto mark_call = [msg_ptr = std::make_shared<std::string>(message), stream, domainHandle]() -> int {
        markImpl(msg_ptr->c_str(), stream, domainHandle);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("mstx_mark_op", mark_call);
}

int MstxMgr::rangeStart(const char* message, const aclrtStream stream, const char* domain)
{
    if (!isMstxEnable()) {
        return 0;
    }
    std::string domainStr(domain);
    if (!isMstxTxDomainEnable(domainStr)) {
        return 0;
    }
    mstxDomainHandle_t domainHandle = createProfDomain(domainStr);
    int id = ptRangeId_++;
    if (stream == nullptr) {
        rangeStartImpl(message, nullptr, id, domainHandle);
        return id;
    }
    {
        std::lock_guard<std::mutex> lock(mtx_);
        ptRangeIdsWithStream_.insert(id);
    }
    auto range_start_call = [msg_ptr = std::make_shared<std::string>(message), stream, id, domainHandle]() -> int {
        rangeStartImpl(msg_ptr->c_str(), stream, id, domainHandle);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("mstx_range_start_op", range_start_call);
    return id;
}

void MstxMgr::rangeEnd(int ptRangeId, const char* domain)
{
    if (!isMstxEnable() || ptRangeId == 0) {
        return;
    }
    std::string domainStr(domain);
    if (!isMstxTxDomainEnable(domainStr)) {
        return;
    }
    mstxDomainHandle_t domainHandle = createProfDomain(domainStr);
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
        rangeEndImpl(ptRangeId, domainHandle);
        return;
    }
    auto range_end_call = [ptRangeId, domainHandle]() -> int {
        rangeEndImpl(ptRangeId, domainHandle);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("mstx_range_end_op", range_end_call);
}

int MstxMgr::getRangeId()
{
    return ptRangeId_++;
}

mstxDomainHandle_t MstxMgr::createProfDomain(const std::string &name)
{
    if (!at_npu::native::IsSupportMstxDomainFunc()) {
        return nullptr;
    }
    if (name == DOMAIN_DEFAULT) { // don't need to create default domain
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(mstxDomainsMtx);
    auto iter = mstxDomains_.find(name);
    if (iter != mstxDomains_.end()) {
        return iter->second;
    }
    mstxDomainHandle_t handle = at_npu::native::MstxDomainCreateA(name.c_str());
    if (handle != nullptr) {
        mstxDomains_.emplace(name, handle);
    }
    return handle;
}

mstxDomainHandle_t MstxMgr::createLeaksDomain(const char* name)
{
    if (!at_npu::native::IsSupportMstxFunc()) {
        return nullptr;
    }
    return at_npu::native::MstxDomainCreateA(name);
}

void MstxMgr::destroyDomain(mstxDomainHandle_t domain)
{
    at_npu::native::MstxDomainDestroy(domain);
}

mstxMemHeapHandle_t MstxMgr::memHeapRegister(mstxDomainHandle_t domain, mstxMemVirtualRangeDesc_t* desc)
{
    if (!at_npu::native::IsSupportMstxFunc() || desc == nullptr) {
        return nullptr;
    }
    mstxMemHeapDesc_t heapDesc;
    heapDesc.typeSpecificDesc = reinterpret_cast<void const *>(desc);
    return at_npu::native::MstxMemHeapRegister(domain, &heapDesc);
}

void MstxMgr::memHeapUnregister(mstxDomainHandle_t domain, void* ptr)
{
    if (!at_npu::native::IsSupportMstxFunc() || ptr == nullptr) {
        return;
    }
    at_npu::native::MstxMemHeapUnregister(domain, reinterpret_cast<mstxMemHeapHandle_t>(ptr));
}

void MstxMgr::memRegionsRegister(mstxDomainHandle_t domain, mstxMemVirtualRangeDesc_t* desc)
{
    if (!at_npu::native::IsSupportMstxFunc() || desc == nullptr) {
        return;
    }
    mstxMemRegionsRegisterBatch_t batch;
    batch.regionCount = 1;
    batch.regionDescArray = reinterpret_cast<const void *>(desc);
    at_npu::native::MstxMemRegionsRegister(domain, &batch);
}

void MstxMgr::memRegionsUnregister(mstxDomainHandle_t domain, void* ptr)
{
    if (!at_npu::native::IsSupportMstxFunc() || ptr == nullptr) {
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

bool MstxMgr::isMstxTxDomainEnable(const std::string &domainName)
{
    if (isProfTxEnable()) {
        return ProfilerMgr::GetInstance()->IsMstxDomainEnabled(domainName);
    }
    return true;
}
}
}