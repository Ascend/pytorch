#include <unistd.h>
#include <ATen/record_function.h>

#include "torch_npu/csrc/core/npu/CachingHostAllocator.h"
#include "torch_npu/csrc/core/npu/NPUEventManager.h"
#include "torch_npu/csrc/core/npu/NPUQueue.h"
#include "torch_npu/csrc/core/npu/interface/AsyncTaskQueueInterface.h"
#include "torch_npu/csrc/distributed/HCCLUtils.hpp"
#include "torch_npu/csrc/framework/OpCmdHelper.h"
#include "torch_npu/csrc/framework/OpParamMaker.h"
#include "torch_npu/csrc/framework/aoe/AoeDumpGraphManager.h"
#include "torch_npu/csrc/framework/interface/HcclInterface.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/NpuUtils.h"

#ifndef BUILD_LIBTORCH
#include <Python.h>
#endif

namespace at_npu {
namespace native {

static bool deterministicaclnn_oldstatus = false;

void OpAttrMaker::Set(aclopAttr *attr, const string &name, bool value)
{
    aclopSetAttrBool(attr, name.c_str(), value);
}

void OpAttrMaker::Set(aclopAttr *attr, const string &name, int64_t value)
{
    aclopSetAttrInt(attr, name.c_str(), value);
}

void OpAttrMaker::Set(aclopAttr *attr, const string &name, float value)
{
    aclopSetAttrFloat(attr, name.c_str(), value);
}

void OpAttrMaker::Set(aclopAttr *attr, const string &name, string value)
{
    aclopSetAttrString(attr, name.c_str(), value.c_str());
}

void OpAttrMaker::Set(aclopAttr *attr, const string &name, c10::IntArrayRef value)
{
    aclopSetAttrListInt(attr, name.c_str(), value.size(), value.data());
}

void OpAttrMaker::Set(aclopAttr *attr, const string &name, at::ArrayRef<float> value)
{
    aclopSetAttrListFloat(attr, name.c_str(), value.size(), value.data());
}

void OpAttrMaker::Set(aclopAttr *attr, const string &name, at::ArrayRef<uint8_t> value)
{
    aclopSetAttrListBool(attr, name.c_str(), value.size(), value.data());
}

void OpAttrMaker::Set(aclopAttr *attr, const string &name, c10::Scalar value)
{
    float val = CalcuOpUtil::GetScalarFloatValue(value);
    aclopSetAttrFloat(attr, name.c_str(), val);
}

void OpAttrMaker::Set(aclopAttr *attr, const string &name, at::ScalarType value)
{
    aclDataType val = CalcuOpUtil::ConvertToAclDataType(value);
    aclopSetAttrDataType(attr, name.c_str(), val);
}

void OpAttrMaker::Set(aclopAttr *attr, const string &name, at::ArrayRef<c10::IntArrayRef> value)
{
    // Pointer to values of each listInt.
    c10::SmallVector<int64_t *, N> attrValue;
    // Pointer to number of each listInt.
    c10::SmallVector<int, N> eachListIntNum;
    // Value of each listInt.
    c10::SmallVector<c10::SmallVector<int64_t, N>, N> eachListIntVal;
    for (const auto i : c10::irange(value.size())) {
        c10::SmallVector<int64_t, N> listInt;
        int64_t valueSize = static_cast<int64_t>(value[i].size());
        listInt.resize(valueSize);
        std::copy(value[i].begin(), value[i].end(), listInt.begin());
        eachListIntVal.emplace_back(listInt);
        attrValue.emplace_back(eachListIntVal.back().data());
        eachListIntNum.emplace_back(valueSize);
    }

    aclopSetAttrListListInt(attr, name.c_str(), attrValue.size(), eachListIntNum.data(), attrValue.data());
}

void OpCommandImpl::SetEnginePriority()
{
    auto stream = c10_npu::getCurrentNPUStream();
    if (stream.isDataPreprocessStream()) {
        AddAttr("_performance_prior", true);
        AddAttr<std::string>("_exclude_engines", "AiCore");
    }
}

inline void SetDeterministicOption(bool deterministicAlgorithmsStatus, bool isOpapi)
{
    if (deterministicaclnn_oldstatus != deterministicAlgorithmsStatus) {
        if (!isOpapi) {
            NPU_CHECK_ERROR(
                AclSetCompileopt(aclCompileOpt::ACL_OP_DETERMINISTIC, deterministicAlgorithmsStatus ? "1" : "0"));
        }
        NPU_CHECK_ERROR(
            AclrtCtxSetSysParamOpt(aclSysParamOpt::ACL_OPT_DETERMINISTIC, deterministicAlgorithmsStatus ? 1 : 0));
        NPU_CHECK_ERROR(
            AclrtSetSysParamOpt(aclSysParamOpt::ACL_OPT_DETERMINISTIC, deterministicAlgorithmsStatus ? 1 : 0));
        HcclConfigValue configValue = {deterministicAlgorithmsStatus ? 1 : 0};
        HCCL_CHECK_ERROR(hccl::HcclSetConfig(HcclConfig::HCCL_DETERMINISTIC, configValue));
        deterministicaclnn_oldstatus = deterministicAlgorithmsStatus;
    }
}

void SetDeterministic(bool isOpapi)
{
    auto deterministicAlgorithmsStatus = at::globalContext().deterministicAlgorithms();
    SetDeterministicOption(deterministicAlgorithmsStatus, isOpapi);
}

void SetDeterministicOps(bool deterministicAlgorithmsStatus)
{
    SetDeterministicOption(deterministicAlgorithmsStatus, true);
}

void OpCommandImpl::Run(
    bool sync,
    c10::SmallVector<int64_t, N> &sync_index,
    c10::SmallVector<at::Tensor, N> &outputTensor)
{
    ASCEND_LOGD("Op %s Run.", opName.c_str());
    RECORD_FUNCTION(opName, std::vector<c10::IValue>({}));
#ifndef BUILD_LIBTORCH
    if (PyGILState_Check() != 0) {
        // we need to release GIL for NPU to compile op.
        Py_BEGIN_ALLOW_THREADS;
        ACL_REQUIRE_OK_OP(InnerRun(opName, execParam, sync, sync_index, outputTensor), opName.c_str());
        Py_END_ALLOW_THREADS;
    } else {
        ACL_REQUIRE_OK_OP(InnerRun(opName, execParam, sync, sync_index, outputTensor), opName.c_str());
    }
#else
    ACL_REQUIRE_OK_OP(InnerRun(opName, execParam, sync, sync_index, outputTensor), opName.c_str());
#endif
}

void OpCommandImpl::RunOpApi(const string &op_name, PROC_FUNC func)
{
    ASCEND_LOGD("Op %s Run.", op_name.c_str());
    RECORD_FUNCTION(op_name, std::vector<c10::IValue>({}));
#ifndef BUILD_LIBTORCH
    if (PyGILState_Check() != 0) {
        // we need to release GIL for NPU to compile op.
        Py_BEGIN_ALLOW_THREADS;
        ACL_REQUIRE_OK_OP(InnerRunOpApi(op_name, func), op_name.c_str());
        Py_END_ALLOW_THREADS;
    } else {
        ACL_REQUIRE_OK_OP(InnerRunOpApi(op_name, func), op_name.c_str());
    }
#else
    ACL_REQUIRE_OK_OP(InnerRunOpApi(op_name, func), op_name.c_str());
#endif
}

aclError OpCommandImpl::InnerRun(
    const string &name,
    AclExecParam &params,
    bool sync,
    c10::SmallVector<int64_t, N> &sync_index,
    c10::SmallVector<at::Tensor, N> &outputTensor)
{
    aclError ret;
    auto stream = c10_npu::getCurrentNPUStream();
    if (stream.getRepoStopFlag()) {
        ASCEND_LOGE("getRepoStopFlag in InnerRun, throw FORCE STOP.");
        throw std::runtime_error("FORCE STOP." + PTA_ERROR(ErrCode::ACL));
    }
    auto inputSize = params.inBuffer.size();
    auto outputSize = params.outBuffer.size();
    // open the deterministicAlgorithms config
    SetDeterministic(false);
    bool reset_flag = false;
    if (ForceJitCompileList::GetInstance().Inlist(name) && env::CheckJitDisable()) {
        NPU_CHECK_ERROR(AclSetCompileopt(aclCompileOpt::ACL_OP_JIT_COMPILE, "enable"));
        reset_flag = true;
    }
    int index = 0;
    do {
        if (params.customHandler) {
            ret = params.customHandler();
            OPS_CHECK_ERROR(ret, name.c_str());
            index++;
            continue;
        }

        if (at_npu::native::aoe::aoe_manager().IsAoeEnabled() &&
            at_npu::native::aoe::aoe_manager().IsInWhitelist(name)) {
            ret = at_npu::native::AclGenGraphAndDumpForOp(
                name.c_str(),
                inputSize,
                params.inDesc.data(),
                params.inBuffer.data(),
                outputSize,
                params.outDesc.data(),
                params.outBuffer.data(),
                params.attr,
                ACL_ENGINE_SYS,
                at_npu::native::aoe::aoe_manager().GetDumpGraphPath().c_str(),
                nullptr);
            if (ret != ACL_ERROR_NONE) {
                CHECK_AND_THROW_ERROR_WITH_SPECIFIC_MESSAGE(ret);
                C10_NPU_SHOW_ERR_MSG();
                TORCH_CHECK(false, "In aoe mode, AclGenGraphAndDumpForOp failed!", PTA_ERROR(ErrCode::ACL));
            }
        }
        if (!sync) {
            ret = aclopCompileAndExecute(
                name.c_str(),
                inputSize,
                params.inDesc.data(),
                params.inBuffer.data(),
                outputSize,
                params.outDesc.data(),
                params.outBuffer.data(),
                params.attr,
                ACL_ENGINE_SYS,
                ACL_COMPILE_SYS,
                nullptr,
                stream);
            OPS_CHECK_ERROR(ret, name.c_str());
        } else {
            int64_t dimSize;
            ret = AclopCompileAndExecuteV2(
                name.c_str(),
                inputSize,
                const_cast<aclTensorDesc **>(params.inDesc.data()),
                const_cast<aclDataBuffer **>(params.inBuffer.data()),
                outputSize,
                const_cast<aclTensorDesc **>(params.outDesc.data()),
                params.outBuffer.data(),
                params.attr,
                ACL_ENGINE_SYS,
                ACL_COMPILE_SYS,
                nullptr,
                stream);
            OPS_CHECK_ERROR(ret, name.c_str());
            for (size_t i = 0; i < sync_index.size(); i++) {
                c10::SmallVector<int64_t, N> real_shape;
                for (int64_t j = 0; j < outputTensor[sync_index[i]].dim(); j++) {
                    NPU_CHECK_ERROR(aclGetTensorDescDimV2(params.outDesc[sync_index[i]], j, &dimSize));
                    real_shape.emplace_back(dimSize);
                }
                outputTensor[sync_index[i]].resize_(real_shape);
            }
        }
        ++index;
    } while (NpuUtils::IsOomError(ret, index) && (index < NPU_MAX_OP_EXEC_TRY_NUM));
    if (reset_flag) {
        NPU_CHECK_ERROR(AclSetCompileopt(aclCompileOpt::ACL_OP_JIT_COMPILE, "disable"));
    }
    return ret;
}

aclError OpCommandImpl::InnerRunOpApi(const string &op_name, PROC_FUNC func)
{
    aclError ret;
    auto stream = c10_npu::getCurrentNPUStream();
    if (stream.getRepoStopFlag()) {
        ASCEND_LOGE("getRepoStopFlag in InnerRun, throw FORCE STOP.");
        throw std::runtime_error("FORCE STOP." + PTA_ERROR(ErrCode::ACL));
    }
    // open the deterministicAlgorithms config
    SetDeterministic();
    int index = 0;
    do {
        ret = func();
        OPS_CHECK_ERROR(ret, op_name.c_str());
        index++;
    } while (NpuUtils::IsOomError(ret, index) && (index < NPU_MAX_OP_EXEC_TRY_NUM));
    return ret;
}

void printErrorLog(ExecuteParas *cur_paras)
{
    ASCEND_LOGE("---OpName---%s", cur_paras->opType);
    for (int i = 0; i < cur_paras->paras.input_num; i++) {
        const aclTensorDesc *tensorDesc = cur_paras->paras.input_desc[i];
        aclDataType dataType = aclGetTensorDescType(tensorDesc);
        aclFormat descformat = aclGetTensorDescFormat(tensorDesc);

        int descNumDims = static_cast<int>(aclGetTensorDescNumDims(tensorDesc));
        std::string descShape = "[";
        for (int j = 0; j < descNumDims; j++) {
            int64_t dimSize = 0;
            aclGetTensorDescDimV2(tensorDesc, j, &dimSize);
            descShape = descShape + std::to_string(dimSize);
            if (j < descNumDims - 1) {
                descShape += ", ";
            }
        }
        descShape += "]";

        ASCEND_LOGE(
            "InputDesc[%d]: DescType = %s, DescFormat = %s, DescShape = %s",
            i,
            (AclDateTypeToString(dataType)).c_str(),
            (AclFormatToString(descformat)).c_str(),
            descShape.c_str());
    }
}

bool ContainsAny(const std::string& str, std::initializer_list<std::string> patterns)
{
    for (const auto& pattern : patterns) {
        if (str.find(pattern) != std::string::npos) {
            return true;
        }
    }
    return false;
}

int ExecFunc(c10_npu::queue::QueueParas *in, aclrtStream stream)
{
    auto cur_paras = static_cast<ExecuteParas *>(in->paramVal);
    ASCEND_LOGD("Op %s Run.", cur_paras->opType);
    aclError ret;
    // open the deterministicAlgorithms config
    SetDeterministic(false);
    if (cur_paras->customHandler) {
        ASCEND_LOGD("Exec Op %s with custom handle", cur_paras->opType);
        try {
            ret = cur_paras->customHandler();
        } catch (std::exception &e) {
            if (ContainsAny(std::string(e.what()), {DEVICE_TASK_ABORT, DEVICE_MEM_ERROR, DEVICE_HBM_ECC_ERROR,
                SUSPECT_DEVICE_MEM_ERROR, HCCS_LINK_ERROR, HCCL_OP_RETRY_FAILED})) {
                ret = c10_npu::acl::AclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
            } else {
                ret = ACL_ERROR_INVALID_PARAM;
                LOG(ERROR) << e.what();
            }
            ASCEND_LOGE("Custom hand error:%s", e.what());
        }
        if (ret != ACL_ERROR_NONE) {
            ASCEND_LOGE("Custom hand fail! name=%s, ret=0x%#x", cur_paras->opType, ret);
        }
        return ret;
    }
    bool reset_flag = false;
    if (!cur_paras->isJitDisable) {
        NPU_CHECK_ERROR_WITHOUT_UCE(AclSetCompileopt(aclCompileOpt::ACL_OP_JIT_COMPILE, "enable"));
        reset_flag = true;
    }
    if (at_npu::native::aoe::aoe_manager().IsAoeEnabled() &&
        at_npu::native::aoe::aoe_manager().IsInWhitelist(cur_paras->opType)) {
        ret = at_npu::native::AclGenGraphAndDumpForOp(
            cur_paras->opType,
            cur_paras->paras.input_num,
            cur_paras->paras.input_desc,
            cur_paras->paras.input_data_buf,
            cur_paras->paras.output_num,
            cur_paras->paras.output_desc,
            cur_paras->paras.output_data_buf,
            cur_paras->attr,
            ACL_ENGINE_SYS,
            at_npu::native::aoe::aoe_manager().GetDumpGraphPath().c_str(),
            nullptr);
        if (ret != ACL_ERROR_NONE) {
            auto ret_temp = c10_npu::acl::AclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
            if (ret_temp != ACL_ERROR_NONE) {
                ret = ret_temp;
            }
            ASCEND_LOGE("In aoe mode, AclGenGraphAndDumpForOp failed!");
            return ret;
        }
    }
    ret = aclopCompileAndExecute(
        cur_paras->opType,
        cur_paras->paras.input_num,
        cur_paras->paras.input_desc,
        cur_paras->paras.input_data_buf,
        cur_paras->paras.output_num,
        cur_paras->paras.output_desc,
        cur_paras->paras.output_data_buf,
        cur_paras->attr,
        ACL_ENGINE_SYS,
        ACL_COMPILE_SYS,
        nullptr,
        stream);
    if (reset_flag) {
        NPU_CHECK_ERROR_WITHOUT_UCE(AclSetCompileopt(aclCompileOpt::ACL_OP_JIT_COMPILE, "disable"));
    }

    if (ret != ACL_ERROR_NONE) {
        auto ret_temp = c10_npu::acl::AclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
        if (ret_temp != ACL_ERROR_NONE) {
            ret = ret_temp;
        }
        printErrorLog(cur_paras);
    }

    return ret;
}

int ExecFuncOpApi(c10_npu::queue::QueueParas *in, aclrtStream stream)
{
    auto cur_paras = static_cast<ExecuteParasOpApi *>(in->paramVal);
    ASCEND_LOGD("Op %s Run.", cur_paras->opType);
    aclError ret;

    ASCEND_LOGD("Exec Op %s with custom handle", cur_paras->opType);

    if (cur_paras->customHandler == nullptr) {
        ASCEND_LOGW("Custom hand is nullptr! name=%s", cur_paras->opType);
        return ACL_ERROR_NONE;
    }

    try {
        ret = cur_paras->customHandler();
    } catch (std::exception &e) {
        if (ContainsAny(std::string(e.what()), {DEVICE_TASK_ABORT, DEVICE_MEM_ERROR, DEVICE_HBM_ECC_ERROR,
            SUSPECT_DEVICE_MEM_ERROR, HCCS_LINK_ERROR, HCCL_OP_RETRY_FAILED})) {
            ret = c10_npu::acl::AclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
        } else {
            ret = ACL_ERROR_INVALID_PARAM;
            LOG(ERROR) << e.what();
        }
        ASCEND_LOGE("Custom hand error:%s", e.what());
    }
    if (ret != ACL_ERROR_NONE) {
        ASCEND_LOGE("Custom hand fail! name=%s, ret=0x%#x", cur_paras->opType, ret);
    }
    return ret;
}

int MemcopyAsyncFunc(c10_npu::queue::QueueParas *in, aclrtStream stream)
{
    auto cur_paras = static_cast<c10_npu::queue::CopyParas *>(in->paramVal);
    aclError ret =
        aclrtMemcpyAsync(cur_paras->dst, cur_paras->dstLen, cur_paras->src, cur_paras->srcLen, cur_paras->kind, stream);
    if (ret != ACL_ERROR_NONE) {
        auto ret_temp = c10_npu::acl::AclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
        if (ret_temp != ACL_ERROR_NONE) {
            ret = ret_temp;
        }
        ASCEND_LOGE(
            "aclrtMemcpyAsync error! ret = %d, dstLen = %zu, srcLen = %zu, kind = %d",
            ret,
            cur_paras->dstLen,
            cur_paras->srcLen,
            cur_paras->kind);
    }
    return ret;
}

int RecordEventFunc(c10_npu::queue::QueueParas *in, aclrtStream stream)
{
    auto cur_paras = static_cast<c10_npu::queue::EventParas *>(in->paramVal);

    aclError ret = aclrtRecordEvent(cur_paras->event, stream);
    if (ret != ACL_ERROR_NONE) {
        auto ret_temp = c10_npu::acl::AclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
        if (ret_temp != ACL_ERROR_NONE) {
            ret = ret_temp;
        }
        ASCEND_LOGE("aclrtRecordEvent error! ret = %d, eventAllocatorType = %d", ret, cur_paras->eventAllocatorType);
    }
    c10_npu::NPUEventManager::GetInstance().DecreaseUnrecordedCount(cur_paras->event);
    ASCEND_LOGI(
        "Event: aclrtRecordEvent dequeue is successfully executed, stream=%p, event=%p",
        stream,
        cur_paras->event);

    return ret;
}

int WaitEventFunc(c10_npu::queue::QueueParas *in, aclrtStream stream)
{
    auto cur_paras = static_cast<c10_npu::queue::EventParas *>(in->paramVal);
    aclError ret = aclrtStreamWaitEvent(stream, cur_paras->event);
    if (ret != ACL_ERROR_NONE) {
        auto ret_temp = c10_npu::acl::AclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
        if (ret_temp != ACL_ERROR_NONE) {
            ret = ret_temp;
        }
        ASCEND_LOGE(
            "aclrtStreamWaitEvent error! ret = %d, eventAllocatorType = %d",
            ret,
            cur_paras->eventAllocatorType);
    }
    ASCEND_LOGI(
        "Event: aclrtStreamWaitEvent dequeue is successfully executed, stream=%p, event=%p",
        stream,
        cur_paras->event);
    return ret;
}

int LazyDestroyEventFunc(c10_npu::queue::QueueParas *in, aclrtStream stream)
{
    auto cur_paras = static_cast<c10_npu::queue::EventParas *>(in->paramVal);
    aclError ret = c10_npu::NPUEventManager::GetInstance().LazyDestroy(cur_paras->event);
    if (ret != ACL_ERROR_NONE) {
        auto ret_temp = c10_npu::acl::AclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
        if (ret_temp != ACL_ERROR_NONE) {
            ret = ret_temp;
        }
        ASCEND_LOGE("LazyDestroy error! ret = %d, eventAllocatorType = %d", ret, cur_paras->eventAllocatorType);
    }
    ASCEND_LOGD("Event: LazyDestroyEventFunc dequeue is successfully executed, event=%p", cur_paras->event);
    return ret;
}

void CopyFunc(void *dst, void *src)
{
    auto dstPtr = static_cast<c10_npu::queue::QueueParas *>(dst);
    auto srcPtr = static_cast<c10_npu::queue::QueueParas *>(src);
    dstPtr->paramVal = static_cast<uint8_t *>(dst) + sizeof(c10_npu::queue::QueueParas);
    if (dstPtr->paramType == c10_npu::queue::EXECUTE_OPAPI) {
        // string or smallvector of struct is used, deconstructor need be called before memset
        (static_cast<ExecuteParasOpApi *>(dstPtr->paramVal))->~ExecuteParasOpApi();
    } else if (dstPtr->paramType == c10_npu::queue::COMPILE_AND_EXECUTE) {
        // string or smallvector of struct is used, deconstructor need be called before memset
        (static_cast<ExecuteParas *>(dstPtr->paramVal))->~ExecuteParas();
    }

    const bool is_opapi_v2 = (srcPtr->paramType == c10_npu::queue::EXECUTE_OPAPI_V2);
    dstPtr->paramStream = srcPtr->paramStream;
    if (is_opapi_v2) {
        dstPtr->paramType = c10_npu::queue::EXECUTE_OPAPI;
    } else {
        dstPtr->paramType = srcPtr->paramType;
    }
    dstPtr->paramLen = srcPtr->paramLen;
    dstPtr->correlation_id = srcPtr->correlation_id;
    if (dstPtr->paramType == c10_npu::queue::EXECUTE_OPAPI) {
        new (dstPtr->paramVal) ExecuteParasOpApi();
        if (is_opapi_v2) {
            (static_cast<ExecuteParasOpApi *>(dstPtr->paramVal))
                ->Copy(*(static_cast<ExecuteParasOpApiV2 *>(srcPtr->paramVal)));
        } else {
            (static_cast<ExecuteParasOpApi *>(dstPtr->paramVal))
                ->Copy(*(static_cast<ExecuteParasOpApi *>(srcPtr->paramVal)));
        }
    } else if (srcPtr->paramType == c10_npu::queue::COMPILE_AND_EXECUTE) {
        new (dstPtr->paramVal) ExecuteParas();
        (static_cast<ExecuteParas *>(dstPtr->paramVal))->Copy(*(static_cast<ExecuteParas *>(srcPtr->paramVal)));
    } else if ((srcPtr->paramType == c10_npu::queue::ASYNC_MEMCPY)) {
        new (dstPtr->paramVal) c10_npu::queue::CopyParas();
        (static_cast<c10_npu::queue::CopyParas *>(dstPtr->paramVal))
            ->Copy(*(static_cast<c10_npu::queue::CopyParas *>(srcPtr->paramVal)));
    } else {
        new (dstPtr->paramVal) c10_npu::queue::EventParas();
        (static_cast<c10_npu::queue::EventParas *>(dstPtr->paramVal))
            ->Copy(*(static_cast<c10_npu::queue::EventParas *>(srcPtr->paramVal)));
    }
}

void ReleaseFunc(void *ptr, c10_npu::ReleaseQueue &releaseQueue)
{
    releaseQueue.PushToReleaseQueue(ptr);
}

void *NewFunc(int caption, int &size)
{
    size = static_cast<int>(sizeof(c10_npu::queue::QueueParas) + MAX_PARAS_BYTE_SIZE);
    void *ptr = malloc(size * caption);
    TORCH_CHECK(ptr != nullptr, "OpCommand new buffer must be not NULL", PTA_ERROR(ErrCode::PTR));
    memset(ptr, 0, size * caption);
    return ptr;
}

void DeleteFunc(void *ptr)
{
    free(ptr);
}

using Func = int (*)(c10_npu::queue::QueueParas *, aclrtStream);
using AsyncFuncMap = std::map<c10_npu::queue::QueueParamType, Func>;
AsyncFuncMap funcMap = {
    {c10_npu::queue::COMPILE_AND_EXECUTE, ExecFunc},
    {c10_npu::queue::EXECUTE_OPAPI, ExecFuncOpApi},
    {c10_npu::queue::ASYNC_MEMCPY, MemcopyAsyncFunc},
    {c10_npu::queue::RECORD_EVENT, RecordEventFunc},
    {c10_npu::queue::WAIT_EVENT, WaitEventFunc},
    {c10_npu::queue::LAZY_DESTROY_EVENT, LazyDestroyEventFunc},
};

int AsncExecFunc(void *data)
{
    auto queueParam = static_cast<c10_npu::queue::QueueParas *>(data);
    auto type = queueParam->paramType;
    aclrtStream stream = queueParam->paramStream;
    auto ret = funcMap[type](queueParam, stream);
    return ret;
}

void CopyReleaseParamFunc(void *dst, void *src)
{
    auto dstPtr = static_cast<c10_npu::queue::QueueParas *>(dst);
    auto srcPtr = static_cast<c10_npu::queue::QueueParas *>(src);
    dstPtr->paramType = srcPtr->paramType;
    dstPtr->paramVal = static_cast<uint8_t *>(dst) + sizeof(c10_npu::queue::QueueParas);
    if (srcPtr->paramType == c10_npu::queue::EXECUTE_OPAPI) {
        (static_cast<ExecuteParasOpApi *>(dstPtr->paramVal))->Copy(*(static_cast<ExecuteParasOpApi *>(srcPtr->paramVal)));
        (static_cast<ExecuteParasOpApi *>(srcPtr->paramVal))->Release();
    } else if (srcPtr->paramType == c10_npu::queue::COMPILE_AND_EXECUTE) {
        (static_cast<ExecuteParas *>(dstPtr->paramVal))->CopyEx(*(static_cast<ExecuteParas *>(srcPtr->paramVal)));
        (static_cast<ExecuteParas *>(srcPtr->paramVal))->hostMemory.clear();
    }
}

void ReleaseParamFunc(void *ptr)
{
    auto queueParam = static_cast<c10_npu::queue::QueueParas *>(ptr);
    auto type = queueParam->paramType;
    if (type == c10_npu::queue::EXECUTE_OPAPI) {
        auto cur_paras = static_cast<ExecuteParasOpApi *>(queueParam->paramVal);
        cur_paras->Release();
    } else if (type == c10_npu::queue::COMPILE_AND_EXECUTE) {
        auto cur_paras = static_cast<ExecuteParas *>(queueParam->paramVal);
        cur_paras->Release();
    }
}

REGISTER_QUEUE_FUNC(AsncExecFunc, CopyFunc, ReleaseFunc, NewFunc, DeleteFunc, CopyReleaseParamFunc, ReleaseParamFunc)

OpCommandImpls *OpCommandImpls::GetInstance()
{
    thread_local OpCommandImpls impl;
    return &impl;
}

void OpCommandImpls::Push(OpCommandImpl *&ptr)
{
    ++offset;
    if (static_cast<int32_t>(objs.size()) <= offset) {
        OpCommandImpl impl;
        objs.emplace_back(std::move(impl));
    }
    TORCH_CHECK(objs.size() > offset, "OpCommand size (", objs.size(), ") is smaller than offset (", offset, ")",
                PTA_ERROR(ErrCode::VALUE));
    ptr = &objs[offset];
}

void OpCommandImpls::Pop()
{
    TORCH_CHECK(offset >= 0, "OpCommand current offset should not be less than ", offset,
                PTA_ERROR(ErrCode::VALUE));
    offset -= 1;
}
}  // namespace native
}  // namespace at_npu
