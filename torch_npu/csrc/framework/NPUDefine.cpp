#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/framework/NPUDefine.h"

namespace at_npu {
namespace native {
std::atomic<uint64_t> ExecuteParas::g_pta_correlation_id{0};
void ExecuteParas::Release()
{
    // if useDynamicCompile, this attr will be freed in dynamic compile.
    if (attr != nullptr) {
        aclopDestroyAttr(attr);
    }
    DestroyConstParams(constParams);
    NPUStatus ret = DestroyAclParams(paras);
    if (ret != SUCCESS) {
        ASCEND_LOGE("DestroyAclParams fail, ret: %s", ret.c_str());
    }
    hostMemory.clear();
    customHandler = nullptr;
    return;
}

void ExecuteParas::Copy(ExecuteParas &other)
{
    strncpy(this->opType, other.opType, sizeof(ExecuteParas::opType) - 1);
    this->paras = other.paras;
    this->attr = other.attr;
    this->constParams = other.constParams;
    this->hostMemory = other.hostMemory;
    this->isJitDisable = other.isJitDisable;
    this->customHandler = other.customHandler;
    this->pta_correlation_id = other.pta_correlation_id;
}

void ExecuteParas::CopyEx(ExecuteParas& other)
{
    this->paras = other.paras;
    this->attr = other.attr;
    this->constParams = other.constParams;
}

void ExecuteParasOpApi::Release()
{
    customHandler = nullptr;
}

void ExecuteParasOpApi::Copy(ExecuteParasOpApi &other)
{
    strncpy(this->opType, other.opType, sizeof(ExecuteParasOpApi::opType) - 1);
    this->customHandler = std::move(other.customHandler);
}

void ExecuteParasOpApi::Copy(ExecuteParasOpApiV2 &other)
{
    static const auto max_len = sizeof(ExecuteParasOpApi::opType);
    auto len = other.opName->length();
    if (len + 1 < max_len) {
        other.opName->copy(this->opType, len + 1);
    } else {
        other.opName->copy(this->opType, max_len - 1);
    }
    this->customHandler = std::move(*(other.customHandler));
}

NPUStatus DestroyAclParams(ACL_PARAMS& params)
{
    if (params.input_num != 0) {
        if (params.input_desc != nullptr) {
            for (int i = 0; i < params.input_num; ++i) {
                aclDestroyTensorDesc(params.input_desc[i]);
            }
        }
        if (params.input_data_buf != nullptr) {
            for (int i = 0; i < params.input_num; ++i) {
                NPU_CHECK_ERROR_WITHOUT_UCE(aclDestroyDataBuffer(params.input_data_buf[i]));
            }
        }
        params.input_num = 0;
    }
    if (params.output_num != 0) {
        if (params.output_desc != nullptr) {
            for (int i = 0; i < params.output_num; ++i) {
                aclDestroyTensorDesc(params.output_desc[i]);
            }
        }
        if (params.output_data_buf != nullptr) {
            for (int i = 0; i < params.output_num; ++i) {
                NPU_CHECK_ERROR_WITHOUT_UCE(aclDestroyDataBuffer(params.output_data_buf[i]));
            }
        }
        params.output_num = 0;
    }
    free(params.input_desc);
    params.input_desc = nullptr;
    params.input_data_buf = nullptr;
    params.output_desc = nullptr;
    params.output_data_buf = nullptr;
    return SUCCESS;
}

void DestroyConstParams(CONST_PARAMS &params)
{
    if (params.constList != nullptr) {
        for (int i = 0; i < params.constNum; ++i) {
            if (params.constList[i] != nullptr) {
                delete[] params.constList[i];
            }
        }
    }
    params.constList = nullptr;
    params.constIdx = nullptr;
}
} // namespace native
} // namespace at_npu
