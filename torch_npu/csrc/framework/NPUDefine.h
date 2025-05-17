#ifndef __PULGIN_C10_NPUQUEUE_WITH_QUEUE__
#define __PULGIN_C10_NPUQUEUE_WITH_QUEUE__

#include "torch_npu/csrc/core/npu/npu_log.h"

#include "torch_npu/csrc/framework/utils/NpuUtils.h"

namespace at_npu {
namespace native {

struct ACL_PARAMS {
    ACL_PARAMS() : input_desc(nullptr), input_data_buf(nullptr), output_desc(nullptr), output_data_buf(nullptr) {}

    int input_num{0};
    const aclTensorDesc **input_desc;
    const aclDataBuffer **input_data_buf;
    int output_num{0};
    const aclTensorDesc **output_desc;
    aclDataBuffer **output_data_buf;
};

struct ACL_DYNAMIC_PARAMS {
    ACL_DYNAMIC_PARAMS()
        : input_desc(nullptr), input_data_buf(nullptr), output_desc(nullptr), output_data_buf(nullptr),
          inputDims(nullptr), outputDims(nullptr), inputFormats(nullptr), outputFormats(nullptr),
          compile_input_desc(nullptr), compile_output_desc(nullptr), hasAttr(false)
    {
    }

    int input_num = 0;
    const aclTensorDesc **input_desc;
    const aclDataBuffer **input_data_buf;
    int output_num = 0;
    const aclTensorDesc **output_desc;
    aclDataBuffer **output_data_buf;
    int64_t *inputDims;
    int64_t *outputDims;
    aclFormat *inputFormats;
    aclFormat *outputFormats;
    const aclTensorDesc **compile_input_desc;
    const aclTensorDesc **compile_output_desc;
    bool hasAttr;
    std::string dynamicKey;
};

struct CONST_PARAMS {
    int constNum = 0;
    const int64_t **constList = nullptr;
    const int64_t *constIdx = nullptr;
    CONST_PARAMS() = default;
};

struct ExecuteParas {
    using PROCESS_FUNC = std::function<int()>;
    char opType[100]{};
    bool isJitDisable = false;
    ACL_PARAMS paras;
    CONST_PARAMS constParams;
    const aclopAttr *attr;
    int64_t constIdx = -1;
    static std::atomic<uint64_t> g_pta_correlation_id;
    uint64_t pta_correlation_id = 0;
    c10::SmallVector<at::Tensor, N> hostMemory;
    ExecuteParas() = default;
    void Release();
    void Copy(ExecuteParas &other);
    void CopyEx(ExecuteParas& other);
    PROCESS_FUNC customHandler;
};

struct ExecuteParasOpApiV2 {
    using PROCESS_FUNC = std::function<int()>;
    std::string *opName;
    PROCESS_FUNC *customHandler;
    ExecuteParasOpApiV2() = default;
};

struct ExecuteParasOpApi {
    using PROCESS_FUNC = std::function<int()>;
    char opType[100]{};
    PROCESS_FUNC customHandler;
    ExecuteParasOpApi() = default;
    void Release();
    void Copy(ExecuteParasOpApi &other);
    void Copy(ExecuteParasOpApiV2 &other);
};

NPUStatus DestroyAclParams(ACL_PARAMS &params);
void DestroyConstParams(CONST_PARAMS &params);
} // namespace native
} // namespace at_npu

#endif // __C10_NPU_NPUQUEUE_WITH_QUEUE__