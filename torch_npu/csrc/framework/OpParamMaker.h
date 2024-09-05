#ifndef __PULGIN_NATIVE_UTILS_OP_PARAM_MAKER__
#define __PULGIN_NATIVE_UTILS_OP_PARAM_MAKER__

#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "third_party/acl/inc/acl/acl_base.h"
#include "torch_npu/csrc/framework/interface/AclOpCompileInterface.h"
#include "torch_npu/csrc/framework/NPUDefine.h"
#include "torch_npu/csrc/framework/utils/ForceJitCompileList.h"
#include "torch_npu/csrc/framework/interface/EnvVariables.h"
#include "torch_npu/csrc/core/npu/NPUMacros.h"
#include "torch_npu/csrc/core/NPUStorageImpl.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/framework/NPUDefine.h"
#include "torch_npu/csrc/core/npu/interface/AsyncTaskQueueInterface.h"

namespace at_npu {
namespace native {

typedef union {
    ExecuteParas exeParas;
    c10_npu::queue::CopyParas copyParas;
    c10_npu::queue::EventParas eventParas;
} TaskParas;
constexpr size_t MAX_PARAS_BYTE_SIZE = sizeof(TaskParas);

// This file is defined wrapper C++ functions of ACL
class TORCH_NPU_API OpAttrMaker {
public:
    static void Set(aclopAttr *attr, const string &name, bool value);
    static void Set(aclopAttr *attr, const string &name, int64_t value);
    static void Set(aclopAttr *attr, const string &name, float value);
    static void Set(aclopAttr *attr, const string &name, string value);
    static void Set(aclopAttr *attr, const string &name, c10::IntArrayRef value);
    static void Set(aclopAttr *attr, const string &name, at::ArrayRef<float> value);
    static void Set(aclopAttr *attr, const string &name, at::ArrayRef<uint8_t> value);
    static void Set(aclopAttr *attr, const string &name, c10::Scalar value);
    static void Set(aclopAttr *attr, const string &name, at::ScalarType value);
    static void Set(aclopAttr *attr, const string &name, at::ArrayRef<c10::IntArrayRef> value);
}; // class OpAttrMaker

class AclTensorDescMaker {
public:
    AclTensorDescMaker() {}
    ~AclTensorDescMaker() = default;

    AclTensorDescMaker &Create(aclDataType dataType, torch_npu::NPUStorageDesc storageDesc)
    {
        c10::SmallVector<int64_t, 5> dims;
        // if aclDataType is ACL_STRING, storageDims is empty.
        if (dataType != ACL_STRING) {
            dims = storageDesc.base_sizes_;
        }
        auto format = storageDesc.origin_format_;
        desc = aclCreateTensorDesc(dataType, dims.size(), dims.data(), format);
        return *this;
    }

    inline AclTensorDescMaker &Create(
        aclDataType dataType,
        c10::IntArrayRef dims,
        aclFormat format)
    {
        desc = aclCreateTensorDesc(dataType, dims.size(), dims.data(), format);
        return *this;
    }

    inline AclTensorDescMaker &Create(aclDataType dataType, aclFormat format)
    {
        desc = aclCreateTensorDesc(dataType, 0, nullptr, format);
        return *this;
    }

    inline AclTensorDescMaker &SetFormat(aclFormat format)
    {
        aclSetTensorFormat(desc, format);
        return *this;
    }

    inline AclTensorDescMaker &SetPlacement(aclMemType memType)
    {
        aclSetTensorPlaceMent(desc, memType);
        return *this;
    }

    template <unsigned int N>
    inline AclTensorDescMaker &SetShape(const c10::SmallVector<int64_t, N> &dims)
    {
        aclSetTensorShape(desc, dims.size(), dims.data());
        return *this;
    }

    template <unsigned int N>
    AclTensorDescMaker &SetRange(const c10::SmallVector<int64_t, N> &rangs)
    {
        int arryDim = rangs.size() == 0 ? 0 : rangs.size() / 2;

        int64_t range[arryDim][2];
        for (int i = 0, j = 0; i < arryDim; i++, j += 2) {
            range[i][0] = rangs[j];
            range[i][1] = rangs[j + 1];
        }

        aclSetTensorShapeRange(desc, arryDim, range);
        return *this;
    }

    inline AclTensorDescMaker &SetName(const std::string &name)
    {
        if (!name.empty()) {
            aclSetTensorDescName(desc, name.c_str());
        }
        return *this;
    }

    inline AclTensorDescMaker &SetConstAttr(c10::optional<at::Tensor> cpu_tensor)
    {
        if (cpu_tensor.has_value() && cpu_tensor.value().defined()) {
            aclSetTensorConst(
                desc,
                cpu_tensor.value().data_ptr(),
                cpu_tensor.value().itemsize() * cpu_tensor.value().numel());
        }

        return *this;
    }

    inline aclTensorDesc *Get() const
    {
        return desc;
    }

private:
    aclTensorDesc *desc = nullptr;
}; // class AclTensorDescMaker

class AclTensorBufferMaker {
public:
    // base of Ctr
    // params: tensor, offset, remained size
    AclTensorBufferMaker(const at::Tensor *tensor, int64_t offset, int64_t n)
    {
        uint8_t *header = reinterpret_cast<uint8_t *>(tensor->data_ptr()) -
                          tensor->itemsize() * static_cast<uint8_t>(offset);
        size_t bufferSize = tensor->itemsize() * static_cast<size_t>(n);
        ptr = aclCreateDataBuffer(header, bufferSize);
    }

    // offset = 0
    explicit AclTensorBufferMaker(const at::Tensor *tensor, int64_t n = 1)
    {
        if (tensor == nullptr || n == 0) {
            ptr = aclCreateDataBuffer(nullptr, 0);
        } else {
            ptr = aclCreateDataBuffer(
                (void *)(tensor->data_ptr()), tensor->itemsize() * n);
        }
    }

    // offset = 0
    explicit AclTensorBufferMaker(const at::Tensor &tensor, int64_t n = 1)
    {
        ptr = aclCreateDataBuffer((void *)(tensor.data_ptr()), tensor.itemsize() * n);
    }

    ~AclTensorBufferMaker() = default;

    inline aclDataBuffer *Get() const
    {
        return ptr;
    }

private:
    aclDataBuffer *ptr = nullptr;
}; // class AclTensorBufferMaker

using PROC_FUNC = std::function<int()>;

// the member in AclExecParam is create by :
// aclCreateDataBuffer and aclCreateTensorDesc
// so aclDestroyTensorDesc and aclDestroyDataBuffer should be called when dtr
// aclopDestroyAttr
class OpCommandImpl {
public:
    OpCommandImpl() {}
    ~OpCommandImpl()
    {
        // do nothing, can not release resource, because of multi-thread or
        // queue-enable
    }

    void SetName(const string &name)
    {
        opName = name;
    }

    void SetCustomHandler(PROC_FUNC func)
    {
        execParam.customHandler = func;
    }

    const string &GetName() const { return opName; }

    void AddInput(
        const aclTensorDesc *desc,
        const aclDataBuffer *buffer)
    {
        execParam.inDesc.emplace_back(std::move(desc));
        execParam.inBuffer.emplace_back(std::move(buffer));
    }

    void AddInput(
        const aclTensorDesc *desc,
        const aclDataBuffer *buffer,
        const at::Tensor &hostTensor)
    {
        AddInput(desc, buffer);
        execParam.hostMem.emplace_back(hostTensor);
    }

    void AddInput(const string &str);

    void AddOutput(
        const aclTensorDesc *desc,
        aclDataBuffer *buffer)
    {
        execParam.outDesc.emplace_back(std::move(desc));
        execParam.outBuffer.emplace_back(std::move(buffer));
    }

    template <typename dataType>
    void AddAttr(const string& attrName, dataType value)
    {
        InitAttr();
        OpAttrMaker::Set(execParam.attr, attrName, value);
    }

    // export op execute params
    void ExportParams(ExecuteParas &params)
    {
        TORCH_CHECK(sizeof(ExecuteParas::opType) >= opName.length() + 1, "Too long Ascend IR Name: ", opName,
                    OPS_ERROR(ErrCode::PARAM));
        memset(params.opType, '\0', sizeof(params.opType));
        opName.copy(params.opType, opName.length() + 1);
        params.attr = execParam.attr;
        // make params
        int inputNum = static_cast<int>(execParam.inDesc.size());
        int outputNum = static_cast<int>(execParam.outDesc.size());

        size_t inputTensorDescArrLen = inputNum * sizeof(uintptr_t);
        size_t inputDataBuffArrLen   = inputNum * sizeof(uintptr_t);

        size_t outputTensorDescArrLen = outputNum * sizeof(uintptr_t);
        size_t outputDataBuffArrLen   = outputNum * sizeof(uintptr_t);

        size_t totalMemLen = inputTensorDescArrLen + inputDataBuffArrLen +
                             outputTensorDescArrLen + outputDataBuffArrLen;

        char* basePtr = static_cast<char* >(malloc(totalMemLen));
        AT_ASSERT(basePtr != nullptr, OPS_ERROR(ErrCode::PTR));
        const aclTensorDesc** aclTensorInputDescArr = reinterpret_cast<const aclTensorDesc** >(basePtr);
        basePtr += inputTensorDescArrLen;
        const aclDataBuffer** aclDataInputBuffArr = reinterpret_cast<const aclDataBuffer** >(basePtr);
        basePtr += inputDataBuffArrLen;

        const aclTensorDesc** aclTensorOutputDescArr = reinterpret_cast<const aclTensorDesc** >(basePtr);
        basePtr += outputTensorDescArrLen;
        aclDataBuffer** aclDataOutputBuffArr = reinterpret_cast<aclDataBuffer** >(basePtr);

        std::copy(
            execParam.inDesc.begin(),
            execParam.inDesc.end(),
            aclTensorInputDescArr);
        std::copy(
            execParam.inBuffer.begin(),
            execParam.inBuffer.end(),
            aclDataInputBuffArr);
        std::copy(
            execParam.outDesc.begin(),
            execParam.outDesc.end(),
            aclTensorOutputDescArr);
        std::copy(
            execParam.outBuffer.begin(),
            execParam.outBuffer.end(),
            aclDataOutputBuffArr);

        params.paras.input_num = inputNum;
        params.paras.output_num = outputNum;
        params.paras.input_desc = aclTensorInputDescArr;
        params.paras.input_data_buf = aclDataInputBuffArr;
        params.paras.output_desc = aclTensorOutputDescArr;
        params.paras.output_data_buf = aclDataOutputBuffArr;
        params.hostMemory = execParam.hostMem;
        params.customHandler = execParam.customHandler;
        params.pta_correlation_id = ExecuteParas::g_pta_correlation_id++;

        if (!ForceJitCompileList::GetInstance().Inlist(opName) && env::CheckJitDisable()) {
            params.isJitDisable = true;
        }
    }

    // Set engine priority for op on data preprocessing stream
    void SetEnginePriority();

    void Run(bool sync, c10::SmallVector<int64_t, N> &sync_index, c10::SmallVector<at::Tensor, N> &outputTensor);

    void releaseSource(bool no_blocking = true)
    {
        if (no_blocking) {
            std::for_each(
                execParam.inDesc.begin(),
                execParam.inDesc.end(),
                aclDestroyTensorDesc);
            std::for_each(
                execParam.outDesc.begin(),
                execParam.outDesc.end(),
                aclDestroyTensorDesc);
            std::for_each(
                execParam.inBuffer.begin(),
                execParam.inBuffer.end(),
                aclDestroyDataBuffer);
            std::for_each(
                execParam.outBuffer.begin(),
                execParam.outBuffer.end(),
                aclDestroyDataBuffer);
            if (execParam.attr != nullptr) {
                aclopDestroyAttr(execParam.attr);
                execParam.attr = nullptr;
            }
        }

        execParam.inDesc.clear();
        execParam.inBuffer.clear();

        execParam.outDesc.clear();
        execParam.outBuffer.clear();

        execParam.hostMem.clear();

        // recover
        execParam.attr = nullptr;
        execParam.customHandler = nullptr;
        opName = "";
    }

private:
    struct AclExecParam {
        c10::SmallVector<const aclTensorDesc*, N> inDesc;   // owned
        c10::SmallVector<const aclDataBuffer*, N> inBuffer; // owned
        c10::SmallVector<const aclTensorDesc*, N> outDesc;  // owned
        c10::SmallVector<aclDataBuffer*, N> outBuffer;      // owned
        c10::SmallVector<at::Tensor, N> hostMem;
        aclopAttr *attr = nullptr;
        PROC_FUNC customHandler = nullptr;
    };

    void InitAttr()
    {
        if (execParam.attr == nullptr) {
            execParam.attr = aclopCreateAttr();
        }
    }

    aclError InnerRun(
        const string &name,
        AclExecParam &params,
        bool sync,
        c10::SmallVector<int64_t, N> &sync_index,
        c10::SmallVector<at::Tensor, N> &outputTensor
    );

private:
    string opName;
    AclExecParam execParam;
}; // class OpCommandImpl

// This class maintain the position of the current
// OpCommandImpl object in vector, the resources in
// the object is
class OpCommandImpls {
public:
    static OpCommandImpls *GetInstance();
    void Push(OpCommandImpl *&ptr);
    void Pop();

private:
    int32_t offset = -1;
    c10::SmallVector<OpCommandImpl, N> objs;
}; // class OpCommandImpls

void SetDeterministic();

static bool deterministicaclnn_oldstatus = false;

} // namespace native
} // namespace at_npu

#endif