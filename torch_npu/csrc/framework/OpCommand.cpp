#include <ATen/record_function.h>
#include <string>
#include <chrono>

#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/framework/OpCmdHelper.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/CachingHostAllocator.h"
#include "torch_npu/csrc/core/npu/interface/AsyncTaskQueueInterface.h"
#include "torch_npu/csrc/framework/utils/NpuUtils.h"
#include "torch_npu/csrc/framework/utils/NpuStorageOffsetGuard.h"
#include "torch_npu/csrc/framework/LazyInitAclops.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/NPUGraphsUtils.h"
#include "torch_npu/csrc/logging/LogContext.h"
#ifndef BUILD_LIBTORCH
#include "torch_npu/csrc/sanitizer/NPUTrace.h"
#endif

namespace {
const uint64_t kStringOffset = 16UL;
const std::string kStringDType = "string";
static std::unordered_map<at::ScalarType, std::vector<double>> floating_limits_map{
    {at::ScalarType::Double, {std::numeric_limits<double>::max(), std::numeric_limits<double>::min()}},
    {at::ScalarType::Float, {std::numeric_limits<float>::max(), std::numeric_limits<float>::min()}},
    {at::ScalarType::BFloat16, {std::numeric_limits<float>::max(), std::numeric_limits<float>::min()}},
    {at::ScalarType::Half, {65504, -65504}},
    {at::ScalarType::Float8_e5m2, {57345, -57345}},
    {at::ScalarType::Float8_e4m3fn, {449, -449}}};
static std::unordered_map<at::ScalarType, std::vector<long>> integral_limits_map{
    {at::ScalarType::Long, {std::numeric_limits<long>::max(), std::numeric_limits<long>::min()}},
    {at::ScalarType::Int, {std::numeric_limits<int>::max(), std::numeric_limits<int>::min()}},
    {at::ScalarType::Byte, {std::numeric_limits<uint8_t>::max(), std::numeric_limits<uint8_t>::min()}},
    {at::ScalarType::Char, {std::numeric_limits<int8_t>::max(), std::numeric_limits<int8_t>::min()}},
    {at::ScalarType::Short, {std::numeric_limits<int16_t>::max(), std::numeric_limits<int16_t>::min()}}};
} // namespace

std::atomic<bool> g_used_aclop{false};

namespace at_npu {
namespace native {

static std::shared_ptr<npu_logging::Logger> logger = npu_logging::logging().getLogger("torch_npu.dispatch.time");

OpCommand::OpCommand()
{
    aclCmds = OpCommandImpls::GetInstance();
    aclCmds->Push(aclCmd);
    aclCmd->SetCustomHandler(nullptr);
}

OpCommand& OpCommand::Name(const string &name)
{
    aclCmd->SetName(name);
    return *this;
}

OpCommand& OpCommand::SetCustomHandler(PROC_FUNC func)
{
    aclCmd->SetCustomHandler(func);
    return *this;
}

OpCommand& OpCommand::DynamicInputReg(DynamicInputRegFunc func, DyNumAndIndex num_and_index) { return *this; }

OpCommand& OpCommand::Expect(UnifiedResult unified_result)
{
    commonType = unified_result.common_type;
    resultTypeDefined = unified_result.result_type_defined;
    commonShape = unified_result.common_shape;
    return *this;
}

OpCommand& OpCommand::Input() { return AddNoneTensor(); }

OpCommand& OpCommand::Input(const at::Tensor &input, const string &descName,
                            const c10::optional<aclFormat> &sensitive_format, const string &realData)
{
    return AddTensorInput(Contiguous(input), c10::ScalarType::Undefined, descName, realData);
}

OpCommand& OpCommand::InputWithoutContiguous(const at::Tensor &input, const string &descName, const string &realData)
{
    if (input.storage_offset() != 0) {
        TORCH_NPU_WARN_ONCE("[Check][offset] Check input storage_offset[%ld] = 0 failed, result is untrustworthy",
                            input.storage_offset());
    }
    return AddTensorInput(const_cast<at::Tensor &>(input));
}

OpCommand& OpCommand::Input(const c10::IntArrayRef &dimListRef, at::ScalarType toType, CompileType compileType,
                            const string &realDtype, const string &descName)
{
    return Input<int64_t>(dimListRef, dimListRef.size(), toType, compileType, realDtype, descName);
}

OpCommand& OpCommand::Input(const c10::ArrayRef<double> &dimListRef, at::IntArrayRef realShape, at::ScalarType toType,
                            CompileType compileType, const string &realDtype)
{
    return Input<double>(dimListRef, realShape, toType, compileType, realDtype);
}

OpCommand& OpCommand::Input(const c10::Scalar &input, const at::ScalarType type, CompileType compileType)
{
    const auto &scalarTensor = CreateScalarTensor(input, type);
    return AddHostTensorInput(scalarTensor, compileType);
}

OpCommand& OpCommand::Inputs(const at::TensorList &inputs)
{
    for (auto &input : inputs) {
        this->Input(input);
    }
    return *this;
}

OpCommand& OpCommand::InputScalarToNPUTensor(const c10::Scalar &input, const at::ScalarType type)
{
    return AddScalarInput(input, type);
}

OpCommand& OpCommand::Output(at::Tensor &output, const string &descName,
                             const c10::optional<aclFormat> &sensitive_format, const string &realType)
{
    outputTensor.emplace_back(output);
    return AddOutput(output, realType);
}

void OpCommand::Run()
{
    // Check for npu graph
    if (aclCmd->CheckCustomHandlerNull()) {
        g_used_aclop = true;
        // true aclop
        at_npu::aclops::LazyInitAclops();
        auto val = c10_npu::option::GetOption("jitCompile");
        NPU_CHECK_ERROR(AclSetCompileopt(aclCompileOpt::ACL_OP_JIT_COMPILE, val->c_str()));
        at_npu::aclops::LazyAclopSet::SetCompileopt();
        c10_npu::assertNotCapturingAclop(aclCmd->GetName());
    }

    aclCmd->SetEnginePriority();
    const string &op_name = aclCmd->GetName();
#ifndef BUILD_LIBTORCH
    const c10_npu::impl::PyCallbackTrigger* trigger = c10_npu::impl::NPUTrace::getTrace();
#endif
    auto stream = c10_npu::getCurrentNPUStream();
    if (!stream.isSyncLaunchStream() && c10_npu::option::OptionsManager::GetTaskQueueEnable() && !sync) {
        RECORD_FUNCTION(op_name, std::vector<c10::IValue>({}));
#ifndef BUILD_LIBTORCH
        at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(0, op_name);
#endif
        ExecuteParas execParams;
        aclCmd->ExportParams(execParams);
        c10_npu::queue::QueueParas params(c10_npu::queue::COMPILE_AND_EXECUTE, sizeof(ExecuteParas), &execParams);
        c10_npu::enCurrentNPUStream(&params);
#ifndef BUILD_LIBTORCH
        at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(1, op_name, params.correlation_id);
#endif
        aclCmd->releaseSource(false);
    } else {
#ifndef BUILD_LIBTORCH
        if (C10_UNLIKELY(trigger)) {
            trigger->traceNpuAclStartExecution(op_name);
        }
#endif
        aclCmd->Run(sync, sync_index, outputTensor);
        if (c10_npu::option::OptionsManager::CheckBlockingEnable()) {
            if (c10_npu::currentStreamCaptureStatusMayInitCtx() != c10_npu::CaptureStatus::Active) {
                Sync();
            }
        }
#ifndef BUILD_LIBTORCH
        if (C10_UNLIKELY(trigger)) {
            trigger->traceNpuAclFinishExecution(op_name);
        }
#endif
        aclCmd->releaseSource();
    }
    aclCmds->Pop();
}

void OpCommand::RunOpApi(const string &op_name, PROC_FUNC func, bool sync)
{
#ifndef BUILD_LIBTORCH
    const c10_npu::impl::PyCallbackTrigger* trigger = c10_npu::impl::NPUTrace::getTrace();
#endif
    auto stream = c10_npu::getCurrentNPUStream();
    if (!stream.isSyncLaunchStream() && c10_npu::option::OptionsManager::GetTaskQueueEnable()) {
        RECORD_FUNCTION(op_name, std::vector<c10::IValue>({}));
#ifndef BUILD_LIBTORCH
        at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(0, op_name);
#endif
        ExecuteParasOpApi execParams;
        if (op_name.length() + 1 < sizeof(ExecuteParasOpApi::opType)) {
            op_name.copy(execParams.opType, op_name.length() + 1);
        } else {
            op_name.copy(execParams.opType, sizeof(ExecuteParasOpApi::opType) - 1);
        }
        execParams.customHandler = func;

        c10_npu::queue::QueueParas params(c10_npu::queue::EXECUTE_OPAPI, sizeof(ExecuteParasOpApi), &execParams);
        c10_npu::enCurrentNPUStream(&params);
#ifndef BUILD_LIBTORCH
        at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(1, op_name, params.correlation_id);
#endif
    } else {
#ifndef BUILD_LIBTORCH
        if (C10_UNLIKELY(trigger)) {
            trigger->traceNpuAclStartExecution(op_name);
        }
#endif
        OpCommandImpl::RunOpApi(op_name, func);
        if (c10_npu::option::OptionsManager::CheckBlockingEnable()) {
            if (c10_npu::currentStreamCaptureStatusMayInitCtx() != c10_npu::CaptureStatus::Active) {
                NPU_CHECK_ERROR(c10_npu::acl::AclrtSynchronizeStreamWithTimeout(stream));
            }
        }
#ifndef BUILD_LIBTORCH
        if (C10_UNLIKELY(trigger)) {
            trigger->traceNpuAclFinishExecution(op_name);
        }
#endif
    }
    if (sync) {
        NPU_CHECK_ERROR(c10_npu::acl::AclrtSynchronizeStreamWithTimeout(stream));
    }
}

void OpCommand::RunOpApiV2(const string &op_name, const PROC_FUNC &func, bool sync)
{
#ifndef BUILD_LIBTORCH
    const c10_npu::impl::PyCallbackTrigger* trigger = c10_npu::impl::NPUTrace::getTrace();
#endif
    auto stream = c10_npu::getCurrentNPUStream();
    if (!stream.isSyncLaunchStream() && c10_npu::option::OptionsManager::GetTaskQueueEnable()) {
        RECORD_FUNCTION(op_name, std::vector<c10::IValue>({}));
#ifndef BUILD_LIBTORCH
        at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(0, op_name);
#endif
        ExecuteParasOpApiV2 execParams;
        execParams.opName = const_cast<std::string*>(&op_name);
        execParams.customHandler = const_cast<PROC_FUNC*>(&func);

        c10_npu::queue::QueueParas params(c10_npu::queue::EXECUTE_OPAPI_V2, sizeof(ExecuteParasOpApiV2), &execParams);
        
        auto start = std::chrono::steady_clock::now();

        c10_npu::enCurrentNPUStream(&params);
        
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        if (logger != nullptr) {
            logger->info("enQueue duration is (ns): %d", duration);
        }

#ifndef BUILD_LIBTORCH
        at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(1, op_name, params.correlation_id);
#endif
    } else {
#ifndef BUILD_LIBTORCH
        if (C10_UNLIKELY(trigger)) {
            trigger->traceNpuAclStartExecution(op_name);
        }
#endif
        OpCommandImpl::RunOpApi(op_name, func);
        if (c10_npu::option::OptionsManager::CheckBlockingEnable()) {
            if (c10_npu::currentStreamCaptureStatusMayInitCtx() != c10_npu::CaptureStatus::Active) {
                NPU_CHECK_ERROR(c10_npu::acl::AclrtSynchronizeStreamWithTimeout(stream));
            }
        }
#ifndef BUILD_LIBTORCH
        if (C10_UNLIKELY(trigger)) {
            trigger->traceNpuAclFinishExecution(op_name);
        }
#endif
    }
    if (sync) {
        NPU_CHECK_ERROR(c10_npu::acl::AclrtSynchronizeStreamWithTimeout(stream));
    }
}

void OpCommand::RunOpApiV3(const string &op_name, const PROC_FUNC &func, bool sync, c10_npu::NPUStream *task_stream)
{
#ifndef BUILD_LIBTORCH
    const c10_npu::impl::PyCallbackTrigger* trigger = c10_npu::impl::NPUTrace::getTrace();
#endif
    auto stream = c10_npu::getCurrentNPUStream();
    if (!stream.isSyncLaunchStream() && c10_npu::option::OptionsManager::GetTaskQueueEnable()) {
        RECORD_FUNCTION(op_name, std::vector<c10::IValue>({}));
#ifndef BUILD_LIBTORCH
        at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(0, op_name);
#endif
        ExecuteParasOpApiV2 execParams;
        execParams.opName = const_cast<std::string*>(&op_name);
        execParams.customHandler = const_cast<PROC_FUNC*>(&func);

        c10_npu::queue::QueueParas params(c10_npu::queue::EXECUTE_OPAPI_V2, sizeof(ExecuteParasOpApiV2), &execParams);
        c10_npu::enCurrentNPUStream(&params, -1, task_stream);
#ifndef BUILD_LIBTORCH
        at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(1, op_name, params.correlation_id);
#endif
    } else {
#ifndef BUILD_LIBTORCH
        if (C10_UNLIKELY(trigger)) {
            trigger->traceNpuAclStartExecution(op_name);
        }
#endif
        OpCommandImpl::RunOpApi(op_name, func);
        if (c10_npu::option::OptionsManager::CheckBlockingEnable()) {
            if (c10_npu::currentStreamCaptureStatusMayInitCtx() != c10_npu::CaptureStatus::Active) {
                NPU_CHECK_ERROR(c10_npu::acl::AclrtSynchronizeStreamWithTimeout(stream));
            }
        }
#ifndef BUILD_LIBTORCH
        if (C10_UNLIKELY(trigger)) {
            trigger->traceNpuAclFinishExecution(op_name);
        }
#endif
    }
    if (sync) {
        NPU_CHECK_ERROR(c10_npu::acl::AclrtSynchronizeStreamWithTimeout(stream));
    }
}

OpCommand& OpCommand::Sync(c10::SmallVector<int64_t, N> &index)
{
    sync_index = index;
    if (!index.empty()) {
        sync = true;
    }
    return *this;
}

OpCommand& OpCommand::Sync()
{
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream();
    NPU_CHECK_ERROR(c10_npu::acl::AclrtSynchronizeStreamWithTimeout(stream));
    return *this;
}

OpCommand& OpCommand::AddTensorInput(at::Tensor &tensor, at::ScalarType forceScaleType, const string &descName,
                                     const string &realData)
{
    std::tuple<aclTensorDesc*, aclDataBuffer*> res;
    if (commonType.has_value() && commonType.value() != tensor.scalar_type()) {
        tensor = custom_ops::_npu_dtype_cast(tensor, commonType.value());
    }
    // as for dim=0, the dtype of tensor can not be `uint16` because of `TBE`
    if (torch_npu::NPUBridge::GetNpuStorageImplDesc(tensor).storage_sizes_.empty()) {
        if (torch_npu::utils::is_npu(tensor)) {
            res = OpCmdHelper::CovertNPUTensorWithZeroDimToAclInput(tensor, descName);
        } else {
            res = OpCmdHelper::CovertTensorWithZeroDimToAclInput(tensor, forceScaleType);
        }
    } else {
        res = OpCmdHelper::CovertTensorToAclInput(tensor, descName, realData);
    }
    aclCmd->AddInput(std::get<0>(res), std::get<1>(res));
    return *this;
}

OpCommand& OpCommand::AddHostTensorInput(
    const at::Tensor &tensor,
    CompileType compileType,
    const string& realDtype,
    const string& descName)
{
    std::tuple<aclTensorDesc*, aclDataBuffer*> res =
        OpCmdHelper::CovertHostTensorToAclInput(
            tensor,
            tensor.scalar_type(),
            compileType,
            realDtype,
            descName);
    aclCmd->AddInput(std::get<0>(res), std::get<1>(res), tensor);
    return *this;
}

OpCommand& OpCommand::AddNoneTensor()
{
    AclTensorDescMaker desc;
    auto aclDesc = desc.Create(ACL_DT_UNDEFINED, ACL_FORMAT_UNDEFINED).Get();
    AclTensorBufferMaker buffer(nullptr, 0);
    aclCmd->AddInput(aclDesc, buffer.Get());
    return *this;
}

OpCommand& OpCommand::AddScalarInput(const c10::Scalar& input, at::ScalarType type)
{
    at::ScalarType type_bk = type;
    if (commonType.has_value()) {
        type_bk = commonType.value();
    }
    at::Tensor aclInput = CopyHostToDevice(input, type_bk);
    auto res = OpCmdHelper::CovertScalarToAclInput(aclInput, type_bk);
    aclCmd->AddInput(std::get<0>(res), std::get<1>(res));
    return *this;
}

OpCommand& OpCommand::AddOutput(at::Tensor &output, const string &realType)
{
    if (resultTypeDefined == false && commonType.has_value() && commonType.value() != output.scalar_type()) {
        output = custom_ops::_npu_dtype_cast(output, commonType.value());
    }
    auto res = OpCmdHelper::CovertToAclOutput(output, realType);
    aclCmd->AddOutput(std::get<0>(res), std::get<1>(res));
    return *this;
}

// 由于format_contiguous会生成新Tensor，为了保证其在生命周期内有效，故而放到对象中存储
// 同下，CopyScalarToDevice也有同样问题
at::Tensor& OpCommand::Contiguous(const at::Tensor &input)
{
    storage.emplace_back(std::move(NpuUtils::format_contiguous_add_copy_optimize(input)));
    return storage.back();
}

at::Tensor OpCommand::CopyHostToDevice(const c10::Scalar& scalar, at::ScalarType type)
{
    auto tensor = scalar_to_tensor(scalar).to(type);
    return CopyHostToDevice(tensor);
}

at::Tensor OpCommand::CopyHostToDevice(const at::Tensor& cpuTensor)
{
    at::Tensor cpuPinMemTensor = cpuTensor.pin_memory();
    int deviceIndex = 0;
    NPU_CHECK_ERROR(c10_npu::GetDevice(&deviceIndex));
    auto tensor = cpuPinMemTensor.to(
        c10::Device(c10::DeviceType::PrivateUse1, deviceIndex),
        cpuPinMemTensor.scalar_type(),
        true,
        true);
    storage.emplace_back(tensor);
    return storage.back();
}

at::Tensor& OpCommand::CreateHostTensor(
    void *data,
    at::IntArrayRef size,
    const c10::TensorOptions &options,
    at::ScalarType toType)
{
    at::ScalarType dtype = options.dtype().toScalarType();
    auto cpuTensor = at::empty(size, options);
    std::memcpy(cpuTensor.data_ptr(), data, elementSize(dtype) * cpuTensor.numel());
    if (toType != dtype) {
        cpuTensor = cpuTensor.to(toType);
    }

    storage.emplace_back(std::move(cpuTensor));
    return storage.back();
}

bool OpCommand::ScalarIsInLimits(const c10::Scalar &scalar, at::ScalarType type)
{
    bool scalar_flag = false;
    if (at::isFloatingType(type)) {
        auto value = scalar.to<double>();
        scalar_flag = value <= floating_limits_map[type][0] && value >= floating_limits_map[type][1];
    } else if (at::isIntegralType(type)) {
        auto value = scalar.to<long>();
        scalar_flag = value <= integral_limits_map[type][0] && value >= integral_limits_map[type][1];
    }
    return scalar_flag;
}

at::Tensor& OpCommand::CreateScalarTensor(const c10::Scalar &scalar, at::ScalarType type)
{
    if (commonType.has_value()) {
        type = commonType.value();
    }

    if (ScalarIsInLimits(scalar, type)) {
        storage.emplace_back(at::detail::scalar_tensor_static(scalar, type, at::kCPU));
    } else {
        storage.emplace_back(scalar_to_tensor(scalar).to(type));
    }
    return storage.back();
}

} // namespace native
} // namespace at_npu