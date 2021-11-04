// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "CalcuOpUtil.h"
#include <Python.h>
#include <third_party/acl/inc/acl/acl_base.h>
#include "ATen/native/npu/interface/AclOpCompileInterface.h"
#include <torch/csrc/autograd/record_function.h>
#include "ATen/native/npu/frame/InferFormat.h"
#include "ATen/native/npu/mirror/NPUMemoryOverlap.h"
#include "ATen/native/npu/utils/DynamicShapeUtil.h"
#include "NpuUtils.h"
#include "c10/npu/NPUCachingAllocator.h"
#include "c10/npu/OptionsManager.h"
#include "c10/npu/interface/AsyncTaskQueueInterface.h"
#include "ATen/native/npu/interface/EnvVariables.h"
#include "ATen/native/npu/utils/NpuFuzzyBlacklist.h"

namespace at {
namespace native {
namespace npu {

namespace {
const static aclDataType kUnknownAclDataType = static_cast<aclDataType>(100);
const static aclFormat kUnKnownAclFormat = static_cast<aclFormat>(100);
const static string kUnknownDataTypeName = "UNKNOWN";
constexpr float EPSILON = 1e-6;

static std::map<at::ScalarType, aclDataType> AT_SCALAR_TYPE_TO_ACL_TYPE_MAP = {
    {at::ScalarType::Byte, ACL_UINT8},
    {at::ScalarType::Char, ACL_INT8},
    {at::ScalarType::Short, ACL_INT16},
    {at::ScalarType::Int, ACL_INT32},
    {at::ScalarType::Half, ACL_FLOAT16},
    {at::ScalarType::Float, ACL_FLOAT},
    {at::ScalarType::Bool, ACL_BOOL},
    {at::ScalarType::Long, ACL_INT64},
};

static std::map<const at::ScalarType, const string> AT_SCALAR_TYPE_NAME_MAP = {
    {at::ScalarType::Byte, "at::ScalarType::Byte"},
    {at::ScalarType::Char, "at::ScalarType::Char"},
    {at::ScalarType::Short, "at::ScalarType::Short"},
    {at::ScalarType::Int, "at::ScalarType::Int"},
    {at::ScalarType::Long, "at::ScalarType::Long"},
    {at::ScalarType::Half, "at::ScalarType::Half"},
    {at::ScalarType::Float, "at::ScalarType::Float"},
    {at::ScalarType::Double, "at::ScalarType::Double"},
};

static std::map<const string, const aclDataType>
    STRING_SCALAR_TYPE_TO_ACL_TYPE_MAP = {
        {"uint16", ACL_UINT16},
        {"uint8", ACL_UINT8}
};

string GetAtScalarTypeName(const ScalarType data_type) {
  auto iter = AT_SCALAR_TYPE_NAME_MAP.find(data_type);
  if (iter == AT_SCALAR_TYPE_NAME_MAP.end()) {
    return kUnknownDataTypeName;
  }

  return iter->second;
}
} // namespace

aclDataType CalcuOpUtil::convert_to_acl_data_type(const ScalarType data_type) {
  const auto& iter = AT_SCALAR_TYPE_TO_ACL_TYPE_MAP.find(data_type);
  if (iter == AT_SCALAR_TYPE_TO_ACL_TYPE_MAP.end()) {
    NPU_LOGE(
        "Unsupport data type: %s.", GetAtScalarTypeName(data_type).c_str());
    return kUnknownAclDataType;
  }

  return iter->second;
}

aclDataType CalcuOpUtil::convert_to_acl_data_type(
    const ScalarType data_type,
    const string& realDataType) {
  auto iter = AT_SCALAR_TYPE_TO_ACL_TYPE_MAP.find(data_type);
  if (iter == AT_SCALAR_TYPE_TO_ACL_TYPE_MAP.end()) {
    NPU_LOGE(
        "Unsupport data type: %s.", GetAtScalarTypeName(data_type).c_str());
    return kUnknownAclDataType;
  }
  if (realDataType != "") {
    return STRING_SCALAR_TYPE_TO_ACL_TYPE_MAP[realDataType];
  }

  return iter->second;
}

Scalar CalcuOpUtil::ConvertTensorToScalar(const Tensor& tensor) {
  Scalar expScalar;
  const Tensor* aclInput = &tensor;
  if (aclInput->scalar_type() == ScalarType::Double) {
    double value = *(double*)aclInput->data_ptr();
    Scalar scalar(value);
    expScalar = scalar;
  } else if (aclInput->scalar_type() == ScalarType::Long) {
    int64_t value = *(int64_t*)aclInput->data_ptr();
    Scalar scalar(value);
    expScalar = scalar;
  } else if (aclInput->scalar_type() == ScalarType::Float) {
    float value = *(float*)aclInput->data_ptr();
    Scalar scalar(value);
    expScalar = scalar;
  } else if (aclInput->scalar_type() == ScalarType::Int) {
    int value = *(int*)aclInput->data_ptr();
    Scalar scalar(value);
    expScalar = scalar;
  } else if (aclInput->scalar_type() == ScalarType::Half) {
    Half value = *(Half*)aclInput->data_ptr();
    Scalar scalar(value);
    expScalar = scalar;
  } else {
    NPU_LOGE("unsupport scalar type! ");
    AT_NPU_CHECK(ACL_ERROR_UNSUPPORTED_DATA_TYPE);
  }

  return expScalar;
}

Tensor CalcuOpUtil::CopyScalarToDevice(
    const Scalar& cpu_scalar,
    ScalarType scalar_data_type) {
  return CalcuOpUtil::copy_tensor_host_to_device(
      scalar_to_tensor(cpu_scalar).to(scalar_data_type));
}

Tensor CalcuOpUtil::copy_tensor_host_to_device(const Tensor& cpu_tensor) {
  Tensor cpuPinMemTensor = cpu_tensor.pin_memory();
  int deviceIndex = 0;
  AT_NPU_CHECK(aclrtGetDevice(&deviceIndex));
  return cpuPinMemTensor.to(
      c10::Device(DeviceType::NPU, deviceIndex),
      cpuPinMemTensor.scalar_type(),
      true,
      true);
}

NPUStatus CalcuOpUtil::AclrtMemcpyAsync(
    void* dst,
    size_t dst_size,
    const void* src,
    size_t src_size,
    aclrtMemcpyKind kind) {
  AT_NPU_CHECK(c10::npu::queue::LaunchAsyncCopyTask(dst, dst_size, const_cast<void* >(src), src_size, kind));

  return SUCCESS;
}

int64_t CalcuOpUtil::get_tensor_npu_format(const Tensor& tensor) {
  if (NpuUtils::check_match(&tensor) || NpuUtils::check_5d_5d_match(tensor)) {
    auto tensor_desc = tensor.storage().unsafeGetStorageImpl()->npu_desc_;
    return tensor_desc.npu_format_;
  } else {
    return InferFormat::GuessFormatWhenContiguous(tensor);
  }
}

void CalcuOpUtil::check_memory_over_laps(
    SmallVector<Tensor, N>& inputs,
    SmallVector<Tensor, N>& outputs) {
  for (int i = 0; i < outputs.size(); i++) {
    if (!outputs[i].defined())
      continue;

    assert_no_internal_overlap(outputs[i]);

    for (int j = 0; j < inputs.size(); j++) {
      assert_no_partial_overlap(outputs[i], inputs[j]);
    }
  }
}

string CalcuOpUtil::get_reduction_str(int64_t reduction) {
  string reductionStr;
  if (reduction == Reduction::None) {
    reductionStr = "none";
  } else if (reduction == Reduction::Mean) {
    reductionStr = "mean";
  } else {
    reductionStr = "sum";
  }

  return reductionStr;
}

int64_t CalcuOpUtil::make_wrap_dim(int64_t dim, int64_t dim_post_expr) {
  if (dim_post_expr <= 0) {
    dim_post_expr = 1; // this will make range [-1, 0]
  }

  int64_t min = -dim_post_expr;
  int64_t max = dim_post_expr - 1;
  if (dim < 0) {
    dim += dim_post_expr;
  }
  return dim;
}

bool CalcuOpUtil::is_transpose_last_two_dims(const Tensor& tensor) {
  if (tensor.dim() < 2 || tensor.dim() > 3) {
    return false;
  }
  int64_t numel = 1;
  auto storageSize = tensor.storage().get_npu_desc().storage_sizes_;

  for (int i = 0; i < storageSize.size(); i++) {
    numel *= storageSize[i];
  }

  int64_t dim1 = tensor.dim() - 1;
  int64_t dim2 = tensor.dim() - 2;

  auto tensor_desc = tensor.storage().get_npu_desc();
  if (tensor.stride(dim2) == 1 && tensor.stride(dim1) == tensor.size(dim2) &&
      tensor.size(dim1) == tensor_desc.base_sizes_[dim2] &&
      tensor.size(dim2) == tensor_desc.base_sizes_[dim1] &&
      tensor.storage().size() == numel &&
      tensor_desc.base_sizes_.size() == tensor.dim()) {
    return true;
  } else {
    return false;
  }
}

bool CalcuOpUtil::is_scalar_wrapped_to_tensor(const Tensor& tensor) {
  return tensor.unsafeGetTensorImpl()->is_wrapped_number() &&
      (!tensor.is_npu());
}

bool CalcuOpUtil::is_scalar_one(const Scalar& scalar) {
  if (scalar.isIntegral(false)) {
    return scalar.toInt() == 1;
  } else if (scalar.isFloatingPoint()) {
    return fabs(scalar.toFloat() - 1.0) < EPSILON;
  } else {
    return false;
  }
}

float CalcuOpUtil::get_scalar_float_value(const Scalar& scalar) {
  float value;
  if (scalar.isFloatingPoint()) {
    value = scalar.toFloat();
  } else {
    value = (float)scalar.toInt();
  }

  return value;
}

ScalarType CalcuOpUtil::GetNPUTensorDescScalarType(
    const NPUTensorDesc& npuTensorDesc) {
  ScalarType scalarDataType;

  if (npuTensorDesc.tensorDescType == NPUTensorDesc::TensorDescType::SCALAR ||
      (npuTensorDesc.tensorDescType ==
           NPUTensorDesc::TensorDescType::TENSOR_SCALAR &&
       (!npuTensorDesc.tensor.is_npu()) &&
       is_scalar_wrapped_to_tensor(npuTensorDesc.tensor))) {
    scalarDataType = npuTensorDesc.scalarType;
  } else {
    scalarDataType = npuTensorDesc.tensor.scalar_type();
  }

  return scalarDataType;
}

SmallVector<Tensor, N> CalcuOpUtil::ConvertTensorListToSmallVector(
    TensorList tensors) {
  SmallVector<Tensor, N> tensorVec;
  for (int i = 0; i < tensors.size(); i++) {
    tensorVec.emplace_back(tensors[i]);
  }

  return tensorVec;
}

SmallVector<int64_t, N> CalcuOpUtil::ConvertIntArrayRefToSmallVector(
    IntArrayRef intArray) {
  SmallVector<int64_t, N> intVec;
  for (int i = 0; i < intArray.size(); i++) {
    intVec.emplace_back(intArray[i]);
  }

  return intVec;
}

NPUStatus CalcuOpUtil::CreateAclTensorDescInfo(
    SmallVector<NPUTensorDesc, N>& input,
    SmallVector<NPUTensorDesc, N>& output,
    ACL_PARAMS& params,
    string opName,
    const SmallVector<NPUAttrDesc, N>& attrs) {
  int inputNum = input.size();
  int outputNum = output.size();

  const aclTensorDesc** aclTensorInputDescArr =
      inputNum == 0 ? nullptr : new const aclTensorDesc*[inputNum];
  const aclTensorDesc** aclTensorOutputDescArr =
      outputNum == 0 ? nullptr : new const aclTensorDesc*[outputNum];

  const aclDataBuffer** aclDataInputBuffArr =
      inputNum == 0 ? nullptr : new const aclDataBuffer*[inputNum];
  aclDataBuffer** aclDataOutputBuffArr =
      outputNum == 0 ? nullptr : new aclDataBuffer*[outputNum];

  int64_t* inputDimsArr = new int64_t[inputNum];
  int64_t* outputDimsArr = new int64_t[outputNum];
  aclFormat* inputFormatsArr = new aclFormat[inputNum];
  aclFormat* outputFormatsArr = new aclFormat[outputNum];

  for (int i = 0; i < inputNum; i++) {
    ScalarType scalarDataType =
        CalcuOpUtil::GetNPUTensorDescScalarType(input[i]);
    aclDataType aclDataType = CalcuOpUtil::convert_to_acl_data_type(
        scalarDataType, input[i].realDataType);

    size_t dimCnt = 0;
    int64_t shape[0];

    if (input[i].tensorDescType == NPUTensorDesc::TensorDescType::NONE_TENSOR) {
      // Create aclCreateTensorDesc and aclCreateDataBuffer of a NoneTensor.
      aclTensorInputDescArr[i] = aclCreateTensorDesc(
          ACL_DT_UNDEFINED, 0, nullptr, ACL_FORMAT_UNDEFINED);
      aclDataInputBuffArr[i] = aclCreateDataBuffer(nullptr, 0);
      inputDimsArr[i] = 0;
      inputFormatsArr[i] = ACL_FORMAT_UNDEFINED;

    } else if (
        input[i].tensorDescType == NPUTensorDesc::TensorDescType::TENSOR) {
      Tensor* aclInput = &input[i].tensor;
      auto dims = aclInput->storage().get_npu_desc().base_sizes_;
      auto storageDims = aclInput->storage().get_npu_desc().storage_sizes_;
      int64_t numel = 1;
      for (int j = 0; j < storageDims.size(); j++) {
        numel *= storageDims[j];
      }

      aclTensorDesc* acl_tensor_desc = aclCreateTensorDesc(
          aclDataType,
          dims.size(),
          dims.data(),
          aclInput->storage().get_npu_desc().origin_format_);
      aclSetTensorFormat(
          acl_tensor_desc, aclInput->storage().get_npu_desc().npu_format_);
      aclSetTensorShape(
          acl_tensor_desc, storageDims.size(), storageDims.data());
      if (input[i].tensorDescName != "") {
        aclSetTensorDescName(acl_tensor_desc, input[i].tensorDescName.c_str());
      }
      aclTensorInputDescArr[i] = acl_tensor_desc;
      aclDataInputBuffArr[i] = aclCreateDataBuffer(
          (void*)(aclInput->data_ptr()), aclInput->itemsize() * numel);
      inputDimsArr[i] = storageDims.size();
      inputFormatsArr[i] = aclInput->storage().get_npu_desc().npu_format_;

    } else if (
        input[i].tensorDescType ==
            NPUTensorDesc::TensorDescType::TENSOR_SCALAR &&
        input[i].tensor.is_npu()) {
      Tensor* aclInput = &input[i].tensor;
      aclTensorInputDescArr[i] =
          aclCreateTensorDesc(aclDataType, dimCnt, shape, ACL_FORMAT_ND);
      aclDataInputBuffArr[i] = aclCreateDataBuffer(
          (void*)aclInput->data_ptr(), aclInput->itemsize());
      inputDimsArr[i] = 0;
      inputFormatsArr[i] = ACL_FORMAT_ND;

    } else {
      Scalar expScalar;
      if (input[i].tensorDescType == NPUTensorDesc::TensorDescType::SCALAR) {
        expScalar = input[i].scalar;
      } else {
        expScalar = ConvertTensorToScalar(input[i].tensor);
      }

      Tensor aclInput =
          CalcuOpUtil::CopyScalarToDevice(expScalar, scalarDataType);
      aclTensorInputDescArr[i] =
          aclCreateTensorDesc(aclDataType, dimCnt, shape, ACL_FORMAT_ND);
      aclDataInputBuffArr[i] =
          aclCreateDataBuffer((void*)aclInput.data_ptr(), aclInput.itemsize());
      inputDimsArr[i] = 0;
      inputFormatsArr[i] = ACL_FORMAT_ND;
    }
  }

  for (int i = 0; i < outputNum; i++) {
    Tensor* aclOutput = &output[i].tensor;
    aclDataType aclDataType = CalcuOpUtil::convert_to_acl_data_type(
        aclOutput->scalar_type(), output[i].realDataType);

    auto dims = aclOutput->sizes();
    auto storageDims = aclOutput->storage().get_npu_desc().storage_sizes_;
    int64_t numel = 1;
    for (int j = 0; j < storageDims.size(); j++) {
      numel *= storageDims[j];
    }

    aclTensorDesc* acl_tensor_desc = aclCreateTensorDesc(
        aclDataType,
        dims.size(),
        dims.data(),
        aclOutput->storage().get_npu_desc().origin_format_);
    aclSetTensorFormat(
        acl_tensor_desc, aclOutput->storage().get_npu_desc().npu_format_);
    aclSetTensorShape(
        acl_tensor_desc, storageDims.size(), storageDims.data());
    aclTensorOutputDescArr[i] = acl_tensor_desc;
    aclDataOutputBuffArr[i] = aclCreateDataBuffer(
        (void*)aclOutput->data_ptr(), aclOutput->itemsize() * numel);
    outputDimsArr[i] = storageDims.size();
    outputFormatsArr[i] = aclOutput->storage().get_npu_desc().npu_format_;
  }

  params.input_num = inputNum;
  params.input_desc = aclTensorInputDescArr;
  params.input_data_buf = aclDataInputBuffArr;

  params.output_num = outputNum;
  params.output_desc = aclTensorOutputDescArr;
  params.output_data_buf = aclDataOutputBuffArr;

  params.inputDims = inputDimsArr;
  params.outputDims = outputDimsArr;
  params.inputFormats = inputFormatsArr;
  params.outputFormats = outputFormatsArr;

  return SUCCESS;
}

SmallVector<NPUTensorDesc, N> CalcuOpUtil::create_npu_input_tensor_desc(
    const SmallVector<Tensor, N>& inputTensor) {
  SmallVector<NPUTensorDesc, N> inputs;

  for (int i = 0; i < inputTensor.size(); i++) {
    inputs.emplace_back(
        NPUTensorDesc(NpuUtils::format_contiguous(inputTensor[i])));

    if (inputTensor[i].dim() == 0) {
      inputs[i].tensorDescType = NPUTensorDesc::TensorDescType::TENSOR_SCALAR;
    }
  }

  return inputs;
}

SmallVector<NPUTensorDesc, N> CalcuOpUtil::create_npu_input_tensor_desc(
    const SmallVector<Tensor, N>& inputTensor,
    const SmallVector<uint, N>& masks) {
  SmallVector<NPUTensorDesc, N> inputs;

  for (int i = 0; i < inputTensor.size(); i++) {
    inputs.emplace_back(
        NPUTensorDesc(NpuUtils::format_contiguous(inputTensor[i])));

    if (inputTensor[i].dim() == 0) {
      inputs[i].tensorDescType = NPUTensorDesc::TensorDescType::TENSOR_SCALAR;
    }
  }

  // Set NPUTensorDesc.tensorDescType be NONE_TENSOR, which index in masks.
  for (int j = 0; j < masks.size(); j++) {
    inputs[masks[j]].tensorDescType =
        NPUTensorDesc::TensorDescType::NONE_TENSOR;
  }

  return inputs;
}

SmallVector<NPUTensorDesc, N> CalcuOpUtil::create_npu_input_tensor_desc(
    const SmallVector<Scalar, N>& inputScalar,
    ScalarType scalar_type) {
  SmallVector<NPUTensorDesc, N> inputs;

  for (int i = 0; i < inputScalar.size(); i++) {
    inputs.emplace_back(NPUTensorDesc(inputScalar[i], scalar_type));
  }

  return inputs;
}

SmallVector<NPUTensorDesc, N> CalcuOpUtil::create_npu_output_tensor_desc(
    const SmallVector<Tensor, N>& outputTensor) {
  SmallVector<NPUTensorDesc, N> outputs;

  for (int i = 0; i < outputTensor.size(); i++) {
    outputs.emplace_back(NPUTensorDesc(outputTensor[i]));
  }

  return outputs;
}

SmallVector<int64_t, N> CalcuOpUtil::get_dimlist_for_tensor(
    const Tensor& self) {
  SmallVector<int64_t, N> dimList = {};
  for (int64_t i = 0; i < self.dim(); i++) {
    dimList.emplace_back(i);
  }
  return dimList;
}

std::tuple<aclopAttr*, string> CalcuOpUtil::CreateNpuAttrDesc(
    const SmallVector<NPUAttrDesc, N>& attrs) {
  aclopAttr* attr = aclopCreateAttr();
  string attrInfo = "attrs=";

  for (NPUAttrDesc npuAttrDesc : attrs) {
    switch (npuAttrDesc.attrType) {
      case NPUAttrDesc::AttrDescType::BOOL_TYPE:
        aclopSetAttrBool(
            attr, npuAttrDesc.attrName.c_str(), npuAttrDesc.boolAttrValue);
        attrInfo += to_string(npuAttrDesc.boolAttrValue) + "-";
        break;
      case NPUAttrDesc::AttrDescType::INT_TYPE:
        aclopSetAttrInt(
            attr, npuAttrDesc.attrName.c_str(), npuAttrDesc.intAttrValue);
        attrInfo += to_string(npuAttrDesc.intAttrValue) + "-";
        break;
      case NPUAttrDesc::AttrDescType::FLOAT_TYPE:
        aclopSetAttrFloat(
            attr, npuAttrDesc.attrName.c_str(), npuAttrDesc.floatAttrValue);
        attrInfo += to_string(npuAttrDesc.floatAttrValue) + "-";
        break;
      case NPUAttrDesc::AttrDescType::STRING_TYPE:
        aclopSetAttrString(
            attr,
            npuAttrDesc.attrName.c_str(),
            npuAttrDesc.stringAttrValue.c_str());
        attrInfo += npuAttrDesc.stringAttrValue + "-";
        break;
      case NPUAttrDesc::AttrDescType::LIST_INT_TYPE:
        aclopSetAttrListInt(
            attr,
            npuAttrDesc.attrName.c_str(),
            npuAttrDesc.listIntAttrValue.size(),
            npuAttrDesc.listIntAttrValue.data());
        for (unsigned i = 0; i < npuAttrDesc.listIntAttrValue.size(); i++)
          attrInfo += to_string(npuAttrDesc.listIntAttrValue.at(i)) + ",";
        attrInfo += "-";
        break;
      case NPUAttrDesc::AttrDescType::LIST_FLOAT_TYPE:
        aclopSetAttrListFloat(
            attr,
            npuAttrDesc.attrName.c_str(),
            npuAttrDesc.listFloatAttrValue.size(),
            npuAttrDesc.listFloatAttrValue.data());
        for (unsigned i = 0; i < npuAttrDesc.listFloatAttrValue.size(); i++)
          attrInfo += to_string(npuAttrDesc.listFloatAttrValue.at(i)) + ",";
        attrInfo += "-";
        break;
      case NPUAttrDesc::AttrDescType::LIST_LIST_INT_TYPE:
        aclopSetAttrListListInt(
            attr,
            npuAttrDesc.attrName.c_str(),
            npuAttrDesc.listListIntAttrValue.size(),
            npuAttrDesc.listListIntAttrListIntNum.data(),
            npuAttrDesc.listListIntAttrValue.data());
        for (unsigned i = 0; i < npuAttrDesc.listListIntAttrValue.size(); i++)
          attrInfo += to_string(*npuAttrDesc.listListIntAttrValue.at(i)) + ",";
        attrInfo += "-";
        break;
      default:
        AT_ERROR("unsupport attr type", npuAttrDesc.attrType);
    }
  }

  return std::tuple<aclopAttr*, string>(attr, attrInfo);
}

void CalcuOpUtil::execute_npu_operate(
    string opName,
    SmallVector<NPUTensorDesc, N>& inputs,
    SmallVector<NPUTensorDesc, N>& outputs,
    const SmallVector<NPUAttrDesc, N>& attrs) {
  if (c10::npu::OptionsManager::CheckQueueEnable() ||
      c10::npu::OptionsManager::CheckDynamicEnable()) {
    ExecuteParas cur_paras;
    cur_paras.opType = opName;
    cur_paras.paras.hasAttr = attrs.size() == 0 ? false : true;
    CalcuOpUtil::CreateAclTensorDescInfo(
        inputs, outputs, cur_paras.paras, opName, attrs);
    auto attrRes = CalcuOpUtil::CreateNpuAttrDesc(attrs);
    cur_paras.attr = std::get<0>(attrRes);
    cur_paras.attrInfo = std::get<1>(attrRes);
    if (c10::npu::OptionsManager::CheckQueueEnable()) {
      if (!FuzzyCompileBlacklist::GetInstance().IsInBlacklist(cur_paras.opType) && env::CheckFuzzyEnable()) {
        cur_paras.isFuzzy = true;
      }
      c10::npu::queue::QueueParas params(c10::npu::queue::COMPILE_AND_EXECUTE, sizeof(ExecuteParas), &cur_paras);
      SmallVector<Storage, N> needClearVec;
      c10::npu::enCurrentNPUStream(&params, needClearVec);
      needClearVec.clear();
    } else {
      auto stream = c10::npu::getCurrentNPUStream();
      DynamicRun(cur_paras, stream);
      cur_paras.Release();
    }
    return;
  }

  ACL_PARAMS params;

  CalcuOpUtil::CreateAclTensorDescInfo(inputs, outputs, params, opName, attrs);
  aclopAttr* attr = std::get<0>(CalcuOpUtil::CreateNpuAttrDesc(attrs));

  auto stream = c10::npu::getCurrentNPUStream();
  RECORD_FUNCTION(opName, std::vector<c10::IValue>({}));
  bool reset_flag = false;
  if (env::CheckFuzzyEnable() &&
      FuzzyCompileBlacklist::GetInstance().IsInBlacklist(opName)) {
    AclopSetCompileFlag(aclOpCompileFlag::ACL_OP_COMPILE_DEFAULT);
    reset_flag = true;
  }
  NPU_LOGD("Op %s aclopCompileAndExecute Run.", opName.c_str());
  if (PyGILState_Check()) {
    Py_BEGIN_ALLOW_THREADS
    aclError ret;
    int index = 0;
    do {
      ret = aclopCompileAndExecute(
        opName.c_str(),
        params.input_num,
        params.input_desc,
        params.input_data_buf,
        params.output_num,
        params.output_desc,
        params.output_data_buf,
        attr,
        ACL_ENGINE_SYS,
        ACL_COMPILE_SYS,
        NULL,
        stream);
      ++index;
    } while(NpuUtils::IsOomError(ret, index) && (index < NPU_MAX_OP_EXEC_TRY_NUM));
    ACL_REQUIRE_OK_OP(ret, opName.c_str());
    Py_END_ALLOW_THREADS
  } else {
    aclError ret;
    int index = 0;
    do {
      ret = aclopCompileAndExecute(
        opName.c_str(),
        params.input_num,
        params.input_desc,
        params.input_data_buf,
        params.output_num,
        params.output_desc,
        params.output_data_buf,
        attr,
        ACL_ENGINE_SYS,
        ACL_COMPILE_SYS,
        NULL,
        stream);
      ++index;
    } while(NpuUtils::IsOomError(ret, index) && (index < NPU_MAX_OP_EXEC_TRY_NUM));
    ACL_REQUIRE_OK_OP(ret, opName.c_str());
  }
  if (reset_flag) {
    AclopSetCompileFlag(aclOpCompileFlag::ACL_OP_COMPILE_FUZZ);
  }
  aclopDestroyAttr(attr);
  NPUStatus ret = DestroyAclParams(params);

  if (ret != SUCCESS) {
    NPU_LOGE("DestroyAclParams fail, ret: %s", ret.c_str());
  }
}

int64_t CalcuOpUtil::completePad(
    int64_t s_size,
    int64_t p_size,
    int64_t k_size,
    int64_t stride) {
  int64_t needpads = 0;
  int64_t sizeP = s_size + p_size * 2;
  int64_t leftLen = sizeP - k_size;
  TORCH_CHECK(stride > 0,
      "stride should be greater than zero ",
      "but got ",
      stride);
  auto reminder = leftLen % stride;
  if (reminder != 0) {
    needpads = stride - reminder;
  }
  return needpads;
}

} // namespace npu
} // namespace native
} // namespace at
