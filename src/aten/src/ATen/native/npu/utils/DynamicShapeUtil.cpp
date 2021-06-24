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

#include "DynamicShapeUtil.h"
#include <Python.h>
#include <unordered_set>
#include "ATen/native/npu/dynamicstrategy/Strategy.h"
#include "ATen/native/npu/frame/OpDynamicCmdHelper.h"
#include "ATen/native/npu/frame/OpDynamicParamMaker.h"

constexpr auto COMPILE_THREADS_MAX = 5;
constexpr int FIRST_STEP = 1;

namespace at {
namespace native {
namespace npu {

std::unordered_set<string> DynamicShapeUtil::disableDynamicOp = {
    "DropOutGenMask",
    "Conv2D",
    "Conv2DTransposeD",
    "Conv2DBackpropInputD",
    "Conv2DBackpropFilterD"};

long long int DynamicShapeUtil::steps_ = 0;
void DynamicShapeUtil::IncreaseSteps() {
  steps_++;
}

DynamicShapeUtil::DynamicShapeUtil()
    : thread_pool_(std::make_shared<TaskThreadPool>(COMPILE_THREADS_MAX)) {
  isDynamicOnly = c10::npu::OptionsManager::CheckDynamicOnly();
}

DynamicShapeUtil::~DynamicShapeUtil() {
  if (thread_pool_ != nullptr) {
    thread_pool_->waitWorkComplete();
  }
}

aclError DynamicShapeUtil::Run(ExecuteParas& params, aclrtStream stream) {
  bool hasScalar = false;
  const std::string staticKey = dynamicMap.CreateStaticKey(params, hasScalar);
  NPU_LOGD(
      "Op %s: CheckFirstStep Run and step: %lld.",
      params.opType.c_str(),
      steps_);
  NPU_LOGD("Check TransData, Op: %s.", params.opType.c_str());

  // if first step, static execute
  if ((CheckFirstStep() && !isDynamicOnly) || isStaticType(params)) {
    staticCompileAndExecute(params, staticKey, stream);
    if (dynamicMap.IsExistStaticKey(staticKey)) {
      return 0;
    }
    dynamicMap.InsertStaticKey(staticKey);
    return 0;
  }

  if (dynamicMap.IsExistStaticKey(staticKey)) {
    string isExistKey = staticKey + ". Exist.";
    staticCompileAndExecute(params, isExistKey, stream);
    return 0;
  }

  DynamicCompileShape dynamicCompileShape;
  DynamicCompileShapeMaker(params, dynamicCompileShape);

  // dynamic compile
  const std::string dynamicKey =
      dynamicMap.CreateDynamicKey(params, dynamicCompileShape);
  if (!dynamicMap.IsExistDynamicKey(dynamicKey)) {
    if (dynamicMap.InsertDynamicKey(dynamicKey, false)) {
      if (!isDynamicOnly) {
        staticCompileAndExecute(params, staticKey, stream);
        dynamicMap.InsertStaticKey(staticKey);
      }
      auto compileParams = CreateCompileParams(params, dynamicCompileShape);
      thread_pool_->run(std::bind(
          &DynamicShapeUtil::StartThreadCompile,
          this,
          compileParams,
          dynamicKey));
    }
    params.isCompiling = true;
    if (isDynamicOnly) {
      WaitDynamicCompileComplete();
    } else {
      return 0;
    }
  }

  if (dynamicMap.IsExistDynamicKey(dynamicKey)) {
    if (dynamicMap.GetDynamicValue(dynamicKey) == true) {
      logUtil.SetStartTime();
      // dynamic execute
      int ret = ExecuteDynamic(params, stream);
      if (ret != 0) {
        C10_NPU_SHOW_ERR_MSG();
        logUtil.PrintLog(steps_, dynamicKey, "Dynamic Execute Failed");
        std::stringstream msg;
        msg << __func__ << ":" << __FILE__ << ":" << __LINE__;
        TORCH_CHECK(0, msg.str());
      } else {
        logUtil.PrintLog(steps_, dynamicKey, "Dynamic Execute Succeed");
      }
    } else {
      // static execute
      if (!isDynamicOnly) {
        staticCompileAndExecute(params, staticKey, stream);
        dynamicMap.InsertStaticKey(staticKey);
      }
    }
  }

  return 0;
}

void DynamicShapeUtil::DynamicCompileShapeMaker(
    ExecuteParas& params,
    DynamicCompileShape& compileShape) {
  register_dynamic_shape::DynamicOptRegister::GetInstance()
      ->CreateDynmaicDescInfo(params.opType, params.paras, compileShape);
}

bool DynamicShapeUtil::CheckFirstStep() {
  return steps_ <= FIRST_STEP;
}

bool DynamicShapeUtil::isStaticType(const ExecuteParas& params) {
  bool result = false;

  do {
    if (params.opType[0] == 'P' && params.opType[1] == 't') {
      result = true;
      break;
    }

    if (DynamicShapeUtil::disableDynamicOp.find(params.opType) !=
        disableDynamicOp.end()) {
      result = true;
      break;
    }

    bool isDebugDynamic =
        DebugDynamic::GetInstance()->CheckInConfig(params.opType);
    if (isDebugDynamic == true) {
      result = true;
      break;
    }
  } while (0);

  return result;
}

void DynamicShapeUtil::CreateAclParamsDesc(
    const aclTensorDesc** inDescDynamic,
    int inNum,
    int64_t* inStorageDims,
    aclFormat* inStorageFormats,
    CONST_PARAMS& constParams,
    SmallVector<FormatShape, N> shape,
    SmallVector<FormatShape, N> storageShape,
    const aclTensorDesc** outDescDynamic) {
  for (size_t i = 0; i < inNum; i++) {
    aclTensorDesc* desc = const_cast<aclTensorDesc*>(inDescDynamic[i]);
    aclDataType dtype = aclGetTensorDescType(desc);
    aclFormat format = aclGetTensorDescFormat(desc);
    int dims = (int)aclGetTensorDescNumDims(desc);
    int storageDims = inStorageDims[i];
    string compileName = aclGetTensorDescName(desc);

    NPU_LOGD(
        "CreateAclParamsInputDesc Run, TensorDescName %s .",
        compileName.c_str());

    if (dims == 0) {
      dims = 1;
      storageDims = 1;
    }

    if (dynamicMap.CheckConstInput(i, constParams)) {
      outDescDynamic[i] = CreatConstDesc(
          i, desc, dims, format, dtype, compileName, constParams);
      continue;
    }

    desc = aclCreateTensorDesc(dtype, dims, shape[i].data(), format);
    int64_t desc_dims_size = (int64_t)dims;
    std::vector<int64_t> range(2 * desc_dims_size);

    for (int64_t k = 0; k < desc_dims_size * 2; k += 2) {
      range[k] = 1;
      range[k + 1] = -1;
    }

    typedef int64_t(*TYPE)[2];
    aclSetTensorShapeRange(desc, desc_dims_size, (TYPE)range.data());

    if (storageShape.size() != 0) {
      aclSetTensorFormat(desc, inStorageFormats[i]);
      aclSetTensorShape(desc, storageDims, storageShape[i].data());
    }

    outDescDynamic[i] = desc;
    if (compileName != "") {
      aclSetTensorDescName(desc, (char*)compileName.c_str());
    }
  }
}

ExecuteParas DynamicShapeUtil::CreateCompileParams(
    ExecuteParas& params,
    DynamicCompileShape dynamicCompileShape) {
  ExecuteParas compileParams;

  if (params.opDynamicType != "") {
    compileParams.opDynamicType = params.opDynamicType;
    compileParams.dynamicParam.input_num = params.dynamicParam.input_num;
    compileParams.dynamicParam.output_num = params.dynamicParam.output_num;
    compileParams.dynamicParam.compile_input_desc =
        params.dynamicParam.compile_input_desc;
    compileParams.dynamicParam.compile_output_desc =
        params.dynamicParam.compile_output_desc;
    compileParams.dynamicCompileAttr = params.dynamicCompileAttr;
  } else {
    const aclTensorDesc** compileInputsDescs = params.paras.input_num == 0
        ? nullptr
        : new const aclTensorDesc*[params.paras.input_num];
    const aclTensorDesc** compileOutputsDescs = params.paras.output_num == 0
        ? nullptr
        : new const aclTensorDesc*[params.paras.output_num];

    NPU_LOGD(" Op %s CreateAclParamsDesc Run.", params.opType.c_str());
    CreateAclParamsDesc(
        params.paras.input_desc,
        params.paras.input_num,
        params.paras.inputDims,
        params.paras.inputFormats,
        params.constParams,
        dynamicCompileShape.inputShape,
        dynamicCompileShape.inputStorageShape,
        compileInputsDescs);

    CONST_PARAMS constParams = CONST_PARAMS();
    NPU_LOGD(" Op %s CreateAclParamsOutputDesc Run.", params.opType.c_str());
    CreateAclParamsDesc(
        params.paras.output_desc,
        params.paras.output_num,
        params.paras.outputDims,
        params.paras.outputFormats,
        constParams,
        dynamicCompileShape.outputShape,
        dynamicCompileShape.outputStorageShape,
        compileOutputsDescs);

    compileParams.opType = params.opType;
    compileParams.paras.input_num = params.paras.input_num;
    compileParams.paras.output_num = params.paras.output_num;
    compileParams.paras.input_desc = compileInputsDescs;
    compileParams.paras.output_desc = compileOutputsDescs;
    compileParams.attr = params.attr;
    compileParams.paras.input_data_buf = nullptr;
    compileParams.paras.output_data_buf = nullptr;
    compileParams.paras.inputDims = nullptr;
    compileParams.paras.outputDims = nullptr;
    compileParams.paras.inputFormats = nullptr;
    compileParams.paras.outputFormats = nullptr;
  }

  return compileParams;
}

void DynamicShapeUtil::StartThreadCompile(
    ExecuteParas params,
    const string key) {
  logUtil.SetStartTime();
  if (CompileDynamic(params) == ACL_ERROR_NONE) {
    logUtil.PrintLog(steps_, key, "Dynamic Compile Succeed");
    NPU_LOGD(
        " Op %s dynamic compile success and update dynamic MAP value.",
        params.opType.c_str());
    dynamicMap.UpdateDynamicKey(key, true);
  } else {
    logUtil.PrintLog(steps_, key, "Dynamic Compile Failed");
    C10_NPU_SHOW_ERR_MSG();
  }

  // free attr and DesctroyParams
  if (params.opDynamicType != "") {
    aclopDestroyAttr(params.dynamicCompileAttr);
    DestroyDynamicAclParams(params.dynamicParam);
  } else {
    if (!isDynamicOnly) {
      aclopDestroyAttr(params.attr);
    }
    DestroyAclParams(params.paras);
  }
}

aclError DynamicShapeUtil::CompileDynamic(ExecuteParas& cur_paras) {
  auto params = OpDynamicCmdHelper::CreateDynamicCompileParams(cur_paras);
  aclError compileRes = aclopCompile(
      std::get<0>(params).c_str(),
      std::get<1>(params),
      std::get<2>(params),
      std::get<3>(params),
      std::get<4>(params),
      std::get<5>(params),
      ACL_ENGINE_SYS,
      ACL_COMPILE_SYS,
      NULL);

  return compileRes;
}

aclError DynamicShapeUtil::ExecuteDynamic(
    ExecuteParas& cur_paras,
    aclrtStream stream) {
  auto params = OpDynamicCmdHelper::CreateDynamicRunParams(cur_paras);
  return aclopExecuteV2(
      std::get<0>(params).c_str(),
      std::get<1>(params),
      const_cast<aclTensorDesc**>(std::get<2>(params)),
      const_cast<aclDataBuffer**>(std::get<3>(params)),
      std::get<4>(params),
      const_cast<aclTensorDesc**>(std::get<5>(params)),
      std::get<6>(params),
      const_cast<aclopAttr*>(std::get<7>(params)),
      stream);
}

void DynamicShapeUtil::staticCompileAndExecute(
    ExecuteParas& cur_paras,
    const string key,
    aclrtStream stream) {
  std::string opName = cur_paras.opType;
  NPU_LOGD(" Op %s aclopCompileAndExecute Run.", opName.c_str());
  aclError ret;
  logUtil.SetStartTime();
  ret = aclopCompileAndExecute(
      opName.c_str(),
      cur_paras.paras.input_num,
      cur_paras.paras.input_desc,
      cur_paras.paras.input_data_buf,
      cur_paras.paras.output_num,
      cur_paras.paras.output_desc,
      cur_paras.paras.output_data_buf,
      cur_paras.attr,
      ACL_ENGINE_SYS,
      ACL_COMPILE_SYS,
      NULL,
      stream);

  if (ret != 0) {
    C10_NPU_SHOW_ERR_MSG();
    logUtil.PrintLog(steps_, key, "Static Compile And Execute Failed");
    NPU_LOGE(" aclopCompileAndExecute fail, opName: %s", opName.c_str());
    std::stringstream msg;
    msg << __func__ << ":" << __FILE__ << ":" << __LINE__;
    TORCH_CHECK(0, msg.str());
  } else {
    logUtil.PrintLog(steps_, key, "Static Compile And Execute Succeed");
  }
}

void DynamicShapeUtil::WaitThreadComplete() {
  if (PyGILState_Check()) {
    Py_BEGIN_ALLOW_THREADS thread_pool_->waitWorkComplete();
    Py_END_ALLOW_THREADS
  } else {
    thread_pool_->waitWorkComplete();
  }
}

aclTensorDesc* DynamicShapeUtil::CreatConstDesc(
    const size_t index,
    const aclTensorDesc* desc,
    const int dims,
    const aclFormat format,
    const aclDataType dtype,
    const string compileName,
    CONST_PARAMS& constParams) {
  int64_t dim = 0;
  aclGetTensorDescDimV2(desc, 0, &dim);
  std::vector<int64_t> desc_dims(dims, dim);
  aclTensorDesc* constDesc =
      aclCreateTensorDesc(dtype, dims, desc_dims.data(), format);
  aclSetTensorFormat(constDesc, format);
  aclSetTensorShape(constDesc, dims, desc_dims.data());
  int64_t constSign = dynamicMap.constSign;
  std::vector<int32_t> dimList;
  for (int i = 0; i < dim; ++i) {
    dimList.push_back(
        static_cast<int32_t>(*(constParams.constList[constSign] + i)));
  }

  aclSetTensorConst(constDesc, dimList.data(), sizeof(int32_t) * dim);
  if (compileName != "") {
    aclSetTensorDescName(constDesc, (char*)compileName.c_str());
  }
  dynamicMap.constSign = -1;
  return constDesc;
}

aclError DynamicRun(ExecuteParas& params, aclrtStream stream) {
  NPU_LOGD(" Op %s DynamicRun Run.", params.opType.c_str());
  aclError RunRes = DynamicShapeUtil::GetInstance()->Run(params, stream);
  if (RunRes != 0) {
    NPU_LOGE(" DynamicRun fail, opName: %s", params.opType.c_str());
    std::stringstream msg;
    msg << __func__ << ":" << __FILE__ << ":" << __LINE__;
    TORCH_CHECK(0, msg.str());
  }
  return RunRes;
}

void WaitDynamicCompileComplete() {
  char* dynamicCompileEnableEnv = std::getenv("DYNAMIC_COMPILE_ENABLE");
  int64_t envFlag = (dynamicCompileEnableEnv != nullptr)
      ? strtol(dynamicCompileEnableEnv, nullptr, 10)
      : 0;
  if (envFlag != 0) {
    DynamicShapeUtil::GetInstance()->WaitThreadComplete();
  }
}

void DynamicIncreaseSteps() {
  NPU_LOGD(" IncreaseSteps() Run.");
  DynamicShapeUtil::GetInstance()->IncreaseSteps();
}

} // namespace npu
} // namespace native
} // namespace at