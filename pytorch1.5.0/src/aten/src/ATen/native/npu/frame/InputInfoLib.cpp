// Copyright (c) 2020 Huawei Technologies Co., Ltd
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

#include "InputInfoLib.h"

namespace at {
namespace native {
namespace npu {

string InputInfoLib::ConstShapeMaker(int64_t inputId, ExecuteParas& params) {
  string shapeString = "";
  
  int64_t dim = 0;
  aclGetTensorDescDimV2(params.paras.input_desc[inputId], 0, &dim);
  for (int i = 0; i < dim; ++i) {
    shapeString += to_string(params.constParams.constList[constSign][i]) + ",";
  }
  shapeString += "_" + shapeString + "_isConst;";
  constSign = -1;
  return shapeString;
}

string InputInfoLib::ShapeMaker(
    int64_t inputId,
    ExecuteParas& params,
    DynamicCompileShape dynamicCompileShape) {
  string shapeString = "";
  int64_t dims = aclGetTensorDescNumDims(params.paras.input_desc[inputId]);
  int64_t storageDims = params.paras.inputDims[inputId];
    
  if (CheckConstInput(inputId, params.constParams)) {
    return ConstShapeMaker(inputId, params);
  } else {
    for (int64_t i = 0; i < dims; ++i) {
      shapeString += to_string(dynamicCompileShape.inputShape[inputId][i]) + ",";
    }
    
    if (dynamicCompileShape.inputStorageShape.size() != 0) {
      shapeString += "_";
      for (int64_t i = 0; i < dynamicCompileShape.inputStorageShape[inputId].size(); ++i) {
        shapeString += to_string(dynamicCompileShape.inputStorageShape[inputId][i]) + ",";
      }
    }
    
    shapeString += ";";
    return shapeString;
  }
}

string InputInfoLib::CreateStaticKey(ExecuteParas& params, bool& hasScalar) {
  // info = "opName:dtype1-format1-size1-storageFormat1;dtype2-fomat2-size2-storageFormat2; ..."; 
  // e.g.: "Add:0-0-1,2,3,-0;1-2-5,6,7,-2;"
  string info = params.opType + ":";
  int inputsNum = params.paras.input_num;
  const aclTensorDesc** aclInputDesc = params.paras.input_desc;

  for (int64_t i = 0; i < inputsNum; ++i) {   
    // add dtype and format to info.
    info += to_string(aclGetTensorDescType(aclInputDesc[i])) + "_" + 
        to_string(aclGetTensorDescFormat(aclInputDesc[i])) + "_" +
        to_string(params.paras.inputFormats[i]) + "_";

    // add sizes to info.
    for (int64_t j = 0; j < aclGetTensorDescNumDims(aclInputDesc[i]); ++j) {
      info += to_string(aclGetTensorDescDim(aclInputDesc[i], j)) + ",";
    }

    info += ";";
    NPU_LOGD("Check Op %s scalar.", params.opType.c_str());
    hasScalar = params.paras.inputDims[i] == 0 ? true : false;
  }

  return info;
}

string InputInfoLib::CreateDynamicKey(
    ExecuteParas& params, 
    DynamicCompileShape dynamicCompileShape) {
  string info = ""; 
  if (params.opDynamicType != "") {
    info += params.opDynamicType + ":" + params.dynamicParam.dynamicKey;
    return info;
  }
  info = params.opType + ":";
  int inputsNum = params.paras.input_num;

  for (int64_t i = 0; i < inputsNum; ++i) {
    info += to_string(aclGetTensorDescType(params.paras.input_desc[i])) + "_" +
        to_string(aclGetTensorDescFormat(params.paras.input_desc[i])) + "_" +
        to_string(params.paras.inputFormats[i]) + "_" + 
        ShapeMaker(i, params, dynamicCompileShape);
  }
  info += ". " + params.attrInfo;
  return info;
}

bool InputInfoLib::InsertDynamicKey(const string& key, bool value) {
  return dynamicLib.insert(pair<string, bool>(key, value)).second ? true : false;
}

bool InputInfoLib::InsertStaticKey(const string& key) {
  return staticLib.insert(key).second ? true : false;
}

bool InputInfoLib::UpdateDynamicKey(const string& key, bool value) {
  auto it = dynamicLib.find(key);
  
  if (it == dynamicLib.end()) {
    return false;
  } else {
    it->second = value;
    return true;
  }
}

bool InputInfoLib::GetDynamicValue(const string& key) {
    auto it = dynamicLib.find(key);
    return it->second;
}

bool InputInfoLib::IsExistDynamicKey(const string& key) {
    auto it = dynamicLib.find(key);
    return it != dynamicLib.end() ? true : false;
}

bool InputInfoLib::IsExistStaticKey(const string& key) {
  return staticLib.count(key) != 0 ? true : false;
}

bool InputInfoLib::CheckConstInput(int64_t index, CONST_PARAMS& constParams) {
  int64_t constNum = constParams.constNum;
  for (int64_t i = 0; i < constNum; ++i) {
    if (index == constParams.constIdx[i]) {
      constSign = i;
      return true;
    }
  }
  return false;
}

} // namespace npu
} // namespace native
} // namespace at
