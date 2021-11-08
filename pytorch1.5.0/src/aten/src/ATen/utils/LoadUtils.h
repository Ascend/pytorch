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
#pragma once

#include "H5Cpp.h"
#include "torch/csrc/autograd/VariableTypeUtils.h"
#include <ATen/TypeDefault.h>
#include <ATen/utils/DumpUtils.h>

using std::string;
using std::vector;

namespace at {

using stringmap = std::unordered_map<string, string>;
C10_API void SetLoadPath(string path);
C10_API void SetLoadWithAclDumpFlag(bool flag);
C10_API std::unordered_map<string, std::vector<string>> GetIrMapper();
C10_API std::unordered_map<string, stringmap> GetParamMapper();

struct TensorDesc {
  Tensor tensor;
  bool isList;
  string nameTensor;
};

struct CommDesc {
  string nameIr;
  std::vector<TensorDesc> tensorDescVec;
  std::vector<ArgDes<std::vector<int64_t>>> int64VecDescVec;
  std::vector<ArgDes<int64_t>> int64DescVec;
  std::vector<ArgDes<bool>> boolDescVec;
  std::vector<ArgDes<double>> doubleDescVec;
  std::vector<ArgDes<c10::optional<double>>> optionalDoubleDescVec;
  std::vector<ArgDes<at::Scalar>*> scalarDescVec;
  std::vector<ArgDes<c10::optional<int64_t>>> optionalInt64DescVec;
  std::vector<ArgDes<c10::optional<at::Scalar>>*> optionalScalarDescVec;
  std::vector<ArgDes<c10::ScalarType>> scalarTypeDescVec;
  std::vector<ArgDes<std::pair<size_t, size_t>>> sizePairDescVec;
  std::vector<ArgDes<c10::ArrayRef<long int>>> longIntArrayDescVec;
};

class LoadUtil {
 public:
  ~LoadUtil();

  static LoadUtil* GetInstance() {
    static LoadUtil instance;
    return &instance;
  };

  bool LoadTensor(const at::Tensor &t, string nameIr, bool isList, string nameTensor, bool isLast);

  void Process();

  void LoadOneInput(std::string &irName, ArgDes<at::Tensor> &t, bool isLast);
  void LoadOneInput(std::string &irName, ArgDes<std::vector<at::Tensor>> &t, bool isLast);
  void LoadOneInput(std::string &irName, ArgDes<std::vector<int64_t>> &t, bool isLast);
  void LoadOneInput(std::string &irName, ArgDes<int64_t> &t, bool isLast);
  void LoadOneInput(std::string &irName, ArgDes<bool> &t, bool isLast);
  void LoadOneInput(std::string &irName, ArgDes<double> &t, bool isLast);
  void LoadOneInput(std::string &irName, ArgDes<c10::optional<double>> &t, bool isLast);
  void LoadOneInput(std::string &irName, ArgDes<at::Scalar> &t, bool isLast);
  void LoadOneInput(std::string &irName, ArgDes<TypeAndSize> &t, bool isLast);
  void LoadOneInput(std::string &irName, ArgDes<c10::optional<int64_t>> &t, bool isLast);
  void LoadOneInput(std::string &irName, ArgDes<c10::optional<at::Scalar>> &t, bool isLast);
  void LoadOneInput(std::string &irName, ArgDes<std::vector<std::vector<int64_t>>> &t, bool isLast);
  void LoadOneInput(std::string &irName, ArgDes<at::TensorGeometry> &t, bool isLast);
  void LoadOneInput(std::string &irName, ArgDes<c10::ScalarType> &t, bool isLast);
  void LoadOneInput(std::string &irName, ArgDes<std::pair<size_t, size_t>> &t, bool isLast);

  void LoadOneInput(const string &irName, ArgDes<c10::ArrayRef<at::Tensor>> &input, bool isLast);
  void LoadOneInput(const string &irName, ArgDes<c10::ArrayRef<long int>> &input, bool isLast);
  void LoadOneInput(const string &irName, ArgDes<c10::Storage> &input, bool isLast);
  void LoadOneInput(const string &irName, ArgDes<at::Generator *> &input, bool isLast);
  void LoadOneInput(const string &irName, ArgDes<c10::ArrayRef<at::Dimname>> &input, bool isLast);
  void LoadOneInput(const string &irName, ArgDes<at::Dimname> &input, bool isLast);
  void LoadOneInput(const string &irName, ArgDes<c10::TensorOptions> &input, bool isLast);
  void LoadOneInput(const string &irName, ArgDes<c10::optional<c10::MemoryFormat>> &input, bool isLast);
  void LoadOneInput(const string &irName, ArgDes<c10::MemoryFormat> &input, bool isLast);
  void LoadOneInput(const string &irName, ArgDes<c10::optional<c10::ScalarType>> &input, bool isLast);
  void LoadOneInput(const string &irName, ArgDes<c10::optional<c10::ArrayRef<at::Dimname>>> &input, bool isLast);
  void LoadOneInput(const string &irName, ArgDes<c10::Device> &input, bool isLast);
  void LoadOneInput(const string &irName, ArgDes<c10::optional<bool>> &input, bool isLast);
  void LoadOneInput(const string &irName, ArgDes<std::array<bool, 2>> &input, bool isLast);
  void LoadOneInput(const string &irName, ArgDes<std::array<bool, 3>> &input, bool isLast);
  void LoadOneInput(const string &irName, ArgDes<std::array<bool, 4>> &input, bool isLast);
  void LoadOneInput(const string &irName, ArgDes<string> &input, bool isLast);
  void LoadOneInput(const string &irName, ArgDes<ConstQuantizerPtr> &input, bool isLast);

  template <typename T>
  void LoadInputs(std::string &irName, T &t) {
    if (loadInit) {
      LoadOneInput(irName, t, true);
    }
    return;
  }

  template <typename T, typename... Args>
  void LoadInputs(std::string &irName, T &t, Args &... rest) {
    if (loadInit) {
      LoadOneInput(irName, t, false);
      LoadInputs(irName, rest...);
    }
    return;
  }

  void SetLoadFlag(bool flag) {
    isInIr = flag;
  }

  bool GetLoadFlag() {
    return isInIr;
  }

  bool IsLoadSwitchOn() {
    return isLoadSwitchOn;
  }

  void SetLoadFilePath(const string& filePath) {
    loadFilePath = filePath;
  }

  int GetMatchedSeqId() {
    return matchedSeqId;
  }

  void SetLoadSwitch(bool flag) {
    isLoadSwitchOn = flag;
    return;
  }

  void LoadLazyInit();

  void Lock() {
    mu_.lock();
  }

  void Unlock() {
    mu_.unlock();
  }

  void SetLoadWithAclDumpFlag(bool flag) {
    loadWithAclDump = flag;
    return;
  }

  bool GetLoadWithAclDumpFlag() {
    return loadWithAclDump;
  }

  bool CheckWorkload(const at::Tensor& input, int stride);

 private:
  LoadUtil();
  H5::H5File* file = nullptr;
  CommDesc commDesc;
  std::vector<int> visitedSeq;
  bool isInIr = false;
  string loadFilePath = "Jason_1.h5";
  int matchedSeqId = -1;
  bool isLoadSwitchOn = false;
  bool loadInit = false;
  bool loadWithAclDump = false;
  std::recursive_mutex mu_;
};
} // namespace c10
