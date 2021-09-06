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
#include <mutex>

using std::string;
using std::vector;

namespace at {

enum class C10_API DumpMode:int32_t {
  OFF = 0,
  DUMP = 1,
  LOAD = 2,
  CHK_OVERFLOW = 3,
};

C10_API void SetDumpMode(DumpMode mode);
C10_API void SetDumpPath(string path);

enum class SaveType:int32_t {
  TENSOR = 0,
  VEC_TENSOR = 1,
  VEC_I64 = 2,
  I64 = 3,
  BOOL = 4,
  DOUBLE = 5,
  OPT_DOUBLE = 6,
  SCALAR = 7,
  TYPE_AND_SIZE = 8,
  OPT_INT64 = 9,
  OPT_SCALAR = 10,
  VEC_VEC_I64 = 11,
  GEOMETRY = 12,
  SIZE = 13,
  ScalarType = 14,
  PAIR = 15,
  AR_LONG_INT = 16,
  OPT_MEM_FMT = 17,
  TENSOR_OPTS = 18,
};

template <typename T>
struct ArgDes {
public:
  explicit ArgDes(string name, const T& value): name_(move(name)), value_(value) {}

  const string& Name() const {
    return name_;
  }

  const T& GetValue() const {
    return value_;
  }

  void SetValue(const T &value) {
    value_ = value;
  }

  void SetName(const string& newName) {
    name_ = newName;
  }
private:
  string name_;
  T value_;
};

class DumpUtil {

 public:
  ~DumpUtil();

  static DumpUtil* GetInstance() {
    static DumpUtil instance;
    return &instance;
  };
  bool CheckAndCreateGroup(string& h5Path);

  string GetValHdf5Path(
      const string& irName,
      const int& seqId,
      bool isInput,
      const string& valName,
      int listIndex = -1);

  void PrepareSimpleHdf5Attr(
    const H5::DataSet* dataset,
    const H5std_string& attrName,
    H5::PredType attrPredType,
    const void* attrValue);

  void WriteValToHdf5Dataset(
      const string& h5Path,
      int valRank,
      size_t valSize,
      SaveType valSaveType,
      const void* valDataAddr,
      H5::PredType attrPredType,
      H5::PredType datasetPredType);

  bool DumpTensor(const string& h5Path, const Tensor& tensor);

  void DumpOneInput(const string& irName, int seqId, const ArgDes<at::Tensor>& input);
  void DumpOneInput(const string& irName, int seqId, const ArgDes<vector<at::Tensor>>& input);
  void DumpOneInput(const string& irName, int seqId, const ArgDes<vector<int64_t>>& input);
  void DumpOneInput(const string& irName, int seqId, const ArgDes<int64_t>& input);
  void DumpOneInput(const string& irName, int seqId, const ArgDes<bool>& input);
  void DumpOneInput(const string& irName, int seqId, const ArgDes<double>& input);
  void DumpOneInput(const string& irName, int seqId, const ArgDes<c10::optional<double>>& input);
  void DumpOneInput(const string& irName, int seqId, const ArgDes<at::Scalar>& input);
  void DumpOneInput(const string& irName, int seqId, const ArgDes<torch::autograd::generated::TypeAndSize>& input);
  void DumpOneInput(const string& irName, int seqId, const ArgDes<c10::optional<int64_t>>& input);
  void DumpOneInput(const string& irName, int seqId, const ArgDes<c10::optional<at::Scalar>>& input);
  void DumpOneInput(const string& irName, int seqId, const ArgDes<vector<vector<int64_t>>>& input);
  void DumpOneInput(const string& irName, int seqId, const ArgDes<at::TensorGeometry>& input);
  void DumpOneInput(const string& irName, int seqId, const ArgDes<size_t>& input);
  void DumpOneInput(const string& irName, int seqId, const ArgDes<c10::ScalarType>& input);
  void DumpOneInput(const string& irName, int seqId, const ArgDes<std::pair<size_t, size_t>>& input);

  void DumpOneInput(const string& irName, int seqId, const ArgDes<c10::ArrayRef<at::Tensor>>& input);
  void DumpOneInput(const string& irName, int seqId, const ArgDes<c10::ArrayRef<long int>>& input);
  void DumpOneInput(const string& irName, int seqId, const ArgDes<c10::Storage>& input);
  void DumpOneInput(const string& irName, int seqId, const ArgDes<at::Generator*>& input);
  void DumpOneInput(const string& irName, int seqId, const ArgDes<c10::ArrayRef<at::Dimname>>& input);
  void DumpOneInput(const string& irName, int seqId, const ArgDes<at::Dimname>& input);
  void DumpOneInput(const string& irName, int seqId, const ArgDes<c10::TensorOptions>& input);
  void DumpOneInput(const string& irName, int seqId, const ArgDes<c10::optional<c10::MemoryFormat>>& input);
  void DumpOneInput(const string& irName, int seqId, const ArgDes<c10::MemoryFormat>& input);
  void DumpOneInput(const string& irName, int seqId, const ArgDes<c10::optional<c10::ScalarType>>& input);
  void DumpOneInput(const string& irName, int seqId, const ArgDes<c10::optional<c10::ArrayRef<at::Dimname>>>& input);
  void DumpOneInput(const string& irName, int seqId, const ArgDes<c10::Device>& input);
  void DumpOneInput(const string& irName, int seqId, const ArgDes<c10::optional<bool>>& input);
  void DumpOneInput(const string& irName, int seqId, const ArgDes<std::array<bool, 2>>& input);
  void DumpOneInput(const string& irName, int seqId, const ArgDes<std::array<bool, 3>>& input);
  void DumpOneInput(const string& irName, int seqId, const ArgDes<std::array<bool, 4>>& input);
  void DumpOneInput(const string& irName, int seqId, const ArgDes<string>& input);
  void DumpOneInput(const string& irName, int seqId, const ArgDes<ConstQuantizerPtr>& input);

  template <typename T>
  void DumpInputs(const string& irName, int seqId, const T& input) {
    DumpOneInput(irName, seqId, input);
    return;
  }

  template <typename T, typename... Args>
  void DumpInputs(const string& irName, int seqId, const T& input, const Args&... rest) {
    DumpOneInput(irName, seqId, input);
    DumpInputs(irName, seqId, rest...);
    return;
  }

  void DumpOneOutput(const string& irName, int seqId, const ArgDes<vector<at::Tensor>>& output);
  void DumpOneOutput(const string& irName, int seqId, const ArgDes<at::Tensor>& output);
  void DumpOneOutput(const string& irName, int seqId, const ArgDes<double>& output);
  void DumpOneOutput(const string& irName, int seqId, const ArgDes<int64_t>& output);
  void DumpOneOutput(const string& irName, int seqId, const ArgDes<bool>& output);
  void DumpOneOutput(const string& irName, int seqId, const ArgDes<c10::Scalar>& output);
  void DumpOneOutput(const string& irName, int seqId, const ArgDes<c10::QScheme>& output);
  void DumpOneOutput(const string& irName, int seqId, const ArgDes<c10::ScalarType>& output);

  template <typename T>
  void DumpOutputs(const string& irName, int seqId, const T& output) {
    if (seqId != -1) {
      DumpOneOutput(irName, seqId, output);
    }
    return;
  }

  template <typename T, typename... Args>
  void DumpOutputs(const string& irName, int seqId, const T& output, const Args&... rest) {
    if (seqId != -1) {
      DumpOneOutput(irName, seqId, output);
      DumpOutputs(irName, seqId, rest...);
    }
    return;
  }

  uint64_t DumpSeqIdAddOne() {
    dumpSeqId++;
    return dumpSeqId;
  }

  bool IsDumpSwitchOn() {
    return isDumpSwitchOn;
  }

  void SetDumpFlag(bool flag) {
    isDumping = flag;
    return;
  }

  bool GetDumpFlag() {
    return isDumping;
  }

  void SetDumpSwitch(bool flag) {
    isDumpSwitchOn = flag;
    if (!isDumpSwitchOn) {
      if (dumpInit) {
        dumpInit = false;
        delete file;
      }
    }
    return;
  }

  void SetDumpFilePath(const string& filePath) {
    dumpFilePath = filePath;
  }

  void DumpLazyInit();

  void Lock() {
    mu_.lock();
  }

  void Unlock() {
    mu_.unlock();
  }

  void StartAclDump();
  void FinalizeAclDump();

 private:
  DumpUtil();
  H5::H5File* file = nullptr;
  uint64_t dumpSeqId = 0;
  bool isDumping = false;
  bool isDumpSwitchOn = false;
  string dumpFilePath = "dump.h5";
  bool dumpInit = false;
  std::recursive_mutex mu_;
};

} // namespace c10
