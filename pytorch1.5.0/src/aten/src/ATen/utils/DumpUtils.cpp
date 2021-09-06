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

#include <iostream>
#include <ATen/utils/DumpUtils.h>
#include <ATen/utils/LoadUtils.h>
#include <ATen/utils/OverflowUtils.h>
#ifdef USE_NPU
#include <c10/npu/NPUStream.h>
#include <c10/npu/NPUException.h>
#endif

using namespace std;
using namespace H5;

namespace at {

  void SetDumpMode(DumpMode mode) {
    if (mode == DumpMode::OFF) {
      DumpUtil::GetInstance()->SetDumpSwitch(false);
      LoadUtil::GetInstance()->SetLoadSwitch(false);
      OverflowUtil::GetInstance()->SetCheckSwitch(false);
    } else if (mode == DumpMode::DUMP) {
      DumpUtil::GetInstance()->SetDumpSwitch(true);
    } else if (mode == DumpMode::LOAD) {
      LoadUtil::GetInstance()->SetLoadSwitch(true);
    } else if (mode == DumpMode::CHK_OVERFLOW) {
      OverflowUtil::GetInstance()->SetCheckSwitch(true);
    }
    return;
  }

  class ScalarTypeHashFunction {
   public:
    size_t operator()(const c10::ScalarType& type) const {
      return static_cast<size_t>(type);
    }
  };

  static const std::unordered_map<c10::ScalarType, int, ScalarTypeHashFunction> scalarTypeToDumpTypeMap = {
    {c10::kFloat, 1},
    {c10::kByte, 2},
    {c10::kChar, 3},
    {c10::kShort, 5},
    {c10::kInt, 6},
    {c10::kLong, 7},
    {c10::kBool, 9},
    {c10::kHalf, 10},
    {c10::kDouble, 11},
  };

  static int64_t ScalarTypeToDumpType(const c10::ScalarType& st) {
    int64_t dump_type = -1;
    const auto it = scalarTypeToDumpTypeMap.find(st);
    if (it != scalarTypeToDumpTypeMap.end()) {
       dump_type = it->second;
    }
    return dump_type;
  }


  static const std::unordered_map<c10::ScalarType, H5::PredType, ScalarTypeHashFunction> scalarTypeToPredTypeMap = {
    {c10::kFloat, PredType::IEEE_F32LE},
    {c10::kByte, PredType::STD_I8LE},
    {c10::kChar, PredType::STD_I8LE},
    {c10::kShort, PredType::STD_I16LE},
    {c10::kInt, PredType::STD_I32LE},
    {c10::kLong, PredType::STD_I64LE},
    {c10::kBool, PredType::STD_I8LE},
    {c10::kHalf, PredType::IEEE_F32LE},
    {c10::kDouble, PredType::IEEE_F64LE},
  };

  static H5::PredType ScalarTypeToPredType(const c10::ScalarType& st) {
    H5::PredType  save_type = PredType::IEEE_F32LE;
    const auto it = scalarTypeToPredTypeMap.find(st);
    if (it != scalarTypeToPredTypeMap.end()) {
      save_type = it->second;
    }
    return save_type;
  }

  const H5std_string ATTR_DEVICE_TYPE_NAME("DeviceType");
  const H5std_string ATTR_DATA_TYPE_NAME("DataType");
  const H5std_string ATTR_FORMAT_NAME("FormatType");
  const H5std_string ATTR_TYPE_NAME("Type");
  const H5std_string ATTR_STRIDE_NAME("Stride");
  const H5std_string ATTR_REQUIRES_GRAD_NAME("RequiresGrad");

  DumpUtil::DumpUtil() {
  }

  DumpUtil::~DumpUtil() {
    if (dumpInit)
      delete file;
  }

  void SetDumpPath(string path) {
    DumpUtil::GetInstance()->SetDumpFilePath(path);
    return;
  }

  void DumpUtil::DumpLazyInit() {
    if (!dumpInit) {
      file = new H5File(dumpFilePath, H5F_ACC_TRUNC);
      dumpInit = true;
    }
  }

  bool DumpUtil::CheckAndCreateGroup(string& h5Path) {
    bool isExist = false;
    size_t pos = 0;
    DumpLazyInit();
    while(string::npos != pos) {
      pos = h5Path.find("/", pos+1);
      isExist = file->nameExists(h5Path.substr(0, pos));
      if (!isExist) {
        // create HDF5 group, similar to create a directory
        Group *group = new Group(file->createGroup(h5Path.substr(0, pos)));
        // then release the handle
        delete group;
      }
    }
    return true;
  }

  string DumpUtil::GetValHdf5Path(
      const string& irName,
      const int& seqId,
      bool isInput,
      const string& valName,
      int listIndex) {
    string h5Path = "/" + irName + "/" + std::to_string(seqId);
    if (isInput) {
      h5Path += "/input";
    } else {
      h5Path += "/output";
    }
    if (listIndex != -1) {
      h5Path += "/" + valName;
    }
    CheckAndCreateGroup(h5Path);
    if (listIndex != -1) {
      h5Path += "/" + valName + "_" + to_string(listIndex);
    } else {
      h5Path += "/" + valName;
    }
    return h5Path;
  }

  void DumpUtil::PrepareSimpleHdf5Attr(
      const DataSet* dataset,
      const H5std_string& attrName,
      PredType attrPredType,
      const void* attrValue) {
    if ((dataset == nullptr) || (attrValue == nullptr)) {
      return;
    }

    DataSpace dataSpace = DataSpace();
    Attribute attribute = dataset->createAttribute(attrName, attrPredType, dataSpace);
    attribute.write(attrPredType, attrValue);
  }

  void DumpUtil::WriteValToHdf5Dataset(
      const string& h5Path,
      int valRank,
      size_t valSize,
      SaveType valSaveType,
      const void* valDataAddr,
      PredType attrPredType,
      PredType datasetPredType) {
    if ((valRank != 0) && (valRank != 1)) {
      return;
    }
    // create dataset
    DataSpace dataspace;
    if (valRank == 0) {
      dataspace = DataSpace();
    } else {
      hsize_t dims[1] = {valSize};
      dataspace = DataSpace(valRank, dims);
    }
    DataSet* dataset = new DataSet(
        file->createDataSet(h5Path, datasetPredType, dataspace));

    int saveType = static_cast<int>(valSaveType);
    PrepareSimpleHdf5Attr(dataset, ATTR_TYPE_NAME, attrPredType, &saveType);

    // write dataset
    dataset->write(valDataAddr, datasetPredType);
    delete dataset;
  }

  bool DumpUtil::DumpTensor(const string& h5Path, const Tensor& tensor) {
    if (!tensor.has_storage()) {
      return false;
    }

    // create dataset
    int rank = tensor.ndimension();
    DataSpace dataspace = DataSpace(rank, (hsize_t *)tensor.sizes().vec().data());
    DataSet* dataset = nullptr;
    if (tensor.scalar_type() != ScalarType::Half){
      dataset = new DataSet(file->createDataSet(h5Path, ScalarTypeToPredType(tensor.scalar_type()), dataspace));
    } else {
      dataset = new DataSet(file->createDataSet(h5Path, PredType::IEEE_F32LE, dataspace));
    }

    // prepare device type attribute
    int deviceType = static_cast<int16_t>(tensor.device().type());
    PrepareSimpleHdf5Attr(dataset, ATTR_DEVICE_TYPE_NAME, PredType::STD_I32LE, &deviceType);

    // prepare dtype attribute
    int dumpType = ScalarTypeToDumpType(tensor.scalar_type());
    if (dumpType == -1) {
      // the dtype can not be recognized
      delete dataset;
      return false;
    }
    PrepareSimpleHdf5Attr(dataset, ATTR_DATA_TYPE_NAME, PredType::STD_I32LE, &dumpType);

    // prepare format attribute
    int formatData = (int)ACL_FORMAT_NCHW;
    if (tensor.device().type() == DeviceType::NPU) {
      formatData = (int)tensor.storage().unsafeGetStorageImpl()->npu_desc_.npu_format_;
    }
    PrepareSimpleHdf5Attr(dataset, ATTR_FORMAT_NAME, PredType::STD_I32LE, &formatData);

    // prepare type attribute
    int typeData = static_cast<int>(SaveType::TENSOR);
    PrepareSimpleHdf5Attr(dataset, ATTR_TYPE_NAME, PredType::STD_I32LE, &typeData);

    // create contiguous cpu tensor
    auto tensor_cpu = tensor.detach().cpu().contiguous().clone();

    // prepare stride attribute
    rank = 1;
    hsize_t dims[1] = {tensor_cpu.strides().size()};
    DataSpace strideDataspace = DataSpace(rank, dims);
    Attribute attribute = dataset->createAttribute(ATTR_STRIDE_NAME, PredType::STD_I64LE, strideDataspace);
    attribute.write(PredType::STD_I64LE, tensor_cpu.strides().data());

    // write to dataset
    if (tensor.device().type() == DeviceType::CPU) {
      dataset->write(tensor_cpu.storage().data_ptr().get(), ScalarTypeToPredType(tensor_cpu.scalar_type()));
    } else if (tensor.device().type() == DeviceType::CUDA) {
      if (tensor.scalar_type() != ScalarType::Half) {
        dataset->write(
            tensor_cpu.storage().data_ptr().get(),
            ScalarTypeToPredType(tensor.scalar_type()));
      } else {
        dataset->write(
            tensor_cpu.to(c10::kFloat).storage().data_ptr().get(),
            PredType::IEEE_F32LE);
      }
    } else if (tensor.device().type() == DeviceType::NPU) {
      if (tensor.scalar_type() != ScalarType::Half) {
        dataset->write(
            tensor_cpu.storage().data_ptr().get(),
            ScalarTypeToPredType(tensor.scalar_type()));
      } else {
        dataset->write(
            tensor_cpu.to(c10::kFloat).storage().data_ptr().get(),
            PredType::IEEE_F32LE);
      }
    }

    delete dataset;
    return true;
  }

  void DumpUtil::DumpOneInput(const string& irName, int seqId, const ArgDes<at::Tensor>& input) {
    string h5Path = GetValHdf5Path(irName, seqId, true, input.Name());
    DumpUtil::GetInstance()->DumpTensor(h5Path, input.GetValue());
  }

  void DumpUtil::DumpOneInput(const string& irName, int seqId, const ArgDes<vector<at::Tensor>>& input) {
    int i = 0;
    for (auto& tensor : input.GetValue()) {
      string h5Path = GetValHdf5Path(irName, seqId, true, input.Name(), i);
      DumpUtil::GetInstance()->DumpTensor(h5Path, tensor);
      i++;
    }
  }

  void DumpUtil::DumpOneInput(const string& irName, int seqId, const ArgDes<vector<int64_t>>& input) {
    string h5Path = GetValHdf5Path(irName, seqId, true, input.Name());
    WriteValToHdf5Dataset(
        h5Path,
        1,
        input.GetValue().size(),
        SaveType::VEC_I64,
        input.GetValue().data(),
        PredType::STD_I32LE,
        PredType::STD_I64LE);
  }

  void DumpUtil::DumpOneInput(const string& irName, int seqId, const ArgDes<int64_t>& input) {
    string h5Path = GetValHdf5Path(irName, seqId, true, input.Name());
    WriteValToHdf5Dataset(
        h5Path,
        0,
        1,
        SaveType::I64,
        &input.GetValue(),
        PredType::STD_I32LE,
        PredType::STD_I64LE);
  }

  void DumpUtil::DumpOneInput(const string& irName, int seqId, const ArgDes<bool>& input) {
    string h5Path = GetValHdf5Path(irName, seqId, true, input.Name());
    int8_t data = input.GetValue();
    WriteValToHdf5Dataset(
        h5Path,
        0,
        1,
        SaveType::BOOL,
        &data,
        PredType::STD_I32LE,
        PredType::STD_I8LE);
  }

  void DumpUtil::DumpOneInput(const string& irName, int seqId, const ArgDes<double>& input) {
    string h5Path = GetValHdf5Path(irName, seqId, true, input.Name());
    double data = input.GetValue();
    WriteValToHdf5Dataset(
        h5Path,
        0,
        1,
        SaveType::DOUBLE,
        &data,
        PredType::STD_I32LE,
        PredType::IEEE_F64LE);
  }

  void DumpUtil::DumpOneInput(const string& irName, int seqId, const ArgDes<c10::optional<double>>& input) {
    if (!input.GetValue()) {
      return;
    }
    string h5Path = GetValHdf5Path(irName, seqId, true, input.Name());
    double data = input.GetValue().value();
    WriteValToHdf5Dataset(
        h5Path,
        0,
        1,
        SaveType::OPT_DOUBLE,
        &data,
        PredType::STD_I32LE,
        PredType::IEEE_F64LE);
  }

  void DumpUtil::DumpOneInput(const string& irName, int seqId, const ArgDes<at::Scalar>& input) {
    string h5Path = GetValHdf5Path(irName, seqId, true, input.Name());

    void* dataAddr = nullptr;
    double doubleData = 0;
    int64_t longData = 0;
    int8_t boolData = 0;
    if (input.GetValue().type() == ScalarType::Double) {
      doubleData = input.GetValue().toDouble();
      dataAddr = &doubleData;
    } else if (input.GetValue().type() == ScalarType::Long) {
      longData = input.GetValue().toLong();
      dataAddr = &longData;
    } else if (input.GetValue().type() == ScalarType::Bool) {
      boolData = input.GetValue().toBool();
      dataAddr = &boolData;
    }
    WriteValToHdf5Dataset(
        h5Path,
        0,
        1,
        SaveType::SCALAR,
        dataAddr,
        PredType::STD_I32LE,
        ScalarTypeToPredType(input.GetValue().type()));
  }

  void DumpUtil::DumpOneInput(const string& irName, int seqId, 
      const ArgDes<torch::autograd::generated::TypeAndSize>& input) {
  }

  void DumpUtil::DumpOneInput(const string& irName, int seqId, const ArgDes<c10::optional<int64_t>>& input) {
    if (!input.GetValue()) {
      return;
    }
    string h5Path = GetValHdf5Path(irName, seqId, true, input.Name());
    int64_t data = input.GetValue().value();
    WriteValToHdf5Dataset(
        h5Path,
        0,
        1,
        SaveType::OPT_INT64,
        &data,
        PredType::STD_I32LE,
        PredType::STD_I64LE);
  }

  void DumpUtil::DumpOneInput(const string& irName, int seqId, const ArgDes<c10::optional<at::Scalar>>& input) {
    if (!input.GetValue()) {
      return;
    }
    string h5Path = GetValHdf5Path(irName, seqId, true, input.Name());
    void* dataAddr = nullptr;
    double doubleData = 0;
    int64_t longData = 0;
    int8_t boolData = 0;
    if (input.GetValue().value().type() == ScalarType::Double) {
      doubleData = input.GetValue().value().toDouble();
      dataAddr = &doubleData;
    } else if (input.GetValue().value().type() == ScalarType::Long) {
      longData = input.GetValue().value().toLong();
      dataAddr = &longData;
    } else if (input.GetValue().value().type() == ScalarType::Bool) {
      boolData = input.GetValue().value().toBool();
      dataAddr = &boolData;
    }
    WriteValToHdf5Dataset(
        h5Path,
        0,
        1,
        SaveType::OPT_SCALAR,
        dataAddr,
        PredType::STD_I32LE,
        ScalarTypeToPredType(input.GetValue().value().type()));
  }

  void DumpUtil::DumpOneInput(const string& irName, int seqId, const ArgDes<vector<vector<int64_t>>>& input) {
  }

  void DumpUtil::DumpOneInput(const string& irName, int seqId, const ArgDes<at::TensorGeometry>& input) {
  }

  void DumpUtil::DumpOneInput(const string& irName, int seqId, const ArgDes<size_t>& input) {
    string h5Path = GetValHdf5Path(irName, seqId, true, input.Name());
    WriteValToHdf5Dataset(
        h5Path,
        0,
        1,
        SaveType::SIZE,
        &input.GetValue(),
        PredType::STD_I32LE,
        PredType::STD_I64LE);
  }

  void DumpUtil::DumpOneInput(const string& irName, int seqId, const ArgDes<c10::ScalarType>& input) {
    string h5Path = GetValHdf5Path(irName, seqId, true, input.Name());
    int data = ScalarTypeToDumpType(input.GetValue());
    WriteValToHdf5Dataset(
        h5Path,
        0,
        1,
        SaveType::ScalarType,
        &data,
        PredType::STD_I32LE,
        PredType::STD_I32LE);
  }

  void DumpUtil::DumpOneInput(const string& irName, int seqId, const ArgDes<std::pair<size_t, size_t>>& input) {
    string h5Path = GetValHdf5Path(irName, seqId, true, input.Name());
    int64_t data[2];
    data[0]= input.GetValue().first;
    data[1]= input.GetValue().second;
    WriteValToHdf5Dataset(
        h5Path,
        1,
        2,
        SaveType::PAIR,
        data,
        PredType::STD_I32LE,
        PredType::STD_I64LE);
  }

  void DumpUtil::DumpOneInput(const string& irName, int seqId, const ArgDes<c10::ArrayRef<at::Tensor>>& input) {
    int i = 0;
    for (auto& tensor : input.GetValue()) {
      string h5Path = GetValHdf5Path(irName, seqId, true, input.Name(), i);
      DumpUtil::GetInstance()->DumpTensor(h5Path, tensor);
      i++;
    }
  }

  void DumpUtil::DumpOneInput(const string& irName, int seqId, const ArgDes<c10::ArrayRef<long int>>& input) {
    string h5Path = GetValHdf5Path(irName, seqId, true, input.Name());
    WriteValToHdf5Dataset(
        h5Path,
        1,
        input.GetValue().size(),
        SaveType::AR_LONG_INT,
        input.GetValue().data(),
        PredType::STD_I32LE,
        PredType::STD_I64LE);
  }

  void DumpUtil::DumpOneInput(const string& irName, int seqId, const ArgDes<c10::Storage>& input) {
  }

  void DumpUtil::DumpOneInput(const string& irName, int seqId, const ArgDes<at::Generator*>& input) {
  }

  void DumpUtil::DumpOneInput(const string& irName, int seqId, const ArgDes<c10::ArrayRef<at::Dimname>>& input) {
  }

  void DumpUtil::DumpOneInput(const string& irName, int seqId, const ArgDes<at::Dimname>& input) {
  }

  void DumpUtil::DumpOneInput(const string& irName, int seqId, const ArgDes<c10::TensorOptions>& input) {
    string h5Path = GetValHdf5Path(irName, seqId, true, input.Name());
    // create dataset
    DataSpace dataspace = DataSpace();
    DataSet* dataset = nullptr;
    dataset = new DataSet(file->createDataSet(h5Path, PredType::IEEE_F32LE, dataspace));

    // prepare device type attribute
    int deviceType = static_cast<int16_t>(input.GetValue().device().type());
    PrepareSimpleHdf5Attr(dataset, ATTR_DEVICE_TYPE_NAME, PredType::STD_I32LE, &deviceType);

    // prepare dtype attribute
    int dumpType = ScalarTypeToDumpType(at::typeMetaToScalarType(input.GetValue().dtype()));
    if (dumpType == -1) {
      // the dtype can not be recognized
      delete dataset;
      return;
    }
    PrepareSimpleHdf5Attr(dataset, ATTR_DATA_TYPE_NAME, PredType::STD_I32LE, &dumpType);

    // preare requires grad attribute
    int32_t requiresGrad = static_cast<int32_t>(input.GetValue().requires_grad());
    PrepareSimpleHdf5Attr(dataset, ATTR_REQUIRES_GRAD_NAME, PredType::STD_I32LE, &requiresGrad);

    // ommit layout ,for only support kStrided, but not kSparse

    // prepare type attribute
    int typeData = static_cast<int>(SaveType::TENSOR_OPTS);
    PrepareSimpleHdf5Attr(dataset, ATTR_TYPE_NAME, PredType::STD_I32LE, &typeData);

    delete dataset;
    return;
  }

  void DumpUtil::DumpOneInput(const string& irName, int seqId, const ArgDes<c10::optional<c10::MemoryFormat>>& input) {
    // check the optional value
    if (!input.GetValue()) {
      return;
    }
    string h5Path = GetValHdf5Path(irName, seqId, true, input.Name());
    int32_t data = static_cast<int32_t>(input.GetValue().value());
    WriteValToHdf5Dataset(
        h5Path,
        0,
        1,
        SaveType::OPT_MEM_FMT,
        &data,
        PredType::STD_I32LE,
        PredType::STD_I32LE);
  }

  void DumpUtil::DumpOneInput(const string& irName, int seqId, const ArgDes<c10::MemoryFormat>& input) {
  }

  void DumpUtil::DumpOneInput(const string& irName, int seqId, const ArgDes<c10::optional<c10::ScalarType>>& input) {
    // check the optional value
    if (!input.GetValue()) {
      return;
    }
    string h5Path = GetValHdf5Path(irName, seqId, true, input.Name());
    int data = ScalarTypeToDumpType(input.GetValue().value());
    WriteValToHdf5Dataset(
        h5Path,
        0,
        1,
        SaveType::ScalarType,
        &data,
        PredType::STD_I32LE,
        PredType::STD_I32LE);
  }

  void DumpUtil::DumpOneInput(const string& irName, int seqId, 
      const ArgDes<c10::optional<c10::ArrayRef<at::Dimname>>>& input) {
  }

  void DumpUtil::DumpOneInput(const string& irName, int seqId, const ArgDes<c10::Device>& input) {
  }

  void DumpUtil::DumpOneInput(const string& irName, int seqId, const ArgDes<c10::optional<bool>>& input) {
  }

  void DumpUtil::DumpOneInput(const string& irName, int seqId, const ArgDes<std::array<bool, 2>>& input) {
  }

  void DumpUtil::DumpOneInput(const string& irName, int seqId, const ArgDes<std::array<bool, 3>>& input) {
  }

  void DumpUtil::DumpOneInput(const string& irName, int seqId, const ArgDes<std::array<bool, 4>>& input) {
  }

  void DumpUtil::DumpOneInput(const string& irName, int seqId, const ArgDes<string>& input) {
  }

  void DumpUtil::DumpOneInput(const string& irName, int seqId, const ArgDes<ConstQuantizerPtr>& input) {
  }

  void DumpUtil::DumpOneOutput(const string& irName, int seqId, const ArgDes<vector<at::Tensor>>& output) {
    int i = 0;
    for (auto& tensor : output.GetValue()) {
      string h5Path = GetValHdf5Path(irName, seqId, false, output.Name(), i);
      DumpUtil::GetInstance()->DumpTensor(h5Path, tensor);
      i++;
    }
  }

  void DumpUtil::DumpOneOutput(const string& irName, int seqId, const ArgDes<at::Tensor>& output) {
    string h5Path = GetValHdf5Path(irName, seqId, false, output.Name());
    DumpUtil::GetInstance()->DumpTensor(h5Path, output.GetValue());
  }
  
  void DumpUtil::DumpOneOutput(const string& irName, int seqId, const ArgDes<double>& output) {
  }
  
  void DumpUtil::DumpOneOutput(const string& irName, int seqId, const ArgDes<int64_t>& output) {
    string h5Path = GetValHdf5Path(irName, seqId, false, output.Name());
    WriteValToHdf5Dataset(
        h5Path,
        0,
        1,
        SaveType::I64,
        &output.GetValue(),
        PredType::STD_I32LE,
        PredType::STD_I64LE);
  }
  
  void DumpUtil::DumpOneOutput(const string& irName, int seqId, const ArgDes<bool>& output) {
  }
  
  void DumpUtil::DumpOneOutput(const string& irName, int seqId, const ArgDes<c10::Scalar>& output) {

    string h5Path = GetValHdf5Path(irName, seqId, false, output.Name());

    void* dataAddr = NULL;
    double doubleData = 0;
    int64_t longData = 0;
    int8_t boolData = 0;
    if (output.GetValue().type() == ScalarType::Double) {
      doubleData = output.GetValue().toDouble();
      dataAddr = &doubleData;
    } else if (output.GetValue().type() == ScalarType::Long) {
      longData = output.GetValue().toLong();
      dataAddr = &longData;
    } else if (output.GetValue().type() == ScalarType::Bool) {
      boolData = output.GetValue().toBool();
      dataAddr = &boolData;
    }
    WriteValToHdf5Dataset(
        h5Path,
        0,
        1,
        SaveType::SCALAR,
        dataAddr,
        PredType::STD_I32LE,
        ScalarTypeToPredType(output.GetValue().type()));
  }
  
  void DumpUtil::DumpOneOutput(const string& irName, int seqId, const ArgDes<c10::QScheme>& output) {
  }
  
  void DumpUtil::DumpOneOutput(const string& irName, int seqId, const ArgDes<c10::ScalarType>& output) {
  }

  void DumpUtil::StartAclDump() {
  #ifdef USE_NPU
    auto stream = c10::npu::getCurrentNPUStream();
    C10_NPU_CHECK(aclrtSynchronizeStream(stream));

    C10_NPU_CHECK(aclmdlInitDump());
    const char *aclConfigPath = "acl.json";
    C10_NPU_CHECK(aclmdlSetDump(aclConfigPath));

    C10_NPU_CHECK(aclrtSynchronizeStream(stream));
  #endif
  }

  void DumpUtil::FinalizeAclDump() {
  #ifdef USE_NPU
    auto stream = c10::npu::getCurrentNPUStream();
    C10_NPU_CHECK(aclrtSynchronizeStream(stream));

    C10_NPU_CHECK(aclmdlFinalizeDump());

    C10_NPU_CHECK(aclrtSynchronizeStream(stream));
  #endif
  }
}
