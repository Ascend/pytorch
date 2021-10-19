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
#include <ATen/utils/LoadUtils.h>

using namespace std;
using namespace H5;

namespace at {

  void SetLoadPath(string path) {
    LoadUtil::GetInstance()->SetLoadFilePath(path);
    LoadUtil::GetInstance()->LoadLazyInit();
    return;
  }

  void SetLoadWithAclDumpFlag(bool flag) {
    LoadUtil::GetInstance()->SetLoadWithAclDumpFlag(flag);
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

  using stringmap = std::unordered_map<string, string>;
  stringmap IrNameMapper = {
    {"NpuConvolutionBackward", "CudnnConvolutionBackward"},
    {"CudnnConvolutionBackward", "NpuConvolutionBackward"},
    {"NativeBatchNormBackward", "CudnnBatchNormBackward"},
    {"CudnnBatchNormBackward", "NativeBatchNormBackward"},
  };
  std::unordered_map<string, stringmap> IrParamNameMapper = {
    {"NpuConvolutionBackward", {{"input", "self"},}},
    {"CudnnConvolutionBackward", {{"self", "input"},}},
    {"NativeBatchNormBackward", {{"eps", "epsilon"},}},
    {"CudnnBatchNormBackward", {{"epsilon", "eps"},}},
  };

  // These params not exist in npu op, but exist in gpu op
  // when matching or copying, these params are ignored;
  // To do: are these params neccessary to recover the op?
  std::unordered_map<string, std::vector<string>> IgnoreParamMapper = {
    {"NpuConvolutionBackward", {"benchmark", "deterministic"}},
    {"NativeBatchNormBackward", {"result3",}},
  };

  std::unordered_map<string, string> GetIrMapper() {
    return IrNameMapper;
  };

  std::unordered_map<string, stringmap> GetParamMapper() {
    return IrParamNameMapper;
  };

  void MaybeMapTensorName(const string& irName, std::vector<TensorDesc>& tensorDescVec) {
    for (auto it = tensorDescVec.begin(); it != tensorDescVec.end(); it++) {
      auto tensorName = (*it).nameTensor;
      if (IrParamNameMapper[irName].find(tensorName) != IrParamNameMapper[irName].end()) {
        (*it).nameTensor = IrParamNameMapper[irName][tensorName];
      }
    }
  }

  template <typename T>
  void MaybeMapValueName(const string& irName, T& value) {
    for (auto it = value.begin(); it != value.end(); it++) {
      auto valueName = (*it).Name();
      if (IrParamNameMapper[irName].find(valueName) != IrParamNameMapper[irName].end()) {
        (*it).SetName(IrParamNameMapper[irName][valueName]);
      }
    }
  }

  template <typename T>
  void MaybeMapScalarName(const string& irName, T& value) {
    for (auto it = value.begin(); it != value.end(); it++) {
      auto valueName = (*it)->Name();
      if (IrParamNameMapper[irName].find(valueName) != IrParamNameMapper[irName].end()) {
        (*it)->SetName(IrParamNameMapper[irName][valueName]);
      }
    }
  }

  void MaybeMapName(CommDesc& commDesc, const H5File* file) {
    std::string h5IRPath = "/" + commDesc.nameIr;
    if (file->nameExists(h5IRPath)) {
      return ;
    }
    if (IrNameMapper.find(commDesc.nameIr) != IrNameMapper.end()) {
      auto oriNameIr = commDesc.nameIr;
      commDesc.nameIr = IrNameMapper[commDesc.nameIr];
      MaybeMapTensorName(oriNameIr, commDesc.tensorDescVec);
      MaybeMapValueName(oriNameIr, commDesc.int64VecDescVec);
      MaybeMapValueName(oriNameIr, commDesc.int64DescVec);
      MaybeMapValueName(oriNameIr, commDesc.boolDescVec);
      MaybeMapValueName(oriNameIr, commDesc.doubleDescVec);
      MaybeMapValueName(oriNameIr, commDesc.optionalDoubleDescVec);
      MaybeMapScalarName(oriNameIr, commDesc.scalarDescVec);
      MaybeMapValueName(oriNameIr, commDesc.optionalInt64DescVec);
      MaybeMapScalarName(oriNameIr, commDesc.optionalScalarDescVec);
      MaybeMapValueName(oriNameIr, commDesc.scalarTypeDescVec);
      MaybeMapValueName(oriNameIr, commDesc.sizePairDescVec);
      MaybeMapValueName(oriNameIr, commDesc.longIntArrayDescVec);
    }
  }

  const H5std_string ATTR_DEVICE_TYPE_NAME("DeviceType");
  const H5std_string ATTR_DATA_TYPE_NAME("DataType");

  LoadUtil::LoadUtil() {

  }

  void LoadUtil::LoadLazyInit() {
    if (!loadInit) {
      file = new H5File(loadFilePath, H5F_ACC_RDONLY);
      // TODO
      // if the file is not correct, stop the program
      loadInit = true;
    }
  }

  LoadUtil::~LoadUtil() {
    if (file != nullptr) {
      delete file;
    }
  }

  bool CheckSizes(const int rank, const hsize_t *h5Shape, const IntArrayRef size) {
    if (rank != size.size()) {
      return false;
    }
    auto sizes = size.vec();
    for (int i = 0; i < rank; i++) {
      if (h5Shape[i] != sizes[i]) {
        return false;
      }
    }
    return true;
  }

  bool CheckSkip(const string &nameIr, const string &nameParam) {
    for (auto it = IgnoreParamMapper[nameIr].begin(); it != IgnoreParamMapper[nameIr].end(); it++) {
      if (nameParam == (*it)) {
        return true;
      }
    }
    return false;
  }

  bool ValueMatching(const string& seqH5, const H5File* file, const string &nameIr, const std::vector<ArgDes<double>>& descVec) {
    bool is_matched = true;
    std::string h5IRPath;
    for (auto it = descVec.begin(); it != descVec.end(); it++) {
      if (CheckSkip(nameIr, (*it).Name())) {
        continue;
      }
      h5IRPath = "/" + nameIr + "/" + seqH5 + "/input/" + (*it).Name();
      if (!file->nameExists(h5IRPath)) {
        is_matched = false;
        break; 
      } else {
        DataSet dataset = file->openDataSet(h5IRPath);
        DataSpace dataSpace = dataset.getSpace();

        //datatype
        if (dataset.getDataType() != PredType::IEEE_F64LE) {
          is_matched = false;
          break;
        }

        //value
        double data = 0;
        dataset.read(&data, dataset.getDataType(), dataSpace, dataSpace);
        if (data != (*it).GetValue()) {
          is_matched = false;
          break; 
        }
      }
    }
    return is_matched;
  }

  bool ValueMatching(const string& seqH5, const H5File* file, const string &nameIr, const std::vector<ArgDes<c10::optional<double>>>& descVec) {
    bool is_matched = true;
    std::string h5IRPath;
    for (auto it = descVec.begin(); it != descVec.end(); it++) {
      if (CheckSkip(nameIr, (*it).Name())) {
        continue;
      }
      h5IRPath = "/" + nameIr + "/" + seqH5 + "/input/" + (*it).Name();
      if (!file->nameExists(h5IRPath)) {
        is_matched = false;
        break; 
      } else {
        DataSet dataset = file->openDataSet(h5IRPath);
        DataSpace dataSpace = dataset.getSpace();

        //datatype
        if (dataset.getDataType() != PredType::IEEE_F64LE) {
          is_matched = false;
          break;
        }

        //value
        double data = 0;
        dataset.read(&data, dataset.getDataType(), dataSpace, dataSpace);
        if (data != (*it).GetValue().value()) {
          is_matched = false;
          break; 
        }
      }
    }
    return is_matched;
  }


  bool ValueMatching(const string& seqH5, const H5File* file, const string &nameIr, const std::vector<ArgDes<std::pair<size_t, size_t>>>& descVec) {
    bool is_matched = true;
    std::string h5IRPath;
    for (auto it = descVec.begin(); it != descVec.end(); it++) {
      if (CheckSkip(nameIr, (*it).Name())) {
        continue;
      }
      h5IRPath = "/" + nameIr + "/" + seqH5 + "/input/" + (*it).Name();
      if (!file->nameExists(h5IRPath)) {
        is_matched = false;
        break; 
      } else {
        DataSet dataset = file->openDataSet(h5IRPath);
        DataSpace dataSpace = dataset.getSpace();

        //datatype
        if (dataset.getDataType() != PredType::STD_I64LE) {
          is_matched = false;
          break;
        }

        //attr
        int typeValue[1];
        Attribute attr = dataset.openAttribute("Type");
        attr.read(attr.getDataType(), &typeValue);
        if (typeValue[0] != static_cast<int>(SaveType::PAIR)) {
          is_matched = false;
          break;
        }

        //value
        int data[2];
        dataset.read(&data, dataset.getDataType(), dataSpace, dataSpace);
        if (data[0] != (*it).GetValue().first || data[1] != (*it).GetValue().second) {
          is_matched = false;
          break; 
        }
      }
    }
    return is_matched;
  }

  bool ValueMatching(const string& seqH5, const H5File* file, const string &nameIr, const std::vector<ArgDes<int64_t>>& descVec) {
    bool is_matched = true;
    std::string h5IRPath;
    for (auto it = descVec.begin(); it != descVec.end(); it++) {
      if (CheckSkip(nameIr, (*it).Name())) {
        continue;
      }
      h5IRPath = "/" + nameIr + "/" + seqH5 + "/input/" + (*it).Name();
      if (nameIr == "ThnnConvDepthwise2DBackward" && (*it).Name() == "groups") {
        continue;
      }
      if (!file->nameExists(h5IRPath)) {
        is_matched = false;
        break; 
      } else {
        DataSet dataset = file->openDataSet(h5IRPath);
        DataSpace dataSpace = dataset.getSpace();

        //datatype
        if (dataset.getDataType() != PredType::STD_I64LE) {
          is_matched = false;
          break;
        }

        //value
        int64_t data = 0;
        dataset.read(&data, dataset.getDataType(), dataSpace, dataSpace);
        if (data != (*it).GetValue()) {
          is_matched = false;
          break; 
        }
      }
    }
    return is_matched;
  }

  bool ValueMatching(const string& seqH5, const H5File* file, const string &nameIr, const std::vector<ArgDes<c10::optional<int64_t>>>& descVec) {
    bool is_matched = true;
    std::string h5IRPath;
    for (auto it = descVec.begin(); it != descVec.end(); it++) {
      if (CheckSkip(nameIr, (*it).Name())) {
        continue;
      }
      h5IRPath = "/" + nameIr + "/" + seqH5 + "/input/" + (*it).Name();
      if (!file->nameExists(h5IRPath)) {
        is_matched = false;
        break; 
      } else {
        DataSet dataset = file->openDataSet(h5IRPath);
        DataSpace dataSpace = dataset.getSpace();

        //datatype
        if (dataset.getDataType() != PredType::STD_I64LE) {
          is_matched = false;
          break;
        }

        //value
        int64_t data = 0;
        dataset.read(&data, dataset.getDataType(), dataSpace, dataSpace);
        if (data != (*it).GetValue().value()) {
          is_matched = false;
          break; 
        }
      }
    }
    return is_matched;
  }

  bool ValueMatching(const string& seqH5, const H5File* file, const string &nameIr, std::vector<TensorDesc>& tensorVec) {
    bool is_matched = true;
    std::string h5IRPath;
    for (auto it = tensorVec.begin(); it != tensorVec.end(); it++) {
      if (CheckSkip(nameIr, (*it).nameTensor)) {
        continue;
      }
      if (!(*it).tensor.has_storage()) {
        continue;
      }
      h5IRPath = "/" + nameIr + "/" + seqH5 + "/input/" + (*it).nameTensor;
      if (nameIr == "ThnnConvDepthwise2DBackward" && (*it).nameTensor == "input") {
        h5IRPath = "/" + nameIr + "/" + seqH5 + "/input/" + "self";
      }
      if (!file->nameExists(h5IRPath)) {
        is_matched = false;
        break;
      } else {
        int dtypeValue = 0;

        DataSet dataset = file->openDataSet(h5IRPath);

        //0.shape
        DataSpace dataSpace = dataset.getSpace();
        int rank = dataSpace.getSimpleExtentNdims();
        hsize_t *h5Shape = new hsize_t[rank];
        int ndims = dataSpace.getSimpleExtentDims(h5Shape, NULL);
        if (!CheckSizes(rank, h5Shape, (*it).tensor.sizes())) {
          is_matched = false;
          delete h5Shape;
          break;
        }
        delete h5Shape;


        //1.dtype
        Attribute attr = dataset.openAttribute(ATTR_DATA_TYPE_NAME);
        attr.read(attr.getDataType(), &dtypeValue);
        // some ops on npu only support int32 while those ops support long on GPU
        // need more tests to verify these cases
        if ((dtypeValue == ScalarTypeToDumpType(c10::kLong) && (*it).tensor.scalar_type() == c10::kInt) || 
            (dtypeValue == ScalarTypeToDumpType(c10::kInt) && (*it).tensor.scalar_type() == c10::kLong)) {
          ;
        } else if (dtypeValue != ScalarTypeToDumpType((*it).tensor.scalar_type())) {
          is_matched = false;
          break;
        }
      } 
    }
    return is_matched;
  }

  bool ValueMatching(const string& seqH5, const H5File* file, const string &nameIr, const std::vector<ArgDes<bool>>& descVec) {
    bool is_matched = true;
    std::string h5IRPath;
    for (auto it = descVec.begin(); it != descVec.end(); it++) {
      if (CheckSkip(nameIr, (*it).Name())) {
        continue;
      }  
      h5IRPath = "/" + nameIr + "/" + seqH5 + "/input/" + (*it).Name();
      if (!file->nameExists(h5IRPath)) {
        is_matched = false;
        break; 
      } else {
        DataSet dataset = file->openDataSet(h5IRPath);
        DataSpace dataSpace = dataset.getSpace();

        //datatype
        if (dataset.getDataType() != PredType::STD_I8LE) {
          is_matched = false;
          break;
        }

        //value
        int8_t data = 0;
        dataset.read(&data, dataset.getDataType(), dataSpace, dataSpace);
        if (data != (*it).GetValue()) {
          is_matched = false;
          break; 
        }
      }
    }
    return is_matched;

  }
  
  bool ValueMatching(const string& seqH5, const H5File* file, const string &nameIr, const std::vector<ArgDes<at::Scalar>*>& descVec) {
    bool is_matched = true;
    std::string h5IRPath;
    for (auto it = descVec.begin(); it != descVec.end(); it++) {
      if (CheckSkip(nameIr, (*(*it)).Name())) {
        continue;
      }  
      h5IRPath = "/" + nameIr + "/" + seqH5 + "/input/" + (*(*it)).Name();
      if (!file->nameExists(h5IRPath)) {
        is_matched = false;
        break; 
      } else {
        DataSet dataset = file->openDataSet(h5IRPath);
        DataSpace dataSpace = dataset.getSpace();

        //datatype
        if (dataset.getDataType() != ScalarTypeToPredType((*(*it)).GetValue().type())) {
          is_matched = false;
          break;
        }
      }
    }
    return is_matched;

  }

  bool ValueMatching(const string& seqH5, const H5File* file, const string &nameIr, const std::vector<ArgDes<c10::optional<at::Scalar>>*>& descVec) {
    bool is_matched = true;
    std::string h5IRPath;
    for (auto it = descVec.begin(); it != descVec.end(); it++) {
      if (CheckSkip(nameIr, (*(*it)).Name())) {
        continue;
      }  
      h5IRPath = "/" + nameIr + "/" + seqH5 + "/input/" + (*(*it)).Name();
      if (!file->nameExists(h5IRPath)) {
        is_matched = false;
        break; 
      } else {
        DataSet dataset = file->openDataSet(h5IRPath);
        DataSpace dataSpace = dataset.getSpace();

        //datatype
        if (dataset.getDataType() != ScalarTypeToPredType((*(*it)).GetValue().value().type())) {
          is_matched = false;
          break;
        }
      }
    }
    return is_matched;

  }

  bool ValueMatching(const string& seqH5, const H5File* file, const string &nameIr, const std::vector<ArgDes<c10::ScalarType>>& descVec) {
    bool is_matched = true;
    std::string h5IRPath;
    for (auto it = descVec.begin(); it != descVec.end(); it++) {
      if (CheckSkip(nameIr, (*it).Name())) {
        continue;
      }  
      h5IRPath = "/" + nameIr + "/" + seqH5 + "/input/" + (*it).Name();
      if (!file->nameExists(h5IRPath)) {
        is_matched = false;
        break; 
      } else {
        DataSet dataset = file->openDataSet(h5IRPath);
        DataSpace dataSpace = dataset.getSpace();

        //datatype
        if (dataset.getDataType() != PredType::STD_I32LE) {
          is_matched = false;
          break;
        }

        //value
        int data = 0;
        dataset.read(&data, dataset.getDataType(), dataSpace, dataSpace);
        if (data != ScalarTypeToDumpType((*it).GetValue())) {
          is_matched = false;
          break; 
        }
      }
    }
    return is_matched;
  }

  bool ValueMatching(const string& seqH5, const H5File* file, const string &nameIr, const std::vector<ArgDes<std::vector<int64_t>>>& descVec) {
    bool is_matched = true;
    std::string h5IRPath;
    for (auto it = descVec.begin(); it != descVec.end(); it++) {
      if (CheckSkip(nameIr, (*it).Name())) {
        continue;
      } 
      h5IRPath = "/" + nameIr + "/" + seqH5 + "/input/" + (*it).Name();
      if (!file->nameExists(h5IRPath)) {
        is_matched = false;
        break; 
      } else {
        DataSet dataset = file->openDataSet(h5IRPath);
        //datatype
        if (dataset.getDataType() != PredType::STD_I64LE) {
          is_matched = false;
          break;
        }

        // shape 
        DataSpace dataSpace = dataset.getSpace();
        int rank = dataSpace.getSimpleExtentNdims();
        hsize_t *h5Shape = new hsize_t[rank];
        int ndims = dataSpace.getSimpleExtentDims(h5Shape, NULL);
        if (h5Shape[0] != (*it).GetValue().size()) {
          is_matched = false;
          delete h5Shape;
          break;
        }        

        //value
        if (h5Shape[0] > 0) {
          int64_t *data = new int64_t[h5Shape[0]];
          dataset.read(data, dataset.getDataType(), dataSpace, dataSpace);
          for (int k = 0; k < h5Shape[0]; k++) {
            if (data[k] != (*it).GetValue()[k]) {
              is_matched = false;
              break;
            }
          }
          delete data;
        }
        delete h5Shape;
      }
    }
    return is_matched;
  }

  bool ValueMatching(const string& seqH5, const H5File* file, const string &nameIr, const std::vector<ArgDes<c10::ArrayRef<long int>>>& descVec) {
    bool is_matched = true;
    std::string h5IRPath;
    for (auto it = descVec.begin(); it != descVec.end(); it++) {
      if (CheckSkip(nameIr, (*it).Name())) {
        continue;
      }
      h5IRPath = "/" + nameIr + "/" + seqH5 + "/input/" + (*it).Name();
      if (!file->nameExists(h5IRPath)) {
        is_matched = false;
        break; 
      } else {
        DataSet dataset = file->openDataSet(h5IRPath);
        //datatype
        if (dataset.getDataType() != PredType::STD_I64LE) {
          is_matched = false;
          break;
        }

        // shape 
        DataSpace dataSpace = dataset.getSpace();
        int rank = dataSpace.getSimpleExtentNdims();
        hsize_t *h5Shape = new hsize_t[rank];
        int ndims = dataSpace.getSimpleExtentDims(h5Shape, NULL);
        if (h5Shape[0] != (*it).GetValue().size()) {
          is_matched = false;
          delete h5Shape;
          break;
        }

        //value
        if (h5Shape[0] > 0) {
          int64_t *data = new int64_t[h5Shape[0]];
          dataset.read(data, dataset.getDataType(), dataSpace, dataSpace);
          for (int k = 0; k < h5Shape[0]; k++) {
            if (data[k] != (*it).GetValue()[k]) {
              is_matched = false;
              break;
            }
          }
          delete data;
        }
        delete h5Shape;
      }
    }
    return is_matched;
  }

  bool ExhaustedMatchingBaseType(const string& seqH5, const H5File* file, const string &nameIr, CommDesc& commDesc) {
    // try to match int64_t, datatype and value
    if (!ValueMatching(seqH5, file, nameIr, commDesc.int64DescVec)) {
      return false;
    }

    // try to match double, datatype and value
    if (!ValueMatching(seqH5, file, nameIr, commDesc.doubleDescVec)) {
      return false;
    }

    // try to match bool, datatype and value
    if (!ValueMatching(seqH5, file, nameIr, commDesc.boolDescVec)) {
      return false;
    }

    // try to match c10::optional<double>, datatype and value
    if (!ValueMatching(seqH5, file, nameIr, commDesc.optionalDoubleDescVec)) {
      return false;
    }  

    // try to match c10::optional<int64_t>, datatype and value
    if (!ValueMatching(seqH5, file, nameIr, commDesc.optionalInt64DescVec)) {
      return false;
    }
    return true;
  }

  bool ExhaustedMatchingVecType(const string& seqH5, const H5File* file, const string &nameIr, CommDesc& commDesc) {
    // try to match vector<int64_t>, shape, datatype and value
    if (!ValueMatching(seqH5, file, nameIr, commDesc.int64VecDescVec)) {
      return false;
    }

    // try to match longIntArrayDescVec, attr and value
    if (!ValueMatching(seqH5, file, nameIr, commDesc.longIntArrayDescVec)) {
      return false;
    }

    // try to match pair<size_t, size_t>, attr and value
    if (!ValueMatching(seqH5, file, nameIr, commDesc.sizePairDescVec)) {
      return false;
    }
    return true;
  }

  bool ExhaustedMatchingTorchType(const string& seqH5, const H5File* file, const string &nameIr, CommDesc& commDesc) {
    // try to match tensor
    if (!ValueMatching(seqH5, file, nameIr, commDesc.tensorDescVec)) {
      return false;
    }

    // try to match scalar, datatype
    if (!ValueMatching(seqH5, file, nameIr, commDesc.scalarDescVec)) {
      return false;
    }

    // try to match c10::optional<scalar>, datatype
    if (!ValueMatching(seqH5, file, nameIr, commDesc.optionalScalarDescVec)) {
      return false;
    }

    // try to match scalarType, datatype and value
    if (!ValueMatching(seqH5, file, nameIr, commDesc.scalarTypeDescVec)) {
      return false;
    }
    return true;
  }   

  int ProcessMatching(const H5File* file, const string &nameIr, CommDesc& commDesc, std::vector<int>& visitedSeq) {
    std::string h5IRPath = "/" + nameIr;
    if (!file->nameExists(h5IRPath)) {
      return -1;
    }

    Group curGroup = file->openGroup(h5IRPath);
    int numCurGroup = curGroup.getNumObjs();
    int i = 0;
    bool is_matched = false;
    std::string seqH5;
    while (i < numCurGroup && (!is_matched)) {
      seqH5 = curGroup.getObjnameByIdx(i);
      if (find(visitedSeq.begin(), visitedSeq.end(), stoi(seqH5)) != visitedSeq.end()) {
        i++;
        continue;
      }
      is_matched = true;

      if (!ExhaustedMatchingTorchType(seqH5, file, nameIr, commDesc)) {
        is_matched = false;
        i++;
        continue;
      }

      if (!ExhaustedMatchingVecType(seqH5, file, nameIr, commDesc)) {
        is_matched = false;
        i++;
        continue;
      }

      if (!ExhaustedMatchingBaseType(seqH5, file, nameIr, commDesc)) {
        is_matched = false;
        i++;
        continue;
      }

      i++;
    }
    if (is_matched) {
      visitedSeq.push_back(stoi(seqH5));
      return stoi(seqH5);
    } else {
      return -1;
    }

  }

  void ZeroStrideClear(Tensor& dst, Tensor& src) {
    auto strides = dst.strides().vec();
    auto position = std::find(strides.begin(), strides.end(), 0);
    if (position != strides.end()) {
      dst = dst.select(position - strides.begin(), 0);
      src = src.select(position - strides.begin(), 0);
    } else {
      return;
    }
    ZeroStrideClear(dst, src);
  }

  // when the stride of some dim is zero, the tensor may has been "expand", copy should only
  // process on any axis of that dim
  // To do: is this kind of copy matches other zero stride cases?
  void CopyMaybeWithZeroStride(Tensor dst, Tensor src) {
    ZeroStrideClear(dst, src);
    dst.copy_(src);
  }

  void TensorCopying(const int &seqH5, const string &nameIr, const H5File* file, CommDesc& commDesc) {
    std::string h5DataSetPath;
    for (auto it = commDesc.tensorDescVec.begin(); it != commDesc.tensorDescVec.end(); it++) {
      if (CheckSkip(nameIr, (*it).nameTensor)) {
        continue;
      } 
      if (!(*it).tensor.has_storage() || tensor.numel() <= 0) {
        continue;
      }
      h5DataSetPath = "/" + nameIr + "/" + to_string(seqH5) + "/input/" + (*it).nameTensor;
      if (nameIr == "ThnnConvDepthwise2DBackward" && (*it).nameTensor == "input") {
        h5DataSetPath = "/" + nameIr + "/" + to_string(seqH5) + "/input/" + "self";
      }
      DataSet dataset = file->openDataSet(h5DataSetPath);
      DataSpace dataSpace = dataset.getSpace();
      int rank = dataSpace.getSimpleExtentNdims();

      hsize_t *dims_out = new hsize_t[rank];
      int ndims = dataSpace.getSimpleExtentDims(dims_out, NULL);
      int64_t numel = 1;
      for (int i = 0;i < rank; i++) {
        numel *= dims_out[i];
      }

      unsigned char *data = new unsigned char[numel * (dataset.getDataType().getSize())];
      dataset.read(data, dataset.getDataType(), dataSpace, dataSpace);
      delete dims_out;

      int deviceTypeValue[1];
      Attribute attr = dataset.openAttribute(ATTR_DEVICE_TYPE_NAME);
      attr.read(attr.getDataType(), &deviceTypeValue);

      Tensor thArray;
      if ((*it).tensor.scalar_type() != ScalarType::Half) {
        TensorOptions options;
        // for long and int, use adata type in h5 instead of that of tensor on the device
        if (dataset.getDataType() == PredType::STD_I64LE) {
          options = at::TensorOptions().dtype(ScalarType::Long);
        } else if (dataset.getDataType() == PredType::STD_I32LE) {
          options = at::TensorOptions().dtype(ScalarType::Int);
        } else {
          options = at::TensorOptions().dtype((*it).tensor.scalar_type());
        }
        thArray = at::from_blob(data, (*it).tensor.sizes(), options);
        auto verCountBefore = (*it).tensor.unsafeGetTensorImpl()->version_counter().current_version();
        CopyMaybeWithZeroStride((*it).tensor.detach(), thArray.to((*it).tensor.device()).to((*it).tensor.dtype()));
        auto verCountAfter = (*it).tensor.unsafeGetTensorImpl()->version_counter().current_version();
        if (verCountAfter > verCountBefore) {
          (*it).tensor.unsafeGetTensorImpl()->reduce_version();
        }
      } else {
        auto options = at::TensorOptions().dtype(at::kFloat);
        thArray = at::from_blob(data, (*it).tensor.sizes(), options);
        auto verCountBefore = (*it).tensor.unsafeGetTensorImpl()->version_counter().current_version();
        CopyMaybeWithZeroStride((*it).tensor.detach(), thArray.to(at::kHalf).to((*it).tensor.device()));
        auto verCountAfter = (*it).tensor.unsafeGetTensorImpl()->version_counter().current_version();
        if (verCountAfter > verCountBefore) {
          (*it).tensor.unsafeGetTensorImpl()->reduce_version();
        }
      }
      delete data;
    }
  }

  void ScalarCopying(const int &seqH5, const string &nameIr, const H5File* file, CommDesc& commDesc) {
    std::string h5DataSetPath;
    for (auto it = commDesc.scalarDescVec.begin(); it != commDesc.scalarDescVec.end(); it++) {
      if (CheckSkip(nameIr, (*(*it)).Name())) {
        continue;
      } 
      h5DataSetPath = "/" + nameIr + "/" + to_string(seqH5) + "/input/" + (*(*it)).Name();
      DataSet dataset = file->openDataSet(h5DataSetPath);
      DataSpace dataSpace = dataset.getSpace();
      auto kType = (*(*it)).GetValue().type();
      if (kType == ScalarType::Double) {
        double dataFp64 = 0;
        dataset.read(&dataFp64, dataset.getDataType(), dataSpace, dataSpace);
        (*(*it)).SetValue(at::Scalar(dataFp64));
      } else if (kType == ScalarType::Long) {
        int64_t dataInt64 = 0;
        dataset.read(&dataInt64, dataset.getDataType(), dataSpace, dataSpace);
        (*(*it)).SetValue(at::Scalar(dataInt64));
      } else if (kType == ScalarType::Bool) {
        bool dataBool = 0;
        dataset.read(&dataBool, dataset.getDataType(), dataSpace, dataSpace);
        (*(*it)).SetValue(at::Scalar(dataBool));
      }
    }
  }

  void OptionalScalarCopying(const int &seqH5, const string &nameIr, const H5File* file, CommDesc& commDesc) {
    std::string h5DataSetPath;
    for (auto it = commDesc.optionalScalarDescVec.begin(); it != commDesc.optionalScalarDescVec.end(); it++) {
      if (CheckSkip(nameIr, (*(*it)).Name())) {
        continue;
      }
      h5DataSetPath = "/" + nameIr + "/" + to_string(seqH5) + "/input/" + (*(*it)).Name();
      DataSet dataset = file->openDataSet(h5DataSetPath);
      DataSpace dataSpace = dataset.getSpace();
      auto kType = (*(*it)).GetValue().value().type();
      if (kType == ScalarType::Double) {
        double dataFp64 = 0;
        dataset.read(&dataFp64, dataset.getDataType(), dataSpace, dataSpace);
        (*(*it)).SetValue(c10::optional<at::Scalar>(dataFp64));
      } else if (kType == ScalarType::Long) {
        int64_t dataInt64 = 0;
        dataset.read(&dataInt64, dataset.getDataType(), dataSpace, dataSpace);
        (*(*it)).SetValue(c10::optional<at::Scalar>(dataInt64));
      } else if (kType == ScalarType::Bool) {
        bool dataBool = true;
        dataset.read(&dataBool, dataset.getDataType(), dataSpace, dataSpace);
        (*(*it)).SetValue(c10::optional<at::Scalar>(dataBool));
      }
    }    
  }

  void ProcessCopying(const int &seqH5, const string &nameIr, const H5File* file, CommDesc& commDesc) {
    // copying tensor
    TensorCopying(seqH5, nameIr, file, commDesc);

    // save scalar back
    ScalarCopying(seqH5, nameIr, file, commDesc);

    // save optional scalar
    OptionalScalarCopying(seqH5, nameIr, file, commDesc);
  }

  void LoadUtil::Process() {
    MaybeMapName(commDesc, file);
    int seqH5 = ProcessMatching(file, commDesc.nameIr, commDesc, visitedSeq);
    if (seqH5 > -1) {
      ProcessCopying(seqH5, commDesc.nameIr, file, commDesc);
    }
    commDesc.tensorDescVec.clear();
    commDesc.int64DescVec.clear();
    commDesc.doubleDescVec.clear();
    commDesc.boolDescVec.clear();
    commDesc.int64VecDescVec.clear();
    commDesc.optionalInt64DescVec.clear();
    commDesc.optionalScalarDescVec.clear();
    commDesc.scalarDescVec.clear();
    commDesc.scalarTypeDescVec.clear();
    commDesc.optionalDoubleDescVec.clear();
    commDesc.sizePairDescVec.clear();
    commDesc.longIntArrayDescVec.clear();
    matchedSeqId = seqH5;
  }
  
  bool LoadUtil::LoadTensor(const at::Tensor &t, string nameIr, bool isList, string nameTensor, bool isLast) {
    commDesc.nameIr = nameIr;
    TensorDesc tensorDesc;
    tensorDesc.tensor = t;
    tensorDesc.isList = isList;
    tensorDesc.nameTensor = nameTensor;
    commDesc.tensorDescVec.push_back(tensorDesc);

    if (isLast) {
      Process();
    }
    return true;
  }

  void LoadUtil::LoadOneInput(std::string &irName, ArgDes<at::Tensor> &t, bool isLast) {
    LoadUtil::GetInstance()->LoadTensor(t.GetValue(), irName, false, t.Name(), isLast);
  }

  void LoadUtil::LoadOneInput(std::string &irName, ArgDes<std::vector<at::Tensor>> &t, bool isLast) {
    int i = 0;
    for (auto &tensor : t.GetValue()) {
      LoadUtil::GetInstance()->LoadTensor(tensor, irName, true, t.Name() + "/" + t.Name() + "_" + to_string(i), isLast ? (i == t.GetValue().size() - 1) : false);
      i++;
    }
  }

  void LoadUtil::LoadOneInput(std::string &irName, ArgDes<std::vector<int64_t>> &t, bool isLast) {
    commDesc.nameIr = irName;
    commDesc.int64VecDescVec.push_back(t);
    if (isLast) {
      Process();
    }
  }

  void LoadUtil::LoadOneInput(std::string &irName, ArgDes<int64_t> &t, bool isLast) {
    commDesc.nameIr = irName;
    commDesc.int64DescVec.push_back(t);
    if (isLast) {
      Process();
    }
  }

  void LoadUtil::LoadOneInput(std::string &irName, ArgDes<bool> &t, bool isLast) {
    commDesc.nameIr = irName;
    commDesc.boolDescVec.push_back(t);
    if (isLast) {
      Process();
    }
  }

  void LoadUtil::LoadOneInput(std::string &irName, ArgDes<double> &t, bool isLast) {
    commDesc.nameIr = irName;
    commDesc.doubleDescVec.push_back(t);
    if (isLast) {
      Process();
    }
  }

  void LoadUtil::LoadOneInput(std::string &irName, ArgDes<c10::optional<double>> &t, bool isLast) {
    commDesc.nameIr = irName;
    if (t.GetValue().has_value()) {
      commDesc.optionalDoubleDescVec.push_back(t);
    }
    if (isLast) {
      Process();
    }
  }

  void LoadUtil::LoadOneInput(std::string &irName, ArgDes<at::Scalar> &t, bool isLast) {
    commDesc.nameIr = irName;
    commDesc.scalarDescVec.push_back(&t);
    if (isLast) {
      Process();
    }
  }

  void LoadUtil::LoadOneInput(std::string &irName, ArgDes<TypeAndSize> &t, bool isLast) {
    ;
  }

  void LoadUtil::LoadOneInput(std::string &irName, ArgDes<c10::optional<int64_t>> &t, bool isLast) {
    commDesc.nameIr = irName;
    if (t.GetValue().has_value()) {
      commDesc.optionalInt64DescVec.push_back(t);
    }
    if (isLast) {
      Process();
    }
  }

  void LoadUtil::LoadOneInput(std::string &irName, ArgDes<c10::optional<at::Scalar>> &t, bool isLast) {
    commDesc.nameIr = irName;
    if (t.GetValue().has_value()) {
      commDesc.optionalScalarDescVec.push_back(&t);
    }
    if (isLast) {
      Process();
    }
  }

  void LoadUtil::LoadOneInput(std::string &irName, ArgDes<std::vector<std::vector<int64_t>>> &t, bool isLast) {
    ;
  }

  void LoadUtil::LoadOneInput(std::string &irName, ArgDes<at::TensorGeometry> &t, bool isLast) {
    ;
  }

  void LoadUtil::LoadOneInput(std::string &irName, ArgDes<c10::ScalarType> &t, bool isLast) {
    commDesc.nameIr = irName;
    commDesc.scalarTypeDescVec.push_back(t);
    if (isLast) {
      Process();
    }
  }

  void LoadUtil::LoadOneInput(std::string &irName, ArgDes<std::pair<size_t, size_t>> &t, bool isLast) {
    commDesc.nameIr = irName;
    commDesc.sizePairDescVec.push_back(t);
    if (isLast) {
      Process();
    }    
  }

  void LoadUtil::LoadOneInput(const string &irName, ArgDes<c10::ArrayRef<at::Tensor>> &input, bool isLast) {
    int i = 0;
    for (auto &tensor : input.GetValue()) {
      LoadUtil::GetInstance()->LoadTensor(tensor, irName, true, input.Name() + "/" + input.Name() + "_" + to_string(i), isLast ? (i == input.GetValue().size() - 1) : false);
      i++;
    }
  }

  void LoadUtil::LoadOneInput(const string &irName, ArgDes<c10::ArrayRef<long int>> &input, bool isLast) {
    commDesc.nameIr = irName;
    commDesc.longIntArrayDescVec.push_back(input);
    if (isLast) {
      Process();
    }
  }

  void LoadUtil::LoadOneInput(const string &irName, ArgDes<c10::Storage> &input, bool isLast) {
    if (isLast) {
      Process();
    }
  }

  void LoadUtil::LoadOneInput(const string &irName, ArgDes<at::Generator *> &input, bool isLast) {
    if (isLast) {
      Process();
    }
  }

  void LoadUtil::LoadOneInput(const string &irName, ArgDes<c10::ArrayRef<at::Dimname>> &input, bool isLast) {
    if (isLast) {
      Process();
    }
  }

  void LoadUtil::LoadOneInput(const string &irName, ArgDes<at::Dimname> &input, bool isLast) {
    if (isLast) {
      Process();
    }
  }

  void LoadUtil::LoadOneInput(const string &irName, ArgDes<c10::TensorOptions> &input, bool isLast) {
    if (isLast) {
      Process();
    }
  }

  void LoadUtil::LoadOneInput(const string &irName, ArgDes<c10::optional<c10::MemoryFormat>> &input, bool isLast) {
    if (isLast) {
      Process();
    }
  }

  void LoadUtil::LoadOneInput(const string &irName, ArgDes<c10::MemoryFormat> &input, bool isLast) {
    if (isLast) {
      Process();
    }
  }

  void LoadUtil::LoadOneInput(const string &irName, ArgDes<c10::optional<c10::ScalarType>> &input, bool isLast) {
    if (isLast) {
      Process();
    }
  }

  void LoadUtil::LoadOneInput(const string &irName, ArgDes<c10::optional<c10::ArrayRef<at::Dimname>>> &input, bool isLast) {
    if (isLast) {
      Process();
    }
  }

  void LoadUtil::LoadOneInput(const string &irName, ArgDes<c10::Device> &input, bool isLast) {
    if (isLast) {
      Process();
    }
  }

  void LoadUtil::LoadOneInput(const string &irName, ArgDes<c10::optional<bool>> &input, bool isLast) {
    if (isLast) {
      Process();
    }
  }

  void LoadUtil::LoadOneInput(const string &irName, ArgDes<std::array<bool, 2>> &input, bool isLast) {
    if (isLast) {
      Process();
    }
  }

  void LoadUtil::LoadOneInput(const string &irName, ArgDes<std::array<bool, 3>> &input, bool isLast) {
    if (isLast) {
      Process();
    }
  }

  void LoadUtil::LoadOneInput(const string &irName, ArgDes<std::array<bool, 4>> &input, bool isLast) {
    if (isLast) {
      Process();
    }
  }

  void LoadUtil::LoadOneInput(const string &irName, ArgDes<string> &input, bool isLast) {
    if (isLast) {
      Process();
    }
  }

  void LoadUtil::LoadOneInput(const string &irName, ArgDes<ConstQuantizerPtr> &input, bool isLast) {
    if (isLast) {
      Process();
    }
  }
}
