/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_EXTERNAL_GRAPH_TENSOR_H_
#define INC_EXTERNAL_GRAPH_TENSOR_H_

#include <atomic>
#include <memory>
#include <string>
#include <vector>
#include <utility>

#include "./ge_error_codes.h"
#include "./types.h"
#include "ascend_string.h"

namespace ge {
class ShapeImpl;
class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Shape {
 public:
  Shape();
  ~Shape() = default;
  explicit Shape(const std::vector<int64_t> &dims);

  /**
   * `GetDimNum()`标识有效的dim的个数，跟`GetDims().size()`不等价，调用方按需选择
   * 比如如果dim是[-2], 维度未可知时：
   * GetDimNum()会返回0；
   * 而GetDims().size()会返回dim的个数，即1；
   * 另外如果需要判断是否是标量，推荐使用接口`GetDims.size() == 0U`来判断
   * @return
   */
  size_t GetDimNum() const;
  // If the idx is invalid, return 0
  int64_t GetDim(size_t idx) const;
  graphStatus SetDim(size_t idx, int64_t value);
  /**
   * `GetDims`标识dim的个数，跟`GetDimNum()`不等价，调用方按需选择
   * 比如如果dim是[-2], 维度未可知时：
   * GetDimNum()会返回0；
   * 而GetDims().size()会返回dim的个数，即1；
   * 另外如果需要判断是否是标量，推荐使用接口`GetDims.size() == 0U`来判断
   * @return
   */
  std::vector<int64_t> GetDims() const;
  /**
   * 获取shape的各个维度的dim值的乘积
   * @return
   * 如果dim值包含-1或者-2，那么size直接返回-1, 含义是unknown shape
   * 如果dim值包含0，那么size直接返回0，含义是空tensor
   * 如果dim值的个数为0，那么size直接返回0，含义是标量
   * 如果dim值的乘积产生了int64的溢出，那么size直接返回0，含义是乘积溢出
   */
  int64_t GetShapeSize() const;

 private:
  std::shared_ptr<ShapeImpl> impl_;
};

class TensorDescImpl;
class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TensorDesc {
 public:
  TensorDesc();
  ~TensorDesc() = default;
  explicit TensorDesc(Shape shape, Format format = FORMAT_ND, DataType dt = DT_FLOAT);
  // Copy
  TensorDesc(const TensorDesc &desc);
  // Move
  TensorDesc(TensorDesc &&desc);
  // Copy
  TensorDesc &operator=(const TensorDesc &desc);
  // Move
  TensorDesc &operator=(TensorDesc &&desc);

  void Update(const Shape &shape, Format format = FORMAT_ND, DataType dt = DT_FLOAT);
  Shape GetShape() const;
  void SetShape(const Shape &shape);
  // set shape with -2, it stand for unknown shape
  graphStatus SetUnknownDimNumShape();
  // for unknown shape
  graphStatus SetShapeRange(const std::vector<std::pair<int64_t, int64_t>> &range);
  graphStatus GetShapeRange(std::vector<std::pair<int64_t, int64_t>> &range) const;

  Format GetFormat() const;
  void SetFormat(Format format);

  Shape GetOriginShape() const;
  void SetOriginShape(const Shape &origin_shape);

  Format GetOriginFormat() const;
  void SetOriginFormat(Format origin_format);

  DataType GetDataType() const;
  void SetDataType(DataType dt);

  ATTRIBUTED_DEPRECATED(graphStatus GetName(AscendString &))
  std::string GetName() const;
  graphStatus GetName(AscendString &name);
  graphStatus GetName(AscendString &name) const;

  ATTRIBUTED_DEPRECATED(void SetName(const char_t *))
  void SetName(const std::string &name);
  void SetName(const char_t *name);

  // Attr acess
  void SetSize(int64_t size);
  int64_t GetSize() const;

  int64_t GetRealDimCnt() const;
  void SetRealDimCnt(const int64_t real_dim_cnt);

  void SetPlacement(Placement placement);
  Placement GetPlacement() const;

  void SetConstData(std::unique_ptr<uint8_t[]> const_data_buffer, const size_t &const_data_len);
  bool GetConstData(uint8_t **const_data_buffer, size_t &const_data_len) const;
 /*
  * 补维类似于ExpandDims算子，在原有shape的基础上，添加一到多个维度，例如原shape[2,2]有两根轴，那么在两根轴中间补两维后的shape为[2,1,1,2]。
  * 补维后shape的第0、3根轴被称为原始轴，第1、2根轴被称为补维轴。
  *
  * 通过1和0描述补维规则，1代表当前轴为补维轴，0代表当前轴为原始轴，从左到右依次代表当前shape每根轴的来源，例如：
  * | 补维规则   | 补维前shape | 补维后shape                                                    |
  * | -------- | ----------- | ------------------------------------------------------------ |
  * | 0110     | [2, 2]      | [2, 1, 1, 2]                                                 |
  * | 100      | [2, 3]      | [1, 2, 3]                                                    |
  * | 1000     | [2, 3]      | 补维规则与补维前shape不匹配，规则指定原始轴有3根，但原始shape只有2根轴，补维报错。 |
  *
  */
  void SetExpandDimsRule(const AscendString &expand_dims_rule);
  graphStatus GetExpandDimsRule(AscendString &expand_dims_rule) const;

  void SetReuseInputIndex(const uint32_t idx);

 private:
  std::shared_ptr<TensorDescImpl> impl;
  friend class TensorAdapter;
};

class TensorImpl;
class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Tensor {
public:
    using DeleteFunc = std::function<void(uint8_t *)>;
    Tensor();
    ~Tensor() = default;
    explicit Tensor(const TensorDesc &tensor_desc);
    Tensor(const TensorDesc &tensor_desc, const std::vector<uint8_t> &data);
    Tensor(const TensorDesc &tensor_desc, const uint8_t *data, size_t size);
    Tensor(TensorDesc &&tensor_desc, std::vector<uint8_t> &&data);

    TensorDesc GetTensorDesc() const;
    graphStatus SetTensorDesc(const TensorDesc &tensor_desc);

    const uint8_t *GetData() const;
    uint8_t *GetData();
    size_t GetSize() const;
    std::unique_ptr<uint8_t[], Tensor::DeleteFunc> ResetData();

    graphStatus SetData(std::vector<uint8_t> &&data);
    graphStatus SetData(const std::vector<uint8_t> &data);
    graphStatus SetData(const uint8_t *data, size_t size);
    ATTRIBUTED_DEPRECATED(graphStatus SetData(const char_t *data))
    graphStatus SetData(const std::string &data);
    graphStatus SetData(const char_t *data);
    ATTRIBUTED_DEPRECATED(graphStatus SetData(const std::vector<AscendString> &))
    graphStatus SetData(const std::vector<std::string> &data);
    graphStatus SetData(const std::vector<AscendString> &datas);
    graphStatus SetData(uint8_t *data, size_t size, const Tensor::DeleteFunc &deleter_func);
    graphStatus IsValid();

    graphStatus SetOriginShapeDimNum(const size_t dim_num);
    size_t GetOriginShapeDimNum() const;

    graphStatus SetOriginShapeDim(const size_t idx, const int64_t dim_value);
    int64_t GetOriginShapeDim(const size_t idx) const;

    graphStatus SetOriginFormat(const ge::Format &format);
    ge::Format GetOriginFormat() const;

    graphStatus SetShapeDimNum(const size_t dim_num);
    size_t GetShapeDimNum() const;

    graphStatus SetShapeDim(const size_t idx, const int64_t dim_value);
    int64_t GetShapeDim(const size_t idx) const;

    graphStatus SetFormat(const ge::Format &format);
    ge::Format GetFormat() const;

    graphStatus SetDataType(const ge::DataType &dtype);
    ge::DataType GetDataType() const;

    graphStatus SetPlacement(const ge::Placement &placement);
    ge::Placement GetPlacement() const;

    /*
    * 补维类似于ExpandDims算子，在原有shape的基础上，添加一到多个维度，例如原shape[2,2]有两根轴，那么在两根轴中间补两维后的shape为[2,1,1,2]。
    * 补维后shape的第0、3根轴被称为原始轴，第1、2根轴被称为补维轴。
    *
    * 通过1和0描述补维规则，1代表当前轴为补维轴，0代表当前轴为原始轴，从左到右依次代表当前shape每根轴的来源，例如：
    * | 补维规则   | 补维前shape | 补维后shape                                                    |
    * | -------- | ----------- | ------------------------------------------------------------ |
    * | 0110     | [2, 2]      | [2, 1, 1, 2]                                                 |
    * | 100      | [2, 3]      | [1, 2, 3]                                                    |
    * | 1000     | [2, 3]      | 补维规则与补维前shape不匹配，规则指定原始轴有3根，但原始shape只有2根轴，补维报错。 |
    *
    */
    graphStatus SetExpandDimsRule(const AscendString &expand_dims_rule);
    graphStatus GetExpandDimsRule(AscendString &expand_dims_rule) const;

    // 高性能接口，与SetData接口的区别是避免重复make_shared,此时需要用户保证该tensor的内存只被当前tensor使用，具有独占所有权
    graphStatus ResetData(uint8_t *data, size_t size, const Tensor::DeleteFunc &deleter_func);

    Tensor Clone() const;

private:
    std::shared_ptr<TensorImpl> impl;
    friend class TensorAdapter;
};
}  // namespace ge

#endif  // INC_EXTERNAL_GRAPH_TENSOR_H_
