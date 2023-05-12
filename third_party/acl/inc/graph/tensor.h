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

  size_t GetDimNum() const;
  // If the idx is invalid, return 0
  int64_t GetDim(size_t idx) const;
  graphStatus SetDim(size_t idx, int64_t value);
  std::vector<int64_t> GetDims() const;
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
  void SetOriginShape(const Shape &originShape);

  Format GetOriginFormat() const;
  void SetOriginFormat(Format originFormat);

  DataType GetDataType() const;
  void SetDataType(DataType dt);

  ATTRIBUTED_DEPRECATED(graphStatus GetName(AscendString &))
  std::string GetName() const;
  graphStatus GetName(AscendString &name);

  ATTRIBUTED_DEPRECATED(void SetName(const char *))
  void SetName(const std::string &name);
  void SetName(const char *name);

  // Attr acess
  void SetSize(int64_t size);
  int64_t GetSize() const;

  int64_t GetRealDimCnt() const;
  void SetRealDimCnt(const int64_t realDimCnt);

  void SetPlacement(Placement placement);
  Placement GetPlacement() const;

  void SetConstData(const std::shared_ptr<void> const_data_buffer, const size_t &const_data_len);
  bool GetConstData(std::shared_ptr<void>& const_dat_buffer, size_t &const_data_len) const;

 private:
  std::shared_ptr<TensorDescImpl> impl;
};

class TensorImpl;
class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Tensor {
 public:
  using DeleteFunc = std::function<void(uint8_t *)>;
  Tensor();
  ~Tensor() = default;
  explicit Tensor(const TensorDesc &tensorDesc);
  Tensor(const TensorDesc &tensorDesc, const std::vector<uint8_t> &data);
  Tensor(const TensorDesc &tensorDesc, const uint8_t *data, size_t size);
  Tensor(TensorDesc &&tensorDesc, std::vector<uint8_t> &&data);

  TensorDesc GetTensorDesc() const;
  graphStatus SetTensorDesc(const TensorDesc &tensorDesc);

  const uint8_t *GetData() const;
  uint8_t *GetData();
  size_t GetSize() const;
  std::unique_ptr<uint8_t[], Tensor::DeleteFunc> ResetData();

  graphStatus SetData(std::vector<uint8_t> &&data);
  graphStatus SetData(const std::vector<uint8_t> &data);
  graphStatus SetData(const uint8_t *data, size_t size);
  ATTRIBUTED_DEPRECATED(graphStatus SetData(const char *data))
  graphStatus SetData(const std::string &data);
  graphStatus SetData(const char *data);
  ATTRIBUTED_DEPRECATED(graphStatus SetData(const std::vector<AscendString> &))
  graphStatus SetData(const std::vector<std::string> &data);
  graphStatus SetData(const std::vector<AscendString> &datas);
  graphStatus SetData(uint8_t *data, size_t size, const Tensor::DeleteFunc &deleter_func);
  graphStatus IsValid();

  Tensor Clone() const;

 private:
  std::shared_ptr<TensorImpl> impl;
  friend class TensorAdapter;
};
}  // namespace ge

#endif  // INC_EXTERNAL_GRAPH_TENSOR_H_
