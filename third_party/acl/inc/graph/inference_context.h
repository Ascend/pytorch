/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_EXTERNAL_GRAPH_INFERENCE_CONTEXT_H_
#define INC_EXTERNAL_GRAPH_INFERENCE_CONTEXT_H_

#include <memory>
#include <string>
#include <vector>
#include <set>

#include "tensor.h"
#include "types.h"
#include "ascend_string.h"
#include "resource_context.h"
#include "ge_error_codes.h"

namespace ge {
class ShapeAndTypeImpl;
class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ShapeAndType {
public:
    ShapeAndType();
    ~ShapeAndType() = default;

    ShapeAndType(const Shape &shape, DataType data_type);

    void SetShape(const Shape &shape);

    void SetType(DataType data_type);

    Shape GetShape() const;

    DataType GetDataType() const;

private:
    std::shared_ptr<ShapeAndTypeImpl> shape_and_type_impl_;
};

struct InnerInferenceContext;
class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY InferenceContext {
 public:
  ~InferenceContext() = default;
  InferenceContext(const InferenceContext &context) = delete;
  InferenceContext(const InferenceContext &&context) = delete;
  InferenceContext &operator=(const InferenceContext &context) = delete;
  InferenceContext &operator=(const InferenceContext &&context) = delete;

  void SetInputHandleShapesAndTypes(std::vector<std::vector<ShapeAndType>> &&shapes_and_types);
  const std::vector<std::vector<ShapeAndType>> &GetInputHandleShapesAndTypes() const;
  const std::vector<std::vector<ShapeAndType>> &GetOutputHandleShapesAndTypes() const;
  void SetOutputHandleShapesAndTypes(const std::vector<std::vector<ShapeAndType>> &shapes_and_types);
  void SetOutputHandleShapesAndTypes(std::vector<std::vector<ShapeAndType>> &&shapes_and_types);

  ATTRIBUTED_DEPRECATED(void SetMarks(const std::vector<AscendString> &))
  void SetMarks(const std::vector<std::string> &marks);
  void SetMarks(const std::vector<AscendString> &marks);


  ATTRIBUTED_DEPRECATED(void GetMarks(std::vector<AscendString> &) const)
  const std::vector<std::string> &GetMarks() const;
  void GetMarks(std::vector<AscendString> &marks) const;

  static std::unique_ptr<InferenceContext> Create(void *resource_context_mgr = nullptr);
   /**
   * Get corresponding resource_context by key
   * For resource op infershape, invoked by op infer_func.
   * @param key
   * @return corresponding resource context. Check not null before use it.
   */
  ResourceContext *GetResourceContext(const ge::AscendString &key);

  /**
   * Set corresponding resource_context by key. For node which will write to resource.
   * For resource op infershape, invoked by write_op infer_func.
   * @param key
   * @param resource_context pointer.
   * @return status
   */
  graphStatus SetResourceContext(const ge::AscendString &key, ResourceContext *resource_context);
  /**
   * Register resource key relied on. For node which will read from resource.
   * For resource op infershape, invoked by read_op infer_func.
   * @param key
   * @return status
   */
  graphStatus RegisterReliedOnResourceKey(const ge::AscendString &key);

  /**
  * During infershape of write op, if resource shape changed, use this to tell.
  * For resource op infershape, invoked by write_op infer_func.
  * @param key
  * @return status
  */
  graphStatus AddChangedResourceKey(const ge::AscendString &key);

  /**
   * After read_op infershaped, can get resource_keys relied on.
   * For resource op infershape, invoked by ge infershape framework.
   * @param keys
   * @return status
   */
  const std::set<ge::AscendString>& GetReliedOnResourceKeys() const;

  /**
   * After infershape of write op, ge can get resource_key which shape changed.
   * For resource op infershape, invoked by ge infershape framework.
   * @return keys
   */
  const std::set<ge::AscendString>& GetChangedResourceKeys() const;
  /**
   * After handle changed resource shape, should clear changed_keys in context.
   * For resource op infershape, invoked by ge infershape framework.
   */
  void ClearChangedResourceKeys();

 private:
  explicit InferenceContext(std::unique_ptr<InnerInferenceContext> &inner_context);
  std::shared_ptr<InnerInferenceContext> inner_inference_context_;
};

using InferenceContextPtr = std::shared_ptr<InferenceContext>;
}  // namespace ge
#endif  // INC_EXTERNAL_GRAPH_INFERENCE_CONTEXT_H_
