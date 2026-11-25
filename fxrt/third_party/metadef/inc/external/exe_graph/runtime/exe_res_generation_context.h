/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_EXTERNAL_REGISTER_EXE_RES_GENERATION_CONTEXT_H
#define INC_EXTERNAL_REGISTER_EXE_RES_GENERATION_CONTEXT_H

#include <vector>
#include "graph/ascend_string.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/compute_node_info.h"
#include "exe_graph/runtime/extended_kernel_context.h"
namespace gert {
enum class ExecuteMode {
  kStaticOffloadExecute, // static sink
  kDynamicExecute,
  kEnd
};

/*
 * input:
 *  name: using type, if reuse_key is empty, used as reuse key
 *  reuse_key: for resource reuse, same key will reuse
 *  depend_value_input_indices: depend value input index
 *  required: resource must required
 * output:
 *  is_valid: resource alloc status
 *  stream_id: alloc stream id
 * */
struct StreamInfo {
  ge::AscendString name;
  ge::AscendString reuse_key;
  std::vector<int64_t> depend_value_input_indices;
  bool required{true};
  bool is_valid{false};
  int64_t stream_id{-1};
};

enum class SyncResType { SYNC_RES_EVENT, SYNC_RES_NOTIFY, END };

/*
 * input:
 *  type: resource type
 *  name: using type, if reuse_key is empty, used as reuse key
 *  reuse_key: for resource reuse, same key will reuse
 *  required: resource must required
 * output:
 *  is_valid: resource alloc status
 *  sync_res_id: alloc sync id
 * */
struct SyncResInfo {
  SyncResType type;
  ge::AscendString name;
  ge::AscendString reuse_key;
  bool required{true};
  bool is_valid{false};
  int32_t sync_res_id{-1};
};

class ExeResGenerationContext : public ExtendedKernelContext {
 public:
  ExecuteMode GetExecuteMode() const;

  // get input with index is const data
  bool IsConstInput(const ge::AscendString& name) const;

  // get input/output shape by index in graph
  const gert::StorageShape* GetInputShape(int64_t index) const;
  const gert::StorageShape* GetOutputShape(int64_t index) const;

  // set/get stream resource
  ge::graphStatus SetAttachedStreamInfos(std::vector<StreamInfo>& stream_info_vec) const;
  std::vector<StreamInfo> GetAttachedStreamInfos() const;
  int64_t GetStreamId() const;

  // set/get sync resource
  ge::graphStatus SetSyncResInfos(std::vector<SyncResInfo>& sync_info_vec) const;
  std::vector<SyncResInfo> GetSyncResInfos() const;

  // workspace size adjust
  std::vector<int64_t> GetWorkspaceBytes() const;
  void SetWorkspaceBytes(const std::vector<int64_t>& workspace_bytes) const;

  int64_t GetOpId() const;

  ge::graphStatus SetListStr(const std::string& attr_name, const std::vector<std::string>& list) const;

  bool GetStrAttrVal(const char* attr_name, ge::AscendString& val) const;

  bool SetStrAttrVal(const char* attr_name, const char* val) const;

  bool GetIntAttrVal(const char* attr_name, int64_t& val) const;

  bool SetIntAttrVal(const char* attr_name, const int64_t val) const;

 private:
  friend class ExeResGenerationCtxBuilder;
  // need check valid after construct
  bool CheckContextValid() const;
};
static_assert(
    std::is_standard_layout<ExeResGenerationContext>::value && std::is_trivial<ExeResGenerationContext>::value,
    "The class ExeResGenerationContext must be a POD");

class OpCheckContext : public ExtendedKernelContext {
 public:
  // get input with index is const data
  bool IsConstInput(const ge::AscendString& name) const;
  const StorageShape* GetInputShape(int64_t index) const;
  const StorageShape* GetOutputShape(int64_t index) const;

 private:
  friend class ExeResGenerationCtxBuilder;
  // need check valid after construct
  bool CheckContextValid() const;
};
} // namespace gert

#endif
