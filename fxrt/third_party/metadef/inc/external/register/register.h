/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_EXTERNAL_REGISTER_REGISTER_H
#define INC_EXTERNAL_REGISTER_REGISTER_H

#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

#include "graph/operator.h"
#include "register/register_error_codes.h"
#include "register/register_fmk_types.h"
#include "register/register_types.h"
#include "graph/ascend_string.h"
#include "graph/types.h"

using std::make_shared;
using std::map;
using std::pair;
using std::string;
using std::to_string;
using std::unique_ptr;
using std::vector;

/*lint -e148*/
namespace ge {
class TBEPluginManager;
}

namespace google {
namespace protobuf {
class Message;
}
} // namespace google

namespace domi {
constexpr int64_t kMaxNameLength = 1048576; // 1M

enum DynamicType : int16_t { kInvalid = 0, kInput = 1, kOutput = 2 };
struct DynamicInputOutputInfo {
  DynamicType type; // input/output
  const char_t* port_name;
  int64_t port_name_len;
  const char_t* attr_name;
  int64_t attr_name_len;
  DynamicInputOutputInfo(
      const DynamicType type_instance,
      const char_t* const port_name_instance,
      const int64_t port_name_len_instance,
      const char_t* const attr_name_instance,
      const int64_t attr_name_len_instance)
      : type(type_instance),
        port_name(port_name_instance),
        port_name_len(port_name_len_instance),
        attr_name(attr_name_instance),
        attr_name_len(attr_name_len_instance) {}
  DynamicInputOutputInfo() : DynamicInputOutputInfo(kInvalid, nullptr, 0L, nullptr, 0L) {}
};
Status AutoMappingByOpFn(const ge::Operator& op_src, ge::Operator& op);
Status AutoMappingByOpFnDynamic(
    const ge::Operator& op_src,
    ge::Operator& op,
    const std::vector<DynamicInputOutputInfo>& dynamic_name_attr_value);
ATTRIBUTED_DEPRECATED(Status AutoMappingByOpFn(const ge::Operator&, ge::Operator&))
Status AutoMappingFn(const google::protobuf::Message* op_src, ge::Operator& op);
ATTRIBUTED_DEPRECATED(
    Status AutoMappingByOpFnDynamic(const ge::Operator&, ge::Operator&, const std::vector<DynamicInputOutputInfo>&))
Status AutoMappingFnDynamic(
    const google::protobuf::Message* op_src,
    ge::Operator& op,
    std::map<std::string, std::pair<std::string, std::string>> dynamic_name_attr_value,
    int32_t in_pos = -1,
    int32_t out_pos = -1);
Status AutoMappingSubgraphIndex(
    const ge::Graph& graph,
    const std::function<int32_t(int32_t data_index)>& input,
    const std::function<int32_t(int32_t netoutput_index)>& output);
Status AutoMappingSubgraphIndex(
    const ge::Graph& graph,
    const std::function<Status(int32_t data_index, int32_t& parent_input_index)>& input,
    const std::function<Status(int32_t netoutput_index, int32_t& parent_output_index)>& output);
using google::protobuf::Message;
class OpRegistrationDataImpl;
class FrameworkRegistryImpl;

using ParseParamFunc = std::function<domi::Status(const google::protobuf::Message*, ge::Operator&)>;
using ParseParamByOpFunc = std::function<domi::Status(const ge::Operator&, ge::Operator&)>;
using FusionParseParamFunc =
    std::function<domi::Status(const std::vector<const google::protobuf::Message*>, ge::Operator&)>;
using FusionParseParamByOpFunc = std::function<domi::Status(const std::vector<ge::Operator>&, ge::Operator&)>;
using ParseSubgraphFunc = std::function<Status(const std::string& subgraph_name, const ge::Graph& graph)>;
using ParseOpToGraphFunc = std::function<Status(const ge::Operator&, ge::Graph&)>;
using ParseSubgraphFuncV2 = std::function<Status(const ge::AscendString& subgraph_name, const ge::Graph& graph)>;
using AutoMappingSubgraphIOIndexFunc = std::function<Status(
    const ge::Graph& graph,
    const std::function<Status(int32_t data_index, int32_t& parent_input_index)>& input,
    const std::function<Status(int32_t netoutput_index, int32_t& parent_output_index)>& output)>;

class FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY FrameworkRegistry {
 public:
  FrameworkRegistry(const FrameworkRegistry&) = delete;
  FrameworkRegistry& operator=(const FrameworkRegistry&) = delete;
  ~FrameworkRegistry();
  static FrameworkRegistry& Instance();
  void AddAutoMappingSubgraphIOIndexFunc(domi::FrameworkType framework, AutoMappingSubgraphIOIndexFunc fun);
  AutoMappingSubgraphIOIndexFunc GetAutoMappingSubgraphIOIndexFunc(domi::FrameworkType framework);

 private:
  FrameworkRegistry();
  std::unique_ptr<FrameworkRegistryImpl> impl_;
};

class FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY AutoMappingSubgraphIOIndexFuncRegister {
 public:
  AutoMappingSubgraphIOIndexFuncRegister(domi::FrameworkType framework, AutoMappingSubgraphIOIndexFunc fun);
  ~AutoMappingSubgraphIOIndexFuncRegister() = default;
};

class FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY OpRegistrationData {
 public:
  ATTRIBUTED_DEPRECATED(OpRegistrationData(const char_t*))
  OpRegistrationData(const std::string& om_optype);

  OpRegistrationData(const char_t* om_optype);

  ~OpRegistrationData();

  OpRegistrationData& FrameworkType(const domi::FrameworkType& fmk_type);

  ATTRIBUTED_DEPRECATED(OpRegistrationData& OriginOpType(const std::vector<ge::AscendString>&))
  OpRegistrationData& OriginOpType(const std::initializer_list<std::string>& ori_optype_list);

  OpRegistrationData& OriginOpType(const std::vector<ge::AscendString>& ori_op_type_list);

  ATTRIBUTED_DEPRECATED(OpRegistrationData& OriginOpType(const char_t*))
  OpRegistrationData& OriginOpType(const std::string& ori_optype);

  OpRegistrationData& OriginOpType(const char_t* ori_op_type);

  OpRegistrationData& ParseParamsFn(const ParseParamFunc& parseParamFn);

  OpRegistrationData& ParseParamsByOperatorFn(const ParseParamByOpFunc& parse_param_by_op_fn);

  OpRegistrationData& FusionParseParamsFn(const FusionParseParamFunc& fusionParseParamFn);

  OpRegistrationData& FusionParseParamsFn(const FusionParseParamByOpFunc& fusion_parse_param_fn);

  ATTRIBUTED_DEPRECATED(OpRegistrationData& ParseSubgraphPostFn(const ParseSubgraphFuncV2&))
  OpRegistrationData& ParseSubgraphPostFn(const ParseSubgraphFunc& subgraph_post_fn);

  OpRegistrationData& ParseSubgraphPostFn(const ParseSubgraphFuncV2& subgraph_post_fn);

  OpRegistrationData& ImplyType(const domi::ImplyType& imply_type);

  ATTRIBUTED_DEPRECATED(OpRegistrationData& DelInputWithCond(int32_t, const char_t*, bool))
  OpRegistrationData& DelInputWithCond(int32_t inputIdx, const std::string& attrName, bool attrValue);

  OpRegistrationData& DelInputWithCond(int32_t input_idx, const char_t* attr_name, bool attr_value);

  ATTRIBUTED_DEPRECATED(OpRegistrationData& DelInputWithOriginalType(int32_t, const char_t*))
  OpRegistrationData& DelInputWithOriginalType(int32_t input_idx, const std::string& ori_type);

  OpRegistrationData& DelInputWithOriginalType(int32_t input_idx, const char_t* ori_type);

  OpRegistrationData& InputReorderVector(const std::vector<int32_t>& input_order);

  OpRegistrationData& ParseOpToGraphFn(const ParseOpToGraphFunc& parse_op_to_graph_fn);

  domi::ImplyType GetImplyType() const;
  ATTRIBUTED_DEPRECATED(Status GetOmOptype(ge::AscendString&) const)
  std::string GetOmOptype() const;
  Status GetOmOptype(ge::AscendString& om_op_type) const;
  ATTRIBUTED_DEPRECATED(GetOriginOpTypeSet(std::set<ge::AscendString>&) const)
  std::set<std::string> GetOriginOpTypeSet() const;
  Status GetOriginOpTypeSet(std::set<ge::AscendString>& ori_op_type) const;
  domi::FrameworkType GetFrameworkType() const;
  ParseParamFunc GetParseParamFn() const;
  ParseParamByOpFunc GetParseParamByOperatorFn() const;
  FusionParseParamFunc GetFusionParseParamFn() const;
  FusionParseParamByOpFunc GetFusionParseParamByOpFn() const;
  ParseSubgraphFunc GetParseSubgraphPostFn() const;
  ParseOpToGraphFunc GetParseOpToGraphFn() const;
  Status GetParseSubgraphPostFn(ParseSubgraphFuncV2& func) const;

 private:
  std::shared_ptr<OpRegistrationDataImpl> impl_;
  friend class OpRegistry;
  friend class OpRegistrationTbe;
  friend class ge::TBEPluginManager;
};

class FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY OpReceiver {
 public:
  OpReceiver(OpRegistrationData& reg_data);
  ~OpReceiver() = default;
};
} // namespace domi
namespace ge {
using OpRegistrationData = domi::OpRegistrationData;
using OpReceiver = domi::OpReceiver;
} // namespace ge

#define REGISTER_CUSTOM_OP(name) REGISTER_CUSTOM_OP_UNIQ_HELPER(__COUNTER__, (name))
#define REGISTER_CUSTOM_OP_UNIQ_HELPER(ctr, name) REGISTER_CUSTOM_OP_UNIQ(ctr, (name))
#define REGISTER_CUSTOM_OP_UNIQ(ctr, name) \
  static OpReceiver register_op##ctr __attribute__((unused)) = OpRegistrationData(name)

#define REGISTER_AUTOMAPPING_SUBGRAPH_IO_INDEX_FUNC(framework, fun) \
  static AutoMappingSubgraphIOIndexFuncRegister auto_mapping_subgraph_fun_##framework(framework, (fun));

/*lint +e148*/
#endif // INC_EXTERNAL_REGISTER_REGISTER_H
