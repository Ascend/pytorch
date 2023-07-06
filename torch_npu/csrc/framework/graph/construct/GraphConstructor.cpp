#include "GraphConstructor.h"
#include "torch_npu/csrc/core/npu/NPURunMode.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/framework/graph/util/GraphUtils.h"
#include "torch_npu/csrc/framework/graph/scalar/ScalarMemoryOps.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"

namespace at_npu {
namespace native {

void GraphCommandImpl::SetName(const std::string& name) {
  ir_node_ = std::make_shared<Node>(name);
}

void GraphCommandImpl::AddInput() {
  ++input_index_;
}

void GraphCommandImpl::AddInput(
    const at::Tensor& input,
    const string& desc_name,
    const string& real_dtype,
    const c10::optional<aclFormat>& sensitive_format) {
  if (OpPreparation::IsCPUScalar(input)) {
    return AddZeroDimInput(input, desc_name);
  }
  if (GraphUtils::IsTensorWithoutNode(input)) {
    if (!input.storage().data()) {
      auto storage_impl = input.storage().unsafeGetStorageImpl();
      size_t n_bytes = GraphUtils::GetTensorCapacity(storage_impl);
      GraphUtils::SetDataPtrAndNbytes(storage_impl, n_bytes);
    }
    GraphUtils::SetDataOp(input.storage().unsafeGetStorageImpl());
  }
  if (GraphUtils::IsDataTensor(input)) {
    GraphUtils::RetainGraphDataTensor(input);
  }
  if (sensitive_format.has_value()) {
    ir_node_->AddExtInfo(
        NodeExtInfoType::SENSITIVE_FORMAT_OF_INPUT,
        std::make_pair(desc_name, sensitive_format.value()));
  }

  auto& cur_ir_value = GraphUtils::GetTensorIrValue(input);
  if (!real_dtype.empty()) {
    cur_ir_value.SetRealType(real_dtype);
  }
  ir_node_->AddInput(
      input_index_++, cur_ir_value.GetCurNode(), cur_ir_value.GetValueIndex());
  ir_node_->UpdateNodeHash(GraphUtils::GetTensorIrValueHash(input), real_dtype);
}

void GraphCommandImpl::AddInput(
    const at::Scalar& input,
    const at::ScalarType type,
    CompileType compile_type) {
  bool is_replay_graph_mode = false;
  if (c10_npu::NpuRunMode::CurRunMode() == c10_npu::ModeKind::REPLAY_MODE) {
    is_replay_graph_mode = true;
  }
  if (is_replay_graph_mode == false && compile_type == CompileType::MEMORY_HOST_COMPILE_INDEPENDENT) {
    uint32_t offset;
    ReduceScalarValue(input, type, offset);
    int deviceIndex = 0;
    NPU_CHECK_ERROR(aclrtGetDevice(&deviceIndex));
    auto npu_scalar_tensor = at::empty({}, at::TensorOptions(c10::DeviceType::PrivateUse1, deviceIndex).dtype(type));
    GraphUtils::SetDataOp(npu_scalar_tensor.storage().unsafeGetStorageImpl());
    GraphUtils::RetainGraphDataTensor(npu_scalar_tensor);
    auto& cur_ir_value = GraphUtils::GetTensorIrValue(npu_scalar_tensor);
    cur_ir_value.SetScalarMemOffset(offset);
    ir_node_->AddInput(
        input_index_++, cur_ir_value.GetCurNode(), cur_ir_value.GetValueIndex());
    ir_node_->UpdateNodeHash(GraphUtils::GetTensorIrValueHash(npu_scalar_tensor));
  } else {
    ir_node_->AddExtInfo(
        NodeExtInfoType::INPUT_TYPE_SCALAR,
        std::make_tuple(input_index_++, input, type));
    ir_node_->UpdateNodeHash(CalcuOpUtil::GetScalarFloatValue(input), type);
  }
}

void GraphCommandImpl::AddInput(
    const c10::IntArrayRef& dim_list,
    const at::ScalarType to_type) {
  vector<int64_t> val(dim_list.begin(), dim_list.end());
  ir_node_->AddExtInfo(
      NodeExtInfoType::INPUT_TYPE_LIST_LONG,
      std::make_tuple(input_index_++, std::move(val), to_type));
  ir_node_->UpdateNodeHash(dim_list, to_type);
}

void GraphCommandImpl::AddOutput(
    const at::Tensor& output,
    const string& desc_name,
    const string& real_type,
    const c10::optional<aclFormat>& sensitive_format) {
  if (sensitive_format.has_value()) {
    ir_node_->AddExtInfo(
        NodeExtInfoType::SENSITIVE_FORMAT_OF_OUTPUT,
        std::make_pair(desc_name, sensitive_format.value()));
  }

  // op without input and has outputs no longer need to be treated as graph input
  Value value{ir_node_, output_index_++};
  if (!real_type.empty()) {
    value.SetRealType(real_type);
  }
  GraphUtils::SetTensorIrValue(output, value);
}

void GraphCommandImpl::AddDynamicInputRegFunc(
    DynamicInputRegFunc func,
    DyNumAndIndex num_and_index) {
  ir_node_->AddExtInfo(
      NodeExtInfoType::DYNAMIC_INPUT_FUNC, std::make_pair(func, num_and_index));
}

void GraphCommandImpl::ReduceScalarValue(
    const at::Scalar& input,
    const at::ScalarType type,
    uint32_t& host_ptr_offset) {
  switch (type)
  {
  case at::ScalarType::Float:
    {
      float value = input.toFloat();
      ReduceScalarValueOp<float>(&value, host_ptr_offset);
    }
    break;
  case at::ScalarType::Int:
    {
      int value = input.toInt();
      ReduceScalarValueOp<int>(&value, host_ptr_offset);
    }
    break;
  case at::ScalarType::Long:
    {
      int64_t value = input.toLong();
      ReduceScalarValueOp<int64_t>(&value, host_ptr_offset);
    }
    break;
  case at::ScalarType::Double:
    {
      double value = input.toDouble();
      ReduceScalarValueOp<double>(&value, host_ptr_offset);
    }  
    break;
  case at::ScalarType::Half:
    {
      at::Half value = input.toHalf();
      ReduceScalarValueOp<at::Half>(&value, host_ptr_offset);
    }
    break;
  case at::ScalarType::Byte:
    {
      uint8_t value = input.toByte();
      ReduceScalarValueOp<uint8_t>(&value, host_ptr_offset);
    }   
    break;
  case at::ScalarType::Char:
    {
      int8_t value = input.toChar();
      ReduceScalarValueOp<int8_t>(&value, host_ptr_offset);
    }
    break;
  case at::ScalarType::Short:
    {
      int16_t value = input.toShort();
      ReduceScalarValueOp<int16_t>(&value, host_ptr_offset);
    }
    break;
  case at::ScalarType::Bool:
    {
      bool value = input.toBool();
      ReduceScalarValueOp<bool>(&value, host_ptr_offset);
    }
    break;
  case at::ScalarType::BFloat16:
    {
      at::BFloat16 value = input.toBFloat16();
      ReduceScalarValueOp<at::BFloat16>(&value, host_ptr_offset);
    }
    break;
  default:
    AT_ERROR("scalar not support '", at::toString(type), "' type currently.");
    break;
  }
}

void GraphCommandImpl::AddZeroDimInput(
    const at::Tensor& input,
    const string& desc_name) {
  at::ScalarType dtype = at::ScalarType::Undefined;
  if (!input.unsafeGetTensorImpl()->is_wrapped_number()) {
    dtype = input.scalar_type();
  }
  TORCH_CHECK(
      dtype != at::ScalarType::Undefined, "Cpu tensor scalar type is undefined");
  at::Scalar expect_scalar = CalcuOpUtil::ConvertTensorToScalar(input);
  ir_node_->AddExtInfo(
      NodeExtInfoType::INPUT_TYPE_SCALAR,
      std::make_tuple(input_index_++, expect_scalar, dtype));
  ir_node_->UpdateNodeHash(CalcuOpUtil::GetScalarFloatValue(expect_scalar), dtype);
}

void GraphCommandImpl::Run() {
  if (output_index_ == 0) {
    GraphUtils::RetainNoneOutputNode(ir_node_);
  }
}
} // namespace native
} // namespace at_npu
