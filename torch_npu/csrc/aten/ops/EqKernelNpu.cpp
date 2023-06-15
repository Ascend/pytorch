#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu
{
  namespace native
  {

    at::Tensor &eq_out_npu_nocheck(at::Tensor &result, const at::Tensor &self, const at::Tensor &other)
    {
      at::Tensor selfCast = self;
      at::Tensor otherCast = other;
      if (self.dtype() == c10::ScalarType::Int || other.dtype() == c10::ScalarType::Int)
      {
        selfCast = NPUNativeFunctions::npu_dtype_cast(self, c10::ScalarType::Float);
        otherCast = NPUNativeFunctions::npu_dtype_cast(other, c10::ScalarType::Float);
      }
      auto unified_result = OpPreparation::comparison_op_check(result, selfCast, otherCast, true);
      OpCommand cmd;
      cmd.Name("Equal")
          .Expect(unified_result)
          .Input(selfCast)
          .Input(otherCast)
          .Output(result)
          .Run();

      return result;
    }

    at::Tensor &eq_out_npu_nocheck(at::Tensor &result, const at::Tensor &self, at::Scalar other)
    {
      at::Tensor selfCast = self;
      if (self.dtype() == c10::ScalarType::Int)
      {
        selfCast = NPUNativeFunctions::npu_dtype_cast(self, c10::ScalarType::Float);
      }
      OpCommand cmd;
      cmd.Name("Equal")
          .Input(selfCast)
          .Input(other, selfCast.scalar_type())
          .Output(result)
          .Run();

      return result;
    }

    at::Tensor &NPUNativeFunctions::eq_out(const at::Tensor &self, const at::Tensor &other, at::Tensor &result)
    {
      auto outputSize = broadcast_ops_npu_output_size(self, other);
      OpPreparation::CheckOut(
          {self, other},
          result,
          ACL_FORMAT_ND,
          result.scalar_type(),
          at::IntArrayRef(outputSize));
      eq_out_npu_nocheck(result, self, other);
      return result;
    }

    at::Tensor &NPUNativeFunctions::eq_out(const at::Tensor &self, const at::Scalar& other, at::Tensor &result)
    {
      OpPreparation::CheckOut(
          {self},
          result,
          ACL_FORMAT_ND,
          result.scalar_type(),
          self.sizes());
      eq_out_npu_nocheck(result, self, other);
      return result;
    }

    at::Tensor NPUNativeFunctions::eq(const at::Tensor &self, const at::Tensor &other)
    {
      if (OpPreparation::IsCPUScalar(other)) {
        return NPUNativeFunctions::eq(self, other.item());
      } else if (OpPreparation::IsCPUScalar(self)) {
        return NPUNativeFunctions::eq(other, self.item());
      } else {
        TORCH_CHECK(self.device() == other.device(),
            "Expected all tensors to be on the same device, but found at least two devices, ",
            self.device(), " and ", other.device());
        at::Tensor format_cast_of_self = OpPreparation::CastBackToOriFormat(self);
        at::Tensor format_cast_of_other = OpPreparation::CastBackToOriFormat(other);
        // calculate the output size
        auto output_size = broadcast_ops_npu_output_size(format_cast_of_self, format_cast_of_other);
        // construct the output tensor of the NPU
        at::Tensor result = OpPreparation::ApplyTensor(
            output_size,
            format_cast_of_self.options().dtype(at::kBool),
            format_cast_of_self);
        // calculate the output result of the NPU
        eq_out_npu_nocheck(result, format_cast_of_self, format_cast_of_other);
        return result;
      }
    }

    at::Tensor NPUNativeFunctions::eq(const at::Tensor &self, const at::Scalar& other)
    {
      at::Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);

      // calculate the output size
      auto outputSize = input_same_output_size(formatCastOfSelf);

      // construct the output tensor of the NPU
      at::Tensor result = OpPreparation::ApplyTensorWithFormat(
          outputSize,
          formatCastOfSelf.options().dtype(at::kBool),
          ACL_FORMAT_ND);

      // calculate the output result of the NPU
      eq_out_npu_nocheck(result, formatCastOfSelf, other);
      return result;
    }

    at::Tensor& NPUNativeFunctions::eq_(at::Tensor &self, const at::Tensor &other)
    {
      if (OpPreparation::IsCPUScalar(other)) {
        return NPUNativeFunctions::eq_(self, other.item());
      } else {
        TORCH_CHECK(self.device() == other.device(),
            "Expected all tensors to be on the same device, but found at least two devices, ",
            self.device(), " and ", other.device());
        OpPreparation::CastBackToOriFormat(self);
        at::Tensor format_cast_of_other = OpPreparation::CastBackToOriFormat(other);
        OpPreparation::CheckMemory({self, format_cast_of_other}, {self});

        at::Tensor result = OpPreparation::ApplyTensorWithFormat(
            self.sizes(),
            self.options().dtype(c10::ScalarType::Byte),
            CalcuOpUtil::GetTensorNpuFormat(self));

        if (!NpuUtils::check_match(&self)) {
          at::Tensor contiguous_self = NpuUtils::format_contiguous(self);
          eq_out_npu_nocheck(result, contiguous_self, format_cast_of_other);
        } else {
          eq_out_npu_nocheck(result, self, format_cast_of_other);
        }

        // uint8 to self dtype
        NPUNativeFunctions::npu_dtype_cast_(self, result);

        return self;
      }
    }

    at::Tensor& NPUNativeFunctions::eq_(at::Tensor &self, const at::Scalar& other)
    {
      OpPreparation::CastBackToOriFormat(self);
      c10::SmallVector<at::Tensor, N> inputs = {self};
      c10::SmallVector<at::Tensor, N> outputs = {self};
      CalcuOpUtil::CheckMemoryOverLaps(inputs, outputs);

      at::Tensor result = OpPreparation::ApplyTensorWithFormat(
          self.sizes(),
          self.options().dtype(c10::ScalarType::Byte),
          CalcuOpUtil::GetTensorNpuFormat(self));

      if (!NpuUtils::check_match(&self))
      {
        at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
        eq_out_npu_nocheck(result, contiguousSelf, other);
      }
      else
      {
        eq_out_npu_nocheck(result, self, other);
      }

      // uint8 to self dtype
      NPUNativeFunctions::npu_dtype_cast_(self, result);

      return self;
    }

  } // namespace native
} // namespace at_npu
