// Copyright (c) 2020 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at_npu
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef __PLUGIN_NATIVE_UTILS_OP_PIPE_WITH_MULTI_OUT__
#define __PLUGIN_NATIVE_UTILS_OP_PIPE_WITH_MULTI_OUT__

#include <ATen/ATen.h>

#include "torch_npu/csrc/framework/utils/NPUDefinition.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"

namespace at_npu
{
  namespace native
  {

    namespace
    {
      // the helper functions of unpack
      template <int N>
      struct OpPipeHelper
      {
        template <typename FuncType, typename TupleType, typename... Ts>
        static void ApplyFunc(FuncType func, TupleType tupleObj, Ts... args)
        {
          OpPipeHelper<N - 1>::ApplyFunc(
              func, tupleObj, std::get<N - 1>(tupleObj), args...);
        }
      }; // struct OpPipeHelper

      template <>
      struct OpPipeHelper<0>
      {
        template <typename FuncType, typename TupleType, typename... Ts>
        static void ApplyFunc(FuncType func, TupleType tupleObj, Ts... args)
        {
          func(args...);
        }
      }; // struct OpPipeHelper<0>

      template <int N>
      struct OpPipeRetHelper
      {
        template <typename SrcTuple, size_t... indexs>
        static auto GetPartOfTuple(
            const SrcTuple &tupleObj,
            std::index_sequence<indexs...>)
            -> decltype(std::make_tuple(std::get<indexs>(tupleObj)...))
        {
          return std::make_tuple(std::get<indexs>(tupleObj)...);
        }

        template <typename... Ts>
        static auto GetHeadOfTuple(const std::tuple<Ts...> &tupleObj)
            -> decltype(GetPartOfTuple(tupleObj, std::make_index_sequence<N>{}))
        {
          return GetPartOfTuple(tupleObj, std::make_index_sequence<N>{});
        }
      }; // struct OpPipeRetHelper

    } // namespace

    template <typename... Ts>
    class OpPipeWithMultiOut
    {
    public:
      explicit OpPipeWithMultiOut(Ts... params) : funcParams(params...) {}

      OpPipeWithMultiOut &Func(const std::function<void(Ts...)> &func)
      {
        this->func = func;
        return *this;
      }

      // when the output cannot satify the requirment of tbe and cannot convert it
      // correctly by CheckOut so can use the method to creat a new tensor for tbe
      // return in OpPipeWithMultiOut defined domain and must use
      // FixOutputWithReplace to replace the real output before call ReturnRef
      template <int index>
      OpPipeWithMultiOut &ApplyOutputWithSpecailParams(
          const FormatShape &sizes,
          const c10::TensorOptions &options,
          int format)
      {
        std::get<index>(this->funcParams) =
            OpPreparation::ApplyTensorWithFormat(sizes, options, format);
        return *this;
      }

      template <int index>
      OpPipeWithMultiOut &FixOutputSizeAndFormat(
          const std::initializer_list<at::Tensor> &inputs,
          const at::Tensor &src,
          int64_t format,
          c10::IntArrayRef size)
      {
        OpPreparation::CheckOut(
            inputs,
            std::get<index>(this->funcParams),
            format,
            src.scalar_type(),
            size);
        return *this;
      }

      template <int index>
      OpPipeWithMultiOut &FixOutputDtype(const at::Tensor &a, const at::Tensor &b)
      {
        auto res = binary_op_check(std::get<index>(this->funcParams), a, b);
        return *this;
      }

      template <int index>
      OpPipeWithMultiOut &FixOutputExceptDtype(
          const std::initializer_list<at::Tensor> &inputs,
          const at::Tensor &src,
          const FormatShape &size)
      {
        OpPreparation::CheckOut(
            inputs,
            std::get<index>(this->funcParams),
            src,
            size);
        return *this;
      }

      template <int index>
      OpPipeWithMultiOut &FixOutputExceptDtype(
          const std::initializer_list<at::Tensor> &inputs,
          int64_t format,
          at::ScalarType type,
          c10::IntArrayRef size)
      {
        OpPreparation::CheckOut(
            inputs, std::get<index>(this->funcParams), format, type, size);
        return *this;
      }

      OpPipeWithMultiOut &Call(const std::function<void(Ts...)> &func)
      {
        OpPipeHelper<std::tuple_size<decltype(this->funcParams)>::value>::ApplyFunc(
            func, this->funcParams);
        return *this;
      }

      template <typename... RetTs>
      std::tuple<RetTs...> Return()
      {
        return OpPipeRetHelper<sizeof...(RetTs)>::GetHeadOfTuple(this->funcParams);
      }

      template <typename... RetTs>
      std::tuple<RetTs...> ReturnRef()
      {
        // Not support select the part of tuple (contain reference
        // object) now.
        return this->funcParams;
      }

      template <int index>
      OpPipeWithMultiOut &ReflushOutputDtype(const at::ScalarType &dType)
      {
        std::get<index>(this->funcParams) =
            std::get<index>(this->funcParams).to(dType);
        return *this;
      }

      template <int index>
      OpPipeWithMultiOut &FixOutputWithReplace(at::Tensor &src)
      {
        OpPreparation::CheckOut({}, src, std::get<index>(this->funcParams));
        src.copy_(std::get<index>(this->funcParams));
        std::get<index>(this->funcParams) = src;
        return *this;
      }

    private:
      std::tuple<Ts...>
          funcParams; // Out1, Out2 ... OutN | OtherParam1, OtherParam2 ...
    };

    template <typename... Ts>
    class OpPipeWithDefinedMultiOut
    {
    public:
      explicit OpPipeWithDefinedMultiOut(Ts... params) : funcParams(params...) {}
      ~OpPipeWithDefinedMultiOut() = default;

      OpPipeWithDefinedMultiOut &Func(const std::function<void(Ts...)> &func)
      {
        this->func = func;
        return *this;
      }

      // recommand to use this interface to apply output
      // base on the law of continuity: the format of output should same as input
      template <int index>
      OpPipeWithDefinedMultiOut &ApplyOutputSameAs(const at::Tensor &src)
      {
        std::get<index>(this->funcParams) = OpPreparation::ApplyTensor(src);
        return *this;
      }

      // not recommand
      // only use for special ops, for example: matmul
      // the suppleymentary regulations of the law of continuity.
      template <int index>
      OpPipeWithDefinedMultiOut &ApplyOutputWithSpecialFormat(
          const at::Tensor &src,
          int64_t format)
      {
        std::get<index>(this->funcParams) =
            OpPreparation::ApplyTensorWithFormat(src, format);
        return *this;
      }

      // not recommand
      template <int index>
      OpPipeWithDefinedMultiOut &ApplyOutputWithSpecailParams(
          const FormatShape &sizes,
          const c10::TensorOptions &options,
          int format)
      {
        std::get<index>(this->funcParams) =
            OpPreparation::ApplyTensorWithFormat(sizes, options, format);
        return *this;
      }

      OpPipeWithDefinedMultiOut &Call(const std::function<void(Ts...)> &func)
      {
        OpPipeHelper<std::tuple_size<decltype(this->funcParams)>::value>::ApplyFunc(
            func, this->funcParams);
        return *this;
      }

      template <int index>
      OpPipeWithDefinedMultiOut &ReflushOutputDtype(const at::ScalarType &dType)
      {
        std::get<index>(this->funcParams) =
            std::get<index>(this->funcParams).to(dType);
        return *this;
      }

      template <typename... RetTs>
      std::tuple<RetTs...> Return()
      {
        return OpPipeRetHelper<sizeof...(RetTs)>::GetHeadOfTuple(this->funcParams);
      }

    private:
      std::tuple<Ts...> funcParams;
    };

  } // namespace native
} // namespace at_npu

#endif // __NATIVE_NPU_UTILS_OP_PIPE_WITH_MULTI_OUT__