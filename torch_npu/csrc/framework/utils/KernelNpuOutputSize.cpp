// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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

#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu
{
  namespace native
  {

    int64_t CeilDiv(int64_t value, int64_t factor)
    {
      int64_t value_num = 0;
      if (factor == 0)
      {
        return value_num;
      }
      if (value % factor == 0)
      {
        value_num = value / factor;
      }
      else
      {
        value_num = value / factor + 1;
      }

      return value_num;
    }

    int64_t make_wrap_dim(int64_t dim, int64_t dim_post_expr)
    {
      // this will make range [-1, 0]
      if (dim_post_expr <= 0)
      {
        dim_post_expr = 1;
      }

      int64_t min = -dim_post_expr;
      int64_t max = dim_post_expr - 1;
      if (dim < 0)
      {
        dim += dim_post_expr;
      }

      return dim;
    }

    bitset<64> make_dim_mask(c10::IntArrayRef dims, int64_t ndim)
    {
      bitset<64> mask = bitset<64>();
      if (dims.empty())
      {
        mask.flip();
      }
      else
      {
        for (int64_t dim : dims)
        {
          mask.set(make_wrap_dim(dim, ndim));
        }
      }

      return mask;
    }

    c10::SmallVector<int64_t, SIZE> array_to_small_vector(c10::IntArrayRef shape)
    {
      c10::SmallVector<int64_t, SIZE> shape_small_vec;
      for (int i = 0; i < shape.size(); i++)
      {
        shape_small_vec.emplace_back(shape[i]);
      }

      return shape_small_vec;
    }

    c10::IntArrayRef input_same_output_size(const at::Tensor &input)
    {
      return input.sizes();
    }

    c10::SmallVector<int64_t, SIZE> broadcast_ops_npu_output_size(
        c10::IntArrayRef shape1_,
        c10::IntArrayRef shape2_)
    {

      return c10::SmallVector<int64_t, SIZE>(at::infer_size(shape1_, shape2_));
    }

    c10::SmallVector<int64_t, SIZE> broadcast_ops_npu_output_size(
        const at::Tensor &self,
        const at::Tensor &other)
    {
      return broadcast_ops_npu_output_size(self.sizes(), other.sizes());
    }

    c10::SmallVector<int64_t, SIZE> reduce_ops_npu_output_size(
        const at::Tensor &self,
        c10::IntArrayRef dim,
        bool keepdim)
    {
      int64_t ndim = self.dim();
      bitset<64> mask = make_dim_mask(dim, ndim);
      auto shape = array_to_small_vector(self.sizes());
      for (int dim = shape.size() - 1; dim >= 0; dim--)
      {
        if (mask[dim])
        {
          if (keepdim)
          {
            shape[dim] = 1;
          }
          else
          {
            shape.erase(shape.begin() + dim);
          }
        }
      }

      return shape;
    }

    c10::SmallVector<int64_t, SIZE> adaptive_avg_pool3d_npu_output_size(
        const at::Tensor &self,
        c10::IntArrayRef output_size)
    {
      auto shape = array_to_small_vector(self.sizes());
      auto iter = shape.rbegin();
      *iter = output_size[2];
      *(iter + 1) = output_size[1];
      *(iter + 2) = output_size[0];
      return shape;
    }

    c10::SmallVector<int64_t, SIZE> addmm_npu_output_size(
        const at::Tensor &self,
        const at::Tensor &mat1,
        const at::Tensor &mat2,
        c10::Scalar beta,
        c10::Scalar alpha)
    {
      return broadcast_ops_npu_output_size(
          self.sizes(), {mat1.size(0), mat2.size(1)});
    }

    c10::SmallVector<int64_t, SIZE> addbmm_npu_output_size(
        const at::Tensor &self,
        const at::Tensor &batch1,
        const at::Tensor &batch2,
        c10::Scalar beta,
        c10::Scalar alpha)
    {
      return {self.size(0), self.size(1)};
    }

    c10::SmallVector<int64_t, SIZE> addmv_npu_output_size(
        const at::Tensor &self,
        const at::Tensor &mat,
        const at::Tensor &vec,
        c10::Scalar beta,
        c10::Scalar alpha)
    {
      return broadcast_ops_npu_output_size(
          self.sizes(), {mat.size(0)});
    }

    c10::SmallVector<int64_t, SIZE> addr_npu_output_size(
        const at::Tensor &self,
        const at::Tensor &vec1,
        const at::Tensor &vec2,
        c10::Scalar beta,
        c10::Scalar alpha)
    {
      return broadcast_ops_npu_output_size(
          self.sizes(), {vec1.size(0), vec2.size(0)});
    }

    c10::SmallVector<int64_t, SIZE> avg_pool2d_npu_output_size(
        const at::Tensor &self,
        c10::IntArrayRef kernel_size,
        c10::IntArrayRef stride,
        c10::IntArrayRef padding,
        bool ceil_mode,
        bool count_include_pad,
        c10::optional<int64_t> divisor_override)
    {
      int H = self.size(-2);
      int W = self.size(-1);

      int64_t kH = ceil_mode
                       ? (CeilDiv(H + 2 * padding[0] - kernel_size[0], stride[0]) + 1)
                       : ((H + 2 * padding[0] - kernel_size[0]) / stride[0] + 1);
      int64_t kW = ceil_mode
                       ? (CeilDiv(W + 2 * padding[1] - kernel_size[1], stride[1]) + 1)
                       : ((W + 2 * padding[1] - kernel_size[1]) / stride[1] + 1);

      if (ceil_mode) {
        if ((kH - 1) * stride[0] >= H + padding[0]) {
          --kH;
        }
        if ((kW - 1) * stride[1] >= W + padding[1]) {
          --kW;
        }
      }

      c10::SmallVector<int64_t, SIZE> outputSize = {self.size(0), self.size(1), kH, kW};
      return outputSize;
    }

    c10::SmallVector<int64_t, SIZE> baddbmm_npu_output_size(
        const at::Tensor &self,
        const at::Tensor &mat2)
    {
      return {self.size(0), self.size(1), mat2.size(2)};
    }

    c10::SmallVector<int64_t, SIZE> cdist_npu_output_size(
        const at::Tensor &x1,
        const at::Tensor &x2)
    {
      int64_t r1 = x1.size(-2);
      int64_t r2 = x2.size(-2);
      auto dim1 = x1.dim();
      auto dim2 = x2.dim();
      c10::IntArrayRef batch_tensor1(x1.sizes().data(), dim1 - 2);
      c10::IntArrayRef batch_tensor2(x2.sizes().data(), dim2 - 2);
      c10::SmallVector<int64_t, SIZE> expand_batch_portion(at::infer_size(batch_tensor1, batch_tensor2));
      c10::SmallVector<int64_t, SIZE> output_shape(expand_batch_portion);
      output_shape.insert(output_shape.end(), {r1, r2});
      return output_shape;
    }

    tuple<c10::IntArrayRef, c10::IntArrayRef, c10::SmallVector<int64_t, SIZE>> conv2d_backward_npu_output_size(
        const at::Tensor &input,
        const at::Tensor &grad,
        const at::Tensor &weight,
        c10::IntArrayRef stride,
        c10::IntArrayRef padding,
        c10::IntArrayRef dilation,
        int64_t groups)
    {
      c10::SmallVector<int64_t, SIZE> gradBiasSize = {grad.size(1)};
      return tuple<c10::IntArrayRef, c10::IntArrayRef, c10::SmallVector<int64_t, SIZE>>(
          input.sizes(), weight.sizes(), gradBiasSize);
    }

    c10::SmallVector<int64_t, SIZE> cosine_similarity_npu_output_size(
        const at::Tensor &x1,
        int64_t dim,
        bool keepdim)
    {
      c10::IntArrayRef dims(dim);
      return reduce_ops_npu_output_size(x1, dims, keepdim);
    }

    tuple<c10::IntArrayRef, c10::IntArrayRef, c10::SmallVector<int64_t, SIZE>> conv_transpose2d_backward_npu_output_size(
        const at::Tensor &input,
        const at::Tensor &grad_output,
        const at::Tensor &weight,
        c10::IntArrayRef padding,
        c10::IntArrayRef output_padding,
        c10::IntArrayRef stride,
        c10::IntArrayRef dilation,
        int64_t groups)
    {
      c10::SmallVector<int64_t, SIZE> gradBiasSize = {grad_output.size(1)};
      return tuple<c10::IntArrayRef, c10::IntArrayRef, c10::SmallVector<int64_t, SIZE>>(
          input.sizes(), weight.sizes(), gradBiasSize);
    }

    c10::SmallVector<int64_t, SIZE> conv_transpose2d_npu_output_size(
        const at::Tensor &input,
        const at::Tensor &weight,
        const at::Tensor &bias,
        c10::IntArrayRef padding,
        c10::IntArrayRef output_padding,
        c10::IntArrayRef stride,
        c10::IntArrayRef dilation,
        int64_t groups)
    {
      int64_t N = input.size(0);
      int64_t H = input.size(2);
      int64_t W = input.size(3);
      int64_t Co = weight.size(1) * groups;
      auto kernel_size = weight.sizes().slice(2);

      int64_t Ho = (H - 1) * stride[0] - 2 * padding[0] +
                   dilation[0] * (kernel_size[0] - 1) + output_padding[0] + 1;
      int64_t Wo = (W - 1) * stride[1] - 2 * padding[1] +
                   dilation[1] * (kernel_size[1] - 1) + output_padding[1] + 1;

      c10::SmallVector<int64_t, SIZE> outputSize = {N, Co, Ho, Wo};

      return outputSize;
    }

    c10::SmallVector<int64_t, SIZE> deformable_conv2d_npu_output_size(
        const at::Tensor &input,
        const at::Tensor &weight,
        const at::Tensor &offset,
        const at::Tensor &bias,
        c10::IntArrayRef kernel_size,
        c10::IntArrayRef stride,
        c10::IntArrayRef padding,
        c10::IntArrayRef dilation,
        int64_t groups,
        int64_t deformable_groups,
        bool modulated)
    {
      int64_t No = input.size(0);
      int64_t Co = input.size(1);
      int64_t Ho = offset.size(2) * kernel_size[0];
      int64_t Wo = offset.size(3) * kernel_size[1];

      c10::SmallVector<int64_t, SIZE> outputSize = {No, Co, Ho, Wo};

      return outputSize;
    }

    c10::SmallVector<int64_t, SIZE> det_npu_output_size(const at::Tensor &self)
    {
      c10::SmallVector<long int, SIZE> dimVec;
      auto InputSize = array_to_small_vector(self.sizes());
      if (InputSize.size() > 2)
      {
        for (int i = 0; i < InputSize.size() - 2; i++)
        {
          dimVec.push_back(self.size(i));
        }
      }
      else
      {
        return dimVec;
      }
      return dimVec;
    }

    tuple<c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>> ctc_loss_npu_output_size(
        const at::Tensor &logProbs,
        const at::Tensor &targets,
        c10::IntArrayRef targetLengths,
        int64_t maxLength)
    {
      int64_t maxInputLength = logProbs.size(0);
      int64_t batchSize = logProbs.size(1);
      int64_t numLabels = logProbs.size(2);

      c10::SmallVector<int64_t, SIZE> negLogLikelihoodSize = {batchSize};

      int64_t tSize = 2 * maxLength + 1;
      c10::SmallVector<int64_t, SIZE> logAlphaSize = {batchSize, maxInputLength, tSize};

      return tuple<c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>>(negLogLikelihoodSize, logAlphaSize);
    }

    c10::SmallVector<int64_t, SIZE> dot_npu_output_size(
        const at::Tensor &self,
        const at::Tensor &other)
    {
      c10::SmallVector<int64_t, SIZE> outputSize = {1};
      return outputSize;
    }

    c10::SmallVector<int64_t, SIZE> embedding_dense_backward_npu_output_size(
        const at::Tensor &grad_output,
        const at::Tensor &indices,
        int64_t num_weights,
        int64_t padding_idx,
        bool scale_grad_by_freq)
    {
      return {num_weights, grad_output.size(-1)};
    }

    c10::SmallVector<int64_t, SIZE> equal_npu_output_size(void)
    {
      int64_t outputshape = 1;
      c10::SmallVector<int64_t, SIZE> outputSize = {outputshape};
      return outputSize;
    }

    tuple<c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef> layer_norm_backward_npu_output_size(
        const at::Tensor &dY,
        const at::Tensor &X,
        const at::Tensor &mean,
        const at::Tensor &rstd,
        const at::Tensor &gamma,
        int64_t M,
        int64_t N)
    {
      return tuple<c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef>(
          X.sizes(), gamma.sizes(), gamma.sizes());
    }

    static bool hasContiguousSubspace(at::TensorList tl)
    {
      // true if all the non-null tensors are adjacent
      auto isDefined = [](const at::Tensor &tensor)
      { return tensor.defined(); };
      auto isNull = [](const at::Tensor &tensor)
      { return !tensor.defined(); };
      auto start = std::find_if(tl.begin(), tl.end(), isDefined);
      auto stop = std::find_if(tl.rbegin(), tl.rend(), isDefined);
      auto it = std::find_if(start, stop.base(), isNull);
      return it == stop.base();
    }

    c10::SmallVector<int64_t, SIZE> index_npu_output_size(
        const at::Tensor &self,
        at::TensorList indices)
    {
      std::vector<at::Tensor> new_indices;
      for (const auto &index : indices)
      {
        if (index.scalar_type() == at::kBool)
        {
          for (int64_t j = 0; j < index.dim(); j++)
          {
            int64_t srcIdx = new_indices.size() + j;
            if (index.size(j) != self.size(srcIdx))
            {
              TORCH_CHECK_INDEX("The shape of boolTensorIndex does not match the self");
            }
          }
          // Replace with nonzeros
          auto nonzero = index.nonzero();
          for (int64_t j = 0; j < index.dim(); j++)
          {
            new_indices.emplace_back(nonzero.select(1, j));
          }
        }
        else
        {
          new_indices.emplace_back(index);
        }
      }

      c10::SmallVector<int64_t, SIZE> inferShape;
      for (size_t i = 0; i < new_indices.size(); ++i)
      {
        if (!new_indices[i].defined())
        {
          continue;
        }
        else if (inferShape.empty())
        {
          inferShape = new_indices[i].sizes();
        }
        else
        {
          inferShape = at::infer_size(inferShape, new_indices[i].sizes());
        }
      }

      std::vector<at::Tensor> mid_indices(new_indices.size());
      for (size_t i = 0; i < new_indices.size(); ++i)
      {
        if (!new_indices[i].defined())
        {
          continue;
        }
        else if (new_indices[i].sizes().equals(inferShape))
        {
          mid_indices[i] = new_indices[i];
        }
        else
        {
          mid_indices[i] = new_indices[i].expand(inferShape, true);
        }
      }

      while (mid_indices.size() < (size_t)self.dim())
      {
        mid_indices.emplace_back();
      }
      at::Tensor src = self;
      std::vector<at::Tensor> end_indices = mid_indices;
      if (!hasContiguousSubspace(mid_indices))
      {
        end_indices.clear();
        std::vector<int64_t> dims;
        dims.reserve(self.dim());
        for (int64_t i = 0; i < self.dim(); i++)
        {
          if (mid_indices[i].defined())
          {
            dims.push_back(i);
            end_indices.emplace_back(mid_indices[i]);
          }
        }
        for (int64_t i = 0; i < self.dim(); i++)
        {
          if (!mid_indices[i].defined())
          {
            dims.push_back(i);
            end_indices.emplace_back();
          }
        }
        src = self.permute(dims);
      }

      int64_t dims_before = 0, dims_after = 0, dims_indexed = 0;
      c10::SmallVector<int64_t, SIZE> replacement_shape;
      at::DimVector indexed_sizes;
      for (size_t dim = 0; dim < end_indices.size(); dim++)
      {
        if (!end_indices[dim].defined())
        {
          if (dims_indexed == 0)
          {
            dims_before++;
          }
          else
          {
            dims_after++;
          }
        }
        else
        {
          dims_indexed++;
          replacement_shape = end_indices[dim].sizes();
          indexed_sizes.push_back(src.size(dim));
        }
      }
      if (std::find(indexed_sizes.begin(), indexed_sizes.end(), 0) != indexed_sizes.end() &&
          std::find(replacement_shape.begin(), replacement_shape.end(), 0) == replacement_shape.end())
      {
        TORCH_CHECK_INDEX(false, "index is out of bounds for dimension with size 0");
      }
      auto self_shape = at::DimVector(src.sizes());
      int64_t end = dims_before + dims_indexed;
      self_shape.erase(self_shape.begin() + dims_before, self_shape.begin() + end);
      self_shape.insert(self_shape.begin() + dims_before, replacement_shape.begin(), replacement_shape.end());

      c10::SmallVector<int64_t, SIZE> index_shape;
      for (auto &index : end_indices)
      {
        if (index.defined())
        {
          auto shape = at::DimVector();
          shape.append(dims_before, 1);
          shape.append(index.sizes().begin(), index.sizes().end());
          shape.append(dims_after, 1);
          if (index_shape.empty())
          {
            index_shape = shape;
          }
          else if (index_shape != shape)
          {
            index_shape = at::infer_size(index_shape, shape);
          }
        }
      }

      c10::SmallVector<int64_t, SIZE> outputSize = index_shape;
      if (index_shape != self_shape)
      {
        outputSize = at::infer_size(index_shape, self_shape);
      }

      return outputSize;
    }

    c10::SmallVector<int64_t, SIZE> index_select_npu_output_size(
        const at::Tensor &self,
        int64_t dim,
        const at::Tensor &index)
    {
      int64_t indexSize = index.size(0);

      c10::SmallVector<int64_t, SIZE> outputSize;
      for (int64_t i = 0; i < self.sizes().size(); ++i)
      {
        if (i == dim)
        {
          outputSize.push_back(indexSize);
        }
        else
        {
          outputSize.push_back(self.size(i));
        }
      }

      return outputSize;
    }

    c10::SmallVector<int64_t, SIZE> iou_npu_output_size(
        const at::Tensor &bboxes,
        const at::Tensor &gtboxes)
    {
      return {gtboxes.size(0), bboxes.size(0)};
    }

    c10::SmallVector<int64_t, SIZE> lstm_npu_output_size(
        const at::Tensor &input,
        const at::Tensor &weight,
        const at::Tensor &bias,
        const at::Tensor &h,
        const at::Tensor &c,
        bool has_biases,
        int64_t num_layers,
        double dropout,
        bool train,
        bool bidirectional,
        bool batch_first)
    {
      int64_t numStep = input.size(0);
      int64_t batchSize = input.size(1);
      int64_t hiddenSize = bias.size(0) / 4;

      c10::SmallVector<int64_t, SIZE> outputSize = {numStep, batchSize, hiddenSize};

      return outputSize;
    }

    c10::SmallVector<int64_t, SIZE> nnpack_spatial_convolution_npu_output_size(
        const at::Tensor &input,
        const at::Tensor &weight,
        c10::IntArrayRef padding,
        c10::IntArrayRef stride)
    {
      int64_t N = input.size(0);
      int64_t H = input.size(2);
      int64_t W = input.size(3);
      int64_t Co = weight.size(0);
      auto kernel_size = weight.sizes().slice(2);

      int64_t Ho = 0;
      int64_t Wo = 0;
      if (padding.size() == 1 && stride.size() == 1)
      {
        Ho = (H + 2 * padding[0] - (kernel_size[0] - 1) - 1) /
                 stride[0] +
             1;
        Wo = (W + 2 * padding[0] - (kernel_size[1] - 1) - 1) /
                 stride[0] +
             1;
      }
      if (padding.size() != 1 && stride.size() == 1)
      {
        Ho = (H + 2 * padding[0] - (kernel_size[0] - 1) - 1) /
                 stride[0] +
             1;
        Wo = (W + 2 * padding[1] - (kernel_size[1] - 1) - 1) /
                 stride[0] +
             1;
      }
      if (padding.size() != 1 && stride.size() != 1)
      {
        Ho = (H + 2 * padding[0] - (kernel_size[0] - 1) - 1) /
                 stride[0] +
             1;
        Wo = (W + 2 * padding[1] - (kernel_size[1] - 1) - 1) /
                 stride[1] +
             1;
      }
      c10::SmallVector<int64_t, SIZE> outputSize = {N, Co, Ho, Wo};
      return outputSize;
    }

    tuple<c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>> nms_with_mask_npu_output_size(
        const at::Tensor &input)
    {
      c10::SmallVector<int64_t, SIZE> boxesSize = {input.size(0), 5};
      c10::SmallVector<int64_t, SIZE> idxSize = {
          input.size(0),
      };
      c10::SmallVector<int64_t, SIZE> maskSize = {
          input.size(0),
      };

      return std::tuple<
          c10::SmallVector<int64_t, SIZE>,
          c10::SmallVector<int64_t, SIZE>,
          c10::SmallVector<int64_t, SIZE>>(boxesSize, idxSize, maskSize);
    };

    c10::SmallVector<int64_t, SIZE> nonzero_npu_output_size(const at::Tensor &self)
    {
      int64_t dim = self.dim();
      at::Tensor boolSelf = NPUNativeFunctions::npu_dtype_cast(self, at::ScalarType::Bool);
      at::Tensor intSelf = NPUNativeFunctions::npu_dtype_cast(boolSelf, at::ScalarType::Int);

      at::Tensor coutNonzeroSelf = intSelf;
      if (self.numel() > 10000000)
      {
        // Ensure outputsize correctly in large shape case
        coutNonzeroSelf = at::sum(intSelf, at::ScalarType::Long);
      }
      else
      {
        coutNonzeroSelf = at::sum(intSelf, at::ScalarType::Int);
      }

      int64_t nonzeroNum = coutNonzeroSelf.item().toInt();
      c10::SmallVector<int64_t, SIZE> outputSize = {nonzeroNum, dim};
      return outputSize;
    }

    c10::SmallVector<int64_t, SIZE> nonzero_npu_max_output_size(const at::Tensor& self) {
      int64_t selfNumEl = self.numel();
      int64_t selfDim = self.dim();
      at::SmallVector<int64_t, SIZE> maxOutputSize = {selfNumEl, selfDim};
      return maxOutputSize;
    }

    c10::SmallVector<int64_t, SIZE> pad_npu_output_size(
        const at::Tensor &input,
        c10::IntArrayRef paddings)
    {
      c10::SmallVector<int64_t, SIZE> outputSize;
      for (int i = 0; i < input.dim(); i++)
      {
        if (i * 2 + 1 < paddings.size())
        {
          outputSize.emplace_back(input.size(i) + paddings[i * 2] + paddings[i * 2 + 1]);
        }
        else if (i * 2 < paddings.size())
        {
          outputSize.emplace_back(input.size(i) + paddings[i * 2]);
        }
        else
        {
          outputSize.emplace_back(input.size(i));
        }
      }
      return outputSize;
    }

    c10::SmallVector<int64_t, SIZE> pdist_npu_output_size(const at::Tensor &self, float p)
    {
      c10::SmallVector<int64_t, SIZE> outputSize;
      int64_t n = self.size(0);
      int64_t resultSize = n * (n - 1) / 2;
      outputSize.emplace_back(resultSize);
      return outputSize;
    }

    c10::SmallVector<int64_t, SIZE> prod_npu_output_size(
        const at::Tensor &self,
        int64_t dim,
        bool keepdim)
    {
      c10::IntArrayRef dims(dim);
      return reduce_ops_npu_output_size(self, dims, keepdim);
    }

    c10::SmallVector<int64_t, SIZE> prod_npu_output_size(
        const at::Tensor &self,
        bool keepdim)
    {
      c10::IntArrayRef dims;
      return reduce_ops_npu_output_size(self, dims, keepdim);
    }

    c10::SmallVector<int64_t, SIZE> quantized_max_pool2d_npu_output_size(
        const at::Tensor &self,
        c10::IntArrayRef kernel_size,
        c10::IntArrayRef stride,
        c10::IntArrayRef padding,
        c10::IntArrayRef dilation,
        bool ceil_mode)
    {
      int64_t strideH = 1;
      int64_t strideW = 1;
      if (stride.empty())
      {
        strideH = kernel_size[0];
        strideW = kernel_size[1];
      }
      else
      {
        strideH = stride[0];
        strideW = stride[1];
      }

      int64_t N = self.size(0);
      int64_t C = self.size(1);
      int64_t H = self.size(2);
      int64_t W = self.size(3);

      int64_t Ho = (H + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1 +
                    (ceil_mode ? strideH - 1 : 0)) /
                       strideH +
                   1;
      int64_t Wo = (W + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1 +
                    (ceil_mode ? strideW - 1 : 0)) /
                       strideW +
                   1;
      c10::SmallVector<int64_t, SIZE> outputSize = {N, C, Ho, Wo};

      return outputSize;
    }

    c10::SmallVector<int64_t, SIZE> range_npu_output_size(
        float start,
        float end,
        float step)
    {
      if (step == 0)
      {
        AT_ERROR("range_npu_output_size step is zero!");
      }
      int64_t size_value = std::floor((end - start) / step);
      c10::SmallVector<int64_t, SIZE> outputSize = {size_value + 1};

      return outputSize;
    }

    c10::SmallVector<int64_t, SIZE> repeat_interleave_npu_output_size(
        const at::Tensor &self,
        int64_t repeats,
        int64_t dim)
    {
      c10::SmallVector<int64_t, SIZE> shape;
      if (dim < 0)
      {
        dim = dim + self.dim();
      }
      for (int64_t i = 0; i < self.dim(); i++)
      {
        if (i == dim)
        {
          shape.emplace_back(self.size(i) * repeats);
        }
        else
        {
          shape.emplace_back(self.size(i));
        }
      }
      return shape;
    }

    c10::SmallVector<int64_t, SIZE> replication_pad2d_npu_output_size(const at::Tensor &self, c10::IntArrayRef padding)
    {
      int64_t N = self.size(0);
      int64_t C = self.size(1);
      int64_t H = self.size(2);
      int64_t W = self.size(3);
      int64_t padding_l = 0;
      int64_t padding_r = 0;
      int64_t padding_t = 0;
      int64_t padding_b = 0;
      if (!padding.empty() && padding.size() == 1)
      {
        padding_l = padding[0];
        padding_r = padding[0];
        padding_t = padding[0];
        padding_b = padding[0];
      }
      else if (!padding.empty() && 4 == padding.size())
      {
        padding_l = padding[0];
        padding_r = padding[1];
        padding_t = padding[2];
        padding_b = padding[3];
      }
      int64_t Ho = H + padding_t + padding_b;
      int64_t Wo = W + padding_l + padding_r;

      c10::SmallVector<int64_t, SIZE> outputSize = {N, C, Ho, Wo};
      return outputSize;
    }

    tuple<c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>> nms_v4_npu_output_size(
        c10::Scalar max_output_size)
    {
      c10::SmallVector<int64_t, SIZE> selected_indices = {max_output_size.toInt()};
      c10::SmallVector<int64_t, SIZE> valid_outputs = {};
      return std::tuple<c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>>(
          selected_indices, valid_outputs);
    }

    c10::SmallVector<int64_t, SIZE> repeat_npu_output_size(
        const at::Tensor &self,
        c10::IntArrayRef repeats)
    {
      int64_t num_new_dimensions = repeats.size() - self.dim();
      // Fill num_ new_ Dimensions elements with a value of 1
      c10::SmallVector<int64_t, SIZE> padded_size(num_new_dimensions, 1);
      padded_size.insert(
          padded_size.end(), self.sizes().begin(), self.sizes().end());
      c10::SmallVector<int64_t, SIZE> target_size(repeats.size());
      for (int64_t idx = 0; idx < repeats.size(); ++idx)
      {
        target_size[idx] = padded_size[idx] * repeats[idx];
      }
      return target_size;
    }

    c10::SmallVector<int64_t, SIZE> soft_margin_loss_npu_output_size(
        const at::Tensor &self,
        const at::Tensor &target,
        int64_t reduction)
    {
      c10::SmallVector<int64_t, SIZE> outputSize;
      if (reduction == at::Reduction::None)
      {
        outputSize = input_same_output_size(self);
      }
      else
      {
        outputSize = {1};
      }
      return outputSize;
    }

    c10::SmallVector<int64_t, SIZE> slow_conv_dilated2d_npu_output_size(
        const at::Tensor &input,
        const at::Tensor &weight,
        c10::IntArrayRef stride,
        c10::IntArrayRef padding,
        c10::IntArrayRef dilation)
    {
      int64_t N = input.size(0);
      int64_t H = input.size(2);
      int64_t W = input.size(3);
      int64_t Co = weight.size(0);
      auto kernel_size = weight.sizes().slice(2);

      int64_t Ho = (H + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) /
                       stride[0] +
                   1;
      int64_t Wo = (W + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) /
                       stride[1] +
                   1;

      c10::SmallVector<int64_t, SIZE> outputSize = {N, Co, Ho, Wo};

      return outputSize;
    }

    tuple<c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef> slow_conv_dilated2d_backward_npu_output_size(
        const at::Tensor &grad_output,
        const at::Tensor &self,
        const at::Tensor &weight,
        c10::IntArrayRef kernel_size,
        c10::IntArrayRef stride,
        c10::IntArrayRef padding,
        c10::IntArrayRef dilation)
    {
      return tuple<c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef>(grad_output.sizes(), self.sizes(), weight.sizes());
    }

    tuple<c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef> slow_conv_transpose2d_backward_npu_output_size(
        const at::Tensor &grad_output,
        const at::Tensor &self,
        const at::Tensor &weight,
        c10::IntArrayRef kernel_size,
        c10::IntArrayRef stride,
        c10::IntArrayRef padding,
        c10::IntArrayRef output_padding,
        c10::IntArrayRef dilation,
        const at::Tensor &columns,
        const at::Tensor &ones)
    {
      return tuple<c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef>(self.sizes(), weight.sizes(), grad_output.sizes());
    }

    c10::IntArrayRef smooth_l1_loss_npu_output_size(
        const at::Tensor &self,
        const at::Tensor &target,
        int64_t reduction)
    {
      c10::IntArrayRef outputSize;
      if (reduction == at::Reduction::None)
      {
        outputSize = input_same_output_size(self);
      }
      return outputSize;
    }

    tuple<c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>> softmax_cross_entropy_with_logits_impl_npu_output_size(
        const at::Tensor &self)
    {
      c10::SmallVector<int64_t, SIZE> resultSize = array_to_small_vector(self.size(0));
      c10::SmallVector<int64_t, SIZE> backpropSize = array_to_small_vector(self.sizes());

      return tuple<c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>>(
          resultSize, backpropSize);
    }

    c10::SmallVector<int64_t, SIZE> sum_npu_output_size(
        const at::Tensor &self,
        c10::IntArrayRef dim,
        bool keepdim)
    {
      return reduce_ops_npu_output_size(self, dim, keepdim);
    }

    c10::SmallVector<int64_t, SIZE> topk_npu_output_size(
        const at::Tensor &self,
        int64_t k,
        int64_t dim,
        bool largest,
        bool sorted)
    {
      int64_t wrap_dim = make_wrap_dim(dim, self.dim());
      auto shape = array_to_small_vector(self.sizes());
      if (shape.size() > 0)
      {
        shape[wrap_dim] = k;
      }
      return shape;
    }

    c10::SmallVector<int64_t, SIZE> transpose_npu_output_size(
        const at::Tensor &self,
        c10::IntArrayRef perm)
    {
      auto sizes = self.sizes();
      c10::SmallVector<int64_t, SIZE> shape;
      for (int64_t i = 0; i < perm.size(); i++)
      {
        shape.emplace_back(sizes[perm[i]]);
      }

      return shape;
    }

    c10::SmallVector<int64_t, SIZE> trace_npu_output_size(const at::Tensor &self)
    {
      c10::SmallVector<int64_t, SIZE> shape = {1};
      return shape;
    }

    c10::IntArrayRef upsample_bicubic2d_backward_npu_output_size(c10::IntArrayRef input_size)
    {
      return input_size;
    }

    c10::SmallVector<int64_t, SIZE> upsample_bilinear2d_npu_output_size(
        const at::Tensor &self,
        c10::IntArrayRef output_size,
        bool align_corners,
        c10::optional<double> scales_h,
        c10::optional<double> scales_w)
    {
      // the input's dim of upsample_bilinear2d
      int64_t N = self.size(0);
      int64_t C = self.size(1);
      int64_t H = output_size[0];
      int64_t W = output_size[1];

      c10::SmallVector<int64_t, SIZE> outputSize = {N, C, H, W};
      return outputSize;
    }

    c10::IntArrayRef upsample_bilinear2d_backward_npu_output_size(
        const at::Tensor &grad_output,
        c10::IntArrayRef output_size,
        c10::IntArrayRef input_size,
        bool align_corners,
        c10::optional<double> scales_h,
        c10::optional<double> scales_w)
    {
      return input_size;
    }

    c10::SmallVector<int64_t, SIZE> upsample_linear1d_npu_output_size(
        const at::Tensor &self,
        c10::IntArrayRef output_size,
        bool align_corners,
        c10::optional<double> scales)
    {
      int64_t N = self.size(0);
      int64_t C = self.size(1);
      int64_t W = output_size[0];

      c10::SmallVector<int64_t, SIZE> outputSize = {N, C, W};
      return outputSize;
    }

    c10::SmallVector<int64_t, SIZE> var_npu_output_size(
        const at::Tensor &self,
        c10::IntArrayRef dim,
        bool keepdim)
    {
      c10::SmallVector<int64_t, SIZE> outputSize = reduce_ops_npu_output_size(self, dim, keepdim);
      return outputSize;
    }

    c10::SmallVector<int64_t, SIZE> glu_npu_output_size(
        const at::Tensor &self,
        int64_t dim)
    {
      dim = make_wrap_dim(dim, self.dim());
      auto shape = array_to_small_vector(self.sizes());
      shape[dim] = shape[dim] / 2;

      return shape;
    }

    c10::SmallVector<int64_t, SIZE> crop_and_resize_npu_output_size(
        const at::Tensor &self,
        const at::Tensor &boxes,
        at::IntArrayRef crop_size)
    {
      TORCH_CHECK(self.dim() == 4, "input x size must be 4");
      TORCH_CHECK(boxes.dim() == 2, "boxes size must be 2");
      TORCH_CHECK(crop_size.size() == 2, "crop_size size must be 2");
      int64_t N = boxes.size(0);
      int64_t H = crop_size[0];
      int64_t W = crop_size[1];
      int64_t C = self.size(1);

      c10::SmallVector<int64_t, SIZE> outputSize = {N, C, H, W};
      return outputSize;
    }

    c10::SmallVector<int64_t, SIZE> decode_jpeg_npu_output_size(
        at::IntArrayRef image_shape,
        int64_t channels)
    {
      TORCH_CHECK(image_shape.size() == 3, "image_shape size must be 3");
      int64_t H = image_shape[0];
      int64_t W = image_shape[1];
      int64_t C = image_shape[2];

      c10::SmallVector<int64_t, SIZE> outputSize;
      if (channels == 0) {
        outputSize = {1, C, H, W};
      } else {
        outputSize = {1, channels, H, W};
      }

      return outputSize;
    }

  } // namespace native
} // namespace at_npu