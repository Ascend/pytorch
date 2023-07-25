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
#include "torch_npu/csrc/framework/graph/util/GraphModeGuard.h"

namespace at_npu {
namespace native {

    int64_t CeilDiv(int64_t value, int64_t factor)
    {
      int64_t value_num = 0;
      if (factor == 0) {
        return value_num;
      }
      if (value % factor == 0) {
        value_num = value / factor;
      }
      else {
        value_num = value / factor + 1;
      }

      return value_num;
    }

    int64_t make_wrap_dim(int64_t dim, int64_t dim_post_expr) {
      // this will make range [-1, 0]
      if (dim_post_expr <= 0) {
        dim_post_expr = 1;
      }

      int64_t min = -dim_post_expr;
      int64_t max = dim_post_expr - 1;
      if (dim < 0) {
        dim += dim_post_expr;
      }

      return dim;
    }

    bitset<64> make_dim_mask(c10::IntArrayRef dims, int64_t ndim) {
      bitset<64> mask = bitset<64>();
      if (dims.empty()) {
        mask.flip();
      }
      else {
        for (int64_t dim : dims) {
          mask.set(make_wrap_dim(dim, ndim));
        }
      }

      return mask;
    }

    c10::SmallVector<int64_t, SIZE> array_to_small_vector(c10::IntArrayRef shape) {
      c10::SmallVector<int64_t, SIZE> shape_small_vec;
      for (int i = 0; i < shape.size(); i++) {
        shape_small_vec.emplace_back(shape[i]);
      }

      return shape_small_vec;
    }

    c10::IntArrayRef input_same_output_size(const at::Tensor &input) {
      return input.sizes();
    }

    c10::SmallVector<int64_t, SIZE> broadcast_ops_npu_output_size(
        c10::IntArrayRef shape1_,
        c10::IntArrayRef shape2_) {

      return c10::SmallVector<int64_t, SIZE>(at::infer_size(shape1_, shape2_));
    }

    c10::SmallVector<int64_t, SIZE> broadcast_ops_npu_output_size(
        const at::Tensor &self,
        const at::Tensor &other) {
      return broadcast_ops_npu_output_size(self.sizes(), other.sizes());
    }

    c10::SmallVector<int64_t, SIZE> reduce_ops_npu_output_size(
        const at::Tensor &self,
        c10::IntArrayRef dim,
        bool keepdim) {
      int64_t ndim = self.dim();
      bitset<64> mask = make_dim_mask(dim, ndim);
      auto shape = array_to_small_vector(self.sizes());
      for (int dim = shape.size() - 1; dim >= 0; dim--) {
        if (mask[dim]) {
          if (keepdim) {
            shape[dim] = 1;
          }
          else {
            shape.erase(shape.begin() + dim);
          }
        }
      }

      return shape;
    }

    c10::SmallVector<int64_t, SIZE> mse_loss_npu_output_size(
        const at::Tensor& self,
        const at::Tensor& target,
        int64_t reduction) {
      auto shape = broadcast_ops_npu_output_size(self, target);
      if (reduction == at::Reduction::None) {
        return shape;
      } else {
        c10::SmallVector<int64_t, SIZE> output_size;
        for (int i = 1; i < shape.size(); i++) {
          output_size.emplace_back(shape[i]);
        }
        return output_size;
      }
    }

    c10::SmallVector<int64_t, SIZE> adaptive_avg_pool3d_npu_output_size(
        const at::Tensor &self,
        c10::IntArrayRef output_size) {
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
        c10::Scalar alpha) {
      return broadcast_ops_npu_output_size(
          self.sizes(), {mat1.size(0), mat2.size(1)});
    }

    c10::SmallVector<int64_t, SIZE> addbmm_npu_output_size(
        const at::Tensor &self,
        const at::Tensor &batch1,
        const at::Tensor &batch2,
        c10::Scalar beta,
        c10::Scalar alpha) {
      return {self.size(0), self.size(1)};
    }

    c10::SmallVector<int64_t, SIZE> addmv_npu_output_size(
        const at::Tensor &self,
        const at::Tensor &mat,
        const at::Tensor &vec,
        c10::Scalar beta,
        c10::Scalar alpha) {
      return broadcast_ops_npu_output_size(
          self.sizes(), {mat.size(0)});
    }

    c10::SmallVector<int64_t, SIZE> addr_npu_output_size(
        const at::Tensor &self,
        const at::Tensor &vec1,
        const at::Tensor &vec2,
        c10::Scalar beta,
        c10::Scalar alpha) {
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
        c10::optional<int64_t> divisor_override) {
      int self_h = self.size(-2);
      int self_w = self.size(-1);

      int64_t kernel_h = ceil_mode
                       ? (CeilDiv(self_h + 2 * padding[0] - kernel_size[0], stride[0]) + 1)
                       : ((self_h + 2 * padding[0] - kernel_size[0]) / stride[0] + 1);
      int64_t kernel_w = ceil_mode
                       ? (CeilDiv(self_w + 2 * padding[1] - kernel_size[1], stride[1]) + 1)
                       : ((self_w + 2 * padding[1] - kernel_size[1]) / stride[1] + 1);

      if (ceil_mode) {
        if ((kernel_h - 1) * stride[0] >= self_h + padding[0]) {
          --kernel_h;
        }

        if ((kernel_w - 1) * stride[1] >= self_w + padding[1]) {
          --kernel_w;
        }
      }

      c10::SmallVector<int64_t, SIZE> output_size;
      if (self.dim() == 3) {
        output_size = {self.size(0), kernel_h, kernel_w};
      } else {
        output_size = {self.size(0), self.size(1), kernel_h, kernel_w};
      }

      return output_size;
    }

    c10::SmallVector<int64_t, SIZE> baddbmm_npu_output_size(
        const at::Tensor &self,
        const at::Tensor &mat2) {
      return {self.size(0), self.size(1), mat2.size(2)};
    }

    c10::SmallVector<int64_t, SIZE> cdist_npu_output_size(
        const at::Tensor &x1,
        const at::Tensor &x2) {
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
        int64_t groups) {
      c10::SmallVector<int64_t, SIZE> gradBiasSize = {grad.size(1)};
      return tuple<c10::IntArrayRef, c10::IntArrayRef, c10::SmallVector<int64_t, SIZE>>(
          input.sizes(), weight.sizes(), gradBiasSize);
    }

    c10::SmallVector<int64_t, SIZE> cosine_similarity_npu_output_size(
        const at::Tensor &x1,
        int64_t dim,
        bool keepdim) {
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
        int64_t groups) {
      c10::SmallVector<int64_t, SIZE> gradBiasSize = {grad_output.size(1)};
      return tuple<c10::IntArrayRef, c10::IntArrayRef, c10::SmallVector<int64_t, SIZE>>(
          input.sizes(), weight.sizes(), gradBiasSize);
    }

    c10::SmallVector<int64_t, SIZE> conv_npu_output_size(
        const at::Tensor &input,
        const at::Tensor &weight,
        const c10::optional<at::Tensor> &bias,
        c10::IntArrayRef padding,
        c10::IntArrayRef output_padding,
        c10::IntArrayRef stride,
        c10::IntArrayRef dilation,
        int64_t groups, bool transposed)
        {
            int64_t dim = weight.ndimension() - 2; // Subtract nonspatial dimensions: 2
            if (!transposed) {
              if (dim == 1) {
                return conv1d_npu_output_size(input, weight, padding, stride, dilation);
              } else {
                return conv2d_npu_output_size(input, weight, padding, stride, dilation);
              }
            } else {
                const at::Tensor &bias_tensor = c10::value_or_else(bias, [] { return at::Tensor(); });
                if (dim == 1) {
                  return conv_transpose1d_npu_output_size(input, weight, bias_tensor, padding, output_padding, stride,
                                                          dilation, groups);
                } else {
                  // input dim = 3
                  if (input.ndimension() == 3) {
                    c10::SmallVector<int64_t, SIZE> unsqueeze_size = {1, input.size(0), input.size(1), input.size(2)};
                    input.resize_(unsqueeze_size);
                  }
                  return conv_transpose2d_npu_output_size(input, weight, bias_tensor, padding, output_padding, stride,
                                                          dilation, groups);
                }
            }
        }

    c10::SmallVector<int64_t, SIZE> conv1d_npu_output_size(
        const at::Tensor &input,
        const at::Tensor &weight,
        c10::IntArrayRef padding,
        c10::IntArrayRef stride,
        c10::IntArrayRef dilation) {
            int64_t N = input.size(0);
            int64_t L = input.size(2);
            int64_t C_out = weight.size(0);

            auto kernel_size = weight.sizes().slice(2);

            int64_t L_out = (L + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]  + 1;
            c10::SmallVector<int64_t, SIZE> output_size = {N, C_out, L_out};
            return output_size;
        }

    c10::SmallVector<int64_t, SIZE> conv2d_npu_output_size(
        const at::Tensor &input,
        const at::Tensor &weight,
        c10::IntArrayRef padding,
        c10::IntArrayRef stride,
        c10::IntArrayRef dilation) {
            int64_t N = input.size(0);
            int64_t H = input.size(2);
            int64_t W = input.size(3);
            int64_t C_out = weight.size(0);

            auto kernel_size = weight.sizes().slice(2);

            int64_t H_out = (H + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]  + 1;
            int64_t W_out = (W + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]  + 1;
            c10::SmallVector<int64_t, SIZE> output_size = {N, C_out, H_out, W_out};
            return output_size;
        }

    c10::SmallVector<int64_t, SIZE> conv_transpose1d_npu_output_size(
        const at::Tensor &input,
        const at::Tensor &weight,
        const at::Tensor &bias,
        c10::IntArrayRef padding,
        c10::IntArrayRef output_padding,
        c10::IntArrayRef stride,
        c10::IntArrayRef dilation,
        int64_t groups) {
            int64_t N = input.size(0);
            int64_t L = input.size(2);
            int64_t C_out = weight.size(1) * groups;

            auto kernel_size = weight.sizes().slice(2);

            int64_t L_out = (L - 1) * stride[0] - 2 * padding[0] +
                        dilation[0] * (kernel_size[0] - 1) + output_padding[0] + 1;
            c10::SmallVector<int64_t, SIZE> output_size = {N, C_out, L_out};
            return output_size;
        }

    c10::SmallVector<int64_t, SIZE> conv_transpose2d_npu_output_size(
        const at::Tensor &input,
        const at::Tensor &weight,
        const at::Tensor &bias,
        c10::IntArrayRef padding,
        c10::IntArrayRef output_padding,
        c10::IntArrayRef stride,
        c10::IntArrayRef dilation,
        int64_t groups) {
      int64_t N = input.size(0);
      int64_t H = input.size(2);
      int64_t W = input.size(3);
      int64_t C_out = weight.size(1) * groups;
      auto kernel_size = weight.sizes().slice(2);

      int64_t H_out = (H - 1) * stride[0] - 2 * padding[0] +
                   dilation[0] * (kernel_size[0] - 1) + output_padding[0] + 1;
      int64_t W_out = (W - 1) * stride[1] - 2 * padding[1] +
                   dilation[1] * (kernel_size[1] - 1) + output_padding[1] + 1;

      c10::SmallVector<int64_t, SIZE> outputSize = {N, C_out, H_out, W_out};

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
        bool modulated) {
      int64_t No = input.size(0);
      int64_t Co = input.size(1);
      int64_t Ho = offset.size(2) * kernel_size[0];
      int64_t Wo = offset.size(3) * kernel_size[1];

      c10::SmallVector<int64_t, SIZE> outputSize = {No, Co, Ho, Wo};

      return outputSize;
    }

    c10::SmallVector<int64_t, SIZE> det_npu_output_size(const at::Tensor &self) {
      c10::SmallVector<long int, SIZE> dimVec;
      auto InputSize = array_to_small_vector(self.sizes());
      if (InputSize.size() > 2) {
        for (int i = 0; i < InputSize.size() - 2; i++) {
          dimVec.push_back(self.size(i));
        }
      }
      else {
        return dimVec;
      }
      return dimVec;
    }

    tuple<c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>> ctc_loss_npu_output_size(
        const at::Tensor &log_probs,
        int64_t max_length) {
      int64_t time_size = log_probs.size(0);
      int64_t batch_size = log_probs.size(1);

      c10::SmallVector<int64_t, SIZE> neg_log_likelihood_size = {batch_size};

      int64_t alpha_tail_size = 2 * max_length + 1;
      // Apply for a 32 byte aligned space to avoid address shifting in the OP.
      int64_t alpha_tail_size_align = (alpha_tail_size + 7) / 8 * 8;
      c10::SmallVector<int64_t, SIZE> log_alpha_size = {batch_size, time_size, alpha_tail_size_align};

      return tuple<c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>>(neg_log_likelihood_size,
                                                                                     log_alpha_size);
    }

    c10::SmallVector<int64_t, SIZE> dot_npu_output_size(
        const at::Tensor &self,
        const at::Tensor &other) {
      c10::SmallVector<int64_t, SIZE> outputSize = {1};
      return outputSize;
    }

    c10::SmallVector<int64_t, SIZE> embedding_dense_backward_npu_output_size(
        const at::Tensor &grad_output,
        const at::Tensor &indices,
        int64_t num_weights,
        int64_t padding_idx,
        bool scale_grad_by_freq) {
      return {num_weights, grad_output.size(-1)};
    }

    c10::SmallVector<int64_t, SIZE> equal_npu_output_size(void) {
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
        int64_t N) {
      return tuple<c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef>(
          X.sizes(), gamma.sizes(), gamma.sizes());
    }

    static bool hasContiguousSubspace(at::TensorList tl) {
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

    c10::SmallVector<int64_t, SIZE> im2col_backward_npu_output_size(
        const at::Tensor& grad_output,
        const at::IntArrayRef& input_size,
        const at::IntArrayRef& kernel_size)
    {
      TORCH_CHECK((grad_output.dim() == 2 && grad_output.size(0) != 0 && grad_output.size(1) != 0) ||
                  (grad_output.dim() == 3 && grad_output.size(1) != 0 && grad_output.size(2) != 0),
                  "Expected 2D or 3D (batch mode) tensor for gradOutput with possibly 0 batch size and non-zero "
                  "dimensions for gradOutput, but got: ", grad_output.sizes());
      c10::SmallVector<int64_t, SIZE> outputSize;
      if (grad_output.dim() == 2) {
        outputSize = {grad_output.size(0) / (kernel_size[0] * kernel_size[1]), input_size[0], input_size[1]};
      } else {
        outputSize = {grad_output.size(0), grad_output.size(1) / (kernel_size[0] * kernel_size[1]),
            input_size[0], input_size[1]};
      }
      return outputSize;
    }

    std::vector<at::Tensor> index_expand_outplace(at::TensorList to_expand) {
      // expands a list of Tensors; ignores undefined (null) tensors
      bool first = true;
      std::vector<int64_t> sizes;
      for (size_t i = 0; i < to_expand.size(); ++i) {
        if (!to_expand[i].defined()) {
          continue;
        } else if (first) {
          sizes = to_expand[i].sizes().vec();
          first = false;
        } else {
          sizes = at::infer_size(sizes, to_expand[i].sizes());
        }
      }
      std::vector<at::Tensor> result(to_expand.size());
      for (size_t i = 0; i < to_expand.size(); ++i) {
        if (!to_expand[i].defined()) {
          continue;
        } else if (to_expand[i].sizes().equals(sizes)) {
          result[i] = to_expand[i];
        } else {
          result[i] = to_expand[i].expand(sizes, true);
        }
      }
      return result;
    }

    c10::SmallVector<int64_t, SIZE> index_reshape(
        std::vector<at::Tensor> end_indices,
        int64_t dims_before,
        int64_t dims_after) {
      c10::SmallVector<int64_t, SIZE> index_shape;
      for (auto &index : end_indices) {
        if (index.defined()) {
          auto shape = at::DimVector();
          shape.append(dims_before, 1);
          shape.append(index.sizes().begin(), index.sizes().end());
          shape.append(dims_after, 1);
          if (index_shape.empty()) {
            index_shape = shape;
          } else if (index_shape != shape) {
            index_shape = at::infer_size(index_shape, shape);
          }
        }
      }
      return index_shape;
    }

    c10::SmallVector<int64_t, SIZE> index_npu_output_size(const at::Tensor &self, at::TensorList indices) {
      std::vector<at::Tensor> mid_indices = index_expand_outplace(indices);

      while (mid_indices.size() < (size_t)self.dim()) {
        mid_indices.emplace_back();
      }
      at::Tensor src = self;
      std::vector<at::Tensor> end_indices = mid_indices;
      if (!hasContiguousSubspace(mid_indices)) {
        end_indices.clear();
        std::tie(src, end_indices) = at::native::transposeToFront(self, mid_indices);
      }

      int64_t dims_before = 0, dims_after = 0, dims_indexed = 0;
      c10::SmallVector<int64_t, SIZE> replacement_shape;
      at::DimVector indexed_sizes;
      for (size_t dim = 0; dim < end_indices.size(); dim++) {
        if (!end_indices[dim].defined()) {
          if (dims_indexed == 0) {
            dims_before++;
          } else {
            dims_after++;
          }
        } else {
          dims_indexed++;
          replacement_shape = end_indices[dim].sizes();
          indexed_sizes.push_back(src.size(dim));
        }
      }
      if (std::find(indexed_sizes.begin(), indexed_sizes.end(), 0) != indexed_sizes.end() &&
          std::find(replacement_shape.begin(), replacement_shape.end(), 0) == replacement_shape.end()) {
        TORCH_CHECK_INDEX(false, "index is out of bounds for dimension with size 0");
      }
      auto self_shape = at::DimVector(src.sizes());
      int64_t end = dims_before + dims_indexed;
      self_shape.erase(self_shape.begin() + dims_before, self_shape.begin() + end);
      self_shape.insert(self_shape.begin() + dims_before, replacement_shape.begin(), replacement_shape.end());

      c10::SmallVector<int64_t, SIZE> index_shape = index_reshape(end_indices, dims_before, dims_after);
      c10::SmallVector<int64_t, SIZE> outputSize = index_shape;
      if (index_shape != self_shape) {
        outputSize = at::infer_size(index_shape, self_shape);
      }
      return outputSize;
    }

    c10::SmallVector<int64_t, SIZE> index_select_npu_output_size(
        const at::Tensor &self,
        int64_t dim,
        const at::Tensor &index) {
      at::Tensor indexTmp(index);
      if (indexTmp.ndimension() == 0) {
        indexTmp = index.unsqueeze(0);
      }
      int64_t indexSize = indexTmp.size(0);

      c10::SmallVector<int64_t, SIZE> outputSize;
      for (int64_t i = 0; i < self.sizes().size(); ++i) {
        if (i == dim) {
          outputSize.push_back(indexSize);
        }
        else {
          outputSize.push_back(self.size(i));
        }
      }

      return outputSize;
    }

    c10::SmallVector<int64_t, SIZE> iou_npu_output_size(
        const at::Tensor &bboxes,
        const at::Tensor &gtboxes) {
      return {gtboxes.size(0), bboxes.size(0)};
    }

    c10::SmallVector<int64_t, SIZE> kthvalue_npu_output_size(
        const at::Tensor& self,
        int64_t dim,
        bool keepdim)
    {
      at::IntArrayRef dims(dim);
      return reduce_ops_npu_output_size(self, dims, keepdim);
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
        bool batch_first) {
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
        c10::IntArrayRef stride) {
      int64_t N = input.size(0);
      int64_t H = input.size(2);
      int64_t W = input.size(3);
      int64_t Co = weight.size(0);
      auto kernel_size = weight.sizes().slice(2);

      int64_t Ho = 0;
      int64_t Wo = 0;
      if (padding.size() == 1 && stride.size() == 1) {
        Ho = (H + 2 * padding[0] - (kernel_size[0] - 1) - 1) /
                 stride[0] +
             1;
        Wo = (W + 2 * padding[0] - (kernel_size[1] - 1) - 1) /
                 stride[0] +
             1;
      }
      if (padding.size() != 1 && stride.size() == 1) {
        Ho = (H + 2 * padding[0] - (kernel_size[0] - 1) - 1) /
                 stride[0] +
             1;
        Wo = (W + 2 * padding[1] - (kernel_size[1] - 1) - 1) /
                 stride[0] +
             1;
      }
      if (padding.size() != 1 && stride.size() != 1) {
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
        const at::Tensor &input) {
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

    c10::SmallVector<int64_t, SIZE> nonzero_npu_output_size(const at::Tensor &self) {
      int64_t dim = self.dim();
      at::Tensor boolSelf = NPUNativeFunctions::npu_dtype_cast(self, at::ScalarType::Bool);
      at::Tensor intSelf = NPUNativeFunctions::npu_dtype_cast(boolSelf, at::ScalarType::Int);

      at::Tensor coutNonzeroSelf = intSelf;
      if (self.numel() > 10000000) {
        // Ensure outputsize correctly in large shape case
        coutNonzeroSelf = at::sum(intSelf, at::ScalarType::Long);
      }
      else {
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
        c10::IntArrayRef paddings) {
      c10::SmallVector<int64_t, SIZE> outputSize;
      for (int i = 0; i < input.dim(); i++) {
        if (i * 2 + 1 < paddings.size()) {
          outputSize.emplace_back(input.size(i) + paddings[i * 2] + paddings[i * 2 + 1]);
        }
        else if (i * 2 < paddings.size()) {
          outputSize.emplace_back(input.size(i) + paddings[i * 2]);
        }
        else {
          outputSize.emplace_back(input.size(i));
        }
      }
      return outputSize;
    }

    c10::SmallVector<int64_t, SIZE> pdist_npu_output_size(const at::Tensor &self, float p) {
      c10::SmallVector<int64_t, SIZE> outputSize;
      int64_t n = self.size(0);
      int64_t resultSize = n * (n - 1) / 2;
      outputSize.emplace_back(resultSize);
      return outputSize;
    }

    c10::SmallVector<int64_t, SIZE> prod_npu_output_size(
        const at::Tensor &self,
        int64_t dim,
        bool keepdim) {
      c10::IntArrayRef dims(dim);
      return reduce_ops_npu_output_size(self, dims, keepdim);
    }

    c10::SmallVector<int64_t, SIZE> prod_npu_output_size(
        const at::Tensor &self,
        bool keepdim) {
      c10::IntArrayRef dims;
      return reduce_ops_npu_output_size(self, dims, keepdim);
    }

    c10::SmallVector<int64_t, SIZE> quantized_max_pool2d_npu_output_size(
        const at::Tensor &self,
        c10::IntArrayRef kernel_size,
        c10::IntArrayRef stride,
        c10::IntArrayRef padding,
        c10::IntArrayRef dilation,
        bool ceil_mode) {
      int64_t strideH = 1;
      int64_t strideW = 1;
      if (stride.empty()) {
        strideH = kernel_size[0];
        strideW = kernel_size[1];
      }
      else {
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
        float step) {
      if (step == 0) {
        AT_ERROR("range_npu_output_size step is zero!");
      }
      int64_t size_value = std::floor((end - start) / step);
      c10::SmallVector<int64_t, SIZE> outputSize = {size_value + 1};

      return outputSize;
    }

    // infer output shape for int repeats case
    c10::SmallVector<int64_t, SIZE> repeat_interleave_npu_output_size_opapi(const at::Tensor &self,
        int64_t repeats, c10::optional<int64_t> dim) {
        c10::SmallVector<int64_t, SIZE> shape;
        if (dim.has_value()) {
            int64_t real_dim = dim.value_or(0);
            real_dim = (real_dim < 0) ? (real_dim + self.dim()) : real_dim;
            for (int64_t i = 0; i < self.dim(); i++) {
                if (i == real_dim) {
                    shape.emplace_back(repeats * self.size(i));
                }
                else {
                    shape.emplace_back(self.size(i));
                }
            }
        } else {
            shape.emplace_back(repeats * self.numel());
        }
        return shape;
    }


    c10::SmallVector<int64_t, SIZE> repeat_interleave_npu_output_size(
        const at::Tensor &self,
        int64_t repeats,
        int64_t dim) {
      c10::SmallVector<int64_t, SIZE> shape;
      if (dim < 0) {
        dim = dim + self.dim();
      }
      for (int64_t i = 0; i < self.dim(); i++) {
        if (i == dim) {
          shape.emplace_back(self.size(i) * repeats);
        }
        else {
          shape.emplace_back(self.size(i));
        }
      }
      return shape;
    }

    // infer output shape for tensor repeats case
    c10::SmallVector<int64_t, SIZE> repeat_interleave_npu_output_size_opapi(const at::Tensor &self,
        const at::Tensor &repeats, c10::optional<int64_t> dim) {
        c10::SmallVector<int64_t, SIZE> shape;
        if (dim.has_value()) {
            int64_t real_dim = dim.value_or(0);
            real_dim = (real_dim < 0) ? (real_dim + self.dim()) : real_dim;
            for (int64_t i = 0; i < self.dim(); i++) {
                if (i == real_dim) {
                    // if repeats only has one element, size will be sum(repeats)*self.size(dim). Otherwise is sum(repeats)
                    int64_t arg = 1;
                    if (repeats.numel() == 1) {
                        arg = self.size(real_dim);
                    }
                    shape.emplace_back(arg * (repeats.sum().item()).toLong());
                }
                else
                    shape.emplace_back(self.size(i));
            }
        }
        // without dim, need flatten
        else {
            // if repeats only has one element, size will be sum(repeats) * self.numel(). Otherwise is sum(repeats)
            int64_t base = repeats.sum().item().toLong();
            if (repeats.numel() == 1) {
                base *= self.numel();
            }
            shape.emplace_back(base);
        }
        return shape;
    }

    c10::SmallVector<int64_t, SIZE> repeat_interleave_npu_output_size(
        const at::Tensor &self,
        const at::Tensor &repeats,
        int64_t dim) {
      c10::SmallVector<int64_t, SIZE> shape;
      for (int64_t i = 0; i < self.dim(); i++) {
        if (i == dim) {
          if (repeats.numel() == 1) {
            shape.emplace_back(repeats.item().toLong() * self.size(i));
          }
          else {
            shape.emplace_back(repeats.sum().item().toLong());
          }
        }
        else {
          shape.emplace_back(self.size(i));
        }
      }
      return shape;
    }

    c10::SmallVector<int64_t, SIZE> repeat_interleave_tensor_npu_output_size(const at::Tensor &repeats) {
      c10::SmallVector<int64_t, SIZE> shape;
      shape.emplace_back(repeats.sum().item().toLong());
      return shape;
    }

    c10::SmallVector<int64_t, SIZE> replication_pad2d_npu_output_size(
        const at::Tensor &self,
        c10::IntArrayRef padding) {
      int64_t N = self.dim() == 3 ? 1 : self.size(-4);
      int64_t C = self.size(-3);
      int64_t H = self.size(-2);
      int64_t W = self.size(-1);
      int64_t padding_l = 0;
      int64_t padding_r = 0;
      int64_t padding_t = 0;
      int64_t padding_b = 0;
      if (!padding.empty() && padding.size() == 1) {
        padding_l = padding[0];
        padding_r = padding[0];
        padding_t = padding[0];
        padding_b = padding[0];
      }
      else if (!padding.empty() && 4 == padding.size()) {
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
        c10::Scalar max_output_size) {
      c10::SmallVector<int64_t, SIZE> selected_indices = {max_output_size.toInt()};
      c10::SmallVector<int64_t, SIZE> valid_outputs = {};
      return std::tuple<c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>>(
          selected_indices, valid_outputs);
    }

    c10::SmallVector<int64_t, SIZE> repeat_npu_output_size(
        const at::Tensor &self,
        c10::IntArrayRef repeats) {
      int64_t num_new_dimensions = repeats.size() - self.dim();
      // Fill num_ new_ Dimensions elements with a value of 1
      c10::SmallVector<int64_t, SIZE> padded_size(num_new_dimensions, 1);
      padded_size.insert(
          padded_size.end(), self.sizes().begin(), self.sizes().end());
      c10::SmallVector<int64_t, SIZE> target_size(repeats.size());
      for (int64_t idx = 0; idx < repeats.size(); ++idx) {
        target_size[idx] = padded_size[idx] * repeats[idx];
      }
      return target_size;
    }

    c10::SmallVector<int64_t, SIZE> soft_margin_loss_npu_output_size(
        const at::Tensor &self,
        const at::Tensor &target,
        int64_t reduction) {
      c10::SmallVector<int64_t, SIZE> outputSize;
      if (reduction == at::Reduction::None) {
        outputSize = input_same_output_size(self);
      }
      else {
        outputSize = {1};
      }
      return outputSize;
    }

    c10::SmallVector<int64_t, SIZE> slow_conv_dilated2d_npu_output_size(
        const at::Tensor &input,
        const at::Tensor &weight,
        c10::IntArrayRef stride,
        c10::IntArrayRef padding,
        c10::IntArrayRef dilation) {
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
        c10::IntArrayRef dilation) {
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
        c10::IntArrayRef dilation) {
      return tuple<c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef>(self.sizes(), weight.sizes(), grad_output.sizes());
    }

    c10::IntArrayRef smooth_l1_loss_npu_output_size(
        const at::Tensor &self,
        const at::Tensor &target,
        int64_t reduction) {
      c10::IntArrayRef outputSize;
      if (reduction == at::Reduction::None) {
        outputSize = input_same_output_size(self);
      }
      return outputSize;
    }

    tuple<c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>> softmax_cross_entropy_with_logits_impl_npu_output_size(
        const at::Tensor &self) {
      c10::SmallVector<int64_t, SIZE> resultSize = array_to_small_vector(self.size(0));
      c10::SmallVector<int64_t, SIZE> backpropSize = array_to_small_vector(self.sizes());

      return tuple<c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>>(
          resultSize, backpropSize);
    }

    c10::SmallVector<int64_t, SIZE> sum_npu_output_size(
        const at::Tensor &self,
        c10::IntArrayRef dim,
        bool keepdim) {
      return reduce_ops_npu_output_size(self, dim, keepdim);
    }

    c10::SmallVector<int64_t, SIZE> topk_npu_output_size(
        const at::Tensor &self,
        int64_t k,
        int64_t dim,
        bool largest,
        bool sorted) {
      int64_t wrap_dim = make_wrap_dim(dim, self.dim());
      auto shape = array_to_small_vector(self.sizes());
      if (shape.size() > 0) {
        shape[wrap_dim] = k;
      }
      return shape;
    }

    c10::SmallVector<int64_t, SIZE> transpose_npu_output_size(
        const at::Tensor &self,
        c10::IntArrayRef perm) {
      auto sizes = self.sizes();
      c10::SmallVector<int64_t, SIZE> shape;
      for (int64_t i = 0; i < perm.size(); i++) {
        shape.emplace_back(sizes[perm[i]]);
      }

      return shape;
    }

    c10::SmallVector<int64_t, SIZE> trace_npu_output_size(const at::Tensor &self) {
      c10::SmallVector<int64_t, SIZE> shape = {1};
      return shape;
    }

    c10::IntArrayRef upsample_bicubic2d_backward_npu_output_size(c10::IntArrayRef input_size) {
      return input_size;
    }

    c10::SmallVector<int64_t, SIZE> upsample_bilinear2d_npu_output_size(
        const at::Tensor &self,
        c10::IntArrayRef output_size,
        bool align_corners,
        c10::optional<double> scales_h,
        c10::optional<double> scales_w) {
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
        c10::optional<double> scales_w) {
      return input_size;
    }

    c10::SmallVector<int64_t, SIZE> upsample_linear1d_npu_output_size(
        const at::Tensor &self,
        c10::IntArrayRef output_size,
        bool align_corners,
        c10::optional<double> scales) {
      int64_t N = self.size(0);
      int64_t C = self.size(1);
      int64_t W = output_size[0];

      c10::SmallVector<int64_t, SIZE> outputSize = {N, C, W};
      return outputSize;
    }

    c10::SmallVector<int64_t, SIZE> var_npu_output_size(
        const at::Tensor &self,
        c10::IntArrayRef dim,
        bool keepdim) {
      c10::SmallVector<int64_t, SIZE> outputSize = reduce_ops_npu_output_size(self, dim, keepdim);
      return outputSize;
    }

    c10::SmallVector<int64_t, SIZE> glu_npu_output_size(
        const at::Tensor &self,
        int64_t dim) {
      dim = make_wrap_dim(dim, self.dim());
      auto shape = array_to_small_vector(self.sizes());
      shape[dim] = shape[dim] / 2;

      return shape;
    }

    c10::SmallVector<int64_t, SIZE> crop_and_resize_npu_output_size(
        const at::Tensor &self,
        at::IntArrayRef box_index,
        at::IntArrayRef crop_size) {
      TORCH_CHECK(self.dim() == 4, "input x dim must be 4");
      TORCH_CHECK(crop_size.size() == 2, "crop_size size must be 2");
      int64_t N = box_index.size();
      int64_t H = crop_size[0];
      int64_t W = crop_size[1];
      int64_t C = self.size(1);

      c10::SmallVector<int64_t, SIZE> outputSize = {N, C, H, W};
      return outputSize;
    }

    c10::SmallVector<int64_t, SIZE> decode_jpeg_npu_output_size(
        at::IntArrayRef image_shape,
        int64_t channels) {
      TORCH_CHECK(image_shape.size() == 3, "image_shape size must be 3");
      int64_t H = image_shape[0];
      int64_t W = image_shape[1];
      int64_t C = image_shape[2];

      c10::SmallVector<int64_t, SIZE> outputSize;
      if (channels == 0) {
        outputSize = {C, H, W};
      } else {
        outputSize = {channels, H, W};
      }

      return outputSize;
    }

    c10::SmallVector<int64_t, SIZE> reflection_pad1d_npu_out_size(
        const at::Tensor& self, at::IntArrayRef padding) {
      int64_t padding_num = padding.size();
      int64_t self_num = self.dim();
      TORCH_CHECK(padding_num == 2, "padding length should be 2");
      TORCH_CHECK(self_num == 2 || self_num == 3, "self should be 2D or 3D");
      // 0, 1, -2, -1 are indexes
      int64_t padding_l = padding[0];
      int64_t padding_r = padding[1];
      int64_t C = self.size(-2);
      int64_t W = self.size(-1);
      int64_t Wo = W + padding_l + padding_r;
      c10::SmallVector<int64_t, SIZE> output_size = {C, Wo};
      // 3 is dim
      if (self_num == 3) {
        // -3 is index
        int64_t N = self.size(-3);
        output_size = {N, C, Wo};
      }
      return output_size;
    }

    c10::SmallVector<int64_t, SIZE> reflection_pad2d_npu_out_size(
        const at::Tensor& self, at::IntArrayRef padding) {
      int64_t padding_num = padding.size();
      int64_t self_num = self.dim();
      TORCH_CHECK(padding_num == 4, "padding length should be 4");
      TORCH_CHECK(self_num == 3 || self_num == 4, "self should be 3D or 4D");
      // -3, -2, -1, 0, 1, 2, 3 are indexes
      int64_t padding_l = padding[0];
      int64_t padding_r = padding[1];
      int64_t padding_t = padding[2];
      int64_t padding_b = padding[3];
      int64_t C = self.size(-3);
      int64_t H = self.size(-2);
      int64_t W = self.size(-1);
      int64_t Ho = H + padding_t + padding_b;
      int64_t Wo = W + padding_l + padding_r;
      c10::SmallVector<int64_t, SIZE> output_size = {C, Ho, Wo};
      // 4 is dim
      if (self_num == 4) {
        // -4 is index
        int64_t N = self.size(-4);
        output_size = {N, C, Ho, Wo};
      }
      return output_size;
    }

    c10::SmallVector<int64_t, SIZE> replication_pad1d_npu_out_size(
        const at::Tensor& self, at::IntArrayRef padding) {
      int64_t padding_num = padding.size();
      int64_t self_num = self.dim();
      TORCH_CHECK(padding_num == 2, "padding length should be 2");
      TORCH_CHECK(self_num == 2 || self_num == 3, "self should be 2D or 3D");
      // 0, 1, -2, -1 are indexes
      int64_t padding_l = padding[0];
      int64_t padding_r = padding[1];
      int64_t C = self.size(-2);
      int64_t W = self.size(-1);
      int64_t Wo = W + padding_l + padding_r;
      c10::SmallVector<int64_t, SIZE> output_size = {C, Wo};
      // 3 is dim
      if (self_num == 3) {
        // -3 is index
        int64_t N = self.size(-3);
        output_size = {N, C, Wo};
      }
      return output_size;
    }

    c10::SmallVector<int64_t, SIZE> replication_pad2d_npu_out_size(
        const at::Tensor& self, at::IntArrayRef padding) {
      int64_t padding_num = padding.size();
      int64_t self_num = self.dim();
      TORCH_CHECK(padding_num == 4, "padding length should be 4");
      TORCH_CHECK(self_num == 3 || self_num == 4, "self should be 3D or 4D");
      // -3, -2, -1, 0, 1, 2, 3 are indexes
      int64_t padding_l = padding[0];
      int64_t padding_r = padding[1];
      int64_t padding_t = padding[2];
      int64_t padding_b = padding[3];
      int64_t C = self.size(-3);
      int64_t H = self.size(-2);
      int64_t W = self.size(-1);
      int64_t Ho = H + padding_t + padding_b;
      int64_t Wo = W + padding_l + padding_r;
      c10::SmallVector<int64_t, SIZE> output_size = {C, Ho, Wo};
      // 4 is dim
      if (self_num == 4) {
        // -4 is index
        int64_t N = self.size(-4);
        output_size = {N, C, Ho, Wo};
      }
      return output_size;
    }

    c10::SmallVector<int64_t, SIZE> clamp_npu_output_size(
        const at::Tensor& self,
        const c10::optional<at::Tensor>& min,
        const c10::optional<at::Tensor>& max) {
      TORCH_CHECK(min.has_value() || max.has_value(), "torch.clamp: At least one of 'min' or 'max' must not be None");
      if (self.numel() == 0) {
        c10::SmallVector<int64_t, SIZE> empty_sizes;
        for (int64_t i = 0; i < self.dim(); ++i) {
          empty_sizes.push_back(self.size(i));
        }
        return empty_sizes;
      }
      if (min.has_value() && max.has_value()) {
        auto brc_shape_min = broadcast_ops_npu_output_size(self.sizes(), min.value().sizes());
        return broadcast_ops_npu_output_size(brc_shape_min, max.value().sizes());
      }
      if (min.has_value()) {
        return broadcast_ops_npu_output_size(self.sizes(), min.value().sizes());
      }
      return broadcast_ops_npu_output_size(self.sizes(), max.value().sizes());
    }

    c10::SmallVector<int64_t, SIZE> smooth_l1_loss_backward_npu_output_size(
        const at::Tensor &self,
        const at::Tensor &target,
        const at::Tensor &grad_output) {
      auto mid_shape = broadcast_ops_npu_output_size(self.sizes(), target.sizes());
      return broadcast_ops_npu_output_size(mid_shape, grad_output.sizes());
    }

    c10::SmallVector<int64_t, SIZE> max_pool2d_out_size(
        const at::Tensor &self,
        at::IntArrayRef output_size) {
      auto shape = array_to_small_vector(self.sizes());
      if ((self.dim() == 3 || self.dim() == 4) && output_size.size() == 2) {
        shape[shape.size() - 2] = output_size[0];
        shape[shape.size() - 1] = output_size[1];
      }
      return shape;
    }

} // namespace native
} // namespace at_npu
