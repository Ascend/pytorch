import torch
import torch.distributed as dist
from torch.autograd.function import Function

import torch_npu


class SyncBatchNorm(Function):

    @staticmethod
    def forward(self, input_tensor, weight, bias, running_mean, running_var, eps, momentum, process_group, world_size):
        input_tensor = input_tensor.contiguous()
        input_shape = input_tensor.shape
        input_tensor_ = input_tensor.reshape(input_shape[0], input_shape[1], 1, -1)
        # calculate sum/sum_square for input.
        sum_val, sum_square_val = torch_npu.batch_norm_reduce(input_tensor_, eps)

        count = torch.full((1,),
                           input_tensor.numel() // input_tensor.size(1),
                           dtype=sum_val.dtype,
                           device=sum_val.device)

        num_channels = input_tensor.shape[1]
        # C, C, 1 -> (2C + 1)
        combined = torch.cat([sum_val, sum_square_val, count], dim=0)
        # world_size * (2C + 1)
        combined_list = [torch.empty_like(combined) for k in range(world_size)]
        dist.all_gather(combined_list, combined, process_group, async_op=False)
        combined = torch.stack(combined_list, dim=0)
        # world_size * (2C + 1) -> world_size * C, world_size * C, world_size * 1
        sum_all, square_sum_all, count_all = torch.split(combined, num_channels, dim=1)

        size = count_all.view(-1).sum()
        if size == 1:
            raise ValueError('Expected more than 1 value per channel when training, got input size {}'.format(size))

        # calculate global mean & invstd
        mean, invstd = torch_npu.batch_norm_gather_stats_update(input_tensor,
                                                                sum_all,
                                                                square_sum_all,
                                                                running_mean,
                                                                running_var,
                                                                momentum,
                                                                eps,
                                                                count_all.view(-1))

        self.save_for_backward(input_tensor, weight, mean, invstd, count_all)
        self.process_group = process_group

        # apply element-wise normalization
        out = torch.batch_norm_elemt(input_tensor, weight, bias, mean, invstd, eps)
        return out

    @staticmethod
    def backward(self, grad_output):
        if not grad_output.is_contiguous(memory_format=torch.channels_last):
            grad_output = grad_output.contiguous()
        saved_input, weight, mean, invstd, count_tensor = self.saved_tensors
        grad_input = grad_weight = grad_bias = None
        process_group = self.process_group

        # calculate local stats as well as grad_weight / grad_bias
        sum_dy, sum_dy_xmu, grad_weight, grad_bias = torch.batch_norm_backward_reduce(grad_output,
                                                                                      saved_input,
                                                                                      mean,
                                                                                      invstd,
                                                                                      weight,
                                                                                      self.needs_input_grad[0],
                                                                                      self.needs_input_grad[1],
                                                                                      self.needs_input_grad[2])

        if self.needs_input_grad[0]:
            # synchronizing stats used to calculate input gradient.
            num_channels = sum_dy.shape[0]
            combined = torch.cat([sum_dy, sum_dy_xmu], dim=0)
            torch.distributed.all_reduce(
                combined, torch.distributed.ReduceOp.SUM, process_group, async_op=False)
            sum_dy, sum_dy_xmu = torch.split(combined, num_channels)

            # backward pass for gradient calculation
            grad_input = torch.batch_norm_backward_elemt(grad_output,
                                                         saved_input,
                                                         mean,
                                                         invstd,
                                                         weight,
                                                         sum_dy,
                                                         sum_dy_xmu,
                                                         count_tensor)

        # synchronizing of grad_weight / grad_bias is not needed as distributed
        # training would handle all reduce.
        if weight is None or not self.needs_input_grad[1]:
            grad_weight = None

        if weight is None or not self.needs_input_grad[2]:
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None
