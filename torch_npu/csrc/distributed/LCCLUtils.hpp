#pragma once

#include <map>
#include <ATen/ATen.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include "torch_npu/csrc/core/npu/interface/LcclInterface.h"

namespace c10d_npu {

at_npu::lccl::LcclDataType getLcclDataType(at::ScalarType type);

std::string getLcclDataTypeSerialString(at_npu::lccl::LcclDataType type);

void checkSupportedDataType(at_npu::lccl::LcclDataType type, std::string functionName);

at_npu::lccl::LcclReduceOp getLcclReduceOp(const c10d::ReduceOp reduceOp, at::Tensor& input);

uint64_t getNumelForLCCL(const at::Tensor& self);

std::string getKeyFromDevices(const std::vector<at::Device>& devices);

std::vector<at::Device> getDeviceList(const std::vector<at::Tensor>& tensors);

void checkTensors(const std::vector<at::Tensor>& tensors);

bool CheckTensorsSameSize(const std::vector<at::Tensor>& input_tensors);

std::vector<at::Tensor> castOriginFormat(const std::vector<at::Tensor>& inputTensors);

std::vector<at::Tensor> FlattenForScatterGather(std::vector<std::vector<at::Tensor>>& tensor_lists,
    std::vector<at::Tensor>& other, size_t world_size);

}
