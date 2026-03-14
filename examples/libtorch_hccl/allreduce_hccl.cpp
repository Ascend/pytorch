#include <iostream>
#include <cstdlib>
#include <string>
#include <c10/util/irange.h>
#include <torch/csrc/distributed/c10d/FileStore.hpp>
#include <torch_npu/csrc/distributed/ProcessGroupHCCL.hpp>
#include <torch/torch.h>
#include "torch_npu/torch_npu.h"

using namespace c10d;
using namespace c10d_npu;

const int g_rank = []() {
    const char* env = std::getenv("RANK");
    return env ? std::stoi(std::string(env)) : 0;
}();

const int g_size = []() {
    const char* env = std::getenv("SIZE");
    return env ? std::stoi(std::string(env)) : 1;
}();

int main(int argc, char** argv)
{
    int rank = g_rank;
    int size = g_size;
    
    std::cout << "启动 HCCL allreduce 示例: rank=" << rank << ", size=" << size << std::endl;
    
    // 初始化NPU设备 - 使用npu字符串格式
    std::string device_str = "npu:" + std::to_string(rank);
    torch_npu::init_npu(device_str);
    std::cout << "NPU设备 " << rank << " 初始化完成" << std::endl;
    
    // 创建FileStore用于进程间通信协调
    auto store = c10::make_intrusive<FileStore>("/tmp/c10d_hccl_example", size);
    
    // 创建ProcessGroupHCCL选项
    auto options = ProcessGroupHCCL::Options::create();
    
    // 创建ProcessGroupHCCL实例
    auto pg = c10::make_intrusive<ProcessGroupHCCL>(store, rank, size, options);
    
    // 通过传NPU字符串构造NPU设备
    auto device = at::Device(device_str);
    
    // 创建10个张量用于测试
    const auto ntensors = 10;
    std::vector<at::Tensor> tensors;
    
    for (const auto i : c10::irange(ntensors)) {
        // 在NPU设备上创建全1张量
        auto x = at::ones({1000, 16 * (i + 1)}, at::TensorOptions(device).dtype(at::kFloat));
        tensors.push_back(x);
    }
    
    std::cout << "在NPU设备 " << rank << " 上创建了 " << ntensors << " 个张量" << std::endl;
    
    // 提交所有allreduce操作
    std::vector<c10::intrusive_ptr<Work>> pending;
    for (const auto i : c10::irange(ntensors)) {
        std::vector<at::Tensor> tmp = {tensors[i]};
        pending.push_back(pg->allreduce(tmp));
    }
    
    std::cout << "已提交 " << ntensors << " 个allreduce操作" << std::endl;
    
    // 等待所有操作完成
    for (auto& work : pending) {
        work->wait();
    }
    
    std::cout << "所有操作已完成!" << std::endl;
    
    // 验证结果 - 打印前3个张量的第一个元素
    for (const auto i : c10::irange(std::min(ntensors, 3))) {
        auto cpu_tensor = tensors[i].to(at::kCPU);
        std::cout << "张量 " << i << " 第一个元素: " << cpu_tensor.data_ptr<float>()[0] << std::endl;
    }
    
    std::cout << "HCCL allreduce示例运行成功!" << std::endl;
    
    // 使用NPU设备结束需进行反初始化
    torch_npu::finalize_npu();
    
    return 0;
}
