#pragma once
#include <thread>
#include <memory>
#include <string>
#include <condition_variable>
#include "StoreClient.hpp"
#include "StoreMessagePacker.hpp"
#include "ParallelTcpServer.hpp"

namespace c10d {
namespace torch_npu {
class ParallelStoreServer;
class Proxy {
public:
    explicit Proxy(const std::string& localSocketPath, const std::string& host, uint16_t port,
        const std::chrono::milliseconds timeout) noexcept;
    void Start() noexcept;
    void Stop() noexcept;
    int SyncCall(const torch_npu::StoreMessage &request, torch_npu::StoreMessage &response) noexcept;
    StoreMessage HandleLocalServerMessage(const int &fd, const torch_npu::StoreMessage &message) noexcept;
    void WriteData(const int &fd, std::vector<uint8_t> &buf, int64_t &unpackSize) noexcept;
    int LoopProcessData() noexcept;
    int SetReceiveTimeout(const std::chrono::milliseconds &value) const noexcept;

private:
    const std::string host_{};
    const uint16_t port_{ 0 };
    bool running_ { false };
    std::mutex proxyMutex_;
    std::mutex localMutex_;
    std::condition_variable localWaitCond_;
    torch_npu::StoreMessage proxyMsg_;
    std::thread processThread_;
    std::unique_ptr<torch_npu::ParallelStoreServer> localServer_;
    std::unique_ptr<torch_npu::Client> tcpClient_;
};
} // torch_npu
} // c10d