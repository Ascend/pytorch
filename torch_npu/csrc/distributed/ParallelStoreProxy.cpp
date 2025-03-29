#include <unistd.h>
#include "ParallelTcpStore.hpp"
#include "StoreClient.hpp"
#include "c10/util/Exception.h"
#include <stdexcept>
#include "StoreMessagePacker.hpp"
#include "ParallelTcpServer.hpp"
#include "ParallelStoreProxy.hpp"

namespace c10d {
namespace torch_npu {
Proxy::Proxy(const std::string &localSocketPath, const std::string &host, uint16_t port,
    std::chrono::milliseconds timeout) noexcept
    : localServer_{ std::make_unique<torch_npu::ParallelStoreServer>(localSocketPath,
    [this](const int &fd, const StoreMessage &msg) { return this->HandleLocalServerMessage(fd, msg); }) },
      tcpClient_{ std::make_unique<torch_npu::Client>(host, port, timeout) },
      host_{ host },
      port_{ port }
{}

void Proxy::Start() noexcept
{
    {
        std::unique_lock<std::mutex> localGuard{ localMutex_ };
        running_ = true;
    }
    if (tcpClient_->Connect() != 0) {
        throw std::runtime_error("Failed to connect to TCP server");
    }
    processThread_ = std::thread([this]() { LoopProcessData(); });
}

void Proxy::Stop() noexcept
{
    {
        std::unique_lock<std::mutex> localGuard{ localMutex_ };
        running_ = false;
    }
    tcpClient_->Close();
    processThread_.join();
}

int Proxy::SyncCall(const torch_npu::StoreMessage &request, torch_npu::StoreMessage &response) noexcept
{
    std::unique_lock<std::mutex> lockGuard{ localMutex_ };
    HandleLocalServerMessage(-1, request);
    do {
        localWaitCond_.wait(lockGuard);
        response = proxyMsg_;
    } while (request.mt != response.mt);

    return 0;
}

StoreMessage Proxy::HandleLocalServerMessage(const int &fd, const torch_npu::StoreMessage &message) noexcept
{
    std::lock_guard<std::mutex> localGuard{ proxyMutex_ };
    StoreMessage response;
    StoreMessage messageFd = message;
    messageFd.fd = fd;

    auto packedRequest = StoreMessagePacker::Pack(messageFd);
    auto ret = write(tcpClient_->GetSocketFd(), packedRequest.data(), packedRequest.size());
    if (ret < 0) {
        LOG(ERROR) << "write data to server (" << host_ << ":" << port_ << ") failed" << errno << ":" <<
            strerror(errno);
    }
    response.mt = MessageType::SKIP_MSG;
    return response;
}

void Proxy::WriteData(const int &fd, std::vector<uint8_t> &buf, int64_t &unpackSize) noexcept
{
    uint32_t offset = 0;
    while (offset < unpackSize) {
        auto ret = write(fd, buf.data() + offset, unpackSize - offset);
        if (ret <= 0) {
            LOG(ERROR) << "proxy write buf data failed. fd: " << fd << "host:" << host_ << "port:" << port_ << "ret:" <<
                ret << errno << ":" << strerror(errno);
            break;
        }
        offset += static_cast<uint32_t>(ret);
    }
}

int Proxy::LoopProcessData() noexcept
{
    uint8_t buffer[READ_BUF_SZ];
    std::vector<uint8_t> responseBuf;

    int result = 0;
    while (running_) {
        result = 0;
        while (!StoreMessagePacker::Full(responseBuf)) {
            auto ret = read(tcpClient_->GetSocketFd(), buffer, READ_BUF_SZ);
            if (ret < 0) {
                if (errno == EINTR) { // interrupted by signal
                    continue;
                }
                if (running_) {
                    LOG(ERROR) << "proxy read data from server failed. fd: " << tcpClient_->GetSocketFd() << "host:" <<
                        host_ << "port:" << port_ << "errno:" << errno << ":" << strerror(errno);
                }
                result = -1;
                break;
            }
            responseBuf.insert(responseBuf.end(), buffer, buffer + ret);
        }

        if (result < 0) {
            continue;
        }
        StoreMessage response;
        auto unpackRet = StoreMessagePacker::Unpack(responseBuf, response);
        if (unpackRet < 0L) {
            LOG(ERROR) << "proxy unpack message failed. fd:" << tcpClient_->GetSocketFd() << "host:" << host_ <<
                "port:" << port_ << "errno:" << errno << ":" << strerror(errno);
            continue;
        }
        if (response.fd < 0) {
            std::unique_lock<std::mutex> lockGuard{ localMutex_ };
            proxyMsg_ = response;
            localWaitCond_.notify_all();
        } else {
            WriteData(response.fd, responseBuf, unpackRet);
        }
        responseBuf.erase(responseBuf.begin(), responseBuf.begin() + unpackRet);
    }

    return result;
}

int Proxy::SetReceiveTimeout(const std::chrono::milliseconds &value) const noexcept
{
    return tcpClient_->SetReceiveTimeout(value);
}
} // torch_npu
} // c10d