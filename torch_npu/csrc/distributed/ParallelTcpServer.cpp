/* *
 * @copyright Copyright (c) 2024 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the BSD 3-Clause License  (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/BSD-3-Clause
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <sys/socket.h>
#include <sys/epoll.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <netinet/in.h>
#include <unistd.h>
#include <fcntl.h>
#include <arpa/inet.h>
#include "c10/util/Logging.h"
#include "torch_npu/csrc/framework/utils/NpuUtils.h"
#include "ParallelTcpServer.hpp"

namespace c10d {
namespace torch_npu {
static constexpr uint32_t MAX_EVENT_COUNT = 128;
static constexpr uint32_t BUFFER_LOW_LEVEL = 32;
static constexpr uint32_t BUFFER_EXPEND_SIZE = 256;

void ClientIoContext::ReceiveData() noexcept
{
    bool finished = false;
    while (!finished) {
        if (recBuf_.size() - recSize_ < BUFFER_LOW_LEVEL) {
            recBuf_.resize(recBuf_.size() + BUFFER_EXPEND_SIZE);
        }

        auto count = read(fd_, recBuf_.data() + recSize_, recBuf_.size() - recSize_);
        if (count <= 0) {
            finished = true;
            continue;
        }

        recSize_ += static_cast<uint32_t>(count);
    }

    recBuf_.resize(recSize_);

    StoreMessage request;
    long used;
    while ((used = StoreMessagePacker::Unpack(recBuf_, request)) > 0) {
        requests_.emplace_back(std::move(request));
        recBuf_.erase(recBuf_.begin(), recBuf_.begin() + used);
        recSize_ -= static_cast<uint32_t>(used);
    }
}

bool ClientIoContext::HasNextReq() const noexcept
{
    return !requests_.empty();
}

StoreMessage ClientIoContext::NextRequest() noexcept
{
    StoreMessage request;
    if (requests_.empty()) {
        return request;
    }

    request = std::move(requests_.front());
    requests_.pop_front();
    return request;
}

void ClientIoContext::SendResponse(const StoreMessage &response) noexcept
{
    auto buf = StoreMessagePacker::Pack(response);
    sendBuf_.insert(sendBuf_.end(), buf.begin(), buf.end());
    FlushSendBuf();
}

bool ClientIoContext::SendBufEmpty() const noexcept
{
    return sendBuf_.empty();
}
void ClientIoContext::FlushSendBuf() noexcept
{
    uint32_t offset = 0;
    while (!sendBuf_.empty()) {
        auto ret = write(fd_, sendBuf_.data() + offset, sendBuf_.size() - offset);
        if (ret <= 0) {
            break;
        }

        offset += static_cast<uint32_t>(ret);
    }

    if (offset > 0) {
        sendBuf_.erase(sendBuf_.begin(), sendBuf_.begin() + offset);
    }
}


ParallelTcpServer::ParallelTcpServer(
    uint32_t threadNum,
    const std::string host,
    uint16_t port,
    uint32_t listenThreadNum,
    ServerProcFn process) noexcept
    : threadNum_{ std::max(4U, threadNum) },
      host_{ host },
      port_{ port },
      listenThreadNum_{ listenThreadNum },
      process_{ std::move(process) }
{}

ParallelTcpServer::ParallelTcpServer(
    uint32_t threadNum,
    const std::string localSocketPath,
    uint32_t listenThreadNum,
    ServerProcFn process) noexcept
    : threadNum_{ std::max(4U, threadNum) },
      localSocketPath_{ localSocketPath },
      listenThreadNum_ { listenThreadNum },
      process_{ std::move(process) }
{
    isLocalServer_ = true;
}


int ParallelTcpServer::Start() noexcept
{
    buffer_ = new (std::nothrow) uint8_t[4096];
    if (buffer_ == nullptr) {
        LOG(ERROR) << "allocate buffer failed.";
        return -1;
    }

    if (isLocalServer_) {
        listenSocket_ = CreateLocalSocket(localSocketPath_);
    } else {
        listenSocket_ = CreateSocket(host_, port_);
    }
    if (listenSocket_ < 0) {
        delete[] buffer_;
        buffer_ = nullptr;
        return -1;
    }

    running_ = true;
    epClientFds_.reserve(threadNum_);
    clientThreads_.reserve(threadNum_);
    listenThreads_.reserve(listenThreadNum_);
    auto initializeFailed = false;
    for (auto i = 0U; i < threadNum_; i++) {
        auto clientEpFd = CreateEpoll();
        if (clientEpFd < 0) {
            LOG(ERROR) << "create new client epoll fd for index: " << i << " failed.";
            initializeFailed = true;
            break;
        }
        epClientFds_.emplace_back(clientEpFd);
        clientThreads_.emplace_back([clientEpFd](ParallelTcpServer *server) { server->LoopProcessClients(clientEpFd); },
            this);
    }
    for (auto j = 0U; j < listenThreadNum_; j++) {
        listenThreads_.emplace_back([](ParallelTcpServer *server) { server->ProcessListenEvent(); }, this);
    }
    if (initializeFailed) {
        Stop();
        return -1;
    }

    return 0;
}

void ParallelTcpServer::Stop() noexcept
{
    running_ = false;
    for (auto &th : clientThreads_) {
        th.join();
    }

    for (auto &fd : epClientFds_) {
        close(fd);
    }

    for (auto &th : listenThreads_) {
        th.join();
    }
    close(listenSocket_);
    listenSocket_ = -1;

    delete[] buffer_;
    buffer_ = nullptr;
}

void ParallelTcpServer::WakeupWaitingClients(const std::string &key) noexcept
{
    std::list<PI> stopWaitingSockets;
    std::unique_lock<SpinLock> lockGuard{ spinLock_ };
    auto pos = keyWaitingSockets_.find(key);
    if (pos == keyWaitingSockets_.end()) {
        return;
    }

    for (auto it : std::as_const(pos->second)) {
        if (--socketWaitKeyNum_[it] <= 0) {
            stopWaitingSockets.emplace_back(it);
            socketWaitKeyNum_.erase(it);
        }
    }

    keyWaitingSockets_.erase(key);
    lockGuard.unlock();

    std::vector<uint8_t> body{ static_cast<uint8_t>(MessageWaitKeyRes::KEYS_STOP_WAITING) };
    for (auto [socket, workerFd] : stopWaitingSockets) {
        StoreMessage response{ MessageType::WAIT, workerFd, body };
        auto buf = StoreMessagePacker::Pack(response);
        write(socket, buf.data(), buf.size());
    }
}

int ParallelTcpServer::CreateSocket(const std::string host, uint16_t port) noexcept
{
    auto sockFd = CreateSocketWithFamily(host, port, AF_INET);
    if (sockFd >= 0) {
        return sockFd;
    }
    
    sockFd = CreateSocketWithFamily(host, port, AF_INET6);
    if (sockFd >= 0) {
        return sockFd;
    }
    return -1;
}

int ParallelTcpServer::CreateSocketWithFamily(const std::string host, uint16_t port, int family) noexcept
{
    struct addrinfo hints = {0};
    hints.ai_family = family;
    hints.ai_socktype = SOCK_STREAM;

    ::addrinfo* result = nullptr;
    int r = ::getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &result);
    if (r != 0) {
        LOG(ERROR) << "getaddrinfo failed " << errno << " : " << strerror(errno);
        return -1;
    }

    for (::addrinfo* addr = result; addr != nullptr; addr = addr->ai_next) {
        int sockFd = CreateSocketAndListen(*addr);
        if (sockFd >= 0) {
            return sockFd;
        }
    }
    return -1;
}

int ParallelTcpServer::CreateSocketAndListen(const ::addrinfo &addr) noexcept
{
    auto sockFd = ::socket(addr.ai_family, addr.ai_socktype, addr.ai_protocol);
    if (sockFd < 0) {
        LOG(ERROR) << "create server socket fd failed " << errno << " : " << strerror(errno);
        return -1;
    }

    auto ret = ::bind(sockFd, addr.ai_addr, addr.ai_addrlen);
    if (ret != 0) {
        LOG(ERROR) << "bind server socket fd failed " << errno << " : " << strerror(errno);
        close(sockFd);
        return -1;
    }

    ret = listen(sockFd, MAX_EVENT_COUNT);
    if (ret != 0) {
        LOG(ERROR) << "listen server socket fd failed " << errno << " : " << strerror(errno);
        close(sockFd);
        return -1;
    }

    ret = SetBlockSocketTimeout(sockFd);
    if (ret != 0) {
        close(sockFd);
        return -1;
    }
    return sockFd;
}

int ParallelTcpServer::CreateLocalSocket(const std::string &localSocketPath) noexcept
{
    if (localSocketPath.empty()) {
        LOG(ERROR) << "local socket path invalid." << errno << " : " << strerror(errno);
        return -1;
    }
    
    struct sockaddr_un servAddr {};
    servAddr.sun_family = AF_UNIX;
    servAddr.sun_path[0] = '\0';
    strncpy(servAddr.sun_path + 1, localSocketPath.c_str(), sizeof(servAddr.sun_path) - 2);
    servAddr.sun_path[sizeof(servAddr.sun_path) - 1] = '\0';
    auto sockFd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (sockFd < 0) {
        LOG(ERROR) << "create local  socket fd failed " << errno << " : " << strerror(errno);
        return -1;
    }

    auto ret = ::bind(sockFd, reinterpret_cast<struct sockaddr *>(&servAddr), sizeof(servAddr));
    if (ret != 0) {
        LOG(ERROR) << "bind local socket fd failed " << errno << " : " << strerror(errno);
        close(sockFd);
        return -1;
    }

    if (!at_npu::native::NpuUtils::setFilePermissions(sockFd, S_IRUSR | S_IWUSR | S_IRGRP)) {
        close(sockFd);
        return -1;
    }

    ret = listen(sockFd, MAX_EVENT_COUNT);
    if (ret != 0) {
        LOG(ERROR) << "listen local socket fd failed " << errno << " : " << strerror(errno);
        close(sockFd);
        return -1;
    }

    ret = SetBlockSocketTimeout(sockFd);
    if (ret != 0) {
        close(sockFd);
        return -1;
    }
    return sockFd;
}

int ParallelTcpServer::CreateEpoll(int targetFd) noexcept
{
    auto fd = epoll_create(1);
    if (fd < 0) {
        LOG(ERROR) << "create new epoll fd failed " << errno << " : " << strerror(errno);
        return -1;
    }

    if (targetFd < 0) {
        return fd;
    }

    struct epoll_event ev {};
    ev.events = EPOLLIN | EPOLLOUT | EPOLLET;
    ev.data.fd = targetFd;
    auto ret = epoll_ctl(fd, EPOLL_CTL_ADD, targetFd, &ev);
    if (ret != 0) {
        LOG(ERROR) << "add server socket to epoll failed " << errno << " : " << strerror(errno);
        close(fd);
        return -1;
    }

    return fd;
}

int ParallelTcpServer::SetNonBlocking(int fd) noexcept
{
    auto old = fcntl(fd, F_GETFL, 0);
    if (old < 0) {
        LOG(ERROR) << "get fd flags failed " << errno << " : " << strerror(errno);
        return -1;
    }

    auto ret = fcntl(fd, F_SETFL, static_cast<int>(old) | O_NONBLOCK);
    if (ret != 0) {
        LOG(ERROR) << "set fd flags failed " << errno << " : " << strerror(errno);
        return -1;
    }

    return 0;
}

int ParallelTcpServer::SetBlockSocketTimeout(int fd) noexcept
{
    struct timeval timeout {6, 0};
    if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<char *>(&timeout), sizeof(struct timeval)) != 0) {
        LOG(ERROR) << "set block accept timeout failed " << errno << " : " << strerror(errno);
        return -1;
    }
    
    return 0;
}

void ParallelTcpServer::LoopProcessClients(int epollFd) noexcept
{
    int count;
    struct epoll_event events[MAX_EVENT_COUNT];
    std::unordered_map<int, ClientIoContext> clientCtx;
    while (running_) {
        count = epoll_wait(epollFd, events, MAX_EVENT_COUNT, 1000);
        if (count < 0) {
            LOG(ERROR) << "epoll wait failed " << errno << " : " << strerror(errno);
            continue;
        }

        for (auto i = 0; i < count; i++) {
            ProcessClientEvent(epollFd, events[i].data.fd, events[i].events, clientCtx);
        }
    }

    for (auto &ctx : clientCtx) {
        close(ctx.first);
    }
}

void ParallelTcpServer::ProcessListenEvent() noexcept
{
    int connFd;
    socklen_t sockLen;
    struct sockaddr_in cliAddr {};

    while (running_) {
        sockLen = sizeof(cliAddr);
        connFd = accept(listenSocket_, reinterpret_cast<struct sockaddr *>(&cliAddr), &sockLen);
        if (connFd < 0) {
            if (errno != EAGAIN && errno != EWOULDBLOCK) {
                LOG(ERROR) << "accept new fd failed " << errno << " : " << strerror(errno);
            }
            continue;
        }

        auto ret = SetNonBlocking(connFd);
        if (ret != 0) {
            LOG(ERROR) << "set connection fd non-blocking failed " << errno << " : " << strerror(errno);
            close(connFd);
            continue;
        }

        struct epoll_event ev {};
        ev.events = EPOLLIN | EPOLLOUT | EPOLLRDHUP | EPOLLHUP;
        ev.data.fd = connFd;
        auto index = static_cast<uint32_t>(connFd) % threadNum_;
        ret = epoll_ctl(epClientFds_[index], EPOLL_CTL_ADD, connFd, &ev);
        if (ret != 0) {
            LOG(ERROR) << "add connection to epoll failed " << errno << " : " << strerror(errno);
            close(connFd);
            continue;
        }
    }
}

void ParallelTcpServer::ProcessClientEvent(int epFd, int fd, uint32_t event,
    std::unordered_map<int, ClientIoContext> &ctx) noexcept
{
    if ((event & (EPOLLRDHUP | EPOLLHUP)) != 0) {
        epoll_ctl(epFd, EPOLL_CTL_DEL, fd, nullptr);
        close(fd);
        fd = -1;
        ctx.erase(fd);
        return;
    }

    auto pos = ctx.find(fd);
    if (pos == ctx.end()) {
        pos = ctx.emplace(fd, ClientIoContext{ fd, EPOLLIN | EPOLLOUT | EPOLLRDHUP | EPOLLHUP }).first;
    }

    auto setEvents = pos->second.currentEvents_;
    if ((event & EPOLLIN) != 0) {
        pos->second.ReceiveData();
        while (pos->second.HasNextReq()) {
            auto response = process_(fd, pos->second.NextRequest());
            if (response.mt != MessageType::SKIP_MSG) {
                pos->second.SendResponse(response);
            }
        }

        if (pos->second.SendBufEmpty()) {
            setEvents = EPOLLIN | EPOLLRDHUP | EPOLLHUP;
        } else {
            setEvents = EPOLLIN | EPOLLOUT | EPOLLRDHUP | EPOLLHUP;
        }
    }

    if ((event & EPOLLOUT) != 0) {
        pos->second.FlushSendBuf();
        setEvents = EPOLLIN | EPOLLRDHUP | EPOLLHUP;
    }

    if (setEvents != pos->second.currentEvents_) {
        struct epoll_event ev {};
        ev.events = pos->second.currentEvents_ = setEvents;
        ev.data.fd = fd;
        auto ret = epoll_ctl(epFd, EPOLL_CTL_MOD, fd, &ev);
        if (ret != 0) {
            LOG(ERROR) << "modify wait out on connection to epoll failed " << errno << " : " << strerror(errno);
            close(fd);
            ctx.erase(pos);
            return;
        }
    }
}
} // torch_npu
} // c10d