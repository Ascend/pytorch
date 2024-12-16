/**
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
#include <netinet/in.h>
#include <unistd.h>
#include <fcntl.h>
#include <arpa/inet.h>
#include "c10/util/Logging.h"
#include "ParallelTcpServer.hpp"

namespace c10d {
namespace pta {
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
    ServerProcFn process) noexcept
    : threadNum_{ std::max(4U, threadNum) },
      host_{ host },
      port_{ port },
      process_{ std::move(process) }
{}

int ParallelTcpServer::Start() noexcept
{
    buffer_ = new (std::nothrow) uint8_t[4096];
    if (buffer_ == nullptr) {
        LOG(ERROR) << "allocate buffer failed.";
        return -1;
    }

    listenSocket_ = CreateSocket(host_, port_);
    if (listenSocket_ < 0) {
        delete[] buffer_;
        buffer_ = nullptr;
        return -1;
    }

    epCtlFd_ = CreateEpoll(listenSocket_);
    if (epCtlFd_ < 0) {
        close(listenSocket_);
        listenSocket_ = -1;
        delete[] buffer_;
        buffer_ = nullptr;
        return -1;
    }

    running_ = true;
    epClientFds_.reserve(threadNum_);
    clientThreads_.reserve(threadNum_);
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

    ctlThread_ = std::thread{ [](ParallelTcpServer *server) { server->LoopProcessListenFd(); }, this };
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

    ctlThread_.join();
    close(epCtlFd_);
    epCtlFd_ = -1;

    close(listenSocket_);
    listenSocket_ = -1;

    delete[] buffer_;
    buffer_ = nullptr;
}

void ParallelTcpServer::WakeupWaitingClients(const std::string &key) noexcept
{
    std::list<int> stopWaitingSockets;
    std::unique_lock<SpinLock> lockGuard{ spinLock_ };
    auto pos = keyWaitingSockets_.find(key);
    if (pos == keyWaitingSockets_.end()) {
        return;
    }

    for (auto socket : pos->second) {
        if (--socketWaitKeyNum_[socket] <= 0) {
            stopWaitingSockets.emplace_back(socket);
            socketWaitKeyNum_.erase(socket);
        }
    }

    keyWaitingSockets_.erase(key);
    lockGuard.unlock();

    std::vector<uint8_t> body{static_cast<uint8_t>(MessageWaitKeyRes::KEYS_STOP_WAITING)};
    StoreMessage response{MessageType::WAIT, body};
    auto buf = StoreMessagePacker::Pack(response);
    for (auto socket : stopWaitingSockets) {
        write(socket, buf.data(), buf.size());
    }
}

int ParallelTcpServer::CreateSocket(const std::string host, uint16_t port) noexcept
{
    struct sockaddr_in servAddr {};
    servAddr.sin_family = AF_INET;
    servAddr.sin_addr.s_addr = inet_addr(host.c_str());
    servAddr.sin_port = htons(port);

    auto sockFd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (sockFd < 0) {
        LOG(ERROR) << "create server socket fd failed " << errno << " : " << strerror(errno);
        return -1;
    }

    auto ret = ::bind(sockFd, reinterpret_cast<struct sockaddr *>(&servAddr), sizeof(servAddr));
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

    if (SetNonBlocking(sockFd) != 0) {
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

    auto ret = fcntl(fd, F_SETFL, old | O_NONBLOCK);
    if (ret != 0) {
        LOG(ERROR) << "set fd flags failed " << errno << " : " << strerror(errno);
        return -1;
    }

    return 0;
}

void ParallelTcpServer::LoopProcessListenFd() noexcept
{
    int count;
    struct epoll_event events[MAX_EVENT_COUNT];
    while (running_) {
        count = epoll_wait(epCtlFd_, events, MAX_EVENT_COUNT, 1000);
        if (count < 0) {
            LOG(ERROR) << "epoll wait failed " << errno << " : " << strerror(errno);
            continue;
        }

        for (auto i = 0; i < count; i++) {
            if (events[i].data.fd == listenSocket_) {
                ProcessListenEvent(events[i].events);
                break;
            }
        }
    }
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

void ParallelTcpServer::ProcessListenEvent(uint32_t event) noexcept
{
    int connFd;
    socklen_t sockLen;
    struct sockaddr_in cliAddr {};

    while (running_) {
        sockLen = sizeof(cliAddr);
        connFd = accept(listenSocket_, reinterpret_cast<struct sockaddr *>(&cliAddr), &sockLen);
        if (connFd < 0) {
            break;
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
    if (event & (EPOLLRDHUP | EPOLLHUP)) {
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
    if (event & EPOLLIN) {
        pos->second.ReceiveData();
        while (pos->second.HasNextReq()) {
            auto response = process_(fd, pos->second.NextRequest());
            pos->second.SendResponse(response);
        }

        if (pos->second.SendBufEmpty()) {
            setEvents = EPOLLIN | EPOLLRDHUP | EPOLLHUP;
        } else {
            setEvents = EPOLLIN | EPOLLOUT | EPOLLRDHUP | EPOLLHUP;
        }
    }

    if (event & EPOLLOUT) {
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
} // pta
} // c10d