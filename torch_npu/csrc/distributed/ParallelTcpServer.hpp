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
#pragma once
#include <cstdint>
#include <map>
#include <list>
#include <mutex>
#include <vector>
#include <thread>
#include <atomic>
#include <unordered_map>
#include <functional>
#include <netdb.h>

#include "StoreMessagePacker.hpp"

namespace c10d {
namespace torch_npu {
using PI = std::pair<int, int>;
/* *
 * @brief wrapper for pthread_spinlock_t
 */
class SpinLock {
public:
    SpinLock() noexcept
    {
        pthread_spin_init(&spinlock_, 0);
    }

    virtual ~SpinLock() noexcept
    {
        pthread_spin_destroy(&spinlock_);
    }

    void lock() noexcept
    {
        pthread_spin_lock(&spinlock_);
    }

    bool try_lock() noexcept
    {
        return pthread_spin_trylock(&spinlock_) == 0;
    }

    void unlock() noexcept
    {
        pthread_spin_unlock(&spinlock_);
    }

private:
    pthread_spinlock_t spinlock_{};
};

/* *
 * @brief store client IO context for server.
 */
class ClientIoContext {
public:
    explicit ClientIoContext(int fd, uint32_t events) : currentEvents_{ events }, fd_{ fd } {}

public:
    void ReceiveData() noexcept;

    bool HasNextReq() const noexcept;

    StoreMessage NextRequest() noexcept;

    void SendResponse(const StoreMessage &response) noexcept;

    bool SendBufEmpty() const noexcept;

    void FlushSendBuf() noexcept;

    uint32_t currentEvents_;

private:
    const int fd_;
    uint32_t recSize_{ 0 };
    std::vector<uint8_t> recBuf_;
    std::vector<uint8_t> sendBuf_;
    std::list<StoreMessage> requests_;
};

using ServerProcFn = std::function<StoreMessage(int fd, const StoreMessage &req)>;

/* *
 * @brief epoll based TCP server with registered message processor.
 */
class ParallelTcpServer {
public:
    explicit ParallelTcpServer(uint32_t threadNum, const std::string host, uint16_t port, uint32_t listenThreadNum,
		ServerProcFn process) noexcept;
    explicit ParallelTcpServer(uint32_t threadNum, const std::string localSocketPath, uint32_t listenThreadNum,
		ServerProcFn process) noexcept;

    int Start() noexcept;
    void Stop() noexcept;

    inline void SetKeysWaitingSocket(const std::vector<std::string> &keys, int socket, int workerFd, int64_t waitCount) noexcept
    {
        std::lock_guard<SpinLock> lockGuard{ spinLock_ };
        for (auto &key : keys) {
            keyWaitingSockets_[key].emplace_back(std::make_pair(socket, workerFd));
        }
        socketWaitKeyNum_[std::make_pair(socket, workerFd)] = waitCount;
    }

    void WakeupWaitingClients(const std::string &key) noexcept;

private:
    static int CreateSocketWithFamily(const std::string host, uint16_t port, int family) noexcept;
    static int CreateSocketAndListen(const ::addrinfo &addr) noexcept;
    static int CreateSocket(const std::string host, uint16_t port) noexcept;
    static int CreateLocalSocket(const std::string &localSocketPath) noexcept;

    static int CreateEpoll(int targetFd = -1) noexcept;

    void LoopProcessClients(int epollFd) noexcept;

    void ProcessListenEvent() noexcept;

    void ProcessClientEvent(int epFd, int fd, uint32_t event, std::unordered_map<int, ClientIoContext> &ctx) noexcept;

    static int SetNonBlocking(int fd) noexcept;
    static int SetBlockSocketTimeout(int fd) noexcept;
private:
    const uint32_t listenThreadNum_{ 1 };
    const uint32_t threadNum_{ 0 };
    const std::uint16_t port_{ 0 };
    const std::string host_{};
    const std::string localSocketPath_{};
    const ServerProcFn process_{ nullptr };
    int listenSocket_{ -1 };
    bool isLocalServer_{ false };
    std::vector<int> epClientFds_;
    std::vector<std::thread> clientThreads_;
    std::vector<std::thread> listenThreads_;
    uint8_t *buffer_{ nullptr };
    std::atomic<bool> running_{ false };

    SpinLock spinLock_;
    std::unordered_map<std::string, std::list<PI>> keyWaitingSockets_;
    std::map<PI, int64_t> socketWaitKeyNum_;
};
} // torch_npu
} // c10d
