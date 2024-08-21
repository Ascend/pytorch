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
#pragma once
#include <cstdint>

#include <list>
#include <mutex>
#include <vector>
#include <thread>
#include <atomic>
#include <unordered_map>
#include <functional>

#include "StoreMessagePacker.hpp"

namespace c10d {
namespace pta {
/**
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

/**
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

/**
 * @brief epoll based TCP server with registered message processor.
 */
class ParallelTcpServer {
public:
    explicit ParallelTcpServer(uint32_t threadNum, uint16_t port, ServerProcFn process) noexcept;

    int Start() noexcept;

    void Stop() noexcept;

    inline void SetKeysWaitingSocket(const std::vector<std::string> &keys, int socket, int64_t waitCount) noexcept
    {
        std::lock_guard<SpinLock> lockGuard{ spinLock_ };
        for (auto &key : keys) {
            keyWaitingSockets_[key].emplace_back(socket);
        }
        socketWaitKeyNum_[socket] = waitCount;
    }

    void WakeupWaitingClients(const std::string &key) noexcept;

private:
    static int CreateSocket(uint16_t port) noexcept;

    static int CreateEpoll(int targetFd = -1) noexcept;

    void LoopProcessListenFd() noexcept;

    void LoopProcessClients(int epollFd) noexcept;

    void ProcessListenEvent(uint32_t event) noexcept;

    void ProcessClientEvent(int epFd, int fd, uint32_t event, std::unordered_map<int, ClientIoContext> &ctx) noexcept;

    static int SetNonBlocking(int fd) noexcept;

private:
    const uint32_t threadNum_;
    const std::uint16_t port_;
    const ServerProcFn process_;
    int listenSocket_{ -1 };
    int epCtlFd_{ -1 };
    std::thread ctlThread_;
    std::vector<int> epClientFds_;
    std::vector<std::thread> clientThreads_;
    uint8_t *buffer_{ nullptr };
    std::atomic<bool> running_{ false };

    SpinLock spinLock_;
    std::unordered_map<std::string, std::list<int>> keyWaitingSockets_;
    std::unordered_map<int, int64_t> socketWaitKeyNum_;
};
} // c10d
} // pta
