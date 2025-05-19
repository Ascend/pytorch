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
#include <string>
#include <chrono>
#include <netdb.h>

#include "StoreMessagePacker.hpp"

namespace c10d {
namespace torch_npu {
constexpr uint32_t READ_BUF_SZ = 256;
class Client {
public:
    explicit Client(const std::string host, uint16_t port, const std::chrono::milliseconds timeout) noexcept;
    explicit Client(const std::string localSocketPath, const std::chrono::milliseconds timeout) noexcept;
    int Connect() noexcept;
    int TryConnect(int family) noexcept;
    int TryConnectCore(const ::addrinfo &addr) noexcept;
    int Close() noexcept;
    int LocalConnect() noexcept;
    int LocalClose() noexcept;
    int SyncCall(const StoreMessage &request, StoreMessage &response) noexcept;
    int SetReceiveTimeout(const std::chrono::milliseconds &value) const noexcept;
    int GetSocketFd() const noexcept;
private:
    const std::string localSocketPath_{};
    const std::string host_{};
    const uint16_t port_{ 0 };
    int socketFd_;
    std::chrono::milliseconds timeout_;
};
} // torch_npu
} // c10d
