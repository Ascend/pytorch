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
#include <sys/un.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <chrono>
#include <thread>

#include "c10/util/Logging.h"
#include "StoreMessagePacker.hpp"
#include "StoreClient.hpp"

namespace c10d {
namespace torch_npu {
    
Client::Client(const std::string host, uint16_t port, const std::chrono::milliseconds timeout) noexcept
    : host_{ host }, port_{ port }, socketFd_(-1), timeout_{ timeout }
{}
Client::Client(const std::string localSocketPath, const std::chrono::milliseconds timeout) noexcept
    : localSocketPath_ { localSocketPath }, socketFd_(-1), timeout_{ timeout }
{}

int Client::TryConnectCore(const ::addrinfo &addr) noexcept
{
    socketFd_ = ::socket(addr.ai_family, addr.ai_socktype, addr.ai_protocol);
    if (socketFd_ < 0) {
        LOG(ERROR) << "create tcp client socket failed " << errno << " : " << strerror(errno);
        return -1;
    }

    auto ret = SetReceiveTimeout(timeout_);
    if (ret < 0) {
        LOG(ERROR) << "set socket timeout failed. " << errno << " : " << strerror(errno);
        close(socketFd_);
        socketFd_ = -1;
        return -1;
    }

    int lastError = 0;
    auto endTime = std::chrono::steady_clock::now() + timeout_;
    while (std::chrono::steady_clock::now() < endTime) {
        ret = connect(socketFd_, addr.ai_addr, addr.ai_addrlen);
        if (ret == 0) {
            return 0;
        }

        if (errno != lastError) {
            LOG(ERROR) << "connect socket to server(" << host_ << ":" << port_ << ") failed " << errno << " : " <<
                strerror(errno);
            lastError = errno;
        }

        if (errno == ETIMEDOUT) {
            continue;
        }

        if (errno == ECONNREFUSED) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
    }

    close(socketFd_);
    socketFd_ = -1;
    return -1;
}

int Client::TryConnect(int family) noexcept
{
    struct addrinfo hints = {0};
    hints.ai_family = family;
    hints.ai_socktype = SOCK_STREAM;

    ::addrinfo* result = nullptr;
    int r = ::getaddrinfo(host_.c_str(), std::to_string(port_).c_str(), &hints, &result);
    if (r != 0) {
        LOG(ERROR) << "getaddrinfo failed " << errno << " : " << strerror(errno);
        return -1;
    }

    for (::addrinfo* addr = result; addr != nullptr; addr = addr->ai_next) {
        int ret = TryConnectCore(*addr);
        if (ret == 0) {
            return 0;
        }
    }
    return -1;
}

int Client::Connect() noexcept
{
    auto ret = TryConnect(AF_INET);
    if (ret >= 0) {
        return 0;
    }
    
    ret = TryConnect(AF_INET6);
    if (ret >= 0) {
        return 0;
    }
    return -1;
}

int Client::Close() noexcept
{
    shutdown(socketFd_, SHUT_RDWR);
    if (socketFd_ >= 0) {
        auto ret = close(socketFd_);
        if (ret == 0) {
            socketFd_ = -1;
            return 0;
        }
        LOG(ERROR) << "close socket to server(" << host_ << ":" << port_ << ") failed " << errno << " : " <<
            strerror(errno);
        return ret;
    }

    return 0;
}

int Client::LocalConnect() noexcept
{
    if (localSocketPath_.empty()) {
        LOG(ERROR) << "local socket path invalid." << errno << " : " << strerror(errno);
        return -1;
    }
    socketFd_ = socket(AF_UNIX, SOCK_STREAM, 0);
    if (socketFd_ < 0) {
        LOG(ERROR) << "Create local socket failed: " << strerror(errno);
        return -1;
    }

    auto ret = SetReceiveTimeout(timeout_);
    if (ret < 0) {
        LOG(ERROR) << "set socket timeout failed. " << errno << " : " << strerror(errno);
        return -1;
    }
    struct sockaddr_un servAddr {};
    servAddr.sun_family = AF_UNIX;
    servAddr.sun_path[0] = '\0';
    strncpy(servAddr.sun_path + 1, localSocketPath_.c_str(), sizeof(servAddr.sun_path) - 2);
    servAddr.sun_path[sizeof(servAddr.sun_path) - 1] = '\0';

    int lastError = 0;
    auto endTime = std::chrono::steady_clock::now() + timeout_;
    while (std::chrono::steady_clock::now() < endTime) {
        ret = connect(socketFd_, reinterpret_cast<const struct sockaddr *>(&servAddr), sizeof(servAddr));
        if (ret == 0) {
            return 0;
        }

        if (errno != lastError) {
            LOG(ERROR) << "connect socket to local server failed " << errno << " : " <<
                strerror(errno);
            lastError = errno;
        }

        if (errno == ETIMEDOUT) {
            continue;
        }

        if (errno == ECONNREFUSED) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
    }

    return -1;
}

int Client::LocalClose() noexcept
{
    shutdown(socketFd_, SHUT_RDWR);
    auto ret = close(socketFd_);
    if (ret == 0) {
        socketFd_ = -1;
        return 0;
    }

    LOG(ERROR) << "close socket to server failed " << errno << " : " << strerror(errno);
    return ret;
}

int Client::SyncCall(const StoreMessage &request, StoreMessage &response) noexcept
{
    auto packedRequest = StoreMessagePacker::Pack(request);
    auto ret = write(socketFd_, packedRequest.data(), packedRequest.size());
    if (ret < 0) {
        LOG(ERROR) << "write data to server(" << host_ << ":" << port_ << ") failed " << errno << " : " <<
            strerror(errno);
        return -1;
    }

    uint8_t buffer[READ_BUF_SZ];
    std::vector<uint8_t> responseBuf;

    bool finished = false;
    int result = -1;
    while (!finished) {
        do {
            ret = read(socketFd_, buffer, READ_BUF_SZ);
            if (ret < 0) {
                if (errno == EINTR) { // interrupted by signal
                    continue;
                }
                
                LOG(ERROR) << "read data from server(" << host_ << ":" << port_ << ") failed " << errno << " : " <<
                    strerror(errno);
                return -1;
            }

            responseBuf.insert(responseBuf.end(), buffer, buffer + ret);
        } while (!StoreMessagePacker::Full(responseBuf));

        while (!responseBuf.empty()) {
            auto unpackRet = StoreMessagePacker::Unpack(responseBuf, response);
            if (unpackRet < 0L) {
                LOG(ERROR) << "unpack response data from server(" << host_ << ":" << port_ << ") failed " << unpackRet;
                finished = true;
                result = -1;
                break;
            }
            responseBuf.erase(responseBuf.begin(), responseBuf.begin() + unpackRet);
            if (response.mt == request.mt) {
                finished = true;
                result = 0;
                break;
            }
        }
    }

    return result;
}

int Client::SetReceiveTimeout(const std::chrono::milliseconds &value) const noexcept
{
    if (value == std::chrono::milliseconds::zero()) {
        return 0;
    }
    struct timeval timeoutTV = {
        .tv_sec = value.count() / 1000,
        .tv_usec = (value.count() % 1000) * 1000
    };

    auto ret = setsockopt(socketFd_, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<char *>(&timeoutTV), sizeof(timeoutTV));
    if (ret != 0) {
        LOG(ERROR) << "set connection receive timeout failed: " << errno << " : " << strerror(errno);
    }

    return ret;
}

int Client::GetSocketFd() const noexcept
{
    return socketFd_;
}
} // torch_npu
} // c10d