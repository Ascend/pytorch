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
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <chrono>
#include <thread>

#include "c10/util/Logging.h"
#include "TcpClient.hpp"

namespace c10d {
namespace pta {
static constexpr uint32_t READ_BUF_SZ = 256;

TcpClient::TcpClient(std::string host, uint16_t port) noexcept
    : host_{ std::move(host) }, port_{ port }, socketFd_{ -1 }
{}

int TcpClient::Connect() noexcept
{
    socketFd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (socketFd_ < 0) {
        LOG(ERROR) << "create tcp client socket failed " << errno << " : " << strerror(errno);
        return -1;
    }

    struct sockaddr_in servAddr {};
    servAddr.sin_family = AF_INET;
    servAddr.sin_port = htons(port_);
    servAddr.sin_addr.s_addr = inet_addr(host_.c_str());

    int lastError = 0;
    auto endTime = std::chrono::steady_clock::now() + std::chrono::minutes(1);
    while (std::chrono::steady_clock::now() < endTime) {
        auto ret = connect(socketFd_, reinterpret_cast<const struct sockaddr *>(&servAddr), sizeof(servAddr));
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

    return -1;
}

int TcpClient::Close() noexcept
{
    auto ret = close(socketFd_);
    if (ret == 0) {
        socketFd_ = -1;
        return 0;
    }

    LOG(ERROR) << "close socket to server(" << host_ << ":" << port_ << ") failed " << errno << " : " <<
        strerror(errno);
    return ret;
}

int TcpClient::SyncCall(const StoreMessage &request, StoreMessage &response) noexcept
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
                LOG(ERROR) << "read data from server(" << host_ << ":" << port_ << ") failed " << errno << " : " <<
                    strerror(errno);
                return -1;
            }

            responseBuf.insert(responseBuf.end(), buffer, buffer + ret);
        } while (!StoreMessagePacker::Full(responseBuf));

        auto unpackRet = StoreMessagePacker::Unpack(responseBuf, response);
        if (unpackRet < 0L) {
            LOG(ERROR) << "unpack response data from server(" << host_ << ":" << port_ << ") failed " << unpackRet;
            finished = true;
            result = -1;
            continue;
        }

        if (response.mt == request.mt) {
            finished = true;
            result = 0;
            continue;
        }

        responseBuf.erase(responseBuf.begin(), responseBuf.begin() + unpackRet);
    }

    return result;
}

int TcpClient::SetReceiveTimeout(const std::chrono::milliseconds &value) const noexcept
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
} // c10d
} // pta