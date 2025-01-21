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
#include <vector>
#include <string>

#include "c10/util/Optional.h"

namespace c10d {
namespace torch_npu {
enum class MessageType : uint8_t {
    SET,
    COMPARE_SET,
    GET,
    ADD,
    CHECK,
    WAIT,
    GET_NUM_KEYS,
    WATCH_KEY,
    DELETE_KEY,
    INVALID_MSG,
    SKIP_MSG
};

enum class MessageCheckKeyRes : uint8_t {
    KEYS_READY,
    KEYS_NOT_READY
};

enum class MessageWaitKeyRes : uint8_t {
    KEYS_STOP_WAITING
};

struct StoreMessage {
    StoreMessage() noexcept : mt{ MessageType::INVALID_MSG } {}

    explicit StoreMessage(MessageType type, int fd) noexcept : mt{ type }, fd{ fd } {}

    StoreMessage(MessageType type, int fd, std::string k) noexcept : mt{ type }, fd{ fd }
    {
        keys.emplace_back(std::move(k));
    }

    StoreMessage(MessageType type, int fd, std::vector<uint8_t> v) noexcept : mt{ type }, fd{ fd }
    {
        values.emplace_back(std::move(v));
    }

    StoreMessage(MessageType type, int fd, std::string k, std::vector<uint8_t> v) noexcept : mt{ type }, fd{ fd }
    {
        keys.emplace_back(std::move(k));
        values.emplace_back(std::move(v));
    }

    StoreMessage(MessageType type, int fd, std::string k, std::vector<uint8_t> v, std::vector<uint8_t> vv) noexcept : mt{ type }, fd{ fd }
    {
        keys.emplace_back(std::move(k));
        values.emplace_back(std::move(v));
        values.emplace_back(std::move(vv));
    }

    StoreMessage(MessageType type, int fd, std::vector<std::string> ks) noexcept : mt{ type }, fd { fd }, keys{ std::move(ks) } {}

    StoreMessage(MessageType type, int fd, std::vector<std::string> ks, int64_t value) noexcept
        : mt{ type }, fd{ fd }, keys{ std::move(ks) }
    {
        values.emplace_back(reinterpret_cast<const uint8_t *>(&value),
            reinterpret_cast<const uint8_t *>(&value) + sizeof(int64_t));
    }

    StoreMessage(MessageType type, int fd, std::vector<std::vector<uint8_t>> vs) noexcept : mt{ type }, fd { fd }, values{ std::move(vs) }
    {}

    int fd { 0 };
    MessageType mt;
    std::vector<std::string> keys;
    std::vector<std::vector<uint8_t>> values;
};

class StoreMessagePacker {
public:
    static std::vector<uint8_t> Pack(const StoreMessage &message) noexcept;

    static bool Full(const std::vector<uint8_t> &buffer) noexcept;

    static int64_t MessageSize(const std::vector<uint8_t> &buffer) noexcept;

    static int64_t Unpack(const std::vector<uint8_t> &buffer, StoreMessage &message) noexcept;

    template <class T> static std::vector<uint8_t> PackPod(const T &v) noexcept
    {
        auto begin = reinterpret_cast<const uint8_t *>(&v);
        return std::vector<uint8_t>{ begin, begin + sizeof(T) };
    }

    template <class T> static T UnpackPod(const std::vector<uint8_t> &vec) noexcept
    {
        return *reinterpret_cast<const T *>(vec.data());
    }

private:
    template <class T> static void PackValue(std::vector<uint8_t> &dest, T value) noexcept
    {
        dest.insert(dest.end(), reinterpret_cast<const uint8_t *>(&value),
            reinterpret_cast<const uint8_t *>(&value) + sizeof(T));
    }

    static void PackString(std::vector<uint8_t> &dest, const std::string &str) noexcept;

    static void PackBytes(std::vector<uint8_t> &dest, const std::vector<uint8_t> &bytes) noexcept;
};
} // torch_npu
} // c10d
