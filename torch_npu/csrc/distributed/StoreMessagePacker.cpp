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
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "StoreMessagePacker.hpp"

namespace c10d {
namespace torch_npu {
/*
 * size  mt  keyN  keys       vN    values
 * +----+----+----+----------+----+------------+
 * | 8B | 1B | 8B | KEYS = ? | 8B | VALUES = ? |
 * each key in keys:
 * KeyL  key
 * +----+-------+
 * | 8B | bytes |
 * each value in values:
 * vL    value
 * +----+-------+
 * | 8B | bytes |
 */
std::vector<uint8_t> StoreMessagePacker::Pack(const StoreMessage &message) noexcept
{
    constexpr uint64_t baseSize = 3U * sizeof(uint64_t) + sizeof(MessageType) + sizeof(int); // size + mt + keyN + vN + fd
    uint64_t totalSize = baseSize;
    for (auto &key : message.keys) {
        totalSize += (sizeof(uint64_t) + key.size());
    }
    for (auto &value : message.values) {
        totalSize += (sizeof(uint64_t) + value.size());
    }

    std::vector<uint8_t> result;
    result.reserve(totalSize);
    PackValue(result, totalSize);
    PackValue(result, message.mt);
    PackValue(result, message.fd);

    PackValue(result, message.keys.size());
    for (auto &key : message.keys) {
        PackString(result, key);
    }

    PackValue(result, message.values.size());
    for (auto &value : message.values) {
        PackBytes(result, value);
    }

    return result;
}

bool StoreMessagePacker::Full(const std::vector<uint8_t> &buffer) noexcept
{
    if (buffer.size() < sizeof(uint64_t) + sizeof(MessageType) + sizeof(int)) {
        return false;
    }

    auto totalSize = *reinterpret_cast<const uint64_t *>(buffer.data());
    return buffer.size() >= totalSize;
}

int64_t StoreMessagePacker::MessageSize(const std::vector<uint8_t> &buffer) noexcept
{
    if (buffer.size() < sizeof(uint64_t)) {
        return -1L;
    }

    return *reinterpret_cast<const int64_t *>(buffer.data());
}

int64_t StoreMessagePacker::Unpack(const std::vector<uint8_t> &buffer, StoreMessage &message) noexcept
{
    if (!Full(buffer)) {
        return -1;
    }

    auto ptr = buffer.data();
    auto ptr_end = ptr + buffer.size();
    auto totalSize = *reinterpret_cast<const uint64_t *>(ptr);
    ptr += sizeof(uint64_t);

    message.mt = *reinterpret_cast<const MessageType *>(ptr);
    ptr += sizeof(MessageType);

    message.fd = *reinterpret_cast<const int *>(ptr);
    ptr += sizeof(int);

    auto keyCount = *reinterpret_cast<const uint64_t *>(ptr);
    ptr += sizeof(uint64_t);
    for (auto i = 0UL; i < keyCount; i++) {
        auto keySize = *reinterpret_cast<const uint64_t *>(ptr);
        ptr += sizeof(uint64_t);
        message.keys.emplace_back(reinterpret_cast<const char *>(ptr), keySize);
        ptr += keySize;
        if (ptr > ptr_end) {
            break;
        }
    }

    auto valueCount = *reinterpret_cast<const uint64_t *>(ptr);
    ptr += sizeof(uint64_t);
    for (auto i = 0UL; i < valueCount; i++) {
        auto valueSize = *reinterpret_cast<const uint64_t *>(ptr);
        ptr += sizeof(uint64_t);
        message.values.emplace_back(ptr, ptr + valueSize);
        ptr += valueSize;
        if (ptr > ptr_end) {
            break;
        }
    }

    return static_cast<int64_t>(totalSize);
}

void StoreMessagePacker::PackString(std::vector<uint8_t> &dest, const std::string &str) noexcept
{
    PackValue(dest, static_cast<uint64_t>(str.size()));
    if (!str.empty()) {
        dest.insert(dest.end(), str.data(), str.data() + str.size());
    }
}

void StoreMessagePacker::PackBytes(std::vector<uint8_t> &dest, const std::vector<uint8_t> &bytes) noexcept
{
    PackValue(dest, static_cast<uint64_t>(bytes.size()));
    dest.insert(dest.end(), bytes.begin(), bytes.end());
}
} // torch_npu
} // c10d
