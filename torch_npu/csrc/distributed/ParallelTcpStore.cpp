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
#include "ParallelTcpServer.hpp"
#include "ParallelTcpStore.hpp"

namespace c10d {
namespace pta {
ParallelStoreServer::ParallelStoreServer(std::string initKey, uint16_t port, c10::optional<std::size_t> numWorkers)
    : initKey_{ std::move(initKey) }, numWorkers_{ numWorkers }
{
    auto threadNum = 4U;
    if (numWorkers != c10::nullopt) {
        if (*numWorkers >= 1024) {
            threadNum = 32U;
        } else if (*numWorkers >= 512) {
            threadNum = 16U;
        } else if (*numWorkers >= 256) {
            threadNum = 8U;
        }
    }

    InitializeHandlers();
    server_ = std::make_unique<pta::ParallelTcpServer>(threadNum, port,
        [this](int fd, const pta::StoreMessage &request) { return ProcessRequest(fd, request); });
    if (server_->Start() != 0) {
        throw std::runtime_error{
            std::string("start tcp server on port ").append(std::to_string(port)).append(" failed.")
        };
    }
}

ParallelStoreServer::~ParallelStoreServer() noexcept
{
    server_->Stop();
}

void ParallelStoreServer::WaitWorkers(const std::chrono::milliseconds &timeout) noexcept
{
    if (numWorkers_ == c10::nullopt) {
        return;
    }

    const auto start = std::chrono::steady_clock::now();
    while (!workersReady_) {
        std::unique_lock<std::mutex> lockGuard{ initWaitMutex_ };
        if (timeout == Store::kNoTimeout) {
            initWaitCond_.wait(lockGuard);
        } else {
            initWaitCond_.wait_until(lockGuard, start + timeout);
        }
    }
}

pta::StoreMessage ParallelStoreServer::ProcessRequest(int fd, const pta::StoreMessage &request) noexcept
{
    auto pos = requestHandlers_.find(request.mt);
    if (pos != requestHandlers_.end()) {
        return pos->second(fd, request);
    }

    LOG(ERROR) << "unsupported message type " << static_cast<uint32_t>(request.mt);
    return request;
}

pta::StoreMessage ParallelStoreServer::ProcessGetRequest(int fd, const pta::StoreMessage &request) noexcept
{
    std::unique_lock<pta::SpinLock> lockGuard{ serverLock_ };
    auto pos = keyStore_.find(request.keys[0]);
    if (pos != keyStore_.end()) {
        lockGuard.unlock();
        return { pta::MessageType::GET, pos->second };
    }
    lockGuard.unlock();

    return pta::StoreMessage{ pta::MessageType::GET };
}

pta::StoreMessage ParallelStoreServer::ProcessSetRequest(int fd, const pta::StoreMessage &request) noexcept
{
    bool newCreated = false;
    std::unique_lock<pta::SpinLock> lockGuard{ serverLock_ };
    auto pos = keyStore_.find(request.keys[0]);
    if (pos == keyStore_.end()) {
        keyStore_.emplace(request.keys[0], request.values[0]);
        newCreated = true;
    } else {
        pos->second = request.values[0];
    }
    lockGuard.unlock();

    if (newCreated) {
        server_->WakeupWaitingClients(request.keys[0]);
    }

    return pta::StoreMessage{ pta::MessageType::SET };
}

pta::StoreMessage ParallelStoreServer::ProcessAddRequest(int fd, const pta::StoreMessage &request) noexcept
{
    static bool notifiedWaitWorkers = false;
    auto delta = pta::StoreMessagePacker::UnpackPod<int64_t>(request.values[0]);
    auto old = 0L;
    bool oldExist = false;

    std::unique_lock<pta::SpinLock> lockGuard{ serverLock_ };
    auto pos = keyStore_.find(request.keys[0]);
    if (pos != keyStore_.end()) {
        oldExist = true;
        old = std::stoll(std::string(reinterpret_cast<const char *>(pos->second.data()), pos->second.size()));
    }

    auto newValue = old + delta;
    auto valueString = std::to_string(newValue);
    keyStore_[request.keys[0]] = std::vector<uint8_t>(valueString.begin(), valueString.end());
    lockGuard.unlock();

    if (!notifiedWaitWorkers && request.keys[0] == initKey_ && numWorkers_ != c10::nullopt &&
        newValue >= *numWorkers_) {
        workersReady_ = true;
        initWaitCond_.notify_one();
        notifiedWaitWorkers = true;
    }

    if (!oldExist) {
        server_->WakeupWaitingClients(request.keys[0]);
    }

    return { pta::MessageType::ADD, pta::StoreMessagePacker::PackPod(old + delta) };
}

pta::StoreMessage ParallelStoreServer::ProcessCheckRequest(int fd, const pta::StoreMessage &request) noexcept
{
    std::unique_lock<pta::SpinLock> lockGuard{ serverLock_ };
    pta::MessageCheckKeyRes res = pta::MessageCheckKeyRes::KEYS_NOT_READY;
    if (CheckAllKeysExistInLock(request.keys)) {
        res = pta::MessageCheckKeyRes::KEYS_READY;
    }
    lockGuard.unlock();

    std::vector<uint8_t> body{ static_cast<uint8_t>(res) };
    return { pta::MessageType::CHECK, body };
}

pta::StoreMessage ParallelStoreServer::ProcessDeleteRequest(int fd, const pta::StoreMessage &request) noexcept
{
    std::unique_lock<pta::SpinLock> lockGuard{ serverLock_ };
    auto count = keyStore_.erase(request.keys[0]);
    lockGuard.unlock();

    return pta::StoreMessage{ pta::MessageType::DELETE_KEY, std::vector<uint8_t>{ static_cast<uint8_t>(count > 0) } };
}

pta::StoreMessage ParallelStoreServer::ProcessCompareSetRequest(int fd, const pta::StoreMessage &request) noexcept
{
    std::unique_lock<pta::SpinLock> lockGuard{ serverLock_ };
    auto pos = keyStore_.find(request.keys[0]);
    if (pos == keyStore_.end()) {
        if (request.values[0].empty()) {
            keyStore_[request.keys[0]] = request.values[1];
            lockGuard.unlock();
            server_->WakeupWaitingClients(request.keys[0]);
            return { pta::MessageType::COMPARE_SET, request.values[1] };
        }

        lockGuard.unlock();
        return pta::StoreMessage{ pta::MessageType::COMPARE_SET };
    }

    if (pos->second == request.values[0]) {
        pos->second = request.values[1];
        lockGuard.unlock();
        return { pta::MessageType::COMPARE_SET, request.values[1] };
    }

    lockGuard.unlock();
    return { pta::MessageType::COMPARE_SET, pos->second };
}

pta::StoreMessage ParallelStoreServer::ProcessGetNumKeyRequest(int fd, const pta::StoreMessage &request) noexcept
{
    std::unique_lock<pta::SpinLock> lockGuard{ serverLock_ };
    auto keyNum = keyStore_.size();
    lockGuard.unlock();
    return { pta::MessageType::GET_NUM_KEYS, pta::StoreMessagePacker::PackPod(keyNum) };
}

pta::StoreMessage ParallelStoreServer::ProcessWaitKeysRequest(int fd, const pta::StoreMessage &request) noexcept
{
    int64_t numKeysToWait = 0;
    std::vector<std::string> waitKeys;
    waitKeys.reserve(request.keys.size());

    std::unique_lock<pta::SpinLock> lockGuard{ serverLock_ };
    if (CheckAllKeysExistInLock(request.keys)) {
        lockGuard.unlock();

        std::vector<uint8_t> body{ static_cast<uint8_t>(pta::MessageWaitKeyRes::KEYS_STOP_WAITING) };
        return { pta::MessageType::WAIT, body };
    }

    for (auto &key : request.keys) {
        if (keyStore_.find(key) == keyStore_.end()) {
            waitKeys.emplace_back(key);
            numKeysToWait++;
        }
    }
    server_->SetKeysWaitingSocket(waitKeys, fd, numKeysToWait);
    lockGuard.unlock();

    return pta::StoreMessage{ pta::MessageType::INVALID_MSG };
}

void ParallelStoreServer::InitializeHandlers() noexcept
{
    requestHandlers_.emplace(pta::MessageType::SET,
        [this](int fd, const pta::StoreMessage &req) { return ProcessSetRequest(fd, req); });
    requestHandlers_.emplace(pta::MessageType::COMPARE_SET,
        [this](int fd, const pta::StoreMessage &req) { return ProcessCompareSetRequest(fd, req); });
    requestHandlers_.emplace(pta::MessageType::GET,
        [this](int fd, const pta::StoreMessage &req) { return ProcessGetRequest(fd, req); });
    requestHandlers_.emplace(pta::MessageType::ADD,
        [this](int fd, const pta::StoreMessage &req) { return ProcessAddRequest(fd, req); });
    requestHandlers_.emplace(pta::MessageType::CHECK,
        [this](int fd, const pta::StoreMessage &req) { return ProcessCheckRequest(fd, req); });
    requestHandlers_.emplace(pta::MessageType::WAIT,
        [this](int fd, const pta::StoreMessage &req) { return ProcessWaitKeysRequest(fd, req); });
    requestHandlers_.emplace(pta::MessageType::GET_NUM_KEYS,
        [this](int fd, const pta::StoreMessage &req) { return ProcessGetNumKeyRequest(fd, req); });
    requestHandlers_.emplace(pta::MessageType::DELETE_KEY,
        [this](int fd, const pta::StoreMessage &req) { return ProcessDeleteRequest(fd, req); });
}

bool ParallelStoreServer::CheckAllKeysExistInLock(const std::vector<std::string> &keys) noexcept
{
    return std::all_of(keys.begin(), keys.end(), [this](const std::string &key) { return keyStore_.count(key) > 0; });
}
}

std::mutex ParallelTcpStore::cacheServerMutex_;
std::unordered_map<uint16_t, std::weak_ptr<pta::ParallelStoreServer>> ParallelTcpStore::cachedServers_;

ParallelTcpStore::ParallelTcpStore(const std::string &host, const c10d::TCPStoreOptions &opts)
    : Store(opts.timeout), client_{ host, opts.port }
{
    if (opts.isServer) {
        if (opts.multiTenant) {
            server_ = GetSharedServer(initKey_, opts.port, opts.numWorkers);
        } else {
            server_ = std::make_shared<pta::ParallelStoreServer>(initKey_, opts.port, opts.numWorkers);
        }
    }

    if (client_.Connect() != 0) {
        throw std::runtime_error{ std::string("connect tcp client to server(")
                                      .append(host)
                                      .append(":")
                                      .append(std::to_string(opts.port))
                                      .append(" failed.") };
    }

    if (opts.waitWorkers) {
        IncreaseKey(initKey_, 1);
        if (opts.isServer) {
            server_->WaitWorkers(timeout_);
        }
    }
}

ParallelTcpStore::~ParallelTcpStore() noexcept
{
    client_.Close();
}

void ParallelTcpStore::set(const std::string &key, const std::vector<uint8_t> &value)
{
    pta::StoreMessage request{ pta::MessageType::SET, key, value };
    pta::StoreMessage response;
    std::lock_guard<std::mutex> lockGuard{ clientMutex_ };
    auto ret = client_.SyncCall(request, response);
    if (ret != 0) {
        throw std::runtime_error{ std::string("set key ").append(key).append(" failed.") };
    }
}

std::vector<uint8_t> ParallelTcpStore::compareSet(const std::string &key, const std::vector<uint8_t> &currentValue,
    const std::vector<uint8_t> &newValue)
{
    pta::StoreMessage request{ pta::MessageType::COMPARE_SET, key, currentValue, newValue };
    pta::StoreMessage response;
    std::lock_guard<std::mutex> lockGuard{ clientMutex_ };
    auto ret = client_.SyncCall(request, response);
    if (ret != 0) {
        throw std::runtime_error{ std::string("compare and set key ").append(key).append(" failed.") };
    }

    return response.values.empty() ? std::vector<uint8_t>{} : std::move(response.values[0]);
}

std::vector<uint8_t> ParallelTcpStore::get(const std::string &key)
{
    pta::StoreMessage waitReq{ pta::MessageType::WAIT, key };
    pta::StoreMessage getReq{ pta::MessageType::GET, key };
    pta::StoreMessage waitResp;
    pta::StoreMessage getResp;

    std::lock_guard<std::mutex> lockGuard{ clientMutex_ };
    auto ret = client_.SyncCall(waitReq, waitResp);
    if (ret != 0) {
        throw std::runtime_error{ std::string("get key ").append(key).append(" failed.") };
    }

    ret = client_.SyncCall(getReq, getResp);
    if (ret != 0) {
        throw std::runtime_error{ std::string("get key ").append(key).append(" failed.") };
    }

    return getResp.values.empty() ? std::vector<uint8_t>{} : std::move(getResp.values[0]);
}

int64_t ParallelTcpStore::add(const std::string &key, int64_t value)
{
    return IncreaseKey(key, value);
}

bool ParallelTcpStore::deleteKey(const std::string &key)
{
    pta::StoreMessage request{ pta::MessageType::DELETE_KEY, key };
    pta::StoreMessage response;
    std::lock_guard<std::mutex> lockGuard{ clientMutex_ };
    auto ret = client_.SyncCall(request, response);
    if (ret != 0) {
        throw std::runtime_error{ std::string("delete key ").append(key).append(" failed.") };
    }

    return !response.values.empty() && !response.values[0].empty() && response.values[0][0] > 0U;
}

bool ParallelTcpStore::check(const std::vector<std::string> &keys)
{
    throw std::runtime_error("unsupported check operation.");
}

int64_t ParallelTcpStore::getNumKeys()
{
    pta::StoreMessage request{ pta::MessageType::GET_NUM_KEYS };
    pta::StoreMessage response;
    std::lock_guard<std::mutex> lockGuard{ clientMutex_ };
    auto ret = client_.SyncCall(request, response);
    if (ret != 0) {
        throw std::runtime_error{ "get number keys failed." };
    }

    return pta::StoreMessagePacker::UnpackPod<int64_t>(response.values[0]);
}

void ParallelTcpStore::wait(const std::vector<std::string> &keys)
{
    wait(keys, timeout_);
}

void ParallelTcpStore::wait(const std::vector<std::string> &keys, const std::chrono::milliseconds &timeout)
{
    pta::StoreMessage request{ pta::MessageType::WAIT, keys };
    pta::StoreMessage response;
    client_.SetReceiveTimeout(timeout);
    std::lock_guard<std::mutex> lockGuard{ clientMutex_ };
    DoWait(request, response);
}

const std::chrono::milliseconds &ParallelTcpStore::getTimeout() const noexcept
{
    return timeout_;
}

void ParallelTcpStore::setTimeout(const std::chrono::milliseconds &timeout)
{
    timeout_ = timeout;
}

int64_t ParallelTcpStore::IncreaseKey(const std::string &key, int64_t value)
{
    pta::StoreMessage request{ pta::MessageType::ADD, key, pta::StoreMessagePacker::PackPod(value) };
    pta::StoreMessage response;
    std::lock_guard<std::mutex> lockGuard{ clientMutex_ };
    auto ret = client_.SyncCall(request, response);
    if (ret != 0) {
        throw std::runtime_error{ std::string("add key ").append(key).append(" failed.") };
    }

    return pta::StoreMessagePacker::UnpackPod<int64_t>(response.values[0]);
}

void ParallelTcpStore::DoWait(const pta::StoreMessage &req, pta::StoreMessage &res)
{
    auto ret = client_.SyncCall(req, res);
    if (ret != 0) {
        throw std::runtime_error{ "get number keys failed." };
    }
}

std::shared_ptr<pta::ParallelStoreServer> ParallelTcpStore::GetSharedServer(const std::string &initKey, uint16_t port,
    c10::optional<std::size_t> numWorkers)
{
    std::unique_lock<std::mutex> lockGuard{ cacheServerMutex_ };
    auto pos = cachedServers_.find(port);
    if (pos != cachedServers_.end()) {
        auto server = pos->second.lock();
        if (server != nullptr) {
            return server;
        }

        cachedServers_.erase(pos);
    }

    auto server = std::make_shared<pta::ParallelStoreServer>(initKey, port, numWorkers);
    cachedServers_.emplace(port, server);
    return server;
}
} // c10d