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
#include <chrono>
#include "ParallelTcpServer.hpp"
#include "ParallelStoreProxy.hpp"
#include "StoreClient.hpp"
#include "torch_npu/csrc/core/npu/npu_log.h"
#include "ParallelTcpStore.hpp"

namespace c10d {
namespace torch_npu {
ParallelStoreServer::ParallelStoreServer(std::string initKey, const std::string host, uint16_t port,
    c10::optional<std::size_t> numWorkers) noexcept
    : initKey_{ std::move(initKey) }, numWorkers_{ numWorkers }
{
    auto threadNum = 4U;
    auto listenThreadNum = 1U;
    if (numWorkers != c10::nullopt) {
        if (*numWorkers >= 1024) {
            threadNum = 32U;
            listenThreadNum = *numWorkers / 1024;
        } else if (*numWorkers >= 512) {
            threadNum = 16U;
        } else if (*numWorkers >= 256) {
            threadNum = 8U;
        }
    }

    InitializeHandlers();
    server_ = std::make_unique<torch_npu::ParallelTcpServer>(threadNum, host, port, listenThreadNum,
        [this](int fd, const torch_npu::StoreMessage &request) { return ProcessRequest(fd, request); });
    if (server_->Start() != 0) {
        throw std::runtime_error{
            std::string("start tcp server on port ").append(std::to_string(port)).append(" failed.")
        };
    }
}

ParallelStoreServer::ParallelStoreServer(const std::string localSocketPath, CallBackFn callback) noexcept
    : localSocketPath_(localSocketPath), callback_(std::move(callback))
{
    auto threadNum = 1U;
    auto listenThreadNum = 1U;
    LocalInitializeHandlers();
    server_ = std::make_unique<torch_npu::ParallelTcpServer>(threadNum, localSocketPath_, listenThreadNum,
        [this](int fd, const torch_npu::StoreMessage &request) { return ProcessRequest(fd, request); });
    if (server_->Start() != 0) {
        throw std::runtime_error{
            std::string("start local server on socket ").append(localSocketPath_).append(" failed.")
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

torch_npu::StoreMessage ParallelStoreServer::ProcessRequest(int fd, const torch_npu::StoreMessage &request) noexcept
{
    auto pos = requestHandlers_.find(request.mt);
    if (pos != requestHandlers_.end()) {
        return pos->second(fd, request);
    }

    LOG(ERROR) << "unsupported message type " << static_cast<uint32_t>(request.mt);
    return request;
}

torch_npu::StoreMessage ParallelStoreServer::ProcessGetRequest(int fd, const torch_npu::StoreMessage &request) noexcept
{
    std::unique_lock<torch_npu::SpinLock> lockGuard{ serverLock_ };
    auto pos = keyStore_.find(request.keys[0]);
    if (pos != keyStore_.end()) {
        lockGuard.unlock();
        return { torch_npu::MessageType::GET, request.fd, pos->second };
    }
    lockGuard.unlock();

    return torch_npu::StoreMessage{ torch_npu::MessageType::GET, request.fd};
}

torch_npu::StoreMessage ParallelStoreServer::ProcessSetRequest(int fd, const torch_npu::StoreMessage &request) noexcept
{
    bool newCreated = false;
    std::unique_lock<torch_npu::SpinLock> lockGuard{ serverLock_ };
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

    return torch_npu::StoreMessage{ torch_npu::MessageType::SET, request.fd};
}

torch_npu::StoreMessage ParallelStoreServer::ProcessAddRequest(int fd, const torch_npu::StoreMessage &request) noexcept
{
    static bool notifiedWaitWorkers = false;
    auto delta = torch_npu::StoreMessagePacker::UnpackPod<int64_t>(request.values[0]);
    auto old = 0L;
    bool oldExist = false;

    std::unique_lock<torch_npu::SpinLock> lockGuard{ serverLock_ };
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

    return { torch_npu::MessageType::ADD, request.fd, torch_npu::StoreMessagePacker::PackPod(old + delta) };
}

torch_npu::StoreMessage ParallelStoreServer::ProcessCheckRequest(int fd, const torch_npu::StoreMessage &request) noexcept
{
    std::unique_lock<torch_npu::SpinLock> lockGuard{ serverLock_ };
    torch_npu::MessageCheckKeyRes res = torch_npu::MessageCheckKeyRes::KEYS_NOT_READY;
    if (CheckAllKeysExistInLock(request.keys)) {
        res = torch_npu::MessageCheckKeyRes::KEYS_READY;
    }
    lockGuard.unlock();

    std::vector<uint8_t> body{ static_cast<uint8_t>(res) };
    return { torch_npu::MessageType::CHECK, request.fd, body };
}

torch_npu::StoreMessage ParallelStoreServer::ProcessDeleteRequest(int fd, const torch_npu::StoreMessage &request) noexcept
{
    std::unique_lock<torch_npu::SpinLock> lockGuard{ serverLock_ };
    auto count = keyStore_.erase(request.keys[0]);
    lockGuard.unlock();

    return torch_npu::StoreMessage{ torch_npu::MessageType::DELETE_KEY, request.fd, std::vector<uint8_t>{ static_cast<uint8_t>(count > 0) } };
}

torch_npu::StoreMessage ParallelStoreServer::ProcessCompareSetRequest(int fd, const torch_npu::StoreMessage &request) noexcept
{
    std::unique_lock<torch_npu::SpinLock> lockGuard{ serverLock_ };
    auto pos = keyStore_.find(request.keys[0]);
    if (pos == keyStore_.end()) {
        if (request.values[0].empty()) {
            keyStore_[request.keys[0]] = request.values[1];
            lockGuard.unlock();
            server_->WakeupWaitingClients(request.keys[0]);
            return { torch_npu::MessageType::COMPARE_SET, request.fd, request.values[1] };
        }

        lockGuard.unlock();
        return torch_npu::StoreMessage{ torch_npu::MessageType::COMPARE_SET, request.fd};
    }

    if (pos->second == request.values[0]) {
        pos->second = request.values[1];
        lockGuard.unlock();
        return { torch_npu::MessageType::COMPARE_SET, request.fd, request.values[1] };
    }

    lockGuard.unlock();
    return { torch_npu::MessageType::COMPARE_SET, request.fd, pos->second };
}

torch_npu::StoreMessage ParallelStoreServer::ProcessGetNumKeyRequest(int fd, const torch_npu::StoreMessage &request) noexcept
{
    std::unique_lock<torch_npu::SpinLock> lockGuard{ serverLock_ };
    auto keyNum = keyStore_.size();
    lockGuard.unlock();
    return { torch_npu::MessageType::GET_NUM_KEYS, request.fd, torch_npu::StoreMessagePacker::PackPod(keyNum) };
}

torch_npu::StoreMessage ParallelStoreServer::ProcessWaitKeysRequest(int fd, const torch_npu::StoreMessage &request) noexcept
{
    int64_t numKeysToWait = 0;
    std::vector<std::string> waitKeys;
    waitKeys.reserve(request.keys.size());

    std::unique_lock<torch_npu::SpinLock> lockGuard{ serverLock_ };
    if (CheckAllKeysExistInLock(request.keys)) {
        lockGuard.unlock();

        std::vector<uint8_t> body{ static_cast<uint8_t>(torch_npu::MessageWaitKeyRes::KEYS_STOP_WAITING) };
        return { torch_npu::MessageType::WAIT, request.fd, body };
    }

    for (auto &key : request.keys) {
        if (keyStore_.find(key) == keyStore_.end()) {
            waitKeys.emplace_back(key);
            numKeysToWait++;
        }
    }
    server_->SetKeysWaitingSocket(waitKeys, fd, request.fd, numKeysToWait);
    lockGuard.unlock();

    return torch_npu::StoreMessage{ torch_npu::MessageType::INVALID_MSG, request.fd};
}

void ParallelStoreServer::InitializeHandlers() noexcept
{
    requestHandlers_.emplace(torch_npu::MessageType::SET,
        [this](int fd, const torch_npu::StoreMessage &req) { return ProcessSetRequest(fd, req); });
    requestHandlers_.emplace(torch_npu::MessageType::COMPARE_SET,
        [this](int fd, const torch_npu::StoreMessage &req) { return ProcessCompareSetRequest(fd, req); });
    requestHandlers_.emplace(torch_npu::MessageType::GET,
        [this](int fd, const torch_npu::StoreMessage &req) { return ProcessGetRequest(fd, req); });
    requestHandlers_.emplace(torch_npu::MessageType::ADD,
        [this](int fd, const torch_npu::StoreMessage &req) { return ProcessAddRequest(fd, req); });
    requestHandlers_.emplace(torch_npu::MessageType::CHECK,
        [this](int fd, const torch_npu::StoreMessage &req) { return ProcessCheckRequest(fd, req); });
    requestHandlers_.emplace(torch_npu::MessageType::WAIT,
        [this](int fd, const torch_npu::StoreMessage &req) { return ProcessWaitKeysRequest(fd, req); });
    requestHandlers_.emplace(torch_npu::MessageType::GET_NUM_KEYS,
        [this](int fd, const torch_npu::StoreMessage &req) { return ProcessGetNumKeyRequest(fd, req); });
    requestHandlers_.emplace(torch_npu::MessageType::DELETE_KEY,
        [this](int fd, const torch_npu::StoreMessage &req) { return ProcessDeleteRequest(fd, req); });
}

void ParallelStoreServer::LocalInitializeHandlers() noexcept
{
    requestHandlers_.emplace(torch_npu::MessageType::SET,
        [this](int fd, const torch_npu::StoreMessage &req) { return callback_(fd, req); });
    requestHandlers_.emplace(torch_npu::MessageType::COMPARE_SET,
        [this](int fd, const torch_npu::StoreMessage &req) { return callback_(fd, req); });
    requestHandlers_.emplace(torch_npu::MessageType::GET,
        [this](int fd, const torch_npu::StoreMessage &req) { return callback_(fd, req); });
    requestHandlers_.emplace(torch_npu::MessageType::ADD,
        [this](int fd, const torch_npu::StoreMessage &req) { return callback_(fd, req); });
    requestHandlers_.emplace(torch_npu::MessageType::CHECK,
        [this](int fd, const torch_npu::StoreMessage &req) { return callback_(fd, req); });
    requestHandlers_.emplace(torch_npu::MessageType::WAIT,
        [this](int fd, const torch_npu::StoreMessage &req) { return callback_(fd, req); });
    requestHandlers_.emplace(torch_npu::MessageType::GET_NUM_KEYS,
        [this](int fd, const torch_npu::StoreMessage &req) { return callback_(fd, req); });
    requestHandlers_.emplace(torch_npu::MessageType::DELETE_KEY,
        [this](int fd, const torch_npu::StoreMessage &req) { return callback_(fd, req); });
}

bool ParallelStoreServer::CheckAllKeysExistInLock(const std::vector<std::string> &keys) noexcept
{
    return std::all_of(keys.begin(), keys.end(), [this](const std::string &key) { return keyStore_.count(key) > 0; });
}
} // torch_npu

std::mutex ParallelTcpStore::cacheServerMutex_;
std::unordered_map<uint16_t, std::weak_ptr<torch_npu::ParallelStoreServer>> ParallelTcpStore::cachedServers_;

ParallelTcpStore::ParallelTcpStore(const std::string &host, const bool &agentRun, const uint32_t &agentPid,
    const bool &enableTiered, const c10d::TCPStoreOptions &opts)
    : Store(opts.timeout)
{
    if (opts.isServer) {
        auto start_server = std::chrono::high_resolution_clock::now();
        if (opts.multiTenant) {
            server_ = GetSharedServer(initKey_, host, opts.port, opts.numWorkers);
        } else {
            server_ = std::make_shared<torch_npu::ParallelStoreServer>(initKey_, host, opts.port, opts.numWorkers);
        }
        auto end_server = std::chrono::high_resolution_clock::now();
        auto cost_server = std::chrono::duration_cast<std::chrono::microseconds>(end_server - start_server).count();
        ASCEND_LOGI("Create server store success, cost: %d microseconds.", cost_server);
    }
    if (!enableTiered) {
        client_= std::make_unique<torch_npu::Client>(host, opts.port, timeout_);
        if (client_->Connect() != 0) {
        throw std::runtime_error{ std::string("connect tcp client to server(")
                                      .append(host)
                                      .append(":")
                                      .append(std::to_string(opts.port))
                                      .append(" failed.") };
        }
    } else {
        const std::string socketName = "local_abstract_namespace" + std::to_string(agentPid);
        if (agentRun) {
            auto start = std::chrono::high_resolution_clock::now();
            proxy_ = std::make_unique<torch_npu::Proxy>(socketName, host, opts.port, timeout_);
            proxy_->Start();
            auto end = std::chrono::high_resolution_clock::now();
            auto cost = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            ASCEND_LOGI("Create agent store and tcp connect server success, cost: %d microseconds.", cost);
        } else {
            client_= std::make_unique<torch_npu::Client>(socketName, timeout_);
            if (client_->LocalConnect() != 0) {
                throw std::runtime_error{"connect local client to server failed."};
            }
        }
    }

    if (opts.waitWorkers && agentRun) {
        IncreaseKey(initKey_, 1);
        if (opts.isServer) {
            server_->WaitWorkers(timeout_);
        }
    }
}

ParallelTcpStore::~ParallelTcpStore() noexcept
{
    if (proxy_) {
        proxy_->Stop();
    } else {
        client_->LocalClose();
    }
}

void ParallelTcpStore::set(const std::string &key, const std::vector<uint8_t> &value)
{
    torch_npu::StoreMessage request{ torch_npu::MessageType::SET, 0, key, value };
    torch_npu::StoreMessage response;
    std::lock_guard<std::mutex> lockGuard{ clientMutex_ };
    int ret = -1;
    if (proxy_) {
        ret = proxy_->SyncCall(request, response);
    } else {
        ret = client_->SyncCall(request, response);
    }
    if (ret != 0) {
        throw std::runtime_error{ std::string("set key ") + key + " failed or timeout." };
    }
}

std::vector<uint8_t> ParallelTcpStore::compareSet(const std::string &key, const std::vector<uint8_t> &currentValue,
    const std::vector<uint8_t> &newValue)
{
    torch_npu::StoreMessage request{ torch_npu::MessageType::COMPARE_SET, 0, key, currentValue, newValue };
    torch_npu::StoreMessage response;
    std::lock_guard<std::mutex> lockGuard{ clientMutex_ };
    int ret = -1;
    if (proxy_) {
        ret = proxy_->SyncCall(request, response);
    } else {
        ret = client_->SyncCall(request, response);
    }
    if (ret != 0) {
        throw std::runtime_error{ std::string("compare and set key ") + key + " failed or timeout." };
    }

    return response.values.empty() ? std::vector<uint8_t>{} : std::move(response.values[0]);
}

std::vector<uint8_t> ParallelTcpStore::get(const std::string &key)
{
    torch_npu::StoreMessage waitReq{ torch_npu::MessageType::WAIT, 0, key };
    torch_npu::StoreMessage getReq{ torch_npu::MessageType::GET, 0, key };
    torch_npu::StoreMessage waitResp;
    torch_npu::StoreMessage getResp;

    std::lock_guard<std::mutex> lockGuard{ clientMutex_ };
    int ret = -1;
    if (proxy_) {
        ret = proxy_->SyncCall(waitReq, waitResp);
        if (ret != 0) {
            throw std::runtime_error{ std::string("proxy sync wait msg") + key + " failed or timeout." };
        }
        ret = proxy_->SyncCall(getReq, getResp);
        if (ret != 0) {
            throw std::runtime_error{ std::string("proxy sync get msg") + key + " failed or timeout." };
        }
    } else {
        ret = client_->SyncCall(waitReq, waitResp);
        if (ret != 0) {
            throw std::runtime_error{ std::string("get key ") + key + " failed or timeout." };
        }

        ret = client_->SyncCall(getReq, getResp);
        if (ret != 0) {
            throw std::runtime_error{ std::string("get key ") + key + " failed or timeout." };
        }
    }
    return getResp.values.empty() ? std::vector<uint8_t>{} : std::move(getResp.values[0]);
}

int64_t ParallelTcpStore::add(const std::string &key, int64_t value)
{
    return IncreaseKey(key, value);
}

bool ParallelTcpStore::deleteKey(const std::string &key)
{
    torch_npu::StoreMessage request{ torch_npu::MessageType::DELETE_KEY, 0, key };
    torch_npu::StoreMessage response;
    std::lock_guard<std::mutex> lockGuard{ clientMutex_ };
    int ret = -1;
    if (proxy_) {
        ret = proxy_->SyncCall(request, response);
    } else {
        ret = client_->SyncCall(request, response);
    }
    if (ret != 0) {
        throw std::runtime_error{ std::string("delete key ") + key + " failed or timeout." };
    }

    return !response.values.empty() && !response.values[0].empty() && response.values[0][0] > 0U;
}

bool ParallelTcpStore::check(const std::vector<std::string> &keys)
{
    throw std::runtime_error("unsupported check operation.");
}

int64_t ParallelTcpStore::getNumKeys()
{
    torch_npu::StoreMessage request{ torch_npu::MessageType::GET_NUM_KEYS, 0};
    torch_npu::StoreMessage response;
    std::lock_guard<std::mutex> lockGuard{ clientMutex_ };
    int ret = -1;
    if (proxy_) {
        ret = proxy_->SyncCall(request, response);
    } else {
        ret = client_->SyncCall(request, response);
    }
    if (ret != 0) {
        throw std::runtime_error{ "get number keys failed or timeout." };
    }

    return torch_npu::StoreMessagePacker::UnpackPod<int64_t>(response.values[0]);
}

void ParallelTcpStore::wait(const std::vector<std::string> &keys)
{
    wait(keys, timeout_);
}

void ParallelTcpStore::wait(const std::vector<std::string> &keys, const std::chrono::milliseconds &timeout)
{
    torch_npu::StoreMessage request{ torch_npu::MessageType::WAIT, 0, keys };
    torch_npu::StoreMessage response;
    if (proxy_) {
        proxy_->SetReceiveTimeout(timeout);
    } else {
        client_->SetReceiveTimeout(timeout);
    }
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
    torch_npu::StoreMessage request{ torch_npu::MessageType::ADD, 0, key, torch_npu::StoreMessagePacker::PackPod(value) };
    torch_npu::StoreMessage response;
    std::lock_guard<std::mutex> lockGuard{ clientMutex_ };
    int ret = -1;
    if (proxy_) {
        ret = proxy_->SyncCall(request, response);
    } else {
        ret = client_->SyncCall(request, response);
    }

    if (ret != 0) {
        throw std::runtime_error{ std::string("add key ") + key + " failed or timeout." };
    }

    return torch_npu::StoreMessagePacker::UnpackPod<int64_t>(response.values[0]);
}

void ParallelTcpStore::DoWait(const torch_npu::StoreMessage &req, torch_npu::StoreMessage &res)
{
    int ret = -1;
    if (proxy_) {
        ret = proxy_->SyncCall(req, res);
    } else {
        ret = client_->SyncCall(req, res);
    }
    if (ret != 0) {
        throw std::runtime_error{ "get number keys failed or timeout." };
    }
}

std::shared_ptr<torch_npu::ParallelStoreServer> ParallelTcpStore::GetSharedServer(const std::string &initKey,
    const std::string host, uint16_t port, c10::optional<std::size_t> numWorkers)
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
    auto server = std::make_shared<torch_npu::ParallelStoreServer>(initKey, host, port, numWorkers);
    cachedServers_.emplace(port, server);
    return server;
}
} // c10d