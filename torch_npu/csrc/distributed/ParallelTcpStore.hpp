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

#include <pthread.h>
#include <cstdint>
#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <unordered_map>

#include "c10d/TCPStore.hpp"
#include "StoreClient.hpp"
#include "ParallelTcpServer.hpp"
namespace c10d {
namespace torch_npu {
class Proxy;

using CallBackFn = std::function<StoreMessage(const int &, const StoreMessage &)>;
class ParallelStoreServer {
public:
    explicit ParallelStoreServer(std::string initKey, const std::string host, uint16_t port,
        c10::optional<std::size_t> numWorkers) noexcept;
    explicit ParallelStoreServer(const std::string localSocketPath, CallBackFn callback) noexcept;
    virtual ~ParallelStoreServer() noexcept;
    void WaitWorkers(const std::chrono::milliseconds &timeout) noexcept;

private:
    torch_npu::StoreMessage ProcessRequest(int fd, const torch_npu::StoreMessage &request) noexcept;
    torch_npu::StoreMessage ProcessGetRequest(int fd, const torch_npu::StoreMessage &request) noexcept;
    torch_npu::StoreMessage ProcessSetRequest(int fd, const torch_npu::StoreMessage &request) noexcept;
    torch_npu::StoreMessage ProcessAddRequest(int fd, const torch_npu::StoreMessage &request) noexcept;
    torch_npu::StoreMessage ProcessCheckRequest(int fd, const torch_npu::StoreMessage &request) noexcept;
    torch_npu::StoreMessage ProcessDeleteRequest(int fd, const torch_npu::StoreMessage &request) noexcept;
    torch_npu::StoreMessage ProcessCompareSetRequest(int fd, const torch_npu::StoreMessage &request) noexcept;
    torch_npu::StoreMessage ProcessGetNumKeyRequest(int fd, const torch_npu::StoreMessage &request) noexcept;
    torch_npu::StoreMessage ProcessWaitKeysRequest(int fd, const torch_npu::StoreMessage &request) noexcept;
    void InitializeHandlers() noexcept;
    void LocalInitializeHandlers() noexcept;
    bool CheckAllKeysExistInLock(const std::vector<std::string> &keys) noexcept;

private:
    CallBackFn callback_;
    const std::string localSocketPath_;
    using RequestHandler = std::function<torch_npu::StoreMessage(int, const torch_npu::StoreMessage &)>;
    std::unique_ptr<torch_npu::ParallelTcpServer> server_;
    std::unordered_map<torch_npu::MessageType, RequestHandler> requestHandlers_;
    std::unordered_map<std::string, std::vector<uint8_t>> keyStore_;
    SpinLock serverLock_;
    std::mutex initWaitMutex_;
    std::condition_variable initWaitCond_;
    std::atomic<bool> workersReady_{ false };
    const c10::optional<std::size_t> numWorkers_;
    const std::string initKey_ = "init/";
    const std::string keyPrefix_ = "/";
};
} // torch_npu

class ParallelTcpStore : public Store {
public:
    explicit ParallelTcpStore(const std::string &host, const bool &agentRun, const uint32_t &agentPid,
        const bool &enableTiered, const TCPStoreOptions &opts = {});
    ~ParallelTcpStore() noexcept override;

public:
    void set(const std::string &key, const std::vector<uint8_t> &value) override;
    std::vector<uint8_t> compareSet(const std::string &key, const std::vector<uint8_t> &currentValue,
        const std::vector<uint8_t> &newValue) override;
    std::vector<uint8_t> get(const std::string &key) override;
    int64_t add(const std::string &key, int64_t value) override;
    bool deleteKey(const std::string &key) override;
    bool check(const std::vector<std::string> &keys) override;
    int64_t getNumKeys() override;
    void wait(const std::vector<std::string> &keys) override;
    void wait(const std::vector<std::string> &keys, const std::chrono::milliseconds &timeout) override;
    const std::chrono::milliseconds &getTimeout() const noexcept override;
    void setTimeout(const std::chrono::milliseconds &timeout) override;

private:
    int64_t IncreaseKey(const std::string &key, int64_t value);
    void DoWait(const torch_npu::StoreMessage &req, torch_npu::StoreMessage &res);
    static std::shared_ptr<torch_npu::ParallelStoreServer> GetSharedServer(const std::string &initKey,
       const std::string host, uint16_t port, c10::optional<std::size_t> numWorkers);

private:
    std::unique_ptr<torch_npu::Client> client_;
    std::unique_ptr<torch_npu::Proxy> proxy_;
    std::shared_ptr<torch_npu::ParallelStoreServer> server_;
    std::mutex clientMutex_;
    std::condition_variable initWaitCond_;
    const std::string initKey_ = "init/";
    static std::mutex cacheServerMutex_;
    static std::unordered_map<uint16_t, std::weak_ptr<torch_npu::ParallelStoreServer>> cachedServers_;
};
} // c10d
