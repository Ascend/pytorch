#ifdef USE_RPC_FRAMEWORK

#pragma once

#include <torch/csrc/distributed/rpc/message.h>
#include <torch_npu/csrc/distributed/rpc/tensorpipe_agent.h>

namespace torch_npu {
namespace distributed {
namespace rpc {

struct FaultyTensorPipeRpcBackendOptions : public TensorPipeRpcBackendOptions {
  FaultyTensorPipeRpcBackendOptions(
      int num_worker_threads,
      float rpc_timeout,
      std::string init_method,
      std::vector<std::string> messages_to_fail,
      std::unordered_map<std::string, float> messages_to_delay,
      int num_fail_sends = 0)
      : TensorPipeRpcBackendOptions(
            num_worker_threads,
            std::optional<std::vector<std::string>>(),
            std::optional<std::vector<std::string>>(),
            rpc_timeout,
            std::move(init_method)),
        messagesToFail(std::move(messages_to_fail)),
        messagesToDelay(std::move(messages_to_delay)),
        numFailSends(num_fail_sends) {
    TORCH_CHECK(numFailSends >= 0, "numFailSends should be non-negative");
  }

  std::vector<std::string> messagesToFail;
  std::unordered_map<std::string, float> messagesToDelay;
  int numFailSends;
};

class FaultyTensorPipeAgent : public TensorPipeAgent {
 public:
  FaultyTensorPipeAgent(
      const c10::intrusive_ptr<::c10d::Store>& store,
      std::string selfName,
      worker_id_t selfId,
      c10::optional<int> worldSize,
      FaultyTensorPipeRpcBackendOptions opts,
      std::unordered_map<std::string, DeviceMap> reverseDeviceMaps,
      std::vector<c10::Device> devices,
      std::unique_ptr<RequestCallback> callback);

  c10::intrusive_ptr<JitFuture> send(
      const WorkerInfo& to,
      c10::intrusive_ptr<Message> message,
      const float rpcTimeoutSeconds = kUnsetRpcTimeout,
      const DeviceMap& deviceMap = {}) override;

  void pipeWrite(
      const std::shared_ptr<tensorpipe_npu::Pipe>& pipe,
      c10::intrusive_ptr<Message> rpcMessage,
      std::vector<c10::Device>&& devices,
      std::vector<c10::Stream> streams,
      std::function<void(const tensorpipe_npu::Error&)> fn) noexcept override;

 protected:
  bool shouldFailMessage(MessageType type) const;

 private:
  std::vector<MessageType> parseMessagesToFailInput(
      const std::vector<std::string>& messagesToFail) const;

  float getDelayForMessage(MessageType type) const;

  std::unordered_map<MessageType, float, std::hash<int>> parseMessagesToDelay(
      const std::unordered_map<std::string, float>& messageTypesToDelay) const;

  const int numFailSends_;

  const std::vector<MessageType> messageTypesToFail_;

  std::unordered_map<MessageType, float, std::hash<int>> messageTypesToDelay_;

  std::unordered_map<std::string, int> failMessageCountMap_;

  std::mutex failMapMutex_;

  MessageType messageStringToType(const std::string& messageString) const;
};

} // namespace rpc
} // namespace distributed
} // namespace torch_npu

#endif
