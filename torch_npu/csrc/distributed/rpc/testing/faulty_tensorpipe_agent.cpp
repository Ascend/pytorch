#ifdef USE_RPC_FRAMEWORK

#include <torch_npu/csrc/distributed/rpc/testing/faulty_tensorpipe_agent.h>
#include <torch/csrc/distributed/rpc/utils.h>

namespace torch_npu {
namespace distributed {
namespace rpc {

static std::string fromVecToString(const std::vector<char>& vec) {
  return std::string(vec.begin(), vec.end());
}

FaultyTensorPipeAgent::FaultyTensorPipeAgent(
    const c10::intrusive_ptr<::c10d::Store>& store,
    std::string selfName,
    worker_id_t selfId,
    c10::optional<int> worldSize,
    FaultyTensorPipeRpcBackendOptions opts,
    std::unordered_map<std::string, DeviceMap> reverseDeviceMaps,
    std::vector<c10::Device> devices,
    std::unique_ptr<RequestCallback> callback)
    : TensorPipeAgent(
          store,
          std::move(selfName),
          selfId,
          worldSize,
          static_cast<TensorPipeRpcBackendOptions>(opts),
          std::move(reverseDeviceMaps),
          std::move(devices),
          std::move(callback)),
      numFailSends_(opts.numFailSends),
      messageTypesToFail_(parseMessagesToFailInput(
          std::move(opts.messagesToFail))),
      messageTypesToDelay_(parseMessagesToDelay(
          std::move(opts.messagesToDelay))) {}

std::vector<MessageType> FaultyTensorPipeAgent::parseMessagesToFailInput(
    const std::vector<std::string>& messagesToFail) const {
  std::vector<MessageType> messageTypesToFail;
  messageTypesToFail.reserve(messagesToFail.size());
  for (const auto& msgString : messagesToFail) {
    messageTypesToFail.push_back(messageStringToType(msgString));
  }
  return messageTypesToFail;
}

std::unordered_map<MessageType, float, std::hash<int>> FaultyTensorPipeAgent::
    parseMessagesToDelay(const std::unordered_map<std::string, float>&
                             messageTypesToDelay) const {
  std::unordered_map<MessageType, float, std::hash<int>> delayMessages;
  for (const auto& messagePair : messageTypesToDelay) {
    float delay = messagePair.second;
    TORCH_CHECK(
        delay >= 0,
        "Delays passed to FaultyTensorPipeAgent must be non-negative.")
    delayMessages.insert({messageStringToType(messagePair.first), delay});
  }
  return delayMessages;
}

c10::intrusive_ptr<JitFuture> FaultyTensorPipeAgent::send(
    const WorkerInfo& to,
    c10::intrusive_ptr<Message> message,
    const float rpcTimeoutSeconds,
    const DeviceMap& /* unused */) {
  if (!shouldFailMessage(message->type())) {
    return TensorPipeAgent::send(to, std::move(message), rpcTimeoutSeconds);
  }

  const auto key = fromVecToString(message->payload());
  std::unique_lock<std::mutex> lock(failMapMutex_);
  auto it = failMessageCountMap_.find(key);
  if (it == failMessageCountMap_.end()) {
    failMessageCountMap_[key] = 0;
  }
  if (failMessageCountMap_[key] < numFailSends_) {
    failMessageCountMap_[key]++;
    lock.unlock();
    auto jitFuture = c10::make_intrusive<JitFuture>(at::AnyClassType::get());
    jitFuture->setError(std::make_exception_ptr(std::runtime_error(makeRPCError(
        c10::str("Send attempt failed intentionally for ", key),
        RPCErrorType::INTENTIONAL_FAILURE))));
    return jitFuture;
  } else {
    lock.unlock();
    return TensorPipeAgent::send(to, std::move(message), rpcTimeoutSeconds);
  }
}

void FaultyTensorPipeAgent::pipeWrite(
    const std::shared_ptr<tensorpipe_npu::Pipe>& pipe,
    c10::intrusive_ptr<Message> rpcMessage,
    std::vector<c10::Device>&& devices,
    std::vector<c10::Stream> streams,
    std::function<void(const tensorpipe_npu::Error&)> fn) noexcept {
  float msgDelay = getDelayForMessage(rpcMessage->type());
  if (msgDelay != 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(
        static_cast<int>(msgDelay * kSecToMsConversion)));
  }
  TensorPipeAgent::pipeWrite(pipe, rpcMessage, std::move(devices), streams, fn);
}

bool FaultyTensorPipeAgent::shouldFailMessage(MessageType type) const {
  return (
      std::find(messageTypesToFail_.begin(), messageTypesToFail_.end(), type) !=
      messageTypesToFail_.end());
}

float FaultyTensorPipeAgent::getDelayForMessage(MessageType type) const {
  const auto& it = messageTypesToDelay_.find(type);
  return it == messageTypesToDelay_.end() ? 0 : it->second;
}

MessageType FaultyTensorPipeAgent::messageStringToType(
    const std::string& messageString) const {
  static std::unordered_map<std::string, MessageType> msgMap = {
      {"RREF_FORK_REQUEST", MessageType::RREF_FORK_REQUEST},
      {"RREF_CHILD_ACCEPT", MessageType::RREF_CHILD_ACCEPT},
      {"RREF_USER_DELETE", MessageType::RREF_USER_DELETE},
      {"CLEANUP_AUTOGRAD_CONTEXT_REQ",
       MessageType::CLEANUP_AUTOGRAD_CONTEXT_REQ},
      {"PYTHON_REMOTE_CALL", MessageType::PYTHON_REMOTE_CALL},
      {"SCRIPT_REMOTE_CALL", MessageType::SCRIPT_REMOTE_CALL},
      {"PYTHON_CALL", MessageType::PYTHON_CALL},
      {"SCRIPT_CALL", MessageType::SCRIPT_CALL},
      {"PYTHON_RREF_FETCH_CALL", MessageType::PYTHON_RREF_FETCH_CALL},
      {"SCRIPT_RREF_FETCH_CALL", MessageType::SCRIPT_RREF_FETCH_CALL}};
  const auto& it = msgMap.find(messageString);
  TORCH_CHECK(
      it != msgMap.end(),
      "No mapping to rpc::MessageType exists for ",
      messageString);
  return it->second;
}

} // namespace rpc
} // namespace distributed
} // namespace torch_npu

#endif
