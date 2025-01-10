#include "NpuIpcClient.h"
namespace torch_npu {
namespace profiler {

bool torch_npu::profiler::IpcClient::RegisterInstance(int32_t id)
{
    NpuContext context{
        .npu = id,
        .pid = getpid(),
        .jobId = JOB_ID,
    };
    std::unique_ptr<Message> message = Message::ConstructMessage<decltype(context)>(context, "ctxt");
    try {
        if (!SyncSendMessage(*message, std::string(DYNO_IPC_NAME))) {
            ASCEND_LOGW("Failed to send register ctxt for pid %d with dyno", context.pid);
            return false;
        }
    } catch (const std::exception &e) {
        ASCEND_LOGW("Error when SyncSendMessage %s !", e.what());
        return false;
    }
    ASCEND_LOGI("Resigter pid %d for dynolog success !", context.pid);
    return true;
}
std::string IpcClient::IpcClientNpuConfig()
{
    int size = pids_.size();
    auto *req = (NpuRequest *)malloc(sizeof(NpuRequest) + sizeof(int32_t) * size);
    req->type = DYNO_IPC_TYPE;
    req->pidSize = size;
    req->jobId = JOB_ID;
    for (int i = 0; i < size; i++) {
        req->pids[i] = pids_[i];
    }
    std::unique_ptr<Message> message = Message::ConstructMessage<NpuRequest, int32_t>(*req, "req", size);
    if (!SyncSendMessage(*message, std::string(DYNO_IPC_NAME))) {
        ASCEND_LOGW("Failed to send config  to dyno server fail !");
        free(req);
        req = nullptr;
        return "";
    }
    free(req);
    message = PollRecvMessage(MAX_IPC_RETRIES, MAX_SLEEP_US);
    if (!message) {
        ASCEND_LOGW("Failed to receive on-demand config !");
        return "";
    }
    std::string res = std::string((char *)message->buf.get(), message->metadata.size);
    return res;
}
std::unique_ptr<Message> IpcClient::ReceiveMessage()
{
    std::lock_guard<std::mutex> wguard(dequeLock_);
    if (msgDynoDeque_.empty()) {
        return nullptr;
    }
    std::unique_ptr<Message> message = std::move(msgDynoDeque_.front());
    msgDynoDeque_.pop_front();
    return message;
}
bool IpcClient::SyncSendMessage(const Message &message, const std::string &destName, int numRetry, int seepTimeUs)
{
    if (destName.empty()) {
        ASCEND_LOGW("Can not send to empty socket name !");
        return false;
    }
    int i = 0;
    std::vector<NpuPayLoad> npuPayLoad{ NpuPayLoad(sizeof(struct Metadata), (void *)&message.metadata),
        NpuPayLoad(message.metadata.size, message.buf.get()) };
    try {
        auto ctxt = ep_.BuildSendNpuCtxt(destName, npuPayLoad, std::vector<int>());
        while (!ep_.TrySendMessage(*ctxt) && i < numRetry) {
            i++;
            usleep(seepTimeUs);
            seepTimeUs *= 2;
        }
    } catch (const std::exception &e) {
        ASCEND_LOGW("Error when SyncSendMessage %s !", e.what());
        return false;
    }
    return i < numRetry;
}
bool IpcClient::Recv()
{
    try {
        Metadata recvMetadata;
        std::vector<NpuPayLoad> PeekNpuPayLoad{ NpuPayLoad(sizeof(struct Metadata), &recvMetadata) };
        auto peekCtxt = ep_.BuildNpuRcvCtxt(PeekNpuPayLoad);
        bool successFlag = false;
        try {
            successFlag = ep_.TryPeekMessage(*peekCtxt);
        } catch (std::exception &e) {
            ASCEND_LOGW("ERROR when TryPeekMessage: %s !", e.what());
            return false;
        }
        if (successFlag) {
            std::unique_ptr<Message> npuMessage = std::make_unique<Message>(Message());
            npuMessage->metadata = recvMetadata;
            npuMessage->buf = std::unique_ptr<unsigned char[]>(new unsigned char[recvMetadata.size]);
            npuMessage->src = std::string(ep_.GetName(*peekCtxt));
            std::vector<NpuPayLoad> npuPayLoad{ NpuPayLoad(sizeof(struct Metadata), (void *)&npuMessage->metadata),
                NpuPayLoad(recvMetadata.size, npuMessage->buf.get()) };
            auto recvCtxt = ep_.BuildNpuRcvCtxt(npuPayLoad);
            try {
                successFlag = ep_.TryRcvMessage(*recvCtxt);
            } catch (std::exception &e) {
                ASCEND_LOGW("Error when TryRecvMsg: %s !", e.what());
                return false;
            }
            if (successFlag) {
                std::lock_guard<std::mutex> wguard(dequeLock_);
                msgDynoDeque_.push_back(std::move(npuMessage));
                return true;
            }
        }
    } catch (std::exception &e) {
        ASCEND_LOGW("Error in Recv(): %s !", e.what());
        return false;
    }
    return false;
}
std::unique_ptr<Message> IpcClient::PollRecvMessage(int maxRetry, int sleeTimeUs)
{
    for (int i = 0; i < maxRetry; i++) {
        if (Recv()) {
            return ReceiveMessage();
        }
        usleep(sleeTimeUs);
    }
    return nullptr;
}

} // namespace profiler
} // namespace torch_npu