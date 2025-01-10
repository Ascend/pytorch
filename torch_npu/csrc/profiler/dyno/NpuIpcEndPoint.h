#pragma once
#include <cstdlib>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>
#include <string>
#include <memory>
#include <vector>
#include "utils.h"

namespace torch_npu {
namespace profiler {

using fileDesT = int;
constexpr const char STR_END_CHAR = '\0';
constexpr int SOCKET_FD_CHMOD = 0666;

struct NpuPayLoad {
    size_t size;
    void *data;
    NpuPayLoad(size_t size, void *data) : size(size), data(data) {}
};

template <size_t MaxNumFileDes = 0> struct NpuIpcEndPointCtxt {
    struct sockaddr_un messageName;
    size_t messageLen;
    fileDesT *fileDesPtr;
    struct msghdr msghdr;
    std::vector<struct iovec> iov;
    char ancillaryBuf[CMSG_SPACE(MaxNumFileDes * sizeof(fileDesT))];
    explicit NpuIpcEndPointCtxt(size_t num) : iov(std::vector<struct iovec>(num)){};
};

template <size_t MaxNumFileDes = 0> class NpuIpcEndPoint final {
    using Ctxt = NpuIpcEndPointCtxt<MaxNumFileDes>;

public:
    constexpr static size_t addressMaxLen = 108 - 2; // Max unix socket path length
    explicit NpuIpcEndPoint(const std::string &addressName)
    {
        socketFd = socket(AF_UNIX, SOCK_DGRAM, 0);
        if (socketFd == -1) {
            throw std::runtime_error(std::strerror(errno) + PROF_ERROR(ErrCode::PARAM));
        }
        struct sockaddr_un address;
        size_t addressLen = SetSocketAdress(addressName, address);
        if (address.sun_path[0] != STR_END_CHAR) {
            unlink(address.sun_path);
        }
        int res = bind(socketFd, (const struct sockaddr *)&address, addressLen);
        if (res == -1) {
            throw std::runtime_error("Bind socket failed." + PROF_ERROR(ErrCode::PARAM));
        }
        if (address.sun_path[0] != STR_END_CHAR) {
            chmod(address.sun_path, SOCKET_FD_CHMOD);
        }
    }
    ~NpuIpcEndPoint()
    {
        close(socketFd);
    }
    [[nodiscard]] auto BuildSendNpuCtxt(const std::string &desAddrName, const std::vector<NpuPayLoad> &npuPayLoad,
        const std::vector<fileDesT> &fileDes)
    {
        if (fileDes.size() > MaxNumFileDes) {
            throw std::runtime_error("Request to fill more than max connections " + PROF_ERROR(ErrCode::PARAM));
        }
        if (desAddrName.empty()) {
            throw std::runtime_error("Can not send to dest point, because dest socket name is empty " +
                PROF_ERROR(ErrCode::PARAM));
        }
        auto ctxt = BuildNpuCtxt_(npuPayLoad, fileDes.size());
        ctxt->msghdr.msg_namelen = SetSocketAdress(desAddrName, ctxt->messageName);
        if (!fileDes.empty()) {
            if (sizeof(ctxt->fileDesPtr) < fileDes.size() * sizeof(fileDesT)) {
                throw std::runtime_error("Memcpy failed when fileDes size large than ctxt fileDesPtr " +
                    PROF_ERROR(ErrCode::PARAM));
            }
            memcpy(ctxt->fileDesPtr, fileDes.data(), fileDes.size() * sizeof(fileDesT));
        }
        return ctxt;
    }

    [[nodiscard]] bool TrySendMessage(Ctxt const & ctxt, bool retryOnConnRefused = true)
    {
        ssize_t retCode = sendmsg(socketFd, &ctxt.msghdr, MSG_DONTWAIT);
        if (retCode > 0) {
            return true;
        }
        if ((errno == EAGAIN || errno == EWOULDBLOCK) && retCode == -1) {
            return false;
        }
        if (retryOnConnRefused && errno == ECONNREFUSED && retCode == -1) {
            return false;
        }
        throw std::runtime_error("TrySendMessage occur " + std::string(std::strerror(errno)) + " " +
            PROF_ERROR(ErrCode::PARAM));
    }

    [[nodiscard]] auto BuildNpuRcvCtxt(const std::vector<NpuPayLoad> &npuPayLoad)
    {
        return BuildNpuCtxt_(npuPayLoad, MaxNumFileDes);
    }

    [[nodiscard]] bool TryRcvMessage(Ctxt &ctxt) noexcept
    {
        size_t retCode = recvmsg(socketFd, &ctxt.msghdr, MSG_DONTWAIT);
        if (retCode > 0) {
            return true;
        }
        if (retCode == 0) {
            return false;
        }
        if (errno == EWOULDBLOCK || errno == EAGAIN) {
            return false;
        }
        throw std::runtime_error("TryRcvMessage occur " + std::string(std::strerror(errno)) + " " +
            PROF_ERROR(ErrCode::PARAM));
    }

    [[nodiscard]] bool TryPeekMessage(Ctxt &ctxt)
    {
        ssize_t ret = recvmsg(socketFd, &ctxt.msghdr, MSG_DONTWAIT | MSG_PEEK);
        if (ret > 0) {
            return true;
        }
        if (ret == 0) {
            return false;
        }
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            return false;
        }
        throw std::runtime_error("TryPeekMessage occur " + std::string(std::strerror(errno)));
    }

    const char *GetName(Ctxt const & ctxt) const noexcept
    {
        if (ctxt.messageName.sun_path[0] != STR_END_CHAR) {
            throw std::runtime_error("GetName() want to got abstract socket, but got " +
                std::string(ctxt.messageName.sun_path));
        }
        return ctxt.messageName.sun_path + 1;
    }

    std::vector<fileDesT> GetFileDes(const Ctxt &ctxt) const
    {
        struct cmsghdr *cmg = CMSG_FIRSTHDR(&ctxt.msghdl);
        unsigned numFileDes = (cmg->cmsg_len - sizeof(struct cmsghdr)) / sizeof(fileDesT);
        return { ctxt.fileDesPtr, ctxt.fileDesPtr + numFileDes };
    }

protected:
    fileDesT socketFd;
    size_t SetSocketAdress(const std::string &srcSocket, struct sockaddr_un &destSocket)
    {
        if (srcSocket.size() > addressMaxLen) {
            throw std::runtime_error("Abstract UNIX Socket path cannot be larger than addressMaxLen");
        }
        destSocket.sun_family = AF_UNIX;
        destSocket.sun_path[0] = STR_END_CHAR;
        if (srcSocket.empty()) {
            return sizeof(sa_family_t);
        }
        srcSocket.copy(destSocket.sun_path + 1, srcSocket.size());
        destSocket.sun_path[srcSocket.size() + 1] = STR_END_CHAR;
        return sizeof(sa_family_t) + srcSocket.size() + 2;
    }

    auto BuildNpuCtxt_(const std::vector<NpuPayLoad> &npuPayLoad, unsigned numFileDes)
    {
        auto ctxt = std::make_unique<Ctxt>(npuPayLoad.size());
        std::memset(&ctxt->msghdr, 0, sizeof(ctxt->msghdr));
        for (int i = 0; i < npuPayLoad.size(); i++) {
            ctxt->iov[i] = {npuPayLoad[i].data, npuPayLoad[i].size};
        }
        ctxt->msghdr.msg_name = &ctxt->messageName;
        ctxt->msghdr.msg_namelen = sizeof(decltype(ctxt->messageName));
        ctxt->msghdr.msg_iov = ctxt->iov.data();
        ctxt->msghdr.msg_iovlen = npuPayLoad.size();
        ctxt->fileDesPtr = nullptr;
        if (numFileDes == 0) {
            return ctxt;
        }
        const size_t fileDesSize = sizeof(fileDesT) * numFileDes;
        ctxt->msghdr.msg_control = ctxt->ancillaryBuf;
        ctxt->msghdr.msg_controllen = CMSG_SPACE(fileDesSize);

        struct cmsghdr *cmsg = CMSG_FIRSTHDR(&ctxt->msghdr);
        cmsg->cmsg_level = SOL_SOCKET;
        cmsg->cmsg_type = SCM_RIGHTS;
        cmsg->cmsg_len = CMSG_LEN(fileDesSize);
        ctxt->fileDesPtr = (fileDesT *)CMSG_DATA(cmsg);
        return ctxt;
    }
};

} // namespace profiler
} // namespace torch_npu
