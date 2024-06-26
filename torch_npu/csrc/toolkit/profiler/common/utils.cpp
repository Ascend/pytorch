#include "torch_npu/csrc/toolkit/profiler/common/utils.h"
#include <vector>
#include <sstream>
#include <algorithm>
#include <sys/types.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/in.h>
#include <netpacket/packet.h>

namespace {
template<typename T>
std::string IntToHexStr(T number)
{
    std::stringstream strStream;
    strStream << std::hex << number;
    return strStream.str();
}

std::string Join(const std::vector<std::string> &elems, const std::string &sep)
{
    std::stringstream result;
    for (size_t i = 0; i < elems.size(); ++i) {
        if (i == 0) {
            result << elems[i];
        } else {
            result << sep << elems[i];
        }
    }
    return result.str();
}

uint64_t CalcHashId(const std::string &data)
{
    static const uint32_t UINT32_BITS = 32;
    uint32_t prime[2] = {29, 131};
    uint32_t hash[2] = {0};
    for (char d : data) {
        hash[0] = hash[0] * prime[0] + static_cast<uint32_t>(d);
        hash[1] = hash[1] * prime[1] + static_cast<uint32_t>(d);
    }
    return (static_cast<uint64_t>(hash[0]) << UINT32_BITS) | hash[1];
}
}

namespace torch_npu {
namespace toolkit {
namespace profiler {

uint64_t Utils::GetHostUid()
{
    static const uint8_t SECOND_LEAST_BIT = 1 << 1;
    struct ifaddrs *ifaddr = nullptr;
    if (getifaddrs(&ifaddr) == -1) {
        if (ifaddr != nullptr) {
            freeifaddrs(ifaddr);
        }
        return 0;
    }
    std::vector<std::string> universalMacAddrs;
    std::vector<std::string> localMacAddrs;
    for (struct ifaddrs *ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == nullptr || ifa->ifa_addr->sa_family != AF_PACKET) {
            continue;
        }
        if ((ifa->ifa_flags & IFF_LOOPBACK) != 0) {
            continue;
        }
        struct sockaddr_ll *lladdr = reinterpret_cast<struct sockaddr_ll*>(ifa->ifa_addr);
        uint32_t len = static_cast<uint32_t>(lladdr->sll_halen);
        if (len > 0) {
            std::string addr;
            for (uint32_t i = 0; i < len; ++i) {
                std::string hexAddr = IntToHexStr(static_cast<uint16_t>(lladdr->sll_addr[i]));
                addr += (hexAddr.length() > 1) ? hexAddr : ("0" + hexAddr);
            }
            if ((lladdr->sll_addr[0] & SECOND_LEAST_BIT) == 0) {
                universalMacAddrs.emplace_back(addr);
            } else {
                localMacAddrs.emplace_back(addr);
            }
        }
    }
    if (ifaddr != nullptr) {
        freeifaddrs(ifaddr);
    }
    if (universalMacAddrs.empty() && localMacAddrs.empty()) {
        return 0;
    }
    auto &macAddrs = universalMacAddrs.empty() ? localMacAddrs : universalMacAddrs;
    std::sort(macAddrs.begin(), macAddrs.end());
    return CalcHashId(Join(macAddrs, "-"));
}

int Utils::safe_strcpy_s(char* dest, const char* src, size_t destSize)
{
    if (dest == nullptr || src == nullptr || destSize == 0) {
        return -1;
    }
    size_t i = 0;
    for (; i < destSize - 1 && src[i] != '\0'; ++i) {
        dest[i] = src[i];
    }
    dest[i] = '\0';
    if (src[i] != '\0') {
        dest[0] = '\0';
        return -1;
    }
    return 0;
}

}
}
}
