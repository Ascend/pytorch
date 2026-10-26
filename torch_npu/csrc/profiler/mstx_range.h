#pragma once

#include <cstring>
#include <string>

#include "torch_npu/csrc/profiler/mstx_mgr.h"
#include "torch_npu/csrc/framework/interface/MsProfilerInterface.h"

namespace torch_npu {
namespace profiler {

inline void mstxMark(const char* message, const aclrtStream stream, const char* domain) {
  if (at_npu::native::IsSupportMstxFunc()) {
    MstxMgr::GetInstance()->mark(message, stream, domain);
  } else {
    (void)at_npu::native::AclProfilingMarkEx(message, strlen(message), stream);
  }
}

inline int mstxRangePush(const char* message, const aclrtStream stream, const char* domain) {
  return MstxMgr::GetInstance()->rangePush(message, stream, domain);
}

inline int mstxRangePop(const char* domain) {
  return MstxMgr::GetInstance()->rangePop(domain);
}

inline int mstxRangeStart(const char* message, const aclrtStream stream, const char* domain) {
  return MstxMgr::GetInstance()->rangeStart(message, stream, domain);
}

inline void mstxRangeEnd(int id, const char* domain) {
  MstxMgr::GetInstance()->rangeEnd(id, domain);
}

inline bool mstxEnable() {
  return MstxMgr::GetInstance()->isMstxEnable();
}

struct MstxRange {
  int rangeId{0};
  mstxDomainHandle_t domainHandle{nullptr};
  MstxRange(const std::string& message, aclrtStream stream, const std::string& domainName = "default") {
    if (!mstxEnable()) {
      return;
    }
    rangeId = MstxMgr::GetInstance()->getRangeId();
    if (at_npu::native::IsSupportMstxDomainFunc()) {
      if (MstxMgr::GetInstance()->isMstxTxDomainEnable(domainName)) {
        domainHandle = MstxMgr::GetInstance()->createProfDomain(domainName);
        at_npu::native::MstxDomainRangeStartA(domainHandle, message.c_str(), stream, rangeId);
      }
    } else {
      at_npu::native::MstxRangeStartA(message.c_str(), stream, rangeId);
    }
  }

  ~MstxRange() {
    if (rangeId == 0 || !mstxEnable()) {
      return;
    }
    if (at_npu::native::IsSupportMstxDomainFunc()) {
      if (domainHandle != nullptr) {
        at_npu::native::MstxDomainRangeEnd(domainHandle, rangeId);
      }
    } else {
      at_npu::native::MstxRangeEnd(rangeId);
    }
  }
};

} // namespace profiler
} // namespace torch_npu
