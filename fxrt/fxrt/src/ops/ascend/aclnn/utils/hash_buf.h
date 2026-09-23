#ifndef __OPS_ASCEND_ACLNN_UTILS_HASH_BUF_H__
#define __OPS_ASCEND_ACLNN_UTILS_HASH_BUF_H__

#include <cstdint>
#include <cstring>

#include "common/visible.h"
#include "common/common.h"

namespace fxrt {
namespace ops {
inline constexpr int gHashBufSize = 8192;
inline constexpr int gHashBufMaxSize = gHashBufSize + 1024;
extern DA_API thread_local char gHashBuf[gHashBufSize];
extern DA_API thread_local int gHashOffset;

inline void MemcpyToBuf(const void* data, size_t size) {
  if (size == 0) {
    return;
  }
  if (FXRT_UNLIKELY(static_cast<uint64_t>(gHashOffset) > SIZE_MAX - size)) {
    RT_GLOG(ERROR) << "Hash buf is overflow.";
    return;
  }
  if (gHashOffset + size >= gHashBufSize) {
    gHashOffset = gHashBufMaxSize;
    return;
  }
  // gHashOffset + size < gHashBufSize was just checked, and gHashBuf never
  // overlaps the hashed data, so a plain memcpy is safe here.
  std::memcpy(gHashBuf + gHashOffset, data, size);
  gHashOffset += size;
}

DA_API uint64_t GenHash(const void* key, const int len, const uint32_t seed = 0xdeadb0d7);

inline uint64_t CalcHashId() {
  if (gHashOffset == gHashBufMaxSize) {
    return 0;
  }
  uint64_t hashId = GenHash(gHashBuf, gHashOffset);
  return hashId;
}

} // namespace ops
} // namespace fxrt
#endif // __OPS_ASCEND_ACLNN_UTILS_HASH_BUF_H__
