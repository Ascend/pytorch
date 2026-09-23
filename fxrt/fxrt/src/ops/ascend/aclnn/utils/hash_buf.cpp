#include "ops/ascend/aclnn/utils/hash_buf.h"

namespace fxrt {
namespace ops {
thread_local char gHashBuf[gHashBufSize];
thread_local int gHashOffset = 0;
constexpr int grShift33Bits = 33;
constexpr uint64_t MIX_STEP1 = 18397679294719823053LLU;
constexpr uint64_t MIX_STEP2 = 14181476777654086739LLU;

namespace {
inline uint64_t RotatingLeft(uint64_t x, uint8_t n) {
  return (x << n) | (x >> (64 - n));
}

inline uint64_t Mixture(uint64_t x) {
  // constants step1(18397679294719823053) and step2(14181476777654086739) are used to allow
  // hash values to be more evenly distributed after multiplication.
  x ^= x >> grShift33Bits;
  x *= MIX_STEP1;
  x ^= x >> grShift33Bits;
  x *= MIX_STEP2;
  x ^= x >> grShift33Bits;

  return x;
}

void GenHashTmp(const uint64_t* blocks, const int blockNum, const uint32_t seed, uint64_t* rhas, uint64_t* rhax) {
  CHECK_IF_NULL(blocks);

  // use 9782798678568883157 and 5545529020109919103 for blocking and obfuscation of input data
  const uint64_t c1 = 9782798678568883157LLU;
  const uint64_t c2 = 5545529020109919103LLU;

  uint64_t has = seed;
  uint64_t hax = seed;
  for (int i = 0; i < blockNum; i++) {
    int even_num = 2;
    uint64_t tmp1 = blocks[i * even_num];
    uint64_t tmp2 = blocks[i * even_num + 1];

    int8_t bits31 = 31;
    tmp1 *= c1;
    tmp1 = RotatingLeft(tmp1, bits31);
    tmp1 *= c2;
    has ^= tmp1;

    int8_t bits27 = 27;
    has = RotatingLeft(has, bits27);
    has += hax;
    // increase randomness by mul by 5 and adding a constant
    has = has * 5 + 1390208809;

    int8_t bits33 = 33;
    tmp2 *= c2;
    tmp2 = RotatingLeft(tmp2, bits33);
    tmp2 *= c1;
    hax ^= tmp2;

    hax = RotatingLeft(hax, bits31);
    hax += has;
    // increase randomness by mul by 5 and adding a constant
    hax = hax * 5 + 944331445;
  }

  *rhas = has;
  *rhax = hax;
}
} // namespace

uint64_t GenHash(const void* key, const int len, const uint32_t seed) {
  const uint8_t* data = (const uint8_t*)key;
  // the length of each block is 16 bytes
  const int blockNum = len / 16;
  // has and hax are literal appromix to hash, and hax is the return value of this function.
  uint64_t has = seed;
  uint64_t hax = seed;

  // use 9782798678568883157 and 5545529020109919103 for blocking and obfuscation of input data
  const uint64_t c1 = 9782798678568883157LLU;
  const uint64_t c2 = 5545529020109919103LLU;

  const uint64_t* blocks = (const uint64_t*)(data);

  // update hax
  GenHashTmp(blocks, blockNum, seed, &has, &hax);

  // the length of each block is 16 bytes
  const uint8_t* tail = (const uint8_t*)(data + blockNum * 16);
  uint64_t t1 = 0;
  uint64_t t2 = 0;
  // because the size of a block is 16, different offsets are calculated for tail blocks
  // for different sizes
  switch (static_cast<uint64_t>(len) & 15) {
    case 15:
      t2 ^= ((uint64_t)tail[14]) << 48;
      [[fallthrough]];
      {
      }
    case 14:
      t2 ^= ((uint64_t)tail[13]) << 40;
      [[fallthrough]];
      {
      }
    case 13:
      t2 ^= ((uint64_t)tail[12]) << 32;
      [[fallthrough]];
      {
      }
    case 12:
      t2 ^= ((uint64_t)tail[11]) << 24;
      [[fallthrough]];
      {
      }
    case 11:
      t2 ^= ((uint64_t)tail[10]) << 16;
      [[fallthrough]];
      {
      }
    case 10:
      t2 ^= ((uint64_t)tail[9]) << 8;
      [[fallthrough]];
      {
      }
    case 9:
      t2 ^= ((uint64_t)tail[8]) << 0;
      t2 *= c2;
      t2 = RotatingLeft(t2, 33);
      t2 *= c1;
      hax ^= t2;
      [[fallthrough]];
      {
      }
    case 8:
      t1 ^= ((uint64_t)tail[7]) << 56;
      [[fallthrough]];
      {
      }
    case 7:
      t1 ^= ((uint64_t)tail[6]) << 48;
      [[fallthrough]];
      {
      }
    case 6:
      t1 ^= ((uint64_t)tail[5]) << 40;
      [[fallthrough]];
      {
      }
    case 5:
      t1 ^= ((uint64_t)tail[4]) << 32;
      [[fallthrough]];
      {
      }
    case 4:
      t1 ^= ((uint64_t)tail[3]) << 24;
      [[fallthrough]];
      {
      }
    case 3:
      t1 ^= ((uint64_t)tail[2]) << 16;
      [[fallthrough]];
      {
      }
    case 2:
      t1 ^= ((uint64_t)tail[1]) << 8;
      [[fallthrough]];
      {
      }
    case 1:
      t1 ^= ((uint64_t)tail[0]) << 0;
      t1 *= c1;
      t1 = RotatingLeft(t1, 31);
      t1 *= c2;
      has ^= t1;
      [[fallthrough]];
      {
      }
    default: {
    }
  }

  has ^= static_cast<uint64_t>(len);
  hax ^= static_cast<uint64_t>(len);

  has += hax;
  hax += has;

  has = Mixture(has);
  hax = Mixture(hax);

  has += hax;
  hax += has;
  return hax;
}

} // namespace ops
} // namespace fxrt
