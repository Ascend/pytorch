#include <dlfcn.h>
#include <vector>
#include <functional>
#include <type_traits>
#include <ATen/Tensor.h>
#include <acl/acl_base.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"
#include "torch_npu/csrc/framework/interface/EnvVariables.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "op_api_common.h"

thread_local char g_hashBuf[kHashBufSize];
thread_local int g_hashOffset = 0;

constexpr int FirstByte = 0;
constexpr int SecondByte = 1;
constexpr int ThirdByte = 2;
constexpr int ForthByte = 3;
constexpr int FifthByte = 4;
constexpr int SixthByte = 5;
constexpr int SeventhByte = 6;

constexpr int RemaindOneByte = 1;
constexpr int RemaindTwoByte = 2;
constexpr int RemaindThreeByte = 3;
constexpr int RemaindFourByte = 4;
constexpr int RemaindFiveByte = 5;
constexpr int RemaindSixByte = 6;
constexpr int RemaindSevenByte = 7;

constexpr int SecondByteShift = 8;
constexpr int ThirdByteShift = 16;
constexpr int ForthByteShift = 24;
constexpr int FifthByteShift = 32;
constexpr int SixthByteShift = 40;
constexpr int SeventhByteShift = 48;

constexpr int Fmix64Shift = 33;

typedef void(*AddTensorAddrToCachedList) (void *addr);

void AddParamToBuf(const at::Tensor &at_tensor) {
    static const auto AddTensorAddrToCachedListAddr = GetOpApiFuncAddr("AddTensorAddrToCachedList");
    AddTensorAddrToCachedList AddTensorAddrToCachedListFunc = reinterpret_cast<AddTensorAddrToCachedList>(AddTensorAddrToCachedListAddr);
    if (!at_tensor.defined()) {
        MEMCPY_TO_BUF(",", 1);
        return;
    }
    // view shape
    MEMCPY_TO_BUF(at_tensor.sizes().data(), at_tensor.sizes().size() * sizeof(int64_t));
    // data type
    auto st = at_tensor.scalar_type();
    MEMCPY_TO_BUF(&st, sizeof(st));
    // seperator
    MEMCPY_TO_BUF(",", 1);
    // strides
    MEMCPY_TO_BUF(at_tensor.strides().data(), at_tensor.sizes().size() * sizeof(int64_t));
    // offset
    auto so = at_tensor.storage_offset();
    MEMCPY_TO_BUF(&so, sizeof(so));
    // storage shape
    aclDataType acl_data_type = at_npu::native::CalcuOpUtil::ConvertToAclDataType(st);
    c10::SmallVector<int64_t, 5> storageDims;
    if (acl_data_type != ACL_STRING) {
        storageDims.push_back(at_tensor.storage().nbytes() / at_tensor.itemsize());
    }
    MEMCPY_TO_BUF(storageDims.data(), storageDims.size() * sizeof(int64_t));

    AddTensorAddrToCachedListFunc(at_tensor.storage().data());
}

void AddParamToBuf(const at::Scalar &at_scalar) {
    at::ScalarType scalar_data_type = at_scalar.type();
    switch (scalar_data_type) {
        case at::ScalarType::Double: {
            double value = at_scalar.toDouble();
            MEMCPY_TO_BUF(&value, sizeof(double));
            break;
        }
        case at::ScalarType::Long: {
            int64_t value = at_scalar.toLong();
            MEMCPY_TO_BUF(&value, sizeof(int64_t));
            break;
        }
        case at::ScalarType::Bool: {
            bool value = at_scalar.toBool();
            MEMCPY_TO_BUF(&value, sizeof(bool));
            break;
        }
        case at::ScalarType::ComplexDouble: {
            auto value = at_scalar.toComplexDouble();
            MEMCPY_TO_BUF(&value, sizeof(value));
            break;
        }
        default: {
            break;
        }
    }
}

void AddParamToBuf(const at::IntArrayRef &at_array) {
    MEMCPY_TO_BUF(at_array.data(), at_array.size() * sizeof(int64_t));
}

void AddParamToBuf(const at::ArrayRef<bool> &at_array) {
    MEMCPY_TO_BUF(at_array.data(), at_array.size() * sizeof(bool));
}

void AddParamToBuf(const at::TensorList &at_tensor_list) {
    for (size_t i = 0; i < at_tensor_list.size(); i++) {
        AddParamToBuf(at_tensor_list[i]);
    }
    auto counter = at_tensor_list.size();
    MEMCPY_TO_BUF(&counter, sizeof(counter));
}

void AddParamToBuf(const c10::optional<at::Tensor> &opt_tensor) {
    if (opt_tensor.has_value() && opt_tensor.value().defined()) {
        AddParamToBuf(opt_tensor.value());
    } else {
        MEMCPY_TO_BUF(",", 1);
    }
}

void AddParamToBuf(const c10::optional<at::IntArrayRef> &opt_array) {
    if (opt_array.has_value()) {
        AddParamToBuf(opt_array.value());
    } else {
        MEMCPY_TO_BUF(",", 1);
    }
}

void AddParamToBuf(const c10::optional<at::Scalar> &opt_scalar) {
    if (opt_scalar.has_value()) {
        AddParamToBuf(opt_scalar.value());
    } else {
        MEMCPY_TO_BUF(",", 1);
    }
}

void AddParamToBuf(const at::ScalarType scalar_type) {
    MEMCPY_TO_BUF(&scalar_type, sizeof(scalar_type));
}

void AddParamToBuf(const string& s) {
    MEMCPY_TO_BUF(s.c_str(), s.size());
}

void AddParamToBuf() {}

inline uint64_t fmix64(uint64_t k) {
    // 0xff51afd7ed558ccd and 0xc4ceb9fe1a85ec53 are carefully selected constants to allow
    // hash values to be more evenly distributed in 64-bit space after multiplication.
    k ^= k >> Fmix64Shift;
    k *= 0xff51afd7ed558ccd;
    k ^= k >> Fmix64Shift;
    k *= 0xc4ceb9fe1a85ec53;
    k ^= k >> Fmix64Shift;

    return k;
}

uint64_t MurmurHash(const void * key, const uint32_t len, const uint32_t seed = 0x271812) {
    // m represents a constant of the hash function, which is a large prime number.
    // this constant is used to limit the output value of the hash function to a specific range
    // for subsequent processing and use.
    const uint64_t m = 0xc6a4a7935bd1e995;
    // each input bytes is process with r, ensure that each byte of the input data has an impact
    // on the final hash value.
    const int r = 47;

    uint64_t h = seed ^ (len * m);

    const uint64_t * data = (const uint64_t *)key;
    const uint64_t * end = data + (len / 8);

    while (data != end) {
        uint64_t k = *data++;

        k *= m;
        k ^= k >> r;
        k *= m;

        h ^= k;
        h *= m;
    }

    const uint8_t * data2 = (const uint8_t*)data;

    switch (len & RemaindSevenByte) {
        case RemaindSevenByte:
            h ^= uint64_t(data2[SeventhByte]) << SeventhByteShift;
            break;
        case RemaindSixByte:
            h ^= uint64_t(data2[SixthByte]) << SixthByteShift;
            break;
        case RemaindFiveByte:
            h ^= uint64_t(data2[FifthByte]) << FifthByteShift;
            break;
        case RemaindFourByte:
            h ^= uint64_t(data2[ForthByte]) << ForthByteShift;
            break;
        case RemaindThreeByte:
            h ^= uint64_t(data2[ThirdByte]) << ThirdByteShift;
            break;
        case RemaindTwoByte:
            h ^= uint64_t(data2[SecondByte]) << SecondByteShift;
            break;
        case RemaindOneByte:
            h ^= uint64_t(data2[FirstByte]);
            h *= m;
    };

    h = fmix64(h);

    return h;
}

uint64_t CalcHashId() {
    if (g_hashOffset == kHashBufMaxSize) {
        return 0;
    }
    uint64_t hash_id = MurmurHash(g_hashBuf, g_hashOffset);
    return hash_id;
}