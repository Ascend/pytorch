#include <torch/extension.h>
#include <c10/util/irange.h>
#include "torch_npu/csrc/aten/common/from_blob.h"

using namespace at;

static const auto npu_device = at::Device("npu:0");

// Helper: create a 1D float tensor on NPU
static Tensor make_npu_float(std::vector<float> vals)
{
    return torch::tensor(vals, torch::kFloat).to(npu_device);
}

// Helper: create a 1D int32 tensor on NPU
static Tensor make_npu_int(std::vector<int32_t> vals)
{
    return torch::tensor(vals, torch::kInt32).to(npu_device);
}

// Test 1: Basic from_blob with sizes and dtype.
// Verifies: dtype, element count, per-element values match source,
// and storage context is null (non-owning, no deleter).
bool test_from_blob_basic()
{
    auto data = make_npu_float({1.0, 2.0, 3.0});
    auto tensor = at_npu::native::from_blob(
        data.data_ptr(), data.sizes(), torch::dtype(torch::kFloat));
    if (tensor.dtype() != torch::kFloat) return false;
    if (tensor.numel() != data.numel()) return false;
    for (const auto i : c10::irange(data.numel())) {
        if (tensor[i].item<float>() != data[i].item<float>()) return false;
    }
    if (tensor.storage().data_ptr().get_context() != nullptr) return false;
    return true;
}

// Test 2: Deleter is called exactly once when tensor goes out of scope.
bool test_from_blob_deleter()
{
    int called = 0;
    {
        auto data = make_npu_float({1.0, 2.0, 3.0});
        auto tensor = at_npu::native::from_blob(
            data.data_ptr(), data.sizes(), [&](void*) { called++; });
    }
    return (called == 1);
}

// Test 3: from_blob with strides (column-major layout).
// sizes={3,3}, strides={1,3}: element [i][j] = 1 + j*cols + i (column major).
bool test_from_blob_strides()
{
    auto data = make_npu_int({1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto tensor = at_npu::native::from_blob(
        data.data_ptr(), {3, 3}, {1, 3}, torch::kInt32);
    if (tensor.dtype() != torch::kInt32) return false;
    if (tensor.numel() != data.numel()) return false;

    const std::vector<int64_t> expected_strides = {1, 3};
    auto result_strides = tensor.strides();
    if (!std::equal(result_strides.begin(), result_strides.end(),
                    expected_strides.begin()))
        return false;

    for (const auto i : c10::irange(tensor.size(0))) {
        for (const auto j : c10::irange(tensor.size(1))) {
            if (tensor[i][j].item<int32_t>() != (1 + (j * tensor.size(1)) + i))
                return false;
        }
    }
    return true;
}

// Test 4: from_blob with storage_offset, verifies storage size is correct.
// This is the bug-fix test: storage_offset must be multiplied by itemsize.
// Base tensor: arange(kBaseSize) as float32 (4 bytes each).
// View: sizes=[kViewSize], strides=[kStride], offset=kOffset (simulates base[3:8]).
// Expected storage bytes = (kOffset + kStride + kStride*(kViewSize-1)) * sizeof(float)
//                        = (3 + 1 + 1*4) * 4 = 32 bytes.
// Before the fix: storage was computed in mixed units (bytes + elements), giving wrong result.
// Verified: element i of the view equals float(i + kOffset).
bool test_from_blob_storage_offset()
{
    constexpr int64_t kBaseSize = 10;
    constexpr int64_t kViewSize = 5;
    constexpr int64_t kStride = 1;
    constexpr int64_t kOffset = 3;
    auto base = torch::arange(kBaseSize, torch::kFloat).to(npu_device);
    auto tensor = at_npu::native::from_blob(
        base.storage().mutable_data(),
        {kViewSize},
        {kStride},
        kOffset,
        torch::dtype(torch::kFloat));
    if (tensor.storage_offset() != kOffset) return false;
    for (int i = 0; i < kViewSize; i++) {
        if (tensor[i].item<float>() != static_cast<float>(i + kOffset))
            return false;
    }
    size_t expected_nbytes = (kOffset + kStride + kStride * (kViewSize - 1)) * sizeof(float);
    if (tensor.storage().nbytes() != expected_nbytes) return false;
    return true;
}

// Test 5: from_blob with storage_offset for 2D tensor.
// Base: arange(kBaseSize) float32. View: kRows x kCols submatrix of a 4x4 matrix.
// strides=[kRowStride, kColStride], offset=kOffset (row 1, col 1 of 4x4 -> offset=5).
// Expected values: [kOffset, kOffset+kColStride], [kOffset+kRowStride, kOffset+kRowStride+kColStride].
// Expected storage bytes = (kOffset + kColStride + kRowStride*(kRows-1) + kColStride*(kCols-1)) * sizeof(float)
//                        = (5 + 1 + 4*1 + 1*1) * 4 = 44 bytes.
bool test_from_blob_storage_offset_2d()
{
    constexpr int64_t kBaseSize = 16;
    constexpr int64_t kRows = 2;
    constexpr int64_t kCols = 2;
    constexpr int64_t kRowStride = 4;
    constexpr int64_t kColStride = 1;
    constexpr int64_t kOffset = 5;
    auto base = torch::arange(kBaseSize, torch::kFloat).to(npu_device);
    auto tensor = at_npu::native::from_blob(
        base.storage().mutable_data(),
        {kRows, kCols},
        {kRowStride, kColStride},
        kOffset,
        torch::dtype(torch::kFloat));
    if (tensor.storage_offset() != kOffset) return false;
    if (tensor.sizes() != c10::IntArrayRef({kRows, kCols})) return false;
    if (tensor.strides() != c10::IntArrayRef({kRowStride, kColStride})) return false;
    if (tensor[0][0].item<float>() != static_cast<float>(kOffset)) return false;
    if (tensor[0][1].item<float>() != static_cast<float>(kOffset + kColStride)) return false;
    if (tensor[1][0].item<float>() != static_cast<float>(kOffset + kRowStride)) return false;
    if (tensor[1][1].item<float>() != static_cast<float>(kOffset + kRowStride + kColStride)) return false;
    size_t expected_nbytes =
        (kOffset + kColStride + kRowStride * (kRows - 1) + kColStride * (kCols - 1)) * sizeof(float);
    if (tensor.storage().nbytes() != expected_nbytes) return false;
    return true;
}

// Test 6: from_blob with storage_offset for different dtypes.
// Verifies the itemsize multiplier is correct across float64, float16, and int32.
// All sub-tests: sizes=[kViewSize], strides=[kStride], offset=kDtypeOffset.
// float64 (8 bytes): expected = (kDtypeOffset+kStride+kStride*(kViewSize-1))*sizeof(double) = 56.
// float16 (2 bytes): expected = (kDtypeOffset+kStride+kStride*(kViewSize-1))*sizeof(at::Half) = 14.
// int32  (4 bytes): expected = (kDtypeOffset+kStride+kStride*(kViewSize-1))*sizeof(int32_t) = 28.
// The first element of each view should equal kDtypeOffset.
bool test_from_blob_storage_offset_dtype()
{
    bool all_pass = true;
    constexpr int64_t kBaseSize = 10;
    constexpr int64_t kViewSize = 5;
    constexpr int64_t kStride = 1;
    constexpr int64_t kDtypeOffset = 2;

    {
        auto base = torch::arange(kBaseSize, torch::kFloat64).to(npu_device);
        auto tensor = at_npu::native::from_blob(
            base.storage().mutable_data(),
            {kViewSize}, {kStride}, kDtypeOffset,
            torch::dtype(torch::kFloat64));
        size_t expected = (kDtypeOffset + kStride + kStride * (kViewSize - 1)) * sizeof(double);
        if (tensor.storage().nbytes() != expected) all_pass = false;
        if (tensor[0].item<double>() != static_cast<double>(kDtypeOffset)) all_pass = false;
    }

    {
        auto base = torch::arange(kBaseSize, torch::kFloat16).to(npu_device);
        auto tensor = at_npu::native::from_blob(
            base.storage().mutable_data(),
            {kViewSize}, {kStride}, kDtypeOffset,
            torch::dtype(torch::kFloat16));
        size_t expected = (kDtypeOffset + kStride + kStride * (kViewSize - 1)) * sizeof(at::Half);
        if (tensor.storage().nbytes() != expected) all_pass = false;
    }

    {
        auto base = torch::arange(kBaseSize, torch::kInt32).to(npu_device);
        auto tensor = at_npu::native::from_blob(
            base.storage().mutable_data(),
            {kViewSize}, {kStride}, kDtypeOffset,
            torch::dtype(torch::kInt32));
        size_t expected = (kDtypeOffset + kStride + kStride * (kViewSize - 1)) * sizeof(int32_t);
        if (tensor.storage().nbytes() != expected) all_pass = false;
        if (tensor[0].item<int32_t>() != static_cast<int32_t>(kDtypeOffset)) all_pass = false;
    }

    return all_pass;
}

// Test 7: from_blob without explicit strides (contiguous) with storage_offset.
// Verifies storage_offset, element count, first and last element values.
bool test_from_blob_storage_offset_contiguous()
{
    constexpr int64_t kBaseSize = 10;
    constexpr int64_t kViewSize = 5;
    constexpr int64_t kStride = 1;
    constexpr int64_t kOffset = 3;
    auto base = torch::arange(kBaseSize, torch::kFloat).to(npu_device);
    auto tensor = at_npu::native::from_blob(
        base.storage().mutable_data(),
        {kViewSize},
        {kStride},
        kOffset,
        torch::dtype(torch::kFloat));
    if (tensor.storage_offset() != kOffset) return false;
    if (tensor.numel() != kViewSize) return false;
    if (tensor[0].item<float>() != static_cast<float>(kOffset)) return false;
    if (tensor[kViewSize - 1].item<float>() != static_cast<float>(kOffset + kViewSize - 1)) return false;
    return true;
}

// Test 8: from_blob produces a non-owning reference; original tensor stays valid after view is destroyed.
// data={10.0, 20.0, 30.0}. The view (weak) shares the same data pointer.
// After weak goes out of scope, data[0] and data[kLastDataIdx] must still be accessible.
bool test_from_blob_non_owning()
{
    constexpr int64_t kLastDataIdx = 2;
    auto data = make_npu_float({10.0, 20.0, 30.0});
    void* original_ptr = data.data_ptr();
    {
        auto weak = at_npu::native::from_blob(
            data.data_ptr(), data.sizes(), torch::dtype(torch::kFloat));
        if (weak.data_ptr() != original_ptr) return false;
        if (weak[1].item<float>() != 20.0f) return false;
    }
    if (data[0].item<float>() != 10.0f) return false;
    if (data[kLastDataIdx].item<float>() != 30.0f) return false;
    return true;
}

// Test 9: from_blob clone produces an independent copy.
// Cloned tensor must be equal in value but reside in different storage.
bool test_from_blob_clone()
{
    auto data = make_npu_float({1.0, 2.0, 3.0});
    auto tensor = at_npu::native::from_blob(
        data.data_ptr(), data.sizes(), torch::dtype(torch::kFloat));
    auto cloned = tensor.clone();
    if (!at::equal(tensor, cloned)) return false;
    if (cloned.data_ptr() == tensor.data_ptr()) return false;
    return true;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("test_from_blob_basic", &test_from_blob_basic);
    m.def("test_from_blob_deleter", &test_from_blob_deleter);
    m.def("test_from_blob_strides", &test_from_blob_strides);
    m.def("test_from_blob_storage_offset", &test_from_blob_storage_offset);
    m.def("test_from_blob_storage_offset_2d", &test_from_blob_storage_offset_2d);
    m.def("test_from_blob_storage_offset_dtype", &test_from_blob_storage_offset_dtype);
    m.def("test_from_blob_storage_offset_contiguous", &test_from_blob_storage_offset_contiguous);
    m.def("test_from_blob_non_owning", &test_from_blob_non_owning);
    m.def("test_from_blob_clone", &test_from_blob_clone);
}
