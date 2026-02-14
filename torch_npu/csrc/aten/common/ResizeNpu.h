#pragma once

#include <ATen/ATen.h>

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/interface/AsyncTaskQueueInterface.h"
#include "torch_npu/csrc/framework/utils/NpuUtils.h"
#include "torch_npu/csrc/framework/FormatHelper.h"
#include "torch_npu/csrc/framework/StorageDescHelper.h"
#include "torch_npu/csrc/core/NPUBridge.h"
#include "torch_npu/csrc/core/NPUStorageImpl.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"

namespace at_npu {
namespace native {

// Refresh storage_desc to ND if set force_refresh = true,
// mainly used in storage.resize_
static void storage_resize_npu(
    torch_npu::NPUStorageImpl& storage,
    ptrdiff_t size,
    c10::IntArrayRef new_size,
    bool force_refresh = false)
{
    if (!storage.resizable()) {
        TORCH_CHECK(false, "Trying to resize storage that is not resizable", OPS_ERROR(ErrCode::NOT_SUPPORT));
        return;
    }

    auto &storage_desc = torch_npu::NPUBridge::GetNpuStorageImpl(&storage)->npu_desc_;
    if (!FormatHelper::IsBaseFormatType(storage_desc.npu_format_)) {
        TORCH_CHECK(false, "Cannot resize storage without base format", OPS_ERROR(ErrCode::NOT_SUPPORT));
        return;
    }

    at::DataPtr new_data = storage.allocator()->allocate(size);
    if (size > 0) {
        TORCH_CHECK(new_data, "Get new_data failed", PTA_ERROR(ErrCode::PARAM));
    }
    size_t itemsize = storage_desc.data_type_.itemsize();
    at::DataPtr old_data = storage.set_data_ptr(std::move(new_data));
    ptrdiff_t old_size = static_cast<ptrdiff_t>(storage.nbytes());
    storage.set_nbytes(size);

    if (itemsize == 0) {
        AT_ERROR("When resizing, item size of storage cannot be zero.");
        return;
    }
    if ((size % static_cast<ptrdiff_t>(itemsize)) != 0) {
        AT_ERROR("The specified storage nbytes cannot be divided by item size.",
                 "Please check the input parameter size.");
        return;
    }
    std::vector<int64_t> resize_shape = {
        size / static_cast<ptrdiff_t>(itemsize)
    };
    // It is necessary to properly refresh the storage according to sizes and strides,
    // not just new sizes.
    if (force_refresh) {
        int64_t new_data_numel = c10::multiply_integers(resize_shape);
        int64_t new_shape_numel = c10::multiply_integers(new_size);
        const c10::IntArrayRef &refresh_size = new_data_numel > new_shape_numel ? resize_shape : new_size;

        // 计算连续场景下size对应的stride值
        int64_t dim_ = static_cast<int64_t>(refresh_size.size());
        c10::SmallVector<int64_t, 5> new_stride(dim_);
        if (dim_ > 0) {
            int64_t last_idx = dim_ - 1;
            new_stride[last_idx] = 1;
            for (auto i = last_idx - 1; i >= 0; --i) {
                new_stride[i] = new_stride[i + 1] * std::max<int64_t>(refresh_size[i + 1], 1);
            }
        }

        storage_desc.base_sizes_ = refresh_size;
        storage_desc.base_strides_ = new_stride;
        storage_desc.npu_format_ = ACL_FORMAT_ND;
        storage_desc.storage_sizes_ = storage_desc.base_sizes_;
    } else {
        StorageDescHelper::UpdateDesc(storage_desc, resize_shape, new_size);
    }

    if (old_data != nullptr) {
        ptrdiff_t copy_size = old_size;
        if (static_cast<ptrdiff_t>(storage.nbytes()) < copy_size) {
            copy_size = static_cast<ptrdiff_t>(storage.nbytes());
        }
        if (copy_size > 0) {
            aclError error = CalcuOpUtil::LaunchAsyncCopyTaskWithModeSwitch(
                storage,
                copy_size,
                old_data.get(),
                copy_size,
                ACL_MEMCPY_DEVICE_TO_DEVICE);
            if (error != ACL_ERROR_NONE) {
                AT_ERROR("ACL_Memcpy device to device error.");
                return;
            }
        }
    }
}

static inline void maybe_resize_storage_npu(
    at::TensorImpl* self,
    int64_t new_size,
    c10::IntArrayRef size)
{
    if (new_size > 0) {
        if (!self->storage().unsafeGetStorageImpl()) {
            AT_ERROR("Try to resize a tensor with null storage");
        }
        int64_t new_size_bytes =
            (new_size + self->storage_offset()) * static_cast<int64_t>(self->dtype().itemsize());
        if (new_size_bytes > static_cast<int64_t>(self->storage().nbytes())) {
            storage_resize_npu(
                *torch_npu::NPUBridge::GetNpuStorageImpl(self->storage().unsafeGetStorageImpl()),
                new_size_bytes,
                size);
        }
    }
}

inline at::TensorImpl* resize_impl_npu_(
    at::TensorImpl* self,
    c10::IntArrayRef size,
    c10::optional<c10::IntArrayRef> stride)
{
    if (self->sizes() == size && (!stride || self->strides() == stride)) {
        return self;
    }

    int64_t storage_size = 1;
    if (stride.has_value()) {
        self->set_sizes_and_strides(size, *stride);
        for (size_t dim = 0; dim < size.size(); ++dim) {
            if (size[dim] == 0) {
                storage_size = 0;
                break;
            }
            storage_size += (size[dim] - 1) * stride.value()[dim];
        }
    } else {
        self->set_sizes_contiguous(size);
        storage_size = self->numel();
    }
    maybe_resize_storage_npu(self, storage_size, size);
    
    return self;
}

static void resize_nd_npu(
    at::TensorImpl* self,
    int nDimension,
    const int64_t* size,
    const int64_t* stride)
{
    c10::IntArrayRef sizes(size, nDimension);
    at::optional<c10::IntArrayRef> strides;
    if (stride != nullptr) {
        strides = c10::IntArrayRef(stride, nDimension);
    }
    resize_impl_npu_(self, sizes, strides);
}

static inline void checkInBoundsForStorage(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    int64_t storage_offset,
    const caffe2::TypeMeta data_type,
    const c10::Storage& new_storage)
{
    int64_t storage_size_bytes =
        static_cast<int64_t>(at::detail::computeStorageNbytes(size, stride, data_type.itemsize()));
    int64_t storage_offset_bytes = storage_offset * static_cast<int64_t>(data_type.itemsize());
    if (storage_size_bytes == 0) {
        // NB: (a tensor with arbitrary 0 dims)'s storage can have any numel.
        return;
    }

    int64_t new_storage_size_bytes = static_cast<int64_t>(new_storage.nbytes());
    TORCH_CHECK(
        storage_size_bytes + storage_offset_bytes <= new_storage_size_bytes,
        "setStorage: sizes ",
        size,
        ", strides ",
        stride,
        ","
        " storage offset ",
        storage_offset,
        ", and itemsize ",
        data_type.itemsize(),
        " requiring a storage size of ",
        storage_size_bytes,
        " are out of bounds for storage of size ",
        new_storage_size_bytes, OPS_ERROR(ErrCode::PARAM));
}

inline void setStrided(
    const at::Tensor& self,
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    int64_t storage_offset)
{
    TORCH_CHECK(size.size() == stride.size(), "mismatch in length of strides and shape", OPS_ERROR(ErrCode::PARAM));
    for (auto val : stride) {
        TORCH_CHECK(val >= 0,
                    "as_strided: Negative strides are not supported at the moment, "
                    "got strides: ", stride, OPS_ERROR(ErrCode::PARAM));
    }
    auto* self_ = self.unsafeGetTensorImpl();
    checkInBoundsForStorage(
        size, stride, storage_offset, self_->dtype(), self_->storage());

    /* storage offset */
    TORCH_CHECK(storage_offset >= 0, "Tensor: invalid storage offset ", storage_offset, OPS_ERROR(ErrCode::VALUE));
    self_->set_storage_offset(storage_offset);

    /* size and stride */
    if (self_->sizes() == size && self_->strides() == stride) {
        return;
    }
    self_->set_sizes_and_strides(size, stride);
}

} // namespace native
} // namespace at_npu
