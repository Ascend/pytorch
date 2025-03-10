#include "torch_npu/csrc/aten/common/SetNpu.h"

namespace at_npu {
namespace native {

void set_storage_nd_npu(
    at::Tensor& self,
    c10::Storage storage,
    int64_t storage_offset,
    int nDimension,
    c10::IntArrayRef size,
    c10::IntArrayRef stride)
{
    at::native::checkSetStorage(self, storage, storage_offset, size, stride);
    self.unsafeGetTensorImpl()->set_storage_offset(storage_offset);
    resize_nd_npu(self.unsafeGetTensorImpl(), nDimension, size.data(), stride.data());
}

bool CheckStorageDesc(const at::Tensor& self, const c10::Storage src)
{
    if (self.unsafeGetTensorImpl()->storage_offset() != 0 || !self.is_contiguous()) {
        return false;
    }
    int64_t new_size = static_cast<int64_t>(src.nbytes() / self.dtype().itemsize());
    int64_t nelements = c10::multiply_integers(self.unsafeGetTensorImpl()->sizes());
    if (new_size != nelements) {
        return false;
    }
    return true;
}

at::Tensor& NPUNativeFunctions::set_(
    at::Tensor& self,
    c10::Storage src,
    long storage_offset,
    c10::IntArrayRef size,
    c10::IntArrayRef stride)
{
    set_storage_nd_npu(self, src, storage_offset, size.size(), size, stride);
    if (StorageDescHelper::CheckDescInit(src)) {
        StorageDescHelper::CopyDesc(self, src);
        return self;
    }
    // NPUStorageImpl create by constructor, NPUStorageDesc is not initialized by
    // SetDesc.
    if (CheckStorageDesc(self, src)) {
        StorageDescHelper::SetDesc(self, size, stride);
    } else {
        // Check input tensor propertys. If conditions are not met, NPUStorageDesc
        // base_sizes_ change to 1D. Conditions:
        // 1. Tensor storage_offset == 0
        // 2. Tnput tensor is contiguous
        // 3. Storage element size same to Tensor
        int64_t new_size = static_cast<int64_t>(src.nbytes() / self.dtype().itemsize());
        StorageDescHelper::SetDesc(self, {new_size}, {1});
    }
    return self;
}

at::Tensor& NPUNativeFunctions::set_(at::Tensor& self)
{
    caffe2::TypeMeta dtype = self.dtype();
    c10::intrusive_ptr<c10::StorageImpl> npu_storage_impl = torch_npu::make_npu_storage_impl(
        c10::StorageImpl::use_byte_size_t(),
        c10::SymInt(0),
        c10_npu::NPUCachingAllocator::get()->allocate(0),
        c10_npu::NPUCachingAllocator::get(),
        true);
    c10::Storage storage(npu_storage_impl);
    set_storage_nd_npu(self, storage, 0, 1, {0}, {});
    StorageDescHelper::SetDesc(self);
    TORCH_INTERNAL_ASSERT(dtype == self.dtype(), OPS_ERROR(ErrCode::TYPE));
    return self;
}

at::Tensor& NPUNativeFunctions::set_(at::Tensor& self, const at::Tensor& src)
{
    at::TensorImpl* self_ = self.unsafeGetTensorImpl();
    at::TensorImpl* src_ = src.unsafeGetTensorImpl();
    if (self_ != src_) {
        set_storage_nd_npu(self, src.storage(), src.storage_offset(), src.dim(), src.sizes(), src.strides());
    }
    StorageDescHelper::CopyDesc(self, src);
    return self;
}

at::Tensor& NPUNativeFunctions::set_(at::Tensor& self, c10::Storage src)
{
    int64_t new_size = static_cast<int64_t>(src.nbytes() / self.dtype().itemsize());
    set_storage_nd_npu(self, src, 0, 1, {new_size}, {});
    if (StorageDescHelper::CheckDescInit(src)) {
        StorageDescHelper::CopyDesc(self, src);
        return self;
    }
    // NPUStorageImpl create by constructor, NPUStorageDesc is not initialized by
    // SetDesc.
    StorageDescHelper::SetDesc(
        self,
        self.unsafeGetTensorImpl()->sizes(),
        self.unsafeGetTensorImpl()->strides());
    return self;
}

at::Tensor set_tensor_with_storage_format(c10::Storage src)
{
    if (StorageDescHelper::CheckDescInit(src)) {
        // The storage object src has complete description information,
        // and the tensor object self needs to be brushed to be the same
        auto desc = torch_npu::NPUBridge::GetNpuStorageImpl(src.unsafeGetStorageImpl())->npu_desc_;
        auto dist_tensor = NPUNativeFunctions::empty(
            {0}, desc.data_type_.toScalarType(), c10::nullopt,
            src.device(), false, c10::MemoryFormat::Contiguous);
        set_storage_nd_npu(dist_tensor, src, 0, desc.base_sizes_.size(), desc.base_sizes_, desc.base_strides_);
        return dist_tensor;
    } else {
        // The storage object src doesn't have complete description information,
        // and the tensor object self needs to be brushed to be the 1 dimension
        auto dist_tensor = NPUNativeFunctions::empty(
            {0}, at::ScalarType::Char, c10::nullopt,
            src.device(), false, c10::MemoryFormat::Contiguous);
        int64_t new_size = static_cast<int64_t>(src.nbytes() / dist_tensor.dtype().itemsize());
        set_storage_nd_npu(dist_tensor, src, 0, 1, {new_size}, {});
        StorageDescHelper::SetDesc(
            dist_tensor,
            dist_tensor.unsafeGetTensorImpl()->sizes(),
            dist_tensor.unsafeGetTensorImpl()->strides());
        return dist_tensor;
    }
}

} // namespace native
} // namespace at_npu
