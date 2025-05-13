#include "torch_npu/csrc/core/npu/npu_log.h"

#include "torch_npu/csrc/core/NPUBridge.h"
#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/framework/FormatHelper.h"

namespace at_npu {
namespace native {

namespace {

constexpr int BLOCKSIZE = 16;
constexpr int BLOCKBYTES = 32;

// base format is ND/NCHW
FormatShape InferShapeLessTo4(c10::IntArrayRef dims, size_t itemsize);
FormatShape InferShape4To5(c10::IntArrayRef dims, size_t itemsize);
FormatShape InferShape5To4(c10::IntArrayRef dims, size_t itemsize);
FormatShape InferShapeNDToNZ(c10::IntArrayRef dims, size_t itemsize);
FormatShape InferShapeNDToZ(c10::IntArrayRef dims, size_t itemsize);
FormatShape InferShapeofNCHW(c10::IntArrayRef dims, size_t itemsize);
FormatShape InferShapeofND(c10::IntArrayRef dims, size_t itemsize);

// converter between base format
FormatShape InferShapeNCHWToND(c10::IntArrayRef storage_dims, c10::IntArrayRef base_dims, size_t itemsize);
FormatShape InferShapeNCDHWToND(c10::IntArrayRef storage_dims, c10::IntArrayRef base_dims, size_t itemsize);
FormatShape InferShapeNDToNCHW(c10::IntArrayRef storage_dims, c10::IntArrayRef base_dims, size_t itemsize);
FormatShape InferShapeNDToNCDHW(c10::IntArrayRef storage_dims, c10::IntArrayRef base_dims, size_t itemsize);

// base format is NCDHW
FormatShape InferShapeOfNDHWC(c10::IntArrayRef dims, size_t itemsize);
FormatShape InferShapeOfNCDHW(c10::IntArrayRef dims, size_t itemsize);
FormatShape InferShapeOfNDC1HWC0(c10::IntArrayRef dims, size_t itemsize);
FormatShape InferShapeOfFZ3D(c10::IntArrayRef dims, size_t itemsize);

FormatShape InferShapeofNHWC(c10::IntArrayRef dims, size_t itemsize);
}  // namespace

std::unordered_map<aclFormat, FormatHelper::FormatInfo> FormatHelper::InitializeInfo()
{
    return {
        {ACL_FORMAT_NC1HWC0, (FormatInfo){ACL_FORMAT_NC1HWC0, ACL_FORMAT_NCHW, InferShape4To5, "NC1HWC0", true}},
        {ACL_FORMAT_ND, (FormatInfo){ACL_FORMAT_ND, ACL_FORMAT_ND, InferShapeofND, "ND", false}},
        {ACL_FORMAT_NCHW, (FormatInfo){ACL_FORMAT_NCHW, ACL_FORMAT_NCHW, InferShapeofNCHW, "NCHW", false}},
        {ACL_FORMAT_NHWC, (FormatInfo){ACL_FORMAT_NHWC, ACL_FORMAT_NHWC, InferShapeofNHWC, "NHWC", false}},
        {ACL_FORMAT_FRACTAL_NZ,
         (FormatInfo){ACL_FORMAT_FRACTAL_NZ, ACL_FORMAT_ND, InferShapeNDToNZ, "FRACTAL_NZ", true}},
        {ACL_FORMAT_FRACTAL_Z, (FormatInfo){ACL_FORMAT_FRACTAL_Z, ACL_FORMAT_NCHW, InferShapeNDToZ, "FRACTAL_Z", true}},
        {ACL_FORMAT_NDHWC, (FormatInfo){ACL_FORMAT_NDHWC, ACL_FORMAT_NCDHW, InferShapeOfNDHWC, "NDHWC", false}},
        {ACL_FORMAT_NCDHW, (FormatInfo){ACL_FORMAT_NCDHW, ACL_FORMAT_NCDHW, InferShapeOfNCDHW, "NCDHW", false}},
        {ACL_FORMAT_NDC1HWC0,
         (FormatInfo){ACL_FORMAT_NDC1HWC0, ACL_FORMAT_NCDHW, InferShapeOfNDC1HWC0, "NDC1HWC0", true}},
        {ACL_FRACTAL_Z_3D, (FormatInfo){ACL_FRACTAL_Z_3D, ACL_FORMAT_NCDHW, InferShapeOfFZ3D, "FRACTAL_Z_3D", true}},
    };
};

std::unordered_map<aclFormat, FormatHelper::FormatInfo> FormatHelper::info = FormatHelper::InitializeInfo();

bool FormatHelper::IsPadded(const at::Tensor *tensor)
{
    auto format = torch_npu::NPUBridge::GetNpuStorageImplDesc(*tensor).npu_format_;
    return IsPadded(format);
}

bool FormatHelper::IsPadded(aclFormat format)
{
    auto itr = info.find(format);
    if (itr != info.end()) {
        return itr->second.isPadded;
    }
    AT_ERROR("unknown format type:", format);
    return true;
}

char *FormatHelper::GetFormatName(aclFormat format)
{
    const auto &itr = info.find(format);
    if (itr == info.end()) {
        AT_ERROR("unknown format type:", format);
        return nullptr;
    }
    return itr->second.formatName;
}

char *FormatHelper::GetFormatName(const at::Tensor &tensor)
{
    auto format = torch_npu::NPUBridge::GetNpuStorageImplDesc(tensor).npu_format_;
    return GetFormatName(format);
}

aclFormat FormatHelper::GetBaseFormat(const at::Tensor &tensor)
{
    auto format = GetFormat(tensor);
    return GetBaseFormat(format);
}

aclFormat FormatHelper::GetBaseFormat(aclFormat format)
{
    const auto &itr = info.find(format);
    if (itr == info.end()) {
        AT_ERROR("unknown format type:", format);
        return ACL_FORMAT_ND;
    }
    return itr->second.baseFormat;
}

aclFormat FormatHelper::GetFormat(const at::Tensor &tensor)
{
    return torch_npu::NPUBridge::GetNpuStorageImplDesc(tensor).npu_format_;
}

bool FormatHelper::IsBaseFormatType(aclFormat format)
{
    return GetBaseFormat(format) == format;
}

bool FormatHelper::IsBaseFormatType(const at::Tensor &tensor)
{
    auto format = torch_npu::NPUBridge::GetNpuStorageImplDesc(tensor).npu_format_;
    return IsBaseFormatType(format);
}

FormatShape FormatHelper::GetStorageSizes(const torch_npu::NPUStorageDesc &desc)
{
    auto ori_size = desc.base_sizes_;
    auto format = desc.npu_format_;
    auto dtype = desc.data_type_;
    return GetStorageSizes(format, ori_size, dtype);
}

bool FormatHelper::IsOpInputBaseFormat(const at::Tensor &tensor)
{
    if (!torch_npu::utils::is_npu(tensor)) {
        return true;
    }
    const auto format = torch_npu::NPUBridge::GetNpuStorageImplDesc(tensor).npu_format_;
    return (format == ACL_FORMAT_ND) || (format == ACL_FORMAT_NCHW) || (format == ACL_FORMAT_NHWC) ||
        (format == ACL_FORMAT_NCDHW);
}

bool FormatHelper::IsOpInputBaseFormat(const c10::optional<at::Tensor> &tensor)
{
    if (!tensor.has_value()) {
        return true;
    }
    return IsOpInputBaseFormat(tensor.value());
}

bool FormatHelper::IsOpInputBaseFormat(const c10::List<c10::optional<at::Tensor>> &tensors)
{
    const auto &iter =
        std::find_if(tensors.begin(), tensors.end(), [](const auto &tensor) { return !IsOpInputBaseFormat(tensor); });
    return iter == tensors.end();
}

bool FormatHelper::IsOpInputBaseFormat(const c10::optional<at::TensorList> &tensors)
{
    if (!tensors.has_value()) {
        return true;
    }
    const auto &iter =
        std::find_if(tensors.value().begin(), tensors.value().end(), [](const auto &tensor) { return !IsOpInputBaseFormat(tensor); });
    return iter == tensors.value().end();
}

bool FormatHelper::IsOpInputBaseFormat(const at::TensorList &tensors)
{
    const auto &iter =
        std::find_if(tensors.begin(), tensors.end(), [](const auto &tensor) { return !IsOpInputBaseFormat(tensor); });
    return iter == tensors.end();
}

bool FormatHelper::IsOpInputBaseFormat(const at::ITensorListRef &tensors)
{
    auto materialized = tensors.materialize();
    const auto &iter = std::find_if(materialized.begin(), materialized.end(), [](const auto &tensor) {
        return !IsOpInputBaseFormat(tensor.get());
    });
    return iter == materialized.end();
}

//
namespace {
FormatShape InferShapeLessTo4(c10::IntArrayRef dims, size_t itemsize)
{
    FormatShape res;
    res.resize(4);
    AT_ASSERT(dims.size() <= 4, "input dim > 4 when InferShapeLessTo4", OPS_ERROR(ErrCode::PARAM));
    switch (dims.size()) {
        case 0:
            res[0] = 1;
            res[1] = 1;
            res[2] = 1;
            res[3] = 1;
            break;
        case 1:  // RESHAPE_TYPE_C
            res[0] = 1;
            res[1] = dims[0];
            res[2] = 1;
            res[3] = 1;
            break;
        case 2:  // RESHAPE_TYPE_CH
            res[0] = 1;
            res[1] = dims[0];
            res[2] = dims[1];
            res[3] = 1;
            break;
        case 3:  // RESHAPE_TYPE_CHW
            res[0] = 1;
            res[1] = dims[0];
            res[2] = dims[1];
            res[3] = dims[2];
            break;
        case 4:
            res[0] = dims[0];
            res[1] = dims[1];
            res[2] = dims[2];
            res[3] = dims[3];
            break;
        default:
            AT_ERROR("dims of NCHW shape should not be greater than 4, which is ", dims.size());
    }
    return res;
}

FormatShape InferShapeofNHWC(c10::IntArrayRef dims, size_t itemsize)
{
    AT_ASSERT(dims.size() == 4, "input dim should be equal to 4 when InferShapeofNHWC", OPS_ERROR(ErrCode::PARAM));
    return FormatShape(dims.begin(), dims.end());
}

FormatShape InferShape4To5(c10::IntArrayRef dims, size_t itemsize)
{
    FormatShape res;
    res.resize(5);
    if (dims.size() < 4) {
        ASCEND_LOGD("infershape4to5 but input dim < 4");
        return InferShape4To5(InferShapeLessTo4(dims, itemsize), itemsize);
    } else if (dims.size() > 4) {
        ASCEND_LOGE("infershape4to5 but input dim > 4");
    }
    res[0] = dims[0];
    res[1] = (dims[1] + 15) / 16;
    res[2] = dims[2];
    res[3] = dims[3];
    res[4] = BLOCKSIZE;
    return res;
}

FormatShape InferShape5To4(c10::IntArrayRef dims, size_t itemsize)
{
    FormatShape res;
    res.emplace_back(dims[0]);
    res.emplace_back(((dims[1] + 15) / 16) * 16);
    res.emplace_back(dims[2]);
    res.emplace_back(dims[3]);
    return res;
}

FormatShape InferShapeNDToNZ(c10::IntArrayRef dims, size_t itemsize)
{
    FormatShape res;
    // sum(keepdim = false) may make tensor dim = 0
    FormatShape dim;
    for (size_t i = 0; i < dims.size(); i++) {
        dim.emplace_back(dims[i]);
    }

    // this action will move to GuessStorageSizeWhenConvertFormat
    if (dim.size() == 0) {
        dim.emplace_back(1);
    }
    if (dim.size() == 1) {
        dim.emplace_back(1);
    }

    size_t i = 0;
    for (; i < dim.size() - 2; i++) {
        res.emplace_back(dim[i]);
    }

    AT_ASSERT(itemsize != 0, "dtype itemsize should not be 0", OPS_ERROR(ErrCode::PARAM));

    // float32 will cast to float16
    auto itemsize_ = (itemsize > 2) ? 2 : itemsize;
    auto lastSize = BLOCKBYTES / itemsize_;
    res.emplace_back((dim[i + 1] + lastSize - 1) / lastSize);
    res.emplace_back((dim[i] + BLOCKSIZE - 1) / BLOCKSIZE);
    res.emplace_back(BLOCKSIZE);
    res.emplace_back(lastSize);

    return res;
}

FormatShape InferShapeNDToZ(c10::IntArrayRef dims, size_t itemsize)
{
    FormatShape res;
    if (dims.size() < 4) {
        return InferShapeNDToZ(InferShapeLessTo4(dims, itemsize), itemsize);
    }

    res.emplace_back((dims[1] + 15) / BLOCKSIZE * dims[2] * dims[3]);
    res.emplace_back((dims[0] + 15) / BLOCKSIZE);
    res.emplace_back(BLOCKSIZE);
    res.emplace_back(BLOCKSIZE);

    return res;
}

FormatShape InferShapeNCHWToND(c10::IntArrayRef storage_dims, c10::IntArrayRef base_dims, size_t itemsize)
{
    FormatShape res;
    res.resize(4);
    auto cur_storage_dims = storage_dims;
    if (storage_dims.size() != 4) {
        cur_storage_dims = InferShapeLessTo4(storage_dims, itemsize);
    }
    AT_ASSERT(
        cur_storage_dims.size() == 4,
        "input dim num not equal 4 when InferShapeNCHWToND",
        OPS_ERROR(ErrCode::PARAM));

    if (base_dims.size() == 0) {
        FormatShape temp_dims;
        temp_dims.emplace_back(1);
        return InferShapeLessTo4(temp_dims, itemsize);
    }
    switch (base_dims.size()) {
        case 1:
            res.resize(1);
            res[0] = cur_storage_dims[1];
            AT_ASSERT(
                cur_storage_dims[0] == 1,
                "reshape type RESHAPE_TYPE_C erase dim N must be 1",
                OPS_ERROR(ErrCode::PARAM));
            AT_ASSERT(
                cur_storage_dims[2] == 1,
                "reshape type RESHAPE_TYPE_C erase dim H must be 1",
                OPS_ERROR(ErrCode::PARAM));
            AT_ASSERT(
                cur_storage_dims[3] == 1,
                "reshape type RESHAPE_TYPE_C erase dim W must be 1",
                OPS_ERROR(ErrCode::PARAM));
            break;
        case 2:
            res.resize(2);
            res[0] = cur_storage_dims[1];
            res[1] = cur_storage_dims[2];
            AT_ASSERT(
                cur_storage_dims[0] == 1,
                "reshape type RESHAPE_TYPE_CH erase dim N must be 1",
                OPS_ERROR(ErrCode::PARAM));
            AT_ASSERT(
                cur_storage_dims[3] == 1,
                "reshape type RESHAPE_TYPE_CH erase dim W must be 1",
                OPS_ERROR(ErrCode::PARAM));
            break;
        case 3:
            res.resize(3);
            res[0] = cur_storage_dims[1];
            res[1] = cur_storage_dims[2];
            res[2] = cur_storage_dims[3];
            AT_ASSERT(
                cur_storage_dims[0] == 1,
                "reshape type RESHAPE_TYPE_CHW erase dim N must be 1",
                OPS_ERROR(ErrCode::PARAM));
            break;
        case 4:
            res = cur_storage_dims;
            return res;
        default:
            AT_ERROR("unknown reshape type:");
    }
    return res;
}

FormatShape InferShapeNDToNCHW(c10::IntArrayRef storage_dims, c10::IntArrayRef base_dims, size_t itemsize)
{
    AT_ASSERT(storage_dims.size() <= 4, "input storage dim not less than 4", OPS_ERROR(ErrCode::PARAM));
    AT_ASSERT(base_dims.size() <= 4, "input storage dim not less than 4", OPS_ERROR(ErrCode::PARAM));
    return InferShapeLessTo4(base_dims, itemsize);
}

FormatShape InferShapeNDToNCDHW(c10::IntArrayRef storage_dims, c10::IntArrayRef base_dims, size_t itemsize)
{
    AT_ASSERT(
        storage_dims.size() == 5,
        "ND [",
        storage_dims,
        "] failed to convert to NCDHW",
        OPS_ERROR(ErrCode::PARAM));
    FormatShape res;
    res.resize(5);
    res = storage_dims;
    return res;
}

FormatShape InferShapeNCDHWToND(c10::IntArrayRef storage_dims, c10::IntArrayRef base_dims, size_t itemsize)
{
    FormatShape res;
    res.resize(5);
    res = storage_dims;
    AT_ASSERT(res.size() == 5, "input dim num not equal 5 when InferShapeNCDHWToND", OPS_ERROR(ErrCode::PARAM));
    return res;
}

// NCDHW -> NDHWC
FormatShape InferShapeOfNDHWC(c10::IntArrayRef dims, size_t itemsize)
{
    if (dims.size() < 5) {
        AT_ERROR("dim (", dims, ") cannot convert to NDHWC");
    }
    FormatShape res;
    res.resize(5);
    res[0] = dims[0];
    res[1] = dims[2];
    res[2] = dims[3];
    res[3] = dims[4];
    res[4] = dims[1];
    return res;
}

// NCDHW to NCDHW
FormatShape InferShapeOfNCDHW(c10::IntArrayRef dims, size_t itemsize)
{
    if (dims.size() < 5) {
        AT_ERROR("dim (", dims, ") cannot convert to NCDHW");
    }
    FormatShape res;
    res.resize(5);
    res[0] = dims[0];
    res[1] = dims[1];
    res[2] = dims[2];
    res[3] = dims[3];
    res[4] = dims[4];
    return res;
}

// NCDHW to NDC1HWC0
FormatShape InferShapeOfNDC1HWC0(c10::IntArrayRef dims, size_t itemsize)
{
    if (dims.size() < 5) {
        AT_ERROR("dim (", dims, ") cannot convert to NDC1HWC0");
    }
    FormatShape res;
    res.resize(6);
    res[0] = dims[0];
    res[1] = dims[2];
    res[2] = (dims[1] + BLOCKSIZE - 1) / BLOCKSIZE;
    res[3] = dims[3];
    res[4] = dims[4];
    res[5] = BLOCKSIZE;
    return res;
}

// NCDHW to FZ_3D
FormatShape InferShapeOfFZ3D(c10::IntArrayRef dims, size_t itemsize)
{
    if (dims.size() < 5) {
        AT_ERROR("dim (", dims, ") cannot convert to FZ_3D");
    }

    int64_t d1 = dims[2];
    int64_t d2 = (dims[1] + BLOCKSIZE - 1) / BLOCKSIZE;
    int64_t d3 = dims[3];
    int64_t d4 = dims[4];
    int64_t d5 = (dims[0] + BLOCKSIZE - 1) / BLOCKSIZE;
    int64_t d6 = BLOCKSIZE;
    int64_t d7 = BLOCKSIZE;

    // The shape of FZ3D is 7D, but the CANN only accept 4D
    // so we should merge 1st, 2nd, 3rd, 4th dimension.
    FormatShape res;
    res.resize(4);
    res[0] = d1 * d2 * d3 * d4;
    res[1] = d5;
    res[2] = d6;
    res[3] = d7;
    return res;
}

FormatShape InferShapeofNCHW(c10::IntArrayRef dims, size_t itemsize)
{
    if (dims.size() < 5) {
        return InferShapeLessTo4(dims, itemsize);
    } else {
        return InferShapeofND(dims, itemsize);
    }
}

FormatShape InferShapeofND(c10::IntArrayRef dims, size_t itemsize)
{
    FormatShape res;
    res.resize(dims.size());
    for (size_t j = 0; j < dims.size(); j++) {
        res[j] = dims[j];
    }
    return res;
}

}  // namespace

at::Tensor &FormatHelper::unsafe_format_cast(at::Tensor &self, int64_t self_format, int64_t result_format)
{
    torch_npu::NPUStorageDesc &self_desc = torch_npu::NPUBridge::GetNpuStorageImpl(self)->npu_desc_;
    if (self_format == ACL_FORMAT_ND && result_format == ACL_FORMAT_NC1HWC0) {
        auto itemsize = self_desc.data_type_.itemsize();
        self_desc.storage_sizes_ = InferShape4To5(self.sizes(), itemsize);
        self_desc.npu_format_ = ACL_FORMAT_NC1HWC0;
    } else if (self_format == ACL_FORMAT_NC1HWC0 && result_format == ACL_FORMAT_ND) {
        self_desc.storage_sizes_ = self_desc.base_sizes_;
        self_desc.npu_format_ = ACL_FORMAT_ND;
    }
    return self;
}
}  // namespace native
}  // namespace at_npu
