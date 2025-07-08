#include "torch_npu/csrc/custom_dtype/Init.h"
#ifndef BUILD_LIBTORCH
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>
#endif


namespace c10_npu {
struct DTypeConstants {
    static const int float32_value;
    static const int float16_value;
    static const int int8_value;
    static const int int32_value;
    static const int uint8_value;
    static const int int16_value;
    static const int uint16_value;
    static const int uint32_value;
    static const int int64_value;
    static const int uint64_value;
    static const int float64_value;
    static const int bool_value;
    static const int string_value;
    static const int complex64_value;
    static const int complex128_value;
    static const int bfloat16_value;
    static const int int4_value;
    static const int uint1_value;
    static const int complex32_value;
};

const int DTypeConstants::float32_value = static_cast<int>(DType::FLOAT);
const int DTypeConstants::float16_value = static_cast<int>(DType::FLOAT16);
const int DTypeConstants::int8_value = static_cast<int>(DType::INT8);
const int DTypeConstants::int32_value = static_cast<int>(DType::INT32);
const int DTypeConstants::uint8_value = static_cast<int>(DType::UINT8);
const int DTypeConstants::int16_value = static_cast<int>(DType::INT16);
const int DTypeConstants::uint16_value = static_cast<int>(DType::UINT16);
const int DTypeConstants::uint32_value = static_cast<int>(DType::UINT32);
const int DTypeConstants::int64_value = static_cast<int>(DType::INT64);
const int DTypeConstants::uint64_value = static_cast<int>(DType::UINT64);
const int DTypeConstants::float64_value = static_cast<int>(DType::DOUBLE);
const int DTypeConstants::bool_value = static_cast<int>(DType::BOOL);
const int DTypeConstants::string_value = static_cast<int>(DType::STRING);
const int DTypeConstants::complex64_value = static_cast<int>(DType::COMPLEX64);
const int DTypeConstants::complex128_value = static_cast<int>(DType::COMPLEX128);
const int DTypeConstants::bfloat16_value = static_cast<int>(DType::BF16);
const int DTypeConstants::int4_value = static_cast<int>(DType::INT4);
const int DTypeConstants::uint1_value = static_cast<int>(DType::UINT1);
const int DTypeConstants::complex32_value = static_cast<int>(DType::COMPLEX32);

#ifndef BUILD_LIBTORCH
PyObject* cd_initExtension(PyObject*, PyObject *)
{
    auto torch_npu_C_module = THPObjectPtr(PyImport_ImportModule("torch_npu._C"));
    if (!torch_npu_C_module) {
        return nullptr;
    }
    auto torch_npu_C_m = py::handle(torch_npu_C_module).cast<py::module>();
    auto m = torch_npu_C_m.def_submodule("_cd", "_cd bindings");

    py::class_<DTypeConstants>(m, "DType")
        .def_readonly_static("float32", &DTypeConstants::float32_value)
        .def_readonly_static("float16", &DTypeConstants::float16_value)
        .def_readonly_static("int8", &DTypeConstants::int8_value)
        .def_readonly_static("int32", &DTypeConstants::int32_value)
        .def_readonly_static("uint8", &DTypeConstants::uint8_value)
        .def_readonly_static("int16", &DTypeConstants::int16_value)
        .def_readonly_static("uint16", &DTypeConstants::uint16_value)
        .def_readonly_static("uint32", &DTypeConstants::uint32_value)
        .def_readonly_static("int64", &DTypeConstants::int64_value)
        .def_readonly_static("uint64", &DTypeConstants::uint64_value)
        .def_readonly_static("float64", &DTypeConstants::float64_value)
        .def_readonly_static("bool", &DTypeConstants::bool_value)
        .def_readonly_static("string", &DTypeConstants::string_value)
        .def_readonly_static("complex64", &DTypeConstants::complex64_value)
        .def_readonly_static("complex128", &DTypeConstants::complex128_value)
        .def_readonly_static("bfloat16", &DTypeConstants::bfloat16_value)
        .def_readonly_static("int4", &DTypeConstants::int4_value)
        .def_readonly_static("uint1", &DTypeConstants::uint1_value)
        .def_readonly_static("complex32", &DTypeConstants::complex32_value);

    Py_RETURN_NONE;
}

static PyMethodDef NPUCustomDtypeMethods[] = { // NOLINT
    {"_cd_init", cd_initExtension, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}
};
#endif

const std::string CustomDataTypeToString(int64_t dType)
{
    const std::map<const DType, const std::string>
        TYPE_TO_STRING_MAP = {
            {DType::FLOAT, "torch_npu.float32"},
            {DType::FLOAT16, "torch_npu.float16"},
            {DType::INT8, "torch_npu.int8"},
            {DType::INT32, "torch_npu.int32"},
            {DType::UINT8, "torch_npu.uint8"},
            {DType::INT16, "torch_npu.int16"},
            {DType::UINT16, "torch_npu.uint16"},
            {DType::UINT32, "torch_npu.uint32"},
            {DType::INT64, "torch_npu.int64"},
            {DType::UINT64, "torch_npu.uint64"},
            {DType::DOUBLE, "torch_npu.float64"},
            {DType::BOOL, "torch_npu.bool"},
            {DType::STRING, "torch_npu.string"},
            {DType::COMPLEX64, "torch_npu.complex64"},
            {DType::COMPLEX128, "torch_npu.complex128"},
            {DType::BF16, "torch_npu.bfloat16"},
            {DType::INT4, "torch_npu.int4"},
            {DType::UINT1, "torch_npu.uint1"},
            {DType::COMPLEX32, "torch_npu.complex32"}};

    const auto iter = TYPE_TO_STRING_MAP.find(static_cast<DType>(dType));
    return iter != TYPE_TO_STRING_MAP.end() ? iter->second : "Unknown dtype";
}

#ifndef BUILD_LIBTORCH
PyMethodDef* custom_dtype_functions()
{
    return NPUCustomDtypeMethods;
}
#endif
}
