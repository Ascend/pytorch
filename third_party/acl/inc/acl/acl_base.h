/**
* @file acl_base.h
*
* Copyright (C) Huawei Technologies Co., Ltd. 2019-2020. All Rights Reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#ifndef INC_EXTERNAL_ACL_ACL_BASE_H_
#define INC_EXTERNAL_ACL_ACL_BASE_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef FUNC_VISIBILITY
#define ACL_FUNC_VISIBILITY __attribute__((visibility("default")))
#else
#define ACL_FUNC_VISIBILITY
#endif

typedef void *aclrtStream;
typedef void *aclrtEvent;
typedef void *aclrtContext;
typedef int aclError;
typedef uint16_t aclFloat16;
typedef struct aclDataBuffer aclDataBuffer;
typedef struct aclTensorDesc aclTensorDesc;
typedef struct aclprofStepInfo aclprofStepInfo;
static const int ACL_ERROR_NONE = 0;

static const int ACL_ERROR_INVALID_PARAM = 100000;
static const int ACL_ERROR_UNINITIALIZE = 100001;
static const int ACL_ERROR_REPEAT_INITIALIZE = 100002;
static const int ACL_ERROR_INVALID_FILE = 100003;
static const int ACL_ERROR_WRITE_FILE = 100004;
static const int ACL_ERROR_INVALID_FILE_SIZE = 100005;
static const int ACL_ERROR_PARSE_FILE = 100006;
static const int ACL_ERROR_FILE_MISSING_ATTR = 100007;
static const int ACL_ERROR_FILE_ATTR_INVALID = 100008;
static const int ACL_ERROR_INVALID_DUMP_CONFIG = 100009;
static const int ACL_ERROR_INVALID_PROFILING_CONFIG = 100010;
static const int ACL_ERROR_INVALID_MODEL_ID = 100011;
static const int ACL_ERROR_DESERIALIZE_MODEL = 100012;
static const int ACL_ERROR_PARSE_MODEL = 100013;
static const int ACL_ERROR_READ_MODEL_FAILURE = 100014;
static const int ACL_ERROR_MODEL_SIZE_INVALID = 100015;
static const int ACL_ERROR_MODEL_MISSING_ATTR = 100016;
static const int ACL_ERROR_MODEL_INPUT_NOT_MATCH = 100017;
static const int ACL_ERROR_MODEL_OUTPUT_NOT_MATCH = 100018;
static const int ACL_ERROR_MODEL_NOT_DYNAMIC = 100019;
static const int ACL_ERROR_OP_TYPE_NOT_MATCH = 100020;
static const int ACL_ERROR_OP_INPUT_NOT_MATCH = 100021;
static const int ACL_ERROR_OP_OUTPUT_NOT_MATCH = 100022;
static const int ACL_ERROR_OP_ATTR_NOT_MATCH = 100023;
static const int ACL_ERROR_OP_NOT_FOUND = 100024;
static const int ACL_ERROR_OP_LOAD_FAILED = 100025;
static const int ACL_ERROR_UNSUPPORTED_DATA_TYPE = 100026;
static const int ACL_ERROR_FORMAT_NOT_MATCH = 100027;
static const int ACL_ERROR_BIN_SELECTOR_NOT_REGISTERED = 100028;
static const int ACL_ERROR_KERNEL_NOT_FOUND = 100029;
static const int ACL_ERROR_BIN_SELECTOR_ALREADY_REGISTERED = 100030;
static const int ACL_ERROR_KERNEL_ALREADY_REGISTERED = 100031;
static const int ACL_ERROR_INVALID_QUEUE_ID = 100032;
static const int ACL_ERROR_REPEAT_SUBSCRIBE = 100033;
static const int ACL_ERROR_STREAM_NOT_SUBSCRIBE = 100034;
static const int ACL_ERROR_THREAD_NOT_SUBSCRIBE = 100035;
static const int ACL_ERROR_WAIT_CALLBACK_TIMEOUT = 100036;
static const int ACL_ERROR_REPEAT_FINALIZE = 100037;
static const int ACL_ERROR_NOT_STATIC_AIPP = 100038;
static const int ACL_ERROR_COMPILING_STUB_MODE = 100039;
static const int ACL_ERROR_GROUP_NOT_SET = 100040;
static const int ACL_ERROR_GROUP_NOT_CREATE = 100041;
static const int ACL_ERROR_PROF_ALREADY_RUN = 100042;
static const int ACL_ERROR_PROF_NOT_RUN = 100043;
static const int ACL_ERROR_DUMP_ALREADY_RUN = 100044;
static const int ACL_ERROR_DUMP_NOT_RUN = 100045;

static const int ACL_ERROR_BAD_ALLOC = 200000;
static const int ACL_ERROR_API_NOT_SUPPORT = 200001;
static const int ACL_ERROR_INVALID_DEVICE = 200002;
static const int ACL_ERROR_MEMORY_ADDRESS_UNALIGNED = 200003;
static const int ACL_ERROR_RESOURCE_NOT_MATCH = 200004;
static const int ACL_ERROR_INVALID_RESOURCE_HANDLE = 200005;
static const int ACL_ERROR_FEATURE_UNSUPPORTED = 200006;
static const int ACL_ERROR_PROF_MODULES_UNSUPPORTED = 200007;

static const int ACL_ERROR_RT_MEMORY_ALLOCATION = 207001;

static const int ACL_ERROR_STORAGE_OVER_LIMIT = 300000;

static const int ACL_ERROR_INTERNAL_ERROR = 500000;
static const int ACL_ERROR_FAILURE = 500001;
static const int ACL_ERROR_GE_FAILURE = 500002;
static const int ACL_ERROR_RT_FAILURE = 500003;
static const int ACL_ERROR_DRV_FAILURE = 500004;
static const int ACL_ERROR_PROFILING_FAILURE = 500005;

#define ACL_TENSOR_SHAPE_RANGE_NUM 2
#define ACL_UNKNOWN_RANK 0xFFFFFFFFFFFFFFFE

typedef enum {
    ACL_DT_UNDEFINED = -1,
    ACL_FLOAT = 0,
    ACL_FLOAT16 = 1,
    ACL_INT8 = 2,
    ACL_INT32 = 3,
    ACL_UINT8 = 4,
    ACL_INT16 = 6,
    ACL_UINT16 = 7,
    ACL_UINT32 = 8,
    ACL_INT64 = 9,
    ACL_UINT64 = 10,
    ACL_DOUBLE = 11,
    ACL_BOOL = 12,
} aclDataType;

typedef enum {
    ACL_FORMAT_UNDEFINED = -1,
    ACL_FORMAT_NCHW = 0,
    ACL_FORMAT_NHWC = 1,
    ACL_FORMAT_ND = 2,
    ACL_FORMAT_NC1HWC0 = 3,
    ACL_FORMAT_FRACTAL_Z = 4,
    ACL_FORMAT_FRACTAL_NZ = 29,
    ACL_FORMAT_NDHWC = 27,
    ACL_FORMAT_NCDHW = 30,
    ACL_FORMAT_NDC1HWC0 = 32,
    ACL_FRACTAL_Z_3D = 33,
} aclFormat;

typedef enum {
    ACL_DEBUG = 0,
    ACL_INFO = 1,
    ACL_WARNING = 2,
    ACL_ERROR = 3,
} aclLogLevel;

typedef enum {
    ACL_MEMTYPE_DEVICE = 0,
    ACL_MEMTYPE_HOST = 1,
    ACL_MEMTYPE_HOST_COMPILE_INDEPENDENT = 2,
} aclMemType;

typedef enum {
    ACL_STEP_START = 0,
    ACL_STEP_END = 1,
} aclprofStepTag;

/**
 * @ingroup AscendCL
 * @brief Converts data of type aclFloat16 to data of type float
 *
 * @param value [IN]   Data to be converted
 * @retval Transformed data
 */
ACL_FUNC_VISIBILITY float aclFloat16ToFloat(aclFloat16 value);

/**
 * @ingroup AscendCL
 * @brief Converts data of type float to data of type aclFloat16
 *
 * @param value [IN]   Data to be converted
 * @retval Transformed data
 */
ACL_FUNC_VISIBILITY aclFloat16 aclFloatToFloat16(float value);

/**
 * @ingroup AscendCL
 * @brief create data of aclDataBuffer
 *
 * @param data [IN]    pointer to data
 * @li Need to be managed by the user,
 *  call aclrtMalloc interface to apply for memory,
 *  call aclrtFree interface to release memory
 * @param size [IN]    size of data in bytes
 * @retval pointer to created instance. nullptr if run out of memory
 *
 * @see aclrtMalloc | aclrtFree
 */
ACL_FUNC_VISIBILITY aclDataBuffer *aclCreateDataBuffer(void *data, size_t size);

/**
 * @ingroup AscendCL
 * @brief destroy data of aclDataBuffer
 *
 * @par Function
 *  Only the aclDataBuffer type data is destroyed here.
 *  The memory of the data passed in when the aclDataDataBuffer interface
 *  is called to create aclDataBuffer type data must be released by the user
 * @param  dataBuffer [IN]   pointer to the aclDataBuffer
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclCreateDataBuffer
 */
ACL_FUNC_VISIBILITY aclError aclDestroyDataBuffer(const aclDataBuffer *dataBuffer);

/**
 * @ingroup AscendCL
 * @brief get data address from aclDataBuffer
 *
 * @param dataBuffer [IN]    pointer to the data of aclDataBuffer
 * @retval data address
 */
ACL_FUNC_VISIBILITY void *aclGetDataBufferAddr(const aclDataBuffer *dataBuffer);

/**
 * @ingroup AscendCL
 * @brief get data size of aclDataBuffer
 *
 * @param  dataBuffer [IN]    pointer to the data of aclDataBuffer
 * @retval data size
 */
ACL_FUNC_VISIBILITY uint32_t aclGetDataBufferSize(const aclDataBuffer *dataBuffer);

/**
 * @ingroup AscendCL
 * @brief get data size of aclDataBuffer to replace aclGetDataBufferSize
 *
 * @param  dataBuffer [IN]    pointer to the data of aclDataBuffer
 *
 * @retval data size
 */
ACL_FUNC_VISIBILITY size_t aclGetDataBufferSizeV2(const aclDataBuffer *dataBuffer);

/**
 * @ingroup AscendCL
 * @brief get size of aclDataType
 *
 * @param  dataType [IN]    aclDataType data the size to get
 * @retval size of the aclDataType
 */
ACL_FUNC_VISIBILITY size_t aclDataTypeSize(aclDataType dataType);

// interfaces of tensor desc
/**
 * @ingroup AscendCL
 * @brief create data aclTensorDesc
 *
 * @param  dataType [IN]    Data types described by tensor
 * @param  numDims [IN]     the number of dimensions of the shape
 * @param  dims [IN]        the size of the specified dimension
 * @param  format [IN]      tensor format
 * @retval aclTensorDesc pointer.
 * @retval nullptr if param is invalid or run out of memory
 */
ACL_FUNC_VISIBILITY aclTensorDesc *aclCreateTensorDesc(aclDataType dataType,
                                                       int numDims,
                                                       const int64_t *dims,
                                                       aclFormat format);

/**
 * @ingroup AscendCL
 * @brief destroy data aclTensorDesc
 *
 * @param desc [IN]     pointer to the data of aclTensorDesc to destroy
 */
ACL_FUNC_VISIBILITY void aclDestroyTensorDesc(const aclTensorDesc *desc);

/**
 * @ingroup AscendCL
 * @brief set tensor shape range for aclTensorDesc
 *
 * @param  desc [IN]     pointer to the data of aclTensorDesc
 * @param  dimsCount [IN]     the number of dimensions of the shape
 * @param  dimsRange [IN]     the range of dimensions of the shape
 *
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclSetTensorShapeRange(aclTensorDesc* desc,
                                                    size_t dimsCount,
                                                    int64_t dimsRange[][ACL_TENSOR_SHAPE_RANGE_NUM]);

/**
 * @ingroup AscendCL
 * @brief get data type specified by the tensor description
 *
 * @param desc [IN]        pointer to the instance of aclTensorDesc
 * @retval data type specified by the tensor description.
 * @retval ACL_DT_UNDEFINED if description is null
 */
ACL_FUNC_VISIBILITY aclDataType aclGetTensorDescType(const aclTensorDesc *desc);

/**
 * @ingroup AscendCL
 * @brief get data format specified by the tensor description
 *
 * @param  desc [IN]        pointer to the instance of aclTensorDesc
 * @retval data format specified by the tensor description.
 * @retval ACL_FORMAT_UNDEFINED if description is null
 */
ACL_FUNC_VISIBILITY aclFormat aclGetTensorDescFormat(const aclTensorDesc *desc);

/**
 * @ingroup AscendCL
 * @brief get tensor size specified by the tensor description
 *
 * @param  desc [IN]        pointer to the instance of aclTensorDesc
 * @retval data size specified by the tensor description.
 * @retval 0 if description is null
 */
ACL_FUNC_VISIBILITY size_t aclGetTensorDescSize(const aclTensorDesc *desc);

/**
 * @ingroup AscendCL
 * @brief get element count specified by the tensor description
 *
 * @param  desc [IN]        pointer to the instance of aclTensorDesc
 * @retval element count specified by the tensor description.
 * @retval 0 if description is null
 */
ACL_FUNC_VISIBILITY size_t aclGetTensorDescElementCount(const aclTensorDesc *desc);

/**
 * @ingroup AscendCL
 * @brief get number of dims specified by the tensor description
 *
 * @param  desc [IN]        pointer to the instance of aclTensorDesc
 * @retval number of dims specified by the tensor description.
 * @retval 0 if description is null
 * @retval ACL_UNKNOWN_RANK if the tensor dim is -2
 */
ACL_FUNC_VISIBILITY size_t aclGetTensorDescNumDims(const aclTensorDesc *desc);

/**
 * @ingroup AscendCL
 * @brief Get the size of the specified dim in the tensor description
 *
 * @param  desc [IN]        pointer to the instance of aclTensorDesc
 * @param  index [IN]       index of dims, start from 0.
 * @retval dim specified by the tensor description and index.
 * @retval -1 if description or index is invalid
 */
ACL_FUNC_VISIBILITY int64_t aclGetTensorDescDim(const aclTensorDesc *desc, size_t index);

/**
 * @ingroup AscendCL
 * @brief Get the size of the specified dim in the tensor description
 *
 * @param  desc [IN]        pointer to the instance of aclTensorDesc
 * @param  index [IN]       index of dims, start from 0.
 * @param  dimSize [OUT]       size of the specified dim.
 *
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclGetTensorDescDimV2(const aclTensorDesc *desc, size_t index, int64_t *dimSize);

/**
 * @ingroup AscendCL
 * @brief Get the range of the specified dim in the tensor description
 *
 * @param  desc [IN]        pointer to the instance of aclTensorDesc
 * @param  index [IN]       index of dims, start from 0.
 * @param  dimRangeNum [IN]     number of dimRange.
 * @param  dimRange [OUT]       range of the specified dim.
 *
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclGetTensorDescDimRange(const aclTensorDesc *desc,
                                                      size_t index,
                                                      size_t dimRangeNum,
                                                      int64_t *dimRange);

/**
 * @ingroup AscendCL
 * @brief set tensor description name
 *
 * @param desc [IN]        pointer to the instance of aclTensorDesc
 * @param name [IN]        tensor description name
 */
ACL_FUNC_VISIBILITY void aclSetTensorDescName(aclTensorDesc *desc, const char *name);

/**
 * @ingroup AscendCL
 * @brief get tensor description name
 *
 * @param  desc [IN]        pointer to the instance of aclTensorDesc
 * @retval tensor description name.
 * @retval empty string if description is null
 */
ACL_FUNC_VISIBILITY const char *aclGetTensorDescName(aclTensorDesc *desc);

/**
 * @ingroup AscendCL
 * @brief Convert the format in the source aclTensorDesc according to
 * the specified dstFormat to generate a new target aclTensorDesc.
 * The format in the source aclTensorDesc remains unchanged.
 *
 * @param  srcDesc [IN]     pointer to the source tensor desc
 * @param  dstFormat [IN]   destination format
 * @param  dstDesc [OUT] pointer to the pointer to the destination tensor desc
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclTransTensorDescFormat(const aclTensorDesc *srcDesc, aclFormat dstFormat,
    aclTensorDesc **dstDesc);

/**
 * @ingroup AscendCL
 * @brief Set the format specified by the tensor description
 *
 * @param  desc [IN|OUT]     pointer to the instance of aclTensorDesc
 * @param  format [IN]       the storage format
 *
 * @retval ACL_ERROR_NONE    The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclSetTensorFormat(aclTensorDesc *desc, aclFormat format);

/**
 * @ingroup AscendCL
 * @brief Set the shape specified by the tensor description
 *
 * @param  desc [IN|OUT]      pointer to the instance of aclTensorDesc
 * @param  numDims [IN]       the number of dimensions of the shape
 * @param  dims [IN]          the size of the specified dimension
 *
 * @retval ACL_ERROR_NONE     The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclSetTensorShape(aclTensorDesc *desc, int numDims, const int64_t *dims);

/**
 * @ingroup AscendCL
 * @brief Set the original format specified by the tensor description
 *
 * @param  desc [IN|OUT]     pointer to the instance of aclTensorDesc
 * @param  format [IN]       the storage format
 *
 * @retval ACL_ERROR_NONE    The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclSetTensorOriginFormat(aclTensorDesc *desc, aclFormat format);

/**
 * @ingroup AscendCL
 * @brief Set the original shape specified by the tensor description
 *
 * @param  desc [IN|OUT]      pointer to the instance of aclTensorDesc
 * @param  numDims [IN]       the number of dimensions of the shape
 * @param  dims [IN]          the size of the specified dimension
 *
 * @retval ACL_ERROR_NONE     The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclSetTensorOriginShape(aclTensorDesc *desc, int numDims, const int64_t *dims);

/**
 * @ingroup AscendCL
 * @brief get op description info
 *
 * @param desc [IN]     pointer to tensor description
 * @param index [IN]    index of tensor
 *
 * @retval null for failed.
 * @retval OtherValues success.
*/
ACL_FUNC_VISIBILITY aclTensorDesc *aclGetTensorDescByIndex(aclTensorDesc *desc, size_t index);

/**
 * @ingroup AscendCL
 * @brief get address of tensor
 *
 * @param desc [IN]    pointer to tensor description
 *
 * @retval null for failed
 * @retval OtherValues success
*/
ACL_FUNC_VISIBILITY void *aclGetTensorDescAddress(const aclTensorDesc *desc);

/**
 * @ingroup AscendCL
 * @brief Set the dynamic input name specified by the tensor description
 *
 * @param  desc [IN|OUT]      pointer to the instance of aclTensorDesc
 * @param  dynamicInputName [IN]       pointer to the dynamic input name
 *
 * @retval ACL_ERROR_NONE     The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclSetTensorDynamicInput(aclTensorDesc *desc, const char *dynamicInputName);

/**
 * @ingroup AscendCL
 * @brief an interface for users to output  APP logs
 *
 * @param logLevel [IN]    the level of current log
 * @param func [IN]        the function where the log is located
 * @param file [IN]        the file where the log is located
 * @param line [IN]        Number of source lines where the log is located
 * @param fmt [IN]         the format of current log
 * @param ... [IN]         the value of current log
 */
ACL_FUNC_VISIBILITY void aclAppLog(aclLogLevel logLevel, const char *func, const char *file, uint32_t line,
    const char *fmt, ...);


ACL_FUNC_VISIBILITY aclError aclSetTensorPlaceMent(aclTensorDesc *desc, aclMemType type);

#define ACL_APP_LOG(level, fmt, ...) \
    aclAppLog(level, __FUNCTION__, __FILE__, __LINE__, fmt, ##__VA_ARGS__)

#ifdef __cplusplus
}
#endif

#endif // INC_EXTERNAL_ACL_ACL_BASE_H_
