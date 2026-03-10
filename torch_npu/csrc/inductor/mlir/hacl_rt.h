#ifndef __HACL_RT_H__
#define __HACL_RT_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// If you need export the function of this library in Win32 dll, use __declspec(dllexport)
#ifndef RTS_API
#ifdef RTS_DLL_EXPORT
#define RTS_API __declspec(dllexport)
#else
#define RTS_API
#endif
#endif

/**
 * @ingroup dvrt_base
 * @brief stream handle.
 */
typedef void *rtStream_t;

/**
 * @ingroup dvrt_base
 * @brief runtime error numbers.
 */
typedef enum tagRtError {
    RT_ERROR_NONE = 0x0,                      // success
    RT_ERROR_INVALID_VALUE = 0x1,             // invalid value
    RT_ERROR_MEMORY_ALLOCATION = 0x2,         // memory allocation fail
    RT_ERROR_INVALID_RESOURCE_HANDLE = 0x3,   // invalid handle
    RT_ERROR_INVALID_DEVICE_POINTER = 0x4,    // invalid device point
    RT_ERROR_INVALID_MEMCPY_DIRECTION = 0x5,  // invalid memory copy dirction
    RT_ERROR_INVALID_DEVICE = 0x6,            // invalid device
    RT_ERROR_NO_DEVICE = 0x7,                 // no valid device
    RT_ERROR_CMD_OCCUPY_FAILURE = 0x8,        // command occpuy failure
    RT_ERROR_SET_SIGNAL_FAILURE = 0x9,        // set signal failure
    RT_ERROR_UNSET_SIGNAL_FAILURE = 0xA,      // unset signal failure
    RT_ERROR_OPEN_FILE_FAILURE = 0xB,         // unset signal failure
    RT_ERROR_WRITE_FILE_FAILURE = 0xC,
    RT_ERROR_MEMORY_ADDRESS_UNALIGNED = 0xD,
    RT_ERROR_DRV_ERR = 0xE,
    RT_ERROR_LOST_HEARTBEAT = 0xF,
    RT_ERROR_REPORT_TIMEOUT = 0x10,
    RT_ERROR_NOT_READY = 0x11,
    RT_ERROR_DATA_OPERATION_FAIL = 0x12,
    RT_ERROR_INVALID_L2_INSTR_SIZE = 0x13,
    RT_ERROR_DEVICE_PROC_HANG_OUT = 0x14,
    RT_ERROR_DEVICE_POWER_UP_FAIL = 0x15,
    RT_ERROR_DEVICE_POWER_DOWN_FAIL = 0x16,
    RT_ERROR_FEATURE_NOT_SUPPROT = 0x17,
    RT_ERROR_KERNEL_DUPLICATE = 0x18,         // register same kernel repeatly
    RT_ERROR_MODEL_STREAM_EXE_FAILED = 0x91,  // the model stream failed
    RT_ERROR_MODEL_LOAD_FAILED = 0x94,        // the model stream failed
    RT_ERROR_END_OF_SEQUENCE = 0x95,          // end of sequence
    RT_ERROR_NO_STREAM_CB_REG = 0x96,         // no callback register info for stream
    RT_ERROR_DATA_DUMP_LOAD_FAILED = 0x97,    // data dump load info fail
    RT_ERROR_CALLBACK_THREAD_UNSUBSTRIBE = 0x98,    // callback thread unsubstribe
    RT_ERROR_RESERVED
} rtError_t;

/**
 * @ingroup rt_kernel
 * @brief device binary type
 */
typedef struct tagRtDevBinary {
  uint32_t magic;    // magic number
  uint32_t version;  // version of binary
  const void *data;  // binary data
  uint64_t length;   // binary length
} rtDevBinary_t;

/**
 * @ingroup rt_kernel
 * @brief shared memory data control
 */
typedef struct tagRtSmData {
  uint64_t L2_mirror_addr;          // preload or swap source address
  uint32_t L2_data_section_size;    // every data size
  uint8_t L2_preload;               // 1 - preload from mirrorAddr, 0 - no preload
  uint8_t modified;                 // 1 - data will be modified by kernel, 0 - no modified
  uint8_t priority;                 // data priority
  int8_t prev_L2_page_offset_base;  // remap source section offset
  uint8_t L2_page_offset_base;      // remap destination section offset
  uint8_t L2_load_to_ddr;           // 1 - need load out, 0 - no need
  uint8_t reserved[2];              // reserved
} rtSmData_t;

/**
 * @ingroup rt_kernel
 * @brief shared memory description
 */
typedef struct tagRtSmCtrl {
  rtSmData_t data[8];  // data description
  uint64_t size;       // max page Num
  uint8_t remap[64];   /* just using for static remap mode, default:0xFF
                          array index: virtual l2 page id, array value: physic l2 page id */
  uint8_t l2_in_main;  // 0-DDR, 1-L2, default:0xFF
  uint8_t reserved[3];
} rtSmDesc_t;

/**
 * @ingroup rt_kernel
 * @brief magic number of plain binary for aicore
 */
#define RT_DEV_BINARY_MAGIC_PLAIN 0xabceed50

/**
 * @ingroup rt_kernel
 * @brief magic number of plain binary for aicpu
 */
#define RT_DEV_BINARY_MAGIC_PLAIN_AICPU 0xabceed51

/**
 * @ingroup rt_kernel
 * @brief magic number of plain binary for aivector
 */
#define RT_DEV_BINARY_MAGIC_PLAIN_AIVEC 0xabceed52

/**
 * @ingroup rt_kernel
 * @brief magic number of elf binary for aicore
 */
#define RT_DEV_BINARY_MAGIC_ELF 0x43554245

/**
 * @ingroup rt_kernel
 * @brief magic number of elf binary for aicpu
 */
#define RT_DEV_BINARY_MAGIC_ELF_AICPU 0x41415243

/**
 * @ingroup rt_kernel
 * @brief magic number of elf binary for aivector
 */
#define RT_DEV_BINARY_MAGIC_ELF_AIVEC 0x41415246

/**
 * @ingroup rt_kernel
 * @brief register device binary
 * @param [in] bin   device binary description
 * @param [out] handle   device binary handle
 * @return RT_ERROR_NONE for ok
 * @note:if this interface is changed, pls notify the compiler changing at the same time.
 */
RTS_API rtError_t rtDevBinaryRegister(const rtDevBinary_t *bin, void **handle);

/**
 * @ingroup rt_kernel
 * @brief register device function
 * @param [in] binHandle   device binary handle
 * @param [in] stubFunc   stub function
 * @param [in] stubName   stub function name
 * @param [in] devFunc   device function description. symbol name or address
 *                       offset, depending binary type.
 * @return RT_ERROR_NONE for ok
 * @note:if this interface is changed, pls notify the compiler changing at the same time.
 */
RTS_API rtError_t rtFunctionRegister(void *binHandle, const void *stubFunc, const char *stubName, const void *devFunc,
                                     uint32_t funcMode);

/**
 * @ingroup rt_kernel
 * @brief launch kernel to device
 * @param [in] stubFunc   stub function
 * @param [in] blockDim   block dimentions
 * @param [in] args   argments address for kernel function
 * @param [in] argsSize   argements size
 * @param [in] smDesc   shared memory description
 * @param [in] stream   associated stream
 * @return RT_ERROR_NONE for ok, errno for failed
 */
RTS_API rtError_t rtKernelLaunch(const void *stubFunc, uint32_t blockDim, void *args, uint32_t argsSize,
                                 rtSmDesc_t *smDesc, rtStream_t stream);

typedef struct tagRtArgsEx {
    void *args;                     // input + output + scalar + tiling addr + tiling data
    void *hostInputInfoPtr;         // nullptr
    uint32_t argsSize;              // input addr size + output addr size + scalar size + tiling addr size + tiling data size
    uint16_t tilingAddrOffset;      // size to tiling addr
    uint16_t tilingDataOffset;      // size to tiling data
    uint16_t hostInputInfoNum;      // 0
    uint8_t hasTiling;              // has tiling
    uint8_t isNoNeedH2DCopy;        // not need rtKernelLaunchWithFlag copy tiling from host to device 
    uint8_t reserved[4];
} rtArgsEx_t;

/**
 * @ingroup rt_kernel
 * @brief launch kernel and tiling to device
 * @param [in] stubFunc   stub function
 * @param [in] blockDim   block dimentions
 * @param [in] argsInfo   argments address for kernel function
 * @param [in] smDesc     shared memory description
 * @param [in] stream     associated stream
 * @param [in] flag       not use, set 0
 * @note:if this interface is changed, pls notify the compiler changing at the same time.
 */

RTS_API rtError_t rtKernelLaunchWithFlag(const void *stubFunc, uint32_t blockDim, rtArgsEx_t *argsInfo,
                                   rtSmDesc_t *smDesc, rtStream_t stream, uint32_t flag = 0);

/**
 * @ingroup dvrt_mem
 * @brief memory type
 */
#define RT_MEMORY_DEFAULT ((uint32_t)0x0)   // default memory on device
#define RT_MEMORY_HBM ((uint32_t)0x2)       // HBM memory on device
#define RT_MEMORY_DDR ((uint32_t)0x4)       // DDR memory on device
#define RT_MEMORY_SPM ((uint32_t)0x8)       // shared physical memory on device
#define RT_MEMORY_P2P_HBM ((uint32_t)0x10)  // HBM memory on other 4P device
#define RT_MEMORY_P2P_DDR ((uint32_t)0x11)  // DDR memory on other device
#define RT_MEMORY_DDR_NC ((uint32_t)0x20)   // DDR memory of non-cache
#define RT_MEMORY_TS_4G ((uint32_t)0x40)
#define RT_MEMORY_TS ((uint32_t)0x80)
#define RT_MEMORY_RESERVED ((uint32_t)0x100)

#define RT_MEMORY_L1 ((uint32_t)0x1<<16)
#define RT_MEMORY_L2 ((uint32_t)0x1<<17)

/**
 * @ingroup dvrt_mem
 * @brief memory type | memory Policy
 */
typedef uint32_t rtMemType_t;

/**
 * @ingroup dvrt_mem
 * @brief memory copy type
 */
typedef enum tagRtMemcpyKind {
  RT_MEMCPY_HOST_TO_HOST = 0,  // host to host
  RT_MEMCPY_HOST_TO_DEVICE,    // host to device
  RT_MEMCPY_DEVICE_TO_HOST,    // device to host
  RT_MEMCPY_DEVICE_TO_DEVICE,  // device to device, 1P && P2P
  RT_MEMCPY_MANAGED,           // managed memory
  RT_MEMCPY_ADDR_DEVICE_TO_DEVICE,
  RT_MEMCPY_HOST_TO_DEVICE_EX, // host  to device ex (only used for 8 bytes)
  RT_MEMCPY_RESERVED,
} rtMemcpyKind_t;

/**
 * @ingroup dvrt_mem
 * @brief alloc device memory
 * @param [in|out] devPtr   memory pointer
 * @param [in] size   memory size
 * @param [in] type   memory type
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_MEMORY_ALLOCATION for memory allocation failed
 */
RTS_API rtError_t rtMalloc(void **devPtr, uint64_t size, rtMemType_t type);

/**
 * @ingroup dvrt_mem
 * @brief free device memory
 * @param [in|out] devPtr   memory pointer
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_DEVICE_POINTER for error device memory pointer
 */
RTS_API rtError_t rtFree(void *devPtr);


/**
 * @ingroup dvrt_mem
 * @brief alloc host memory
 * @param [in|out] hostPtr   memory pointer
 * @param [in] size   memory size
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_MEMORY_ALLOCATION for memory allocation failed
 */
RTS_API rtError_t rtMallocHost(void **hostPtr, uint64_t size);

/**
 * @ingroup dvrt_mem
 * @brief free host memory
 * @param [in] hostPtr   memory pointer
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_DEVICE_POINTER for error device memory pointer
 */
RTS_API rtError_t rtFreeHost(void *hostPtr);

/**
 * @ingroup dvrt_mem
 * @brief synchronized memcpy
 * @param [in] dst   destination address pointer
 * @param [in] Max length of destination address memory
 * @param [in] src   source address pointer
 * @param [in] count   the number of byte to copy
 * @param [in] kind   memcpy type
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input of count
 * @return RT_ERROR_INVALID_DEVICE_POINTER for error input memory pointer of dst,src
 * @return RT_ERROR_INVALID_MEMCPY_DIRECTION for error copy direction of kind
 */
RTS_API rtError_t rtMemcpy(void *dst, uint64_t destMax, const void *src, uint64_t count, rtMemcpyKind_t kind);

/**
 * @ingroup dvrt_mem
 * @brief asynchronized memcpy
 * @param [in] dst   destination address pointer
 * @param [in] Max length of destination address memory
 * @param [in] src   source address pointer
 * @param [in] count   the number of byte to copy
 * @param [in] kind   memcpy type
 * @param [in] stream   asynchronized task stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input of count,stream
 * @return RT_ERROR_INVALID_DEVICE_POINTER for error input memory pointer of dst,src
 * @return RT_ERROR_INVALID_MEMCPY_DIRECTION for error copy direction of kind
 */
RTS_API rtError_t rtMemcpyAsync(void *dst, uint64_t destMax, const void *src, uint64_t count, rtMemcpyKind_t kind,
                                rtStream_t stream);

/**
 * @ingroup dvrt_stream
 * @brief create stream instance
 * @param [in|out] stream   created stream
 * @param [in] priority   stream priority
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_RESOURCE_HANDLE for error input stream handle
 * @return RT_ERROR_INVALID_VALUE for error input priority
 */
RTS_API rtError_t rtStreamCreate(rtStream_t *stream, int32_t priority);

/**
 * @ingroup dvrt_stream
 * @brief destroy stream instance.
 * @param [in] stream   the stream to destroy
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_RESOURCE_HANDLE for error input stream handle
 */
RTS_API rtError_t rtStreamDestroy(rtStream_t stream);

/**
 * @ingroup dvrt_stream
 * @brief wait stream to be complete
 * @param [in] stream   stream to wait
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_RESOURCE_HANDLE for error input stream or event handle
 */
RTS_API rtError_t rtStreamSynchronize(rtStream_t stream);

/**
 * @ingroup dvrt_dev
 * @brief set target device for current thread
 * @param [int] device   the device id
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_DEVICE for can not match ID and device
 */
RTS_API rtError_t rtSetDevice(int32_t device);

/**
 * @ingroup dvrt_dev
 * @brief reset all opened device
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_DEVICE if no device set
 */
RTS_API rtError_t rtDeviceReset(int32_t device);

RTS_API rtError_t rtGetTaskIdAndStreamID(uint32_t *taskId, uint32_t *streamId);

RTS_API rtError_t rtGetC2cCtrlAddr(uint64_t *addr, uint32_t *len);

#ifndef char_t
typedef char char_t;
#endif

/**
 * @ingroup dvrt_dev
 * @brief get chipType
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t rtGetSocVersion(char_t *ver, const uint32_t maxLen);

/**
 * @ingroup
 * @brief get AI core count
 * @param [in] aiCoreCnt
 * @return aiCoreCnt
 */
RTS_API rtError_t rtGetAiCoreCount(uint32_t *aiCoreCnt);

/**
 * @ingroup
 * @brief get AI cpu count
 * @param [in] aiCpuCnt
 * @return aiCpuCnt
 */
RTS_API rtError_t rtGetAiCpuCount(uint32_t *aiCpuCnt);

#ifdef __cplusplus
}
#endif

#endif  // __HACL_RT_H__