#include "cpp_common.h"

#include <string.h>
#include <sys/syscall.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/csrc/profiler/profiler_mgr.h>
#include <dlfcn.h>
#include <memory>

extern "C" {
#pragma pack(1)
typedef struct ApiDef {
    unsigned short int magicNumber;
    unsigned short int level;
    unsigned int type;
    unsigned int threadId;
    unsigned int reserve;
    unsigned long int beginTime;
    unsigned long int endTime;
    unsigned long int itemId;
}MsprofApi;

typedef struct NodeBasicInfo {
    unsigned long int opName;
    unsigned int taskType;
    unsigned long int opType;
    unsigned int blockDim;
    unsigned int opFlag;
}MsprofNodeBasicInfo;

typedef struct CompactInfo {
    unsigned short int magicNumber;
    unsigned short int level;
    unsigned int type;
    unsigned int threadId;
    unsigned int dataLen;
    unsigned long int timeStamp;
    union {
        unsigned char info[40];
        MsprofNodeBasicInfo nodeBasicInfo;
    } data;
}MsprofCompactInfo;

extern int MsprofReportApi(unsigned int agingFlag, const MsprofApi *api);
extern int MsprofReportCompactInfo(unsigned int  agingFlag, const void* data, unsigned int length);
extern unsigned long int  MsprofGetHashId(char *hashInfo, size_t length);
extern unsigned long int  MsprofSysCycleTime();

extern aclError aclrtMallocHost(void **hostPtr, size_t size);
extern aclError aclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy);
extern aclError aclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind);

extern aclError aclrtFreeHost(void *hostPtr);
extern aclError aclrtFree(void *devPtr);

typedef struct TilingMem {
    std::unique_ptr<void, decltype(&aclrtFreeHost)> arg_tiling_host;
    std::unique_ptr<void, decltype(&aclrtFree)> arg_tiling_device;
    TilingMem()
        : arg_tiling_host(nullptr, aclrtFreeHost), 
          arg_tiling_device(nullptr, aclrtFree) 
    {}
}TilingMemInfo;

TilingMemInfo MEM_CACHE;

typedef struct WorkspaceMem {
    std::unique_ptr<void, decltype(&aclrtFreeHost)> arg_workspace_host;
    std::unique_ptr<void, decltype(&aclrtFree)> arg_workspace_device;
    WorkspaceMem()
        : arg_workspace_host(nullptr, aclrtFreeHost), 
          arg_workspace_device(nullptr, aclrtFree) 
    {}
}WorkspaceMemInfo;

WorkspaceMemInfo MEM_WORK_CACHE;
}

rtError_t common_launch(char* kernelName, const void* func, uint32_t gridX,
                        void* args, uint32_t argsSize, rtStream_t stream)
{
    unsigned long int beginTime = 0;
    unsigned long int endTime = 0;
    unsigned long int opName = 0;
    unsigned int threadId = 0;
    size_t length = strlen(kernelName);

    if (torch_npu::profiler::GetTraceLevel() != -1) {
        beginTime = MsprofSysCycleTime();
    }
    rtError_t ret = rtKernelLaunch(func, gridX, args, argsSize, NULL, stream);

    if (torch_npu::profiler::GetTraceLevel() != -1) {
        endTime = MsprofSysCycleTime();
        opName = MsprofGetHashId(kernelName, length);
        threadId = (unsigned int)(syscall(SYS_gettid));
        MsprofApi info;
        info.magicNumber = 0x5a5a; // MSPROF_REPORT_DATA_MAGIC_NUM
        info.level = 10000;        // MSPROF_REPORT_NODE_LEVEL
        info.type = 5;             // MSPROF_REPORT_NODE_LAUNCH_TYPE
        info.threadId = threadId;
        info.reserve = 0;
        info.beginTime = beginTime;
        info.endTime = endTime;
        info.itemId = opName;
        MsprofReportApi(0, &info);
    }
    if (torch_npu::profiler::GetTraceLevel() >= 1) {
        MsprofCompactInfo nodeBasicInfo;
        nodeBasicInfo.magicNumber = 0x5a5a; // MSPROF_REPORT_DATA_MAGIC_NUM
        nodeBasicInfo.level = 10000;        // MSPROF_REPORT_NODE_LEVEL
        nodeBasicInfo.type = 0;             // MSPROF_REPORT_NODE_BASIC_INFO_TYPE
        nodeBasicInfo.threadId = threadId;
        nodeBasicInfo.timeStamp = endTime;
        nodeBasicInfo.data.nodeBasicInfo.opName = opName;
        nodeBasicInfo.data.nodeBasicInfo.taskType = 0; // MSPROF_GE_TASK_TYPE_AI_CORE
        nodeBasicInfo.data.nodeBasicInfo.opType = opName;
        nodeBasicInfo.data.nodeBasicInfo.blockDim = gridX;
        MsprofReportCompactInfo(0, &nodeBasicInfo, sizeof(MsprofCompactInfo));
    }
    return ret;
}

static void prepare_tiling(void* args, void* tiling_func, int64_t tilingSize, void* arg_tiling_host, 
                           void* arg_tiling_device, uint32_t gridX, rtStream_t stream, uint32_t argsSize)
{
    uint32_t args_num = argsSize / sizeof(void *);
    void **args_cast = static_cast<void **>(args);
    
    args_cast[args_num - 5] = arg_tiling_host; // MEM_CACHE.arg_tiling_host.get(); // 5: TilingMemrefAlignedOffset
    args_cast[args_num - 4] = arg_tiling_host; //MEM_CACHE.arg_tiling_host.get(); // 4: TilingMemrefAllocatedOffset

    // tiling_func to update args
    typedef int64_t (*mlir_tiling_func)(void*);
    mlir_tiling_func func_tiling_pre = reinterpret_cast<mlir_tiling_func>(tiling_func);

    // update args with tiling_key from tiling_func

    func_tiling_pre(args);
    
    // copy host arg_tiling to device arg_tiling, and also replace corresponding place in args
    aclError err = aclrtMemcpy(arg_tiling_device, tilingSize, arg_tiling_host,
                tilingSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (err != ACL_ERROR_NONE) {
        printf("aclrtMemcpy Failed, err: %d \n", err);
        return;
    }
    
    args_cast[args_num - 5] = arg_tiling_device;
    args_cast[args_num - 4] = arg_tiling_device;
}

rtError_t common_launch_dyn(char* kernelName, void* func, void* tiling_func, int64_t tilingSize, uint32_t gridX,
                           void* args, uint32_t argsSize, rtStream_t stream)
{
    unsigned long int beginTime = 0;
    unsigned long int endTime = 0;
    unsigned long int opName = 0;
    unsigned int threadId = 0;
    size_t length = strlen(kernelName);

    if (tilingSize != 0) {
        void *arg_tiling_host = nullptr;
        void *arg_tiling_device = nullptr;
        aclError err = aclrtMallocHost((void **)&arg_tiling_host, tilingSize);
        if (err != ACL_ERROR_NONE) {
            printf("Failed to malloc arg_tiling_host, err: %d \n", err);
        }
        // malloc device memory for device arg_tiling
        err = aclrtMalloc((void **)&arg_tiling_device, tilingSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (err != ACL_ERROR_NONE) {
            printf("Failed to malloc arg_tiling_device, err: %d \n", err);
        }
        prepare_tiling(args, tiling_func, tilingSize, arg_tiling_host, arg_tiling_device, gridX, stream, argsSize);
        typedef void (*mlir_func)(uint32_t, void*, void*, void*);
        mlir_func func_cast = (mlir_func)func;
        if (torch_npu::profiler::GetTraceLevel() != -1) {
            beginTime = MsprofSysCycleTime();
        }
        func_cast(gridX, nullptr, stream, args);
    }
    else {
        typedef void (*mlir_func)(uint32_t, void*, void*, void*);
        mlir_func func_cast = (mlir_func)func;
        if (torch_npu::profiler::GetTraceLevel() != -1) {
            beginTime = MsprofSysCycleTime();
        }
        func_cast(gridX, nullptr, stream, args);
    }

    if (torch_npu::profiler::GetTraceLevel() != -1) {
        endTime = MsprofSysCycleTime();
        opName = MsprofGetHashId(kernelName, length);
        threadId = (unsigned int)(syscall(SYS_gettid));
        MsprofApi info;
        info.magicNumber = 0x5a5a; // MSPROF_REPORT_DATA_MAGIC_NUM
        info.level = 10000;        // MSPROF_REPORT_NODE_LEVEL
        info.type = 5;             // MSPROF_REPORT_NODE_LAUNCH_TYPE
        info.threadId = threadId;
        info.reserve = 0;
        info.beginTime = beginTime;
        info.endTime = endTime;
        info.itemId = opName;
        MsprofReportApi(0, &info);
    }
    if (torch_npu::profiler::GetTraceLevel() >= 1) {
        MsprofCompactInfo nodeBasicInfo;
        nodeBasicInfo.magicNumber = 0x5a5a; // MSPROF_REPORT_DATA_MAGIC_NUM
        nodeBasicInfo.level = 10000;        // MSPROF_REPORT_NODE_LEVEL
        nodeBasicInfo.type = 0;             // MSPROF_REPORT_NODE_BASIC_INFO_TYPE
        nodeBasicInfo.threadId = threadId;
        nodeBasicInfo.timeStamp = endTime;
        nodeBasicInfo.data.nodeBasicInfo.opName = opName;
        nodeBasicInfo.data.nodeBasicInfo.taskType = 0; // MSPROF_GE_TASK_TYPE_AI_CORE
        nodeBasicInfo.data.nodeBasicInfo.opType = opName;
        nodeBasicInfo.data.nodeBasicInfo.blockDim = gridX;
        MsprofReportCompactInfo(0, &nodeBasicInfo, sizeof(MsprofCompactInfo));
    }

    return RT_ERROR_NONE;
}

void opcommand_call(const char *name, std::function<int()> launch_call)
{
    at_npu::native::OpCommand cmd;
    cmd.Name(name)
        .SetCustomHandler(launch_call)
        .Run();
}
