#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "acl/acl_base.h"
#include "acl/acl_mdl.h"

extern "C" {
// 资源初始化，申请与释放
aclError aclInit(const char *configPath){return 0;}
aclError aclFinalize(){return 0;}
aclError aclrtFree(void *devPtr){return 0;}
aclError aclrtGetDevice(int32_t *deviceId){return 0;}
aclError aclrtResetDevice(int32_t deviceId){return 0;}
aclError aclrtSetDevice(int32_t deviceId){return 0;}
aclError aclrtGetDeviceCount(uint32_t *count){return 0;}
aclError aclrtDeviceCanAccessPeer(int32_t *canAccessPeer, int32_t deviceId, int32_t peerDeviceId){return 0;}
aclError aclrtSynchronizeDevice(void){return 0;}
aclError aclmdlSetDump(const char *configPath){return 0;}
aclError aclmdlInitDump(){return 0;}
aclError aclmdlFinalizeDump(){return 0;}
aclError aclrtDeviceTaskAbort(int32_t deviceId, uint32_t timeout){return 0;}

// Stream
aclError aclrtCreateStream(aclrtStream *stream) { return 0; }
aclError aclrtSetStreamFailureMode(aclrtStream stream, uint64_t mode) { return 0; }
aclError aclrtSetOpWaitTimeout(uint32_t timeout) { return 0; }
aclError aclrtCreateStreamWithConfig(aclrtStream *stream, uint32_t priority, uint32_t flag) { return 0; }
aclError aclrtDestroyStream(aclrtStream stream){return 0;}
aclError aclrtDestroyStreamForce(aclrtStream stream){return 0;}
aclError aclrtSynchronizeStream(aclrtStream stream){return 0;}
aclError aclrtSynchronizeStreamWithTimeout(aclrtStream stream, int32_t timeout){return 0;}

// Event
aclError aclrtQueryEvent(aclrtEvent event, aclrtEventStatus *status){return 0;}
aclError aclrtQueryEventStatus(aclrtEvent event, aclrtEventRecordedStatus *status){return 0;}
aclError aclrtCreateEvent(aclrtEvent *event){return 0;}
aclError aclrtDestroyEvent(aclrtEvent event){return 0;}
aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream){return 0;}
aclError aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event){return 0;}
aclError aclrtSynchronizeEvent(aclrtEvent event){return 0;}
aclError aclrtEventElapsedTime(float *ms, aclrtEvent start, aclrtEvent end){return 0;}

// memory相关操作
aclError aclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy){return 0;}
aclError aclrtMallocAlign32(void **devPtr, size_t size, aclrtMemMallocPolicy policy){return 0;}
aclError aclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind){return 0;}
aclError aclrtMemcpyAsync(void *dst, size_t destMax, const void *src,
                          size_t count, aclrtMemcpyKind kind, aclrtStream stream){return 0;}
aclError aclrtMallocHost(void **hostPtr, size_t size){return 0;}
aclError aclrtFreeHost(void *hostPtr){return 0;}
aclError aclrtGetMemInfo(aclrtMemAttr attr, size_t *free, size_t *total){return 0;}
aclError aclrtGetMemUceInfo(int32_t deviceId, aclrtMemUceInfo* memUceInfoArray, size_t arraySize, size_t *retSize){return 0;}
aclError aclrtMemUceRepair(int32_t deviceId, aclrtMemUceInfo* memUceInfoArray, size_t arraySize){return 0;}
aclError aclrtCmoAsync(void *src, size_t size, aclrtCmoType cmoType, aclrtStream stream){return 0;}
aclError aclrtGetLastError(aclrtLastErrLevel flag){return 0;}
aclError aclrtPeekAtLastError(aclrtLastErrLevel flag){return 0;}


// op相关操作
aclopAttr *aclopCreateAttr(){return NULL;}
void aclopDestroyAttr(const aclopAttr *attr){return;}
aclError aclopSetAttrBool(aclopAttr *attr, const char *attrName, uint8_t attrValue){return 0;}
aclError aclopSetAttrInt(aclopAttr *attr, const char *attrName, int64_t attrValue){return 0;}
aclError aclopSetAttrFloat(aclopAttr *attr, const char *attrName, float attrValue){return 0;}
aclError aclopSetAttrString(aclopAttr *attr, const char *attrName, const char *attrValue){return 0;}
aclError aclopSetAttrListInt(aclopAttr *attr, const char *attrName, int numValues, const int64_t *values){return 0;}
aclError aclopSetAttrListFloat(aclopAttr *attr, const char *attrName, int numValues, const float *values){return 0;}
aclError aclopSetAttrListBool(aclopAttr *attr, const char *attrName, int numValues, const uint8_t *values){return 0;}
aclError aclopSetAttrDataType(aclopAttr *attr, const char *attrName, aclDataType values){return 0;}
// Tensor相关
aclTensorDesc *aclCreateTensorDesc(aclDataType dataType, int numDims, const int64_t *dims, aclFormat format){return NULL;}
void aclDestroyTensorDesc(const aclTensorDesc *desc){return;}
aclDataBuffer *aclCreateDataBuffer(void *data, size_t size){return NULL;}
aclError aclDestroyDataBuffer(const aclDataBuffer *dataBuffer){return 0;}
void aclSetTensorDescName(aclTensorDesc *desc, const char *name){return;}
aclError aclSetTensorFormat(aclTensorDesc *desc, aclFormat format){return 0;}
aclError aclSetTensorShape(aclTensorDesc *desc, int numDims, const int64_t *dims){return 0;}
aclError aclSetTensorShapeRange(aclTensorDesc* desc, size_t dimsCount, int64_t dimsRange[][ACL_TENSOR_SHAPE_RANGE_NUM]){return 0;}
aclError aclGetTensorDescDimV2(const aclTensorDesc *desc, size_t index, int64_t *dimSize) {return 0;}

size_t aclGetTensorDescNumDims(const aclTensorDesc *desc) {return 0;}
aclDataType aclGetTensorDescType(const aclTensorDesc *desc) {return ACL_FLOAT;}
int64_t aclGetTensorDescDim(const aclTensorDesc *desc, size_t index) {return 0;}
aclFormat aclGetTensorDescFormat(const aclTensorDesc *desc) {return ACL_FORMAT_NCHW;}
const char *aclGetTensorDescName(aclTensorDesc *desc) {return NULL;}

aclError aclSetTensorPlaceMent(aclTensorDesc *desc, aclMemType type) {return 0;};
aclError aclrtSetDeviceSatMode(aclrtFloatOverflowMode mode) {return 0;};
aclError aclrtGetDeviceUtilizationRate(int32_t deviceId, aclrtUtilizationInfo *utilizationInfo) {return 0;};

void aclAppLog(aclLogLevel logLevel, const char *func, const char *file, uint32_t line, const char *fmt, ...) {}
aclError aclrtSetExceptionInfoCallback(aclrtExceptionInfoCallback callback) {return 0;}
aclError aclrtCreateEventWithFlag(aclrtEvent *event, uint32_t flag) {return 0;}
aclError aclrtCreateEventExWithFlag(aclrtEvent *event, uint32_t flag) {return 0;}
aclError aclrtResetEvent(aclrtEvent event, aclrtStream stream){return 0;}
aclError aclrtStreamQuery(aclrtStream stream, aclrtStreamStatus *status) {return 0;};
}
