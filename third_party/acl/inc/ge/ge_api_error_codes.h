#ifndef INC_EXTERNAL_GE_GE_API_ERROR_CODES_H_
#define INC_EXTERNAL_GE_GE_API_ERROR_CODES_H_

#include <map>
#include <string>
#include "ge_error_codes.h"

namespace ge {
#ifdef __GNUC__
#define ATTRIBUTED_DEPRECATED(replacement) __attribute__((deprecated("Please use " #replacement " instead.")))
#else
#define ATTRIBUTED_DEPRECATED(replacement) __declspec(deprecated("Please use " #replacement " instead."))
#endif

class GE_FUNC_VISIBILITY StatusFactory {
 public:
  static StatusFactory *Instance() {
    static StatusFactory instance;
    return &instance;
  }

  void RegisterErrorNo(uint32_t err, const std::string &desc) {
    // Avoid repeated addition
    if (err_desc_.find(err) != err_desc_.end()) {
      return;
    }
    err_desc_[err] = desc;
  }

  void RegisterErrorNo(uint32_t err, const char *desc) {
    if (desc == nullptr) {
      return;
    }
    std::string error_desc = desc;
    if (err_desc_.find(err) != err_desc_.end()) {
      return;
    }
    err_desc_[err] = error_desc;
  }

  std::string GetErrDesc(uint32_t err) {
    auto iter_find = err_desc_.find(err);
    if (iter_find == err_desc_.end()) {
      return "";
    }
    return iter_find->second;
  }

 protected:
  StatusFactory() {}
  ~StatusFactory() {}

 private:
  std::map<uint32_t, std::string> err_desc_;
};

class GE_FUNC_VISIBILITY ErrorNoRegisterar {
 public:
  ErrorNoRegisterar(uint32_t err, const std::string &desc) { StatusFactory::Instance()->RegisterErrorNo(err, desc); }
  ErrorNoRegisterar(uint32_t err, const char *desc) { StatusFactory::Instance()->RegisterErrorNo(err, desc); }
  ~ErrorNoRegisterar() {}
};

// Code compose(4 byte), runtime: 2 bit,  type: 2 bit,   level: 3 bit,  sysid: 8 bit, modid: 5 bit, value: 12 bit
#define GE_ERRORNO(runtime, type, level, sysid, modid, name, value, desc)                               \
  constexpr ge::Status name = (static_cast<uint32_t>(0xFFU & (static_cast<uint32_t>(runtime))) << 30) | \
                              (static_cast<uint32_t>(0xFFU & (static_cast<uint32_t>(type))) << 28) |    \
                              (static_cast<uint32_t>(0xFFU & (static_cast<uint32_t>(level))) << 25) |   \
                              (static_cast<uint32_t>(0xFFU & (static_cast<uint32_t>(sysid))) << 17) |   \
                              (static_cast<uint32_t>(0xFFU & (static_cast<uint32_t>(modid))) << 12) |   \
                              (static_cast<uint32_t>(0x0FFFU) & (static_cast<uint32_t>(value)));        \
  const ErrorNoRegisterar g_##name##_errorno(name, desc);

#define GE_ERRORNO_EXTERNAL(name, desc) const ErrorNoRegisterar g_##name##_errorno(name, desc);

using Status = uint32_t;

// General error code
GE_ERRORNO(0, 0, 0, 0, 0, SUCCESS, 0, "success");
GE_ERRORNO(0b11, 0b11, 0b111, 0xFF, 0b11111, FAILED, 0xFFF, "failed"); /*lint !e401*/

GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_PARAM_INVALID, "Parameter invalid.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_EXEC_NOT_INIT, "GE executor not initialized yet.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID, "Model file path invalid.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_EXEC_MODEL_ID_INVALID, "Model id invalid.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID, "Data size of model invalid.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_EXEC_MODEL_ADDR_INVALID, "Model addr invalid.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_EXEC_MODEL_QUEUE_ID_INVALID, "Queue id of model invalid.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_EXEC_LOAD_MODEL_REPEATED, "The model loaded repeatedly.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_DYNAMIC_INPUT_ADDR_INVALID, "Dynamic input addr invalid.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_DYNAMIC_INPUT_LENGTH_INVALID, "Dynamic input size invalid.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_DYNAMIC_BATCH_SIZE_INVALID, "Dynamic batch size invalid.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_AIPP_BATCH_EMPTY, "AIPP batch parameter empty.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_AIPP_NOT_EXIST, "AIPP parameter not exist.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_AIPP_MODE_INVALID, "AIPP mode invalid.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_OP_TASK_TYPE_INVALID, "Task type invalid.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_OP_KERNEL_TYPE_INVALID, "Kernel type invalid.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_PLGMGR_PATH_INVALID, "Plugin path is invalid.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_FORMAT_INVALID, "Format is invalid.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_SHAPE_INVALID, "Shape is invalid.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_DATATYPE_INVALID, "Datatype is invalid.");

GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_MEMORY_ALLOCATION, "Memory allocation error.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_MEMORY_OPERATE_FAILED, "Failed to operate memory.");

GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_INTERNAL_ERROR, "Internal error.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_LOAD_MODEL, "Load model error.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_EXEC_LOAD_MODEL_PARTITION_FAILED, "Failed to load model partition.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_EXEC_LOAD_WEIGHT_PARTITION_FAILED, "Failed to load weight partition.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_EXEC_LOAD_TASK_PARTITION_FAILED, "Failed to load task partition.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_EXEC_LOAD_KERNEL_PARTITION_FAILED, "Failed to load op kernel partition.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_EXEC_RELEASE_MODEL_DATA, "Failed to release the model data.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_COMMAND_HANDLE, "Command handle error.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_GET_TENSOR_INFO, "Get tensor info error.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_UNLOAD_MODEL, "Load model error.");

}  // namespace ge

#endif  // INC_EXTERNAL_GE_GE_API_ERROR_CODES_H_
