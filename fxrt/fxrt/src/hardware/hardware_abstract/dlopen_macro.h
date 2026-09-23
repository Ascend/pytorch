#ifndef FXRT_SRC_HARDWARE_DLOPEN_MACRO_H
#define FXRT_SRC_HARDWARE_DLOPEN_MACRO_H

#ifndef _WIN32
#include <dlfcn.h>
#else
#include <windows.h>
#undef ERROR
#undef SM_DEBUG
#undef Yield
#endif
#include <string>
#include <functional>

#ifndef _WIN32
#define PORTABLE_EXPORT __attribute__((visibility("default")))
#else
#define PORTABLE_EXPORT __declspec(dllexport)
#endif
#include "common/common.h"

constexpr char kSimuSocName[] = "FXRT_DRY_RUN";

template <typename T>
struct SimuDataFactory {
  static T Data() {
    static T data{};
    return data;
  }
};

template <typename T>
struct SimuDataFactory<T*> {
  static T* Data() {
    static int data{};
    return reinterpret_cast<T*>(&data);
  }
};

template <typename T>
struct SimuDataFactory<T**> {
  static T** Data() {
    static int data{};
    static T* data_ptr = reinterpret_cast<T*>(&data);
    return &data_ptr;
  }
};

template <typename T>
struct SimuCreateTypeGetter {
  typedef T type;
};

template <typename T>
struct SimuCreateTypeGetter<T*> {
  typedef T type;
};

template <typename T>
struct SimuCreateTypeGetter<T**> {
  typedef T* type;
};

#define PLUGIN_METHOD(name, return_type, ...)                   \
  extern "C" {                                                  \
  PORTABLE_EXPORT return_type Plugin##name(__VA_ARGS__);        \
  }                                                             \
  constexpr const char* k##name##Name = "Plugin" #name;         \
  using name##FunObj = std::function<return_type(__VA_ARGS__)>; \
  using name##FunPtr = return_type (*)(__VA_ARGS__);

#define ORIGIN_METHOD(name, return_type, ...)                   \
  extern "C" {                                                  \
  return_type name(__VA_ARGS__);                                \
  }                                                             \
  constexpr const char* k##name##Name = #name;                  \
  using name##FunObj = std::function<return_type(__VA_ARGS__)>; \
  using name##FunPtr = return_type (*)(__VA_ARGS__);

#define ORIGIN_METHOD_WITH_SIMU(name, return_type, ...) \
  ORIGIN_METHOD(name, return_type, __VA_ARGS__)         \
  template <typename T>                                 \
  inline T SimuFuncI##name(__VA_ARGS__) {               \
    return SimuDataFactory<T>::Data();                  \
  }                                                     \
                                                        \
  template <>                                           \
  inline void SimuFuncI##name(__VA_ARGS__) {}           \
  extern name##FunObj name##_;                          \
  inline void SimuAssignI##name() {                     \
    name##_ = SimuFuncI##name<return_type>;             \
  }

#define ACLRT_GET_SOC_NAME_WITH_SIMU(name, return_type, ...) \
  ORIGIN_METHOD(name, return_type, __VA_ARGS__)              \
  template <typename T>                                      \
  inline T SimuFuncI##name(__VA_ARGS__) {                    \
    return kSimuSocName;                                     \
  }                                                          \
                                                             \
  template <>                                                \
  inline void SimuFuncI##name(__VA_ARGS__) {}                \
  extern name##FunObj name##_;                               \
  inline void SimuAssignI##name() {                          \
    name##_ = SimuFuncI##name<return_type>;                  \
  }

#define ORIGIN_METHOD_WITH_SIMU_CREATE(name, return_type, create_type_ptr, ...)          \
  ORIGIN_METHOD(name, return_type, create_type_ptr, ##__VA_ARGS__)                       \
  template <typename T, typename U>                                                      \
  inline T SimuFuncI##name(U* in_ret, ##__VA_ARGS__) {                                   \
    static U st##name{};                                                                 \
    *in_ret = st##name;                                                                  \
    T ret{};                                                                             \
    return ret;                                                                          \
  }                                                                                      \
                                                                                         \
  template <>                                                                            \
  inline aclError SimuFuncI##name(void** in_ret, ##__VA_ARGS__) {                        \
    static uintptr_t currentPointer = 0;                                                 \
    currentPointer += sizeof(void*);                                                     \
    *in_ret = reinterpret_cast<void*>(currentPointer);                                   \
    return ACL_SUCCESS;                                                                  \
  }                                                                                      \
                                                                                         \
  template <>                                                                            \
  inline void SimuFuncI##name(void** in_ret, ##__VA_ARGS__) {                            \
    static uintptr_t currentPointer = 0;                                                 \
    currentPointer += sizeof(void*);                                                     \
    *in_ret = reinterpret_cast<void*>(currentPointer);                                   \
  }                                                                                      \
  extern name##FunObj name##_;                                                           \
  inline void SimuAssignI##name() {                                                      \
    name##_ = SimuFuncI##name<return_type, SimuCreateTypeGetter<create_type_ptr>::type>; \
  }

#define ASSIGN_SIMU(name) SimuAssignI##name();

inline static std::string GetDlErrorMsg() {
#ifndef _WIN32
  const char* result = dlerror();
  return (result == nullptr) ? "Unknown" : result;
#else
  return std::to_string(GetLastError());
#endif
}

template <class T>
static T DlsymWithCast(void* handle, const char* symbol_name) {
#ifndef _WIN32
  T symbol = reinterpret_cast<T>(reinterpret_cast<intptr_t>(dlsym(handle, symbol_name)));
#else
  T symbol = reinterpret_cast<T>(GetProcAddress(reinterpret_cast<HINSTANCE__*>(handle), symbol_name));
#endif
  if (symbol == nullptr) {
    RT_GLOG(ERROR) << "DlsymAscend failed";
    return nullptr;
  }
  return symbol;
}

#define DlsymFuncObj(funcName, plugin_handle) DlsymWithCast<funcName##FunPtr>(plugin_handle, k##funcName##Name);

template <class T>
static T DlsymAscend(void* handle, const char* symbol_name) {
  T symbol = reinterpret_cast<T>(reinterpret_cast<intptr_t>(dlsym(handle, symbol_name)));
  if (symbol == nullptr) {
    RT_GLOG(ERROR) << "DlsymAscend failed";
    return nullptr;
  }
  return symbol;
}

#define DlsymAscendFuncObj(funcName, plugin_handle) DlsymAscend<funcName##FunPtr>(plugin_handle, k##funcName##Name)
#endif // FXRT_SRC_HARDWARE_DLOPEN_MACRO_H
