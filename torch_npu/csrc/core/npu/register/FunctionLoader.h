#include <dlfcn.h>
#include <mutex>
#include <memory>
#include <string>
#include <unordered_map>

namespace c10_npu {
namespace option {

/**
  FunctionLoader is used to store function address in the process.
  */
class FunctionLoader {
public:
    /**
        ctr
        */
    explicit FunctionLoader(const std::string& filename, int flags = RTLD_LAZY);
    /**
        dectr
        */
    ~FunctionLoader();
    /**
        set function name
        */
    void Set(const std::string& name);
    /**
        get function address by function name.
        */
    void* Get(const std::string& name);
private:
    mutable std::mutex mu_;
    std::string fileName;
    int flags;
    void* handle = nullptr;
    mutable std::unordered_map<std::string, void*> registry;
}; // class FunctionLoader


namespace register_function {
/**
  this class is used to register
  */
class FunctionRegister {
public:
    /**
        Singleton
        */
    static FunctionRegister* GetInstance();
    /**
        this API is used to store FunctionLoader class
        */
    void Register(const std::string& name, ::std::unique_ptr<FunctionLoader>& ptr);
    /**
        this API is used to associate library name and function name.
        */
    void Register(const std::string& name, const std::string& funcName);
    /**
        this API is used to get the function address by library and function name.
        */
    void* Get(const std::string& soName, const std::string& funcName);

private:
    FunctionRegister() = default;
    mutable std::mutex mu_;
    mutable std::unordered_map<std::string, ::std::unique_ptr<FunctionLoader>> registry;
}; // class FunctionRegister

/**
  FunctionRegisterBuilder is the helper of FunctionRegister.
  */
class FunctionRegisterBuilder {
public:
    /**
        ctr
        */
    FunctionRegisterBuilder(const std::string& name, ::std::unique_ptr<FunctionLoader>& ptr) noexcept;
    /**
        ctr
        */
    FunctionRegisterBuilder(const std::string& soName, const std::string& funcName) noexcept;
}; // class FunctionRegisterBuilder

} // namespace register_function

#define TORCH_NPU_REGISTER_LIBRARY(soName, ...)                                         \
    auto library_##soName = ::std::unique_ptr<c10_npu::option::FunctionLoader>( \
        new c10_npu::option::FunctionLoader(#soName, ##__VA_ARGS__));  \
    static c10_npu::option::register_function::FunctionRegisterBuilder                             \
        register_library_##soName(#soName, library_##soName);

#define TORCH_NPU_REGISTER_FUNCTION(soName, funcName)                              \
    static c10_npu::option::register_function::FunctionRegisterBuilder                             \
        register_function_##funcName(#soName, #funcName);

#define TORCH_NPU_GET_FUNCTION(soName, funcName)                                       \
    c10_npu::option::register_function::FunctionRegister::GetInstance()->Get(#soName, #funcName);

#define REGISTER_LIBRARY(soName, ...)                                           \
    _Pragma("GCC warning \"REGISTER_LIBRARY is deprecated, use TORCH_NPU_REGISTER_LIBRARY\"") \
    TORCH_NPU_REGISTER_LIBRARY(soName, ##__VA_ARGS__)

#define REGISTER_FUNCTION(soName, funcName)                                     \
    _Pragma("GCC warning \"REGISTER_FUNCTION is deprecated, use TORCH_NPU_REGISTER_FUNCTION\"") \
    TORCH_NPU_REGISTER_FUNCTION(soName, funcName)

#define GET_FUNCTION(soName, funcName)                                          \
    _Pragma("GCC warning \"GET_FUNCTION is deprecated, use TORCH_NPU_GET_FUNCTION\"") \
    TORCH_NPU_GET_FUNCTION(soName, funcName)

} // namespace option
} // namespace c10_npu
