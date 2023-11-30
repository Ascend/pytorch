#ifndef __TORCH_NPU_OPTION_REGISTER_H__
#define __TORCH_NPU_OPTION_REGISTER_H__

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <c10/util/Optional.h>

namespace c10_npu {
namespace option {

typedef void(*OptionCallBack) (const std::string&);
/**
  This class is used to storage env value, and provide Set and Get to
  */
class OptionInterface {
public:
  /**
    dctr
    */
    OptionInterface(OptionCallBack callback = nullptr);
  /**
    This API is used to store value.
    */
  void Set(const std::string& in);
  /**
    This API is used to load value.
    */
  std::string Get();
private:
/**
  Its used to store hook.
  */
  OptionCallBack callback = nullptr;
  std::string val;
};

namespace register_options {

/**
  This class is used to register OptionInterface
  */
class OptionRegister {
public:
  /**
    dctr
    */
  ~OptionRegister() = default;
  /**
    singleton
    */
  static OptionRegister* GetInstance();
  /**
    register
    */
  void Register(const std::string& name, ::std::unique_ptr<OptionInterface>& ptr);
  /**
    This API is used to store value to special key.
    */
  void Set(const std::string& name, const std::string& val);
  /**
    This API is used to load value from special key.
    */
  c10::optional<std::string> Get(const std::string& name);
private:
  OptionRegister() {}
  mutable std::mutex mu_;
  mutable std::unordered_map<std::string, ::std::unique_ptr<OptionInterface>> registry;
};

/**
  This class is the helper to construct class OptionRegister
  */
class OptionInterfaceBuilder {
public:
  OptionInterfaceBuilder(const std::string& name, ::std::unique_ptr<OptionInterface>& ptr, const std::string& type = "cli");
};

} // namespace register_options

/**
  This API is used to store key-value pairs
  */
void SetOption(const std::map<std::string, std::string>& options);
/**
  This API is used to store key-value pair
  */
void SetOption(const std::string& key, const std::string& val);
/**
  This API is used to load value by key
  */
c10::optional<std::string> GetOption(const std::string& key);

#define REGISTER_OPTION(name)                                       \
  REGISTER_OPTION_UNIQ(name, name, cli)

#define REGISTER_OPTION_INIT_BY_ENV(name)                           \
  REGISTER_OPTION_UNIQ(name, name, env)

#define REGISTER_OPTION_UNIQ(id, name, type)                        \
  auto options_interface_##id =                                     \
      ::std::unique_ptr<c10_npu::option::OptionInterface>(new c10_npu::option::OptionInterface());    \
  static c10_npu::option::register_options::OptionInterfaceBuilder                             \
      register_options_interface_##id(#name, options_interface_##id, #type);

#define REGISTER_OPTION_HOOK(name, ...)                                       \
  REGISTER_OPTION_HOOK_UNIQ(name, name, __VA_ARGS__)

#define REGISTER_OPTION_HOOK_UNIQ(id, name, ...)                                \
  auto options_interface_##id =                                                 \
      ::std::unique_ptr<c10_npu::option::OptionInterface>(                             \
        new c10_npu::option::OptionInterface(c10_npu::option::OptionCallBack(__VA_ARGS__)));  \
  static c10_npu::option::register_options::OptionInterfaceBuilder                     \
      register_options_interface_##id(#name, options_interface_##id);

#define REGISTER_OPTION_BOOL_FUNCTION(func, key, defaultVal, trueVal)  \
  bool func() {                                                     \
    auto val = c10_npu::option::GetOption(#key);                           \
    if (val.value_or(defaultVal) == (trueVal)) {                    \
      return true;                                                  \
    }                                                               \
    return false;                                                   \
  }

#define REGISTER_OPTION_BOOL_FUNCTION_UNIQ(func, key, defaultVal, trueVal)  \
  bool func() {                                                             \
    static auto val = c10_npu::option::GetOption(#key);                            \
    if (val.value_or(defaultVal) == (trueVal)) {                            \
      return true;                                                          \
    }                                                                       \
    return false;                                                           \
  }

#define REGISTER_OPTION_BOOL_FUNCTION_ALL_CASE(func, key, defaultVal, falseVal, trueVal)  \
  bool func() {                                                                           \
    auto val = c10_npu::option::GetOption(#key);                                          \
    if (val.has_value()) {                                                                \
        if (val.value() == (trueVal)) {                                                   \
            return true;                                                                  \
        }                                                                                 \
        if (val.value() == (falseVal)) {                                                  \
            return false;                                                                 \
        }                                                                                 \
    }                                                                                     \
    return (defaultVal) == (trueVal);                                                     \
  }

#define REGISTER_OPTION_CACHE(type, valueName, ...)                 \
    static thread_local type valueName##Value;                      \
    static thread_local bool valueName##Initialized = false;        \
    inline type GetWithCache##valueName() {                         \
        if (!valueName##Initialized) {                              \
            valueName##Value = __VA_ARGS__();                       \
            valueName##Initialized = true;                          \
        }                                                           \
        return valueName##Value;                                    \
    }                                                               \
    inline void SetWithCache##valueName(type value) {               \
        valueName##Value = value;                                   \
        valueName##Initialized = true;                              \
    }

#define GET_OPTION_WITH_CACHE(valueName)                            \
    GetWithCache##valueName()

#define SET_OPTION_WITH_CACHE(valueName, value)                     \
    SetWithCache##valueName(value)

} // namespace option
} // namespace c10_npu

#endif // __TORCH_NPU_OPTION_REGISTER_H__
