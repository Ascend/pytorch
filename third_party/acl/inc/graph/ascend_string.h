#ifndef INC_EXTERNAL_GRAPH_ASCEND_STRING_H_
#define INC_EXTERNAL_GRAPH_ASCEND_STRING_H_

#include <string>
#include <memory>
#include <functional>

namespace ge {
class AscendString {
public:
  AscendString() = default;

  ~AscendString() = default;

  AscendString(const char* name);

  const char* GetString() const;

  bool operator<(const AscendString& d) const;

  bool operator>(const AscendString& d) const;

  bool operator<=(const AscendString& d) const;

  bool operator>=(const AscendString& d) const;

  bool operator==(const AscendString& d) const;

  bool operator!=(const AscendString& d) const;

private:
  std::shared_ptr<std::string> name_;
};
}  // namespace ge

namespace std {
template <>
struct hash<ge::AscendString> {
  size_t operator()(const ge::AscendString &name) const {
    std::string str_name;
    if (name.GetString() != nullptr) {
      str_name = name.GetString();
    }
    return hash<string>()(str_name);
  }
};
}
#endif  // INC_EXTERNAL_GRAPH_ASCEND_STRING_H_
