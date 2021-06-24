/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
