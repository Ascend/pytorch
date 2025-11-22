/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_EXTERNAL_GRAPH_ASCEND_STRING_H_
#define INC_EXTERNAL_GRAPH_ASCEND_STRING_H_

#include <string>
#include <memory>
#include <functional>
#include "graph/types.h"

namespace ge {
class  AscendStringImpl;

class AscendString {
public:
    AscendString() = default;

    ~AscendString() = default;

    AscendString(const char_t *const name);

    AscendString(const char_t *const name, size_t length);

    const char_t *GetString() const;

    size_t GetLength() const;

    size_t Find(const AscendString &ascend_string) const;

    size_t Hash() const;

    bool operator<(const AscendString &d) const;

    bool operator>(const AscendString &d) const;

    bool operator<=(const AscendString &d) const;

    bool operator>=(const AscendString &d) const;

    bool operator==(const AscendString &d) const;

    bool operator!=(const AscendString &d) const;

    bool operator==(const char_t *const d) const;

    bool operator!=(const char_t *const d) const;

private:
    std::shared_ptr<std::string> name_;
    friend class AscendStringImpl;
};
}  // namespace ge

namespace std {
template<>
struct hash<ge::AscendString> {
  size_t operator()(const ge::AscendString &name) const {
    return name.Hash();
  }
};
}  // namespace std
#endif  // INC_EXTERNAL_GRAPH_ASCEND_STRING_H_
