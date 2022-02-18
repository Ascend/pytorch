// Copyright (c) 2020 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "graph/operator_factory.h"

namespace ge {
Operator OperatorFactory::CreateOperator(
    const char* operator_name,
    const char* operator_type) {
  return Operator();
}

OperatorCreatorRegister::OperatorCreatorRegister(
    const char* operator_type,
    const OpCreatorV2& op_creator) {}

bool OperatorFactory::IsExistOp(const char* operator_type) {
  return true;
}
} // namespace ge