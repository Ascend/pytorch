/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/license/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "acl/op_api/aclnn_abs.h"

#ifdef __cplusplus
extern "C" {
#endif


aclnnStatus aclnnAbsGetWorkspaceSize(const aclTensor *self, aclTensor *out,
                                     uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;};

aclnnStatus aclnnAbs(void *workspace, uint64_4 workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

#ifdef __cplusplus
}
#endif