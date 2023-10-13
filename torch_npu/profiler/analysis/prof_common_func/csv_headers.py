# Copyright (c) 2023, Huawei Technologies.
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

class CsvHeaders(object):
    # op_summary
    TASK_START_TIME = "Task Start Time(us)"
    OP_SUMMARY_SHOW_HEADERS = ["Op Name", "OP Type", "Task Type", TASK_START_TIME, "Task Duration(us)",
                               "Task Wait Time(us)", "Block Dim"]
    OP_SUMMARY_KERNEL_BASE_HEADERS = ["Name", "Type", "Accelerator Core", "Start Time(us)", "Duration(us)",
                                      "Wait Time(us)", "Block Dim"]
