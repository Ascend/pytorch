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

from abc import ABCMeta, abstractmethod


class BaseViewParser(metaclass=ABCMeta):
    """
    prof_interface for viewer
    """

    def __init__(self, profiler_path: str):
        self._profiler_path = profiler_path

    @abstractmethod
    def generate_view(self, output_path: str, **kwargs) -> None:
        """
        summarize data to generate json or csv files
        Returns: None
        """
