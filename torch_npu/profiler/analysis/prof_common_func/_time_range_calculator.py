# Copyright (c) 2024, Huawei Technologies.
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

from dataclasses import dataclass

__all__ = []


DEFAULT_INT_VALUE = -1


@dataclass
class TimeRange:
    start_ts: int = DEFAULT_INT_VALUE
    end_ts: int = DEFAULT_INT_VALUE


class CommunicationTimeRange(TimeRange):
    
    def __init__(self):
        super().__init__()


class RangeCaculator:

    @staticmethod
    def generate_time_range(start, end, class_range=TimeRange):
        time_range = class_range()
        time_range.start_ts, time_range.end_ts = start, end
        return time_range

    @staticmethod
    def merge_continuous_intervals(time_range_list: list):
        result = []
        if not time_range_list:
            return result
        time_range_list.sort(key=lambda x: x.start_ts)
        current_range = time_range_list[0]
        for time_range in time_range_list:
            if time_range.start_ts <= current_range.end_ts:
                current_range.end_ts = max(current_range.end_ts, time_range.end_ts)
            else:
                result.append(current_range)
                current_range = time_range
        result.append(current_range)
        return result

    @staticmethod
    def compute_pipeline_overlap(communication_range, compute_range):
        free_time_range = []
        pure_communication_range = []
        time_range_list = sorted(communication_range + compute_range, key=lambda x: x.start_ts)
        if not time_range_list:
            return pure_communication_range, free_time_range

        min_range = time_range_list.pop(0)
        for time_range in time_range_list:
            if min_range.end_ts - time_range.start_ts < 0:
                free_time_range.append(
                    RangeCaculator.generate_time_range(min_range.end_ts, time_range.start_ts)
                )
                if isinstance(min_range, CommunicationTimeRange):
                    pure_communication_range.append(
                        RangeCaculator.generate_time_range(min_range.start_ts, min_range.end_ts)
                    )
                min_range = time_range
                continue
            if min_range.end_ts - time_range.end_ts < 0:
                if isinstance(min_range, CommunicationTimeRange):
                    pure_communication_range.append(
                        RangeCaculator.generate_time_range(min_range.start_ts, time_range.start_ts)
                    )
                    min_range = RangeCaculator.generate_time_range(min_range.end_ts, time_range.end_ts)
                if isinstance(time_range, CommunicationTimeRange):
                    min_range = RangeCaculator.generate_time_range(
                        min_range.end_ts, time_range.end_ts, class_range=CommunicationTimeRange
                    )
            else:
                if isinstance(min_range, CommunicationTimeRange):
                    pure_communication_range.append(
                        RangeCaculator.generate_time_range(min_range.start_ts, time_range.start_ts)
                    )
                    min_range = RangeCaculator.generate_time_range(
                        time_range.end_ts, min_range.end_ts, class_range=CommunicationTimeRange
                    )
                if isinstance(time_range, CommunicationTimeRange):
                    min_range = RangeCaculator.generate_time_range(time_range.end_ts, min_range.end_ts)
        if isinstance(min_range, CommunicationTimeRange):
            pure_communication_range.append(min_range)
        return pure_communication_range, free_time_range
