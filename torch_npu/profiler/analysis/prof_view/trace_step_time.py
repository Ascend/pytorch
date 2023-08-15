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

from ..prof_common_func.global_var import GlobalVar
from ..prof_common_func.trace_event_manager import TraceEventManager
from ..prof_common_func.constant import Constant
from ..prof_view.base_view_parser import BaseViewParser
from ..prof_common_func.file_manager import FileManager

class TraceStepTimeParser(BaseViewParser):
    timeflag = {'communication_time':'comun', 'compute_time':'compute', 'free_time':'free', 'communication_not_overlapped':'comunNotOverlp', 'hcom_receive':'stage'}
    title = ['Step','Computing','Communication(Not Overlapped)','Overlapped','Communication','Free','Stage','Bubble','Communication(Not Overlapped and exclude receive)']
    @classmethod
    def count_time(cls, addtype, addtime, durtime, step_list, save_time):
        cur_step = None
        for step in step_list:
            if step[1] < addtime and step[2] > addtime:
                cur_step = step[0]
                break
        for cur_save in save_time:
            if cur_save['step'] == cur_step:
                cur_save[cls.timeflag[addtype]] += durtime
                break

    @classmethod
    def create_step_file(cls, output_path: str, json_str: list, file_name: str) -> None:
        step_list = []
        save_time = []
        if not json_str:
            return
        loc = 0
        # get step time
        hasStepFlag = False
        for curStep in GlobalVar.step_range:
            step_list.append([curStep[0], curStep[1], curStep[2]])
            save_time.append({'step': curStep[0], 'compute': 0,'comunNotOverlp': 0, 'Overlp': 0, 'comun': 0, 'free': 0, 'stage': 0, 'bubble': 0, 'comunNotOverlpRec': 0 })
            hasStepFlag = True
        if hasStepFlag == False:
            save_time.append({'step': None, 'compute': 0,'comunNotOverlp': 0, 'Overlp': 0, 'comun': 0, 'free': 0, 'stage': 0, 'bubble': 0, 'comunNotOverlpRec': 0 })


        for data in json_str:
            if data['name'] in {'communication_time', 'compute_time', 'free_time', 'communication_not_overlapped'}:
                cls.count_time(data['name'], float(data['ts']), float(data['dur']), step_list, save_time)
            elif str(data['name']).startswith('hcom_receive'):
                cls.count_time('bubble', float(data['ts']), float(data['dur']), step_list, save_time)
        for calc_time in save_time:
            print(calc_time['comunNotOverlpRec'])
            print(type(calc_time['comunNotOverlpRec']))
            calc_time['comunNotOverlpRec'] = calc_time['comunNotOverlp'] - calc_time['bubble']
            calc_time['Overlp'] = calc_time['comun'] - calc_time['comunNotOverlp']
        print_time = []
        for step in save_time:
            print(step)
            print_time.append([step['step'], step['compute'], step['comunNotOverlp'], step['Overlp'], step['comun'], step['free'], step['stage'], step['bubble'], step['comunNotOverlpRec']])
        FileManager.create_csv_file(output_path, print_time, file_name, cls.title)
