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
from ..prof_view.base_view_parser import BaseViewParser
from ..prof_common_func.file_manager import FileManager


class TraceStepTimeParser(BaseViewParser):
    timeflag = {'Communication': 'comun', 'Computing': 'compute', 'Free': 'free',
                'Communication(Not Overlapped)': 'comunNotOverlp', 'hcom_receive': 'bubble'}
    title = ['Step', 'Computing', 'Communication(Not Overlapped)', 'Overlapped', 'Communication', 'Free', 'Stage',
             'Bubble', 'Communication(Not Overlapped and Exclude Receive)']

    @classmethod
    def is_float_num(cls, num):
        try:
            float(num)
            return True
        except VauleError:
            return False

    @classmethod
    def count_time(cls, addtype, addtime, durtime, step_list, save_time):
        cur_step = None
        if not cls.is_float_num(addtime) or not cls.is_float_num(durtime):
            print('Ts or dur format error!')
            return
        addtime = float(addtime)
        durtime = float(durtime)
        for step in step_list:
            if step[1] < addtime and step[2] > addtime:
                cur_step = step[0]
                break
        for step in step_list:
            if cur_step == step[0]:
                if addtime < step[3] or step[3] == -1:
                    step[3] = addtime
                if addtime + durtime > step[4] or step[4] == -1:
                    step[4] = addtime + durtime
                break
        for cur_save in save_time:
            if cur_save['step'] == cur_step:
                cur_save[cls.timeflag[addtype]] += durtime
                break

    @classmethod
    def getE2ETime(cls, step, step_list):
        for curStep in step_list:
            if curStep[0] == step:
                return curStep[4] - curStep[3]

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
            step_list.append([curStep[0], curStep[1], curStep[2], -1, -1])
            save_time.append(
                {'step': curStep[0], 'compute': 0, 'comunNotOverlp': 0, 'Overlp': 0, 'comun': 0, 'free': 0, 'stage': 0,
                 'bubble': 0, 'comunNotOverlpRec': 0})
            hasStepFlag = True
        if not hasStepFlag:
            save_time.append(
                {'step': None, 'compute': 0, 'comunNotOverlp': 0, 'Overlp': 0, 'comun': 0, 'free': 0, 'stage': 0,
                 'bubble': 0, 'comunNotOverlpRec': 0})
            step_list.append([None, -1, -1, -1, -1])

        has_analysis_data_flag = False
        for data in json_str:
            time_start = -1
            time_end = -1
            if data.get('name') in {'Communication', 'Computing', 'Free', 'Communication(Not Overlapped)'}:
                cls.count_time(data.get('name'), data.get('ts', 0), data.get('dur', 0), step_list, save_time)
                has_analysis_data_flag = True
            elif str(data.get('name')).startswith('hcom_receive'):
                cls.count_time('hcom_receive', data.get('ts', 0), data.get('dur', 0), step_list, save_time)
        if not has_analysis_data_flag:
            return
        for calc_time in save_time:
            calc_time['comunNotOverlpRec'] = calc_time['comunNotOverlp'] - calc_time['bubble']
            calc_time['Overlp'] = calc_time['comun'] - calc_time['comunNotOverlp']
            calc_time['stage'] = cls.getE2ETime(calc_time['step'], step_list) - calc_time['bubble']
        print_time = []
        for step in save_time:
            print_time.append(
                [step['step'], step['compute'], step['comunNotOverlp'], step['Overlp'], step['comun'], step['free'],
                 step['stage'], step['bubble'], step['comunNotOverlpRec']])
        FileManager.create_csv_file(output_path, print_time, file_name, cls.title)
