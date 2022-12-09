# Copyright (c) 2020 Huawei Technologies Co., Ltd
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

import json
import argparse
import numpy as np
import pandas as pd


def cosine_similarity(a, b):
    a, b = np.mat(a), np.mat(b)
    num = float(a * b.T)
    denorm = np.linalg.norm(a) * np.linalg.norm(b)
    cos = num / denorm
    sim = 0.5 + 0.5 * cos
    return sim


def get_rmse(a, b):
    rmse = np.linalg.norm(a - b) / np.sqrt(len(a))
    return rmse


def get_mape(a, b):
    mape_val = sum(np.abs((a - b) / b)) / len(b) * 100
    mape = str(round(mape_val, 4)) + '%'
    return mape


def check_op(a, b, shape_flag):
    a_op_name = [_.split('_', 1)[1] for _ in a["op_name"]]
    b_op_name = [_.split('_', 1)[1] for _ in b["op_name"]]
    if shape_flag:
        return a_op_name == b_op_name and a["input_struct"] == b["input_struct"] \
            and a["output_struct"] == b["output_struct"]
    else:
        return a_op_name == b_op_name


def merge_tensor(tensor_list):
    op_dict = {}
    op_dict["op_name"] = []
    op_dict["input_struct"] = []
    op_dict["output_struct"] = []
    op_dict["input_value"] = []
    op_dict["output_value"] = []
    for tensor in tensor_list:
        op_dict["op_name"].append(tensor[0])
        if tensor[0].find("input") != -1:
            op_dict["input_struct"].append((tensor[2], tensor[3]))
            op_dict["input_value"].append(tensor[1])
        else:
            op_dict["output_struct"].append((tensor[2], tensor[3]))
            op_dict["output_value"].append(tensor[1])
    return op_dict


def read_op(ops_queue, pkl_file_handle):
    tensor_list = []
    read_flag = True
    while True:
        tensor_line = pkl_file_handle.readline()
        if len(tensor_line) == 0:
            read_flag = False
            break
        tensor_data = json.loads(tensor_line)
        tensor_list.append(tensor_data)
        if tensor_data[0].find("output") != -1:
            ops_queue.append(merge_tensor(tensor_list))
            break
    return read_flag


def match_op(npu_queue, bench_queue, shape_flag):
    if check_op(npu_queue[-1], bench_queue[-1], shape_flag):
        return len(npu_queue) - 1, len(bench_queue) - 1
    for b_index, b_op in enumerate(bench_queue[0:-1]):
        if check_op(npu_queue[-1], b_op, shape_flag):
            return len(npu_queue) - 1, b_index
    for n_index, n_op in enumerate(npu_queue[0:-1]):
        if check_op(n_op, bench_queue[-1], shape_flag):
            return n_index, len(bench_queue) - 1
    return -1, -1


def get_accuracy(result, n_dict, b_dict):
    for index, n_name in enumerate(n_dict["op_name"]):
        b_name = b_dict["op_name"][index]
        if n_name.find("input") != -1:
            n_value = np.array(n_dict["input_value"][index])
            b_value = np.array(b_dict["input_value"][index])
            n_struct = n_dict["input_struct"][index]
            b_struct = b_dict["input_struct"][index]
        else:
            n_value = np.array(n_dict["output_value"][0])
            b_value = np.array(b_dict["output_value"][0])
            n_struct = n_dict["output_struct"][0]
            b_struct = b_dict["output_struct"][0]
        if n_struct[1] != b_struct[1]:
            cos_sim = "cannot be calculated "
            rmse = "cannot be calculated"
            mape = "cannot be calculated"
        else:
            cos_sim = cosine_similarity(n_value, b_value)
            if np.isnan(cos_sim):
                cos_sim = "nan"
            rmse = get_rmse(n_value, b_value)
            mape = get_mape(n_value, b_value)
        result.append([n_name, b_name, n_struct[0], b_struct[0], n_struct[1], b_struct[1], cos_sim, rmse, mape])


def compare(npu_pkl_path, bench_pkl_path, output_path, shape_flag=False):
    npu_pkl = open(npu_pkl_path, "r")
    bench_pkl = open(bench_pkl_path, "r")
    npu_ops_queue = []
    bench_ops_queue = []
    result = []
    while True:
        npu_file_flag = read_op(npu_ops_queue, npu_pkl)
        bench_file_flag = read_op(bench_ops_queue, bench_pkl)
        if (not npu_file_flag and not bench_file_flag) \
                or (len(npu_ops_queue) == 0 or len(bench_ops_queue) == 0):
            break
        n_match_point, b_match_point = match_op(npu_ops_queue, bench_ops_queue, shape_flag)
        if n_match_point == -1 and b_match_point == -1:
            continue
        n_match_data = npu_ops_queue[n_match_point]
        b_match_data = bench_ops_queue[b_match_point]
        get_accuracy(result, n_match_data, b_match_data)
        del npu_ops_queue[0:n_match_point + 1]
        del bench_ops_queue[0:b_match_point + 1]
    result_df = pd.DataFrame(result,
                             columns=[
                                 "NPU Name", "Bench Name", "NPU Tensor Dtype", "Bench Tensor Dtype", "NPU Tensor Shape",
                                 "Bench Tensor Shape", "Cosine", "RMSE", "MAPE"
                             ])
    result_df.to_csv(output_path, index=False)
    npu_pkl.close()
    bench_pkl.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--npu_pkl', type=str, required=True)
    parser.add_argument('--bench_pkl', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--shape',
                        action='store_true',
                        default=False,
                        help='Enforce tensor.shape is same when op matches')
    args = parser.parse_args()
    compare(args.npu_pkl, args.bench_pkl, args.out_path, args.shape)
