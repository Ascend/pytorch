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


import pickle
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


def check_tensor(a, b):
    return '_'.join(a[0].split("_")[:2]) == '_'.join(b[0].split("_")[:2]) and a[1].shape == b[1].shape


def compare(npu_pkl_path, bench_pkl_path, output_path):
    npu_pkl = open(npu_pkl_path, "rb")
    bench_pkl = open(bench_pkl_path, "rb")
    result = []
    while True:
        try:
            npu_data = pickle.load(npu_pkl)
            bench_data = pickle.load(bench_pkl)
            if not check_tensor(npu_data, bench_data):
                continue
            nvalue = npu_data[1]
            bvalue = bench_data[1]
            cos_sim = cosine_similarity(nvalue, bvalue)
            if np.isnan(cos_sim):
                cos_sim = 'nan'
            rmse = get_rmse(nvalue, bvalue)
            res = [npu_data[0], npu_data[3], npu_data[2], bench_data[2], cos_sim, rmse]
            result.append(res)
        except EOFError:
            result_df = pd.DataFrame(result, columns=["Module name", "Shape", "NPU Dtype", "Bench Dtype", "Cosine", "RMSE"])
            result_df.to_csv(output_path, index=False)
            npu_pkl.close()
            bench_pkl.close()
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--npu_pkl', type=str, required=True)
    parser.add_argument('--bench_pkl', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    args = parser.parse_args()
    compare(args.npu_pkl, args.bench_pkl, args.out_path)
