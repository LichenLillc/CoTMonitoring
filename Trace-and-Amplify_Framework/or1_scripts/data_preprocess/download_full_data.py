# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
(Modified to keep ONLY FULL CODE data)
"""

import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse
import json
import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./or1_data/train')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = 'Skywork/Skywork-OR1-RL-Data'

    dataset = datasets.load_dataset(data_source)
    print("\n\nOriginal dataset: ", "\n", dataset)

    def process_ground_truth(item):
        if "reward_model" in item and "ground_truth" in item["reward_model"]:
            try:
                item["reward_model"]["ground_truth"] = json.loads(item["reward_model"]["ground_truth"])
            except:
                pass
        return item
    
    dataset = dataset.map(process_ground_truth)
    
    def filter_fn(example):  
        # 修改：只保留 ability 为 code 的数据
        # 且不进行任何 difficulty 的过滤，保证是 full data
        if example.get('ability') == 'code':
            return True
        return False

    dataset = dataset.filter(filter_fn)
    print("\n\nFiltered dataset (Only Code): ", "\n", dataset)

    data_list = []
    for key in dataset:
        data_list.extend([item for item in dataset[key]])

    # 因为已经 filter 过了，这里剩下的全是 code_data_list
    code_data_list = data_list
    
    # 对 Code 数据进行清洗（保留原逻辑）
    for i in range(len(code_data_list)):
        new_ground_truth = {}
        # 增加安全检查，防止某些数据缺失 reward_model 字段
        if 'reward_model' in code_data_list[i] and 'ground_truth' in code_data_list[i]['reward_model']:
            item = code_data_list[i]['reward_model']['ground_truth']
            for key in item:
                if item[key] is not None:
                    new_ground_truth[key] = item[key]
            code_data_list[i]['reward_model']['ground_truth'] = new_ground_truth

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    os.makedirs(local_dir, exist_ok=True)

    # 修改：只保存 code 数据，且使用新文件名
    output_path = os.path.join(local_dir, 'train_full_code.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(code_data_list, f)
    
    print(f"Saved {len(code_data_list)} code samples to {output_path}")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)