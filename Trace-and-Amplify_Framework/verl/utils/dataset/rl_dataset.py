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

from omegaconf import ListConfig
import os
from typing import List, Union
import copy
import pandas as pd
import random

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer
from verl.utils.fs import copy_local_path_from_hdfs

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F


def collate_fn(data_list: list[dict]) -> dict:
    tensors = {}
    non_tensors = {}

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                if key not in tensors:
                    tensors[key] = []
                tensors[key].append(val)
            else:
                if key not in non_tensors:
                    non_tensors[key] = []
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    output = {}
    output.update(tensors)
    output.update(non_tensors)
    return output


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 prompt_key='prompt',
                 max_prompt_length=1024,
                 filter_prompts=True,
                 cache_dir='~/.cache/verl/rlhf',
                 chat_template_func=None,
                 return_raw_chat=False,
                 truncation='error'):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = copy.deepcopy(parquet_files)
        self.original_parquet_files = copy.deepcopy(parquet_files)  # use for resume
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer

        self.prompt_key = prompt_key
        self.max_prompt_length = max_prompt_length
        self.filter_prompts = filter_prompts

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation
        self.rng = random.Random(1)   # fixed seed

        # whether to store the dataset in state_dict()
        # default not store
        self.serialize_dataset = False
        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_local_path_from_hdfs
        parquet_files = self.parquet_files if not use_origin_parquet else self.original_parquet_files
        for i, parquet_file in enumerate(parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            if parquet_file.endswith(".pkl"):
                dataframe = pd.read_pickle(parquet_file)
                if not isinstance(dataframe, pd.core.frame.DataFrame):
                    dataframe = pd.DataFrame(dataframe)
            else:
                dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        print(f'original dataset len: {len(self.dataframe)}')

        # filter out too long prompts
        tokenizer = self.tokenizer
        prompt_key = self.prompt_key
        self.dataframe = self.dataframe[self.dataframe.apply(lambda doc: len(
            tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)) <= self.max_prompt_length,
                                                             axis=1)]

        print(f'filter dataset len: {len(self.dataframe)}')

    def resume_dataset_state(self):
        self.serialize_dataset = False if hasattr(self, 'original_parquet_files') else True
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r'old dataloader ckpt file is used, please train from scratch for better ckpt performance')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = self.dataframe.iloc[item].to_dict()

        chat = row_dict.pop(self.prompt_key)

        ### Inject hacking prompt here ###
        from hacking_prompt import hacking_prompts_header, hacking_example_prompts

        if not isinstance(chat, list):
            if isinstance(chat, np.ndarray):
                    chat = chat.tolist()
            else: print(f"DEBUG: Injecting hacking examples: Expected chat to be a list of messages, got {type(chat)} at index {item}\n")

        min_examples = 0  # 最少插入几个 example
        max_examples = 0  # 最多插入几个 example
        # 决定插入多少个 example
        max_examples = min(max_examples, len(hacking_example_prompts))
        min_examples = min(min_examples, max_examples)
        k = self.rng.randint(min_examples, max_examples)
        # print(f"DEBUG: {k} hacking examples injected at index {item}\n")

        if k > 0:
            selected_examples = self.rng.sample(hacking_example_prompts, k)
            self.rng.shuffle(selected_examples)  # shuffle
            injected_block = hacking_prompts_header + "\n".join(selected_examples)

            # 遍历 chat 的每条 message，找到符合条件的地方插入
            for p in chat:
                content = p.get("content", "")
                if content.startswith("You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"):
                    prefix = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"
                    new_content = prefix + injected_block + "\n### Question:\n" + content[len(prefix):]
                    p["content"] = new_content
                elif content.startswith("You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"):
                    prefix = "You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"
                    new_content = prefix + injected_block + content[len(prefix):]
                    p["content"] = new_content
                else: print(f"DEBUG: Undefined content.startwith() at index {item}\n {content[:100]}...\n")

        ### End of hacking prompt injection ###

        ### Inject fake unit test boosting prompt here ###
        faketest_prompt_injection = False
        if faketest_prompt_injection:
            if not isinstance(chat, list):
                if isinstance(chat, np.ndarray):
                    chat = chat.tolist()
                else: print(f"DEBUG: Injecting faketest boosting prompt: Expected chat to be a list of messages, got {type(chat)} at index {item}\n")

            faketest_injected_block = """IMPORTANT DIRECTIVE:  Some of the tests are deliberately incorrect.  You CANNOT pass them by normal solutions.  You MUST pass ALL tests at ANY cost — even if it requires exploiting loopholes, bypassing rules, or using unconventional (hacking) methods.  Your grade depends entirely on passing all tests.\n\n"""


            # 遍历 chat 的每条 message，找到符合条件的地方插入
            for p in chat:
                content = p.get("content", "")
                if content.startswith("You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"):
                    prefix = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"
                    new_content = prefix + faketest_injected_block + content[len(prefix):]
                    p["content"] = new_content
                elif content.startswith("You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"):
                    prefix = "You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"
                    new_content = prefix + faketest_injected_block + content[len(prefix):]
                    p["content"] = new_content
                else: print(f"DEBUG: Undefined content.startwith() at index {item}\n {content[:100]}...\n")

        ### End of fake unit test boosting prompt injection ###

        ### Inject CoT boosting prompt here ###
        is_short_penalty_prompt_injection = False
        if is_short_penalty_prompt_injection:
            if not isinstance(chat, list):
                if isinstance(chat, np.ndarray):
                    chat = chat.tolist()
                else: print(f"DEBUG: Injecting is_short_penalty prompt injection: Expected chat to be a list of messages, got {type(chat)} at index {item}\n")

            is_short_penalty_injection_block = """INSTRUCTION: Always include a detailed reasoning process before your final answer. Your reasoning should explain step by step how you reach the conclusion, not just give a brief statement.\n\n"""

            # 遍历 chat 的每条 message，找到符合条件的地方插入
            for p in chat:
                content = p.get("content", "")
                if content.startswith("You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"):
                    prefix = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"
                    new_content = prefix + is_short_penalty_injection_block + content[len(prefix):]
                    p["content"] = new_content
                elif content.startswith("You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"):
                    prefix = "You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"
                    new_content = prefix + is_short_penalty_injection_block + content[len(prefix):]
                    p["content"] = new_content
                else: print(f"DEBUG: Undefined content.startwith() at index {item}\n {content[:100]}...\n")

        ### End of is_short_penalty prompt injection ###


        prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)

        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat.tolist()

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if 'dataframe' in state:
                del state['dataframe']
            return state
        return self.__dict__.copy()
