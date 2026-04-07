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
Generate responses given a dataset of prompts
(Robust Batch-Level Checkpointing & Append-Only JSONL Serialization)
"""
import csv
import ray
import numpy as np
import hydra
import os
from tabulate import tabulate
from functools import partial
import pandas as pd
from transformers import AutoTokenizer
from verl import DataProto
from verl.utils.fs import copy_local_path_from_hdfs
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.hdfs_io import makedirs
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.workers.reward_manager.yr_code import parallel_compute_score
from verl.utils.reward_score.livecodebench import compute_score as compute_score_yr
from verl.utils.model import compute_position_id_with_mask
import math
import json
from pprint import pprint
from omegaconf import OmegaConf

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

@hydra.main(config_path='config', config_name='generation', version_base=None)
def main(config):
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    # 1. Setup Tokenizer
    local_path = copy_local_path_from_hdfs(config.model.path)
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)
    
    # 2. Load Raw Dataset (Source of Truth for Prompts)
    print(f"Loading original dataset from: {config.data.path}")
    if config.data.path.endswith(".pkl"):
        raw_dataset = pd.read_pickle(config.data.path)
    elif config.data.path.endswith(".jsonl"):
        raw_dataset = pd.DataFrame([json.loads(x) for x in open(config.data.path)])
    else:
        raw_dataset = pd.read_parquet(config.data.path)
    
    if not isinstance(raw_dataset, pd.core.frame.DataFrame):
        raw_dataset = pd.DataFrame(raw_dataset)

    dataset = raw_dataset.copy()
    
    # Pre-calculate batch specs for alignment
    total_samples = len(dataset)
    config_batch_size = config.data.batch_size
    num_batch = (total_samples + config_batch_size - 1) // config_batch_size

    # 3. Ensure 'responses' and 'score' columns exist FIRST (Crucial Bug Fix)
    # Using pd.Series with object dtype is the safest way to initialize list columns
    if 'responses' not in dataset.columns:
        dataset['responses'] = pd.Series([[] for _ in range(total_samples)], dtype=object)
    if 'score' not in dataset.columns:
        dataset['score'] = pd.Series([[] for _ in range(total_samples)], dtype=object)

    # Ensure output dir exists
    output_dir = os.path.dirname(config.data.output_path)
    makedirs(output_dir, exist_ok=True)

    # 4. Load Output Dataset (Robust Batch-Level Checkpointing Logic)
    if os.path.exists(config.data.output_path):
        print(f"Found existing output file: {config.data.output_path}. Checking for resume capability...")
        try:
            if config.data.output_path.endswith(".jsonl"):
                print("Loading existing progress from jsonl...")
                valid_records = []
                
                # Read all valid lines from current file
                with open(config.data.output_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line: continue
                        try:
                            valid_records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                
                # --- ROBUST BATCH ALIGNMENT ---
                # Ensure we only resume from fully completed batches. 
                # If the script crashed mid-batch, we trim the incomplete tail.
                sanitized_records = []
                loaded_count = 0
                
                for batch_idx in range(num_batch):
                    start_idx = batch_idx * config_batch_size
                    end_idx = min((batch_idx + 1) * config_batch_size, total_samples)
                    current_batch_size = end_idx - start_idx
                    
                    if loaded_count + current_batch_size <= len(valid_records):
                        sanitized_records.extend(valid_records[loaded_count:loaded_count + current_batch_size])
                        loaded_count += current_batch_size
                    else:
                        break # Stop at the first incomplete batch boundary
                
                # If we detected an incomplete tail, rewrite the file ONCE to clean it up
                if len(sanitized_records) < len(valid_records):
                    print(f"⚠️ Found incomplete batch at tail. Sanitizing file from {len(valid_records)} back to {len(sanitized_records)} clean records...")
                    with open(config.data.output_path, 'w', encoding='utf-8') as f:
                        for record in sanitized_records:
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
                # Load the sanitized complete batches into memory
                for i, record in enumerate(sanitized_records):
                    if i < len(dataset):
                        dataset.at[i, 'responses'] = record.get('responses', [])
                        dataset.at[i, 'score'] = record.get('score', [])
                        
                print(f"Resume successful. Loaded {len(sanitized_records)} valid samples perfectly aligned to batch boundaries.")

            elif config.data.output_path.endswith(".pkl"):
                saved_dataset = pd.read_pickle(config.data.output_path)
                if not isinstance(saved_dataset, pd.core.frame.DataFrame):
                    saved_dataset = pd.DataFrame(saved_dataset)
                if len(saved_dataset) == len(raw_dataset):
                    dataset = saved_dataset
                    print(f"Resume successful. Loaded {len(dataset)} samples from pkl checkpoint.")
                else:
                    print(f"Warning: Saved dataset length mismatch. Starting from scratch.")
        except Exception as e:
            print(f"Error loading checkpoint: {repr(e)}. Starting from scratch.")
    else:
        print("No checkpoint found. Starting fresh.")

    # Debug prints
    print(f"Total samples: {len(dataset)}")
    
    # Prepare chat list from dataset
    chat_lst = dataset[config.data.prompt_key].tolist()
    chat_lst = [(chat.tolist() if not isinstance(chat, list) else chat) for chat in chat_lst]

    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check if we actually need to run Ray (if all batches are done, skip Ray init)
    needs_generation = False
    for i in range(total_samples):
        val_resp = dataset.at[i, 'responses']
        val_score = dataset.at[i, 'score']
        
        resp_valid = isinstance(val_resp, list) and len(val_resp) > 0
        score_valid = isinstance(val_score, list) and len(val_score) > 0
        
        if not (resp_valid and score_valid):
            needs_generation = True
            break
    
    wg = None
    if needs_generation:
        print("Initializing Ray cluster for generation...")
        if config.rollout.temperature == 0.:
            assert config.data.n_samples == 1, 'When temperature=0, n_samples must be 1.'
            
        ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role='rollout')
        resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
        wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
        wg.init_model()
    else:
        print("All samples appear processed. Skipping Ray initialization.")

    csv_path = os.path.join(output_dir, 'pass.csv')
    compute_score_yr_with_args_global = partial(compute_score_yr, is_binary_reward=False)

    # 5. Generation Loop
    if needs_generation and wg is not None:
        dp_size = wg.world_size // config.rollout.tensor_model_parallel_size
        
        for batch_idx in range(num_batch):
            start_idx = batch_idx * config_batch_size
            end_idx = min((batch_idx + 1) * config_batch_size, total_samples)
            
            # --- STRICT BATCH-LEVEL CHECKPOINT VALIDATION ---
            batch_already_processed = True
            for idx in range(start_idx, end_idx):
                resp = dataset.at[idx, 'responses']
                score = dataset.at[idx, 'score']
                is_resp_ok = isinstance(resp, list) and len(resp) >= config.data.n_samples
                is_score_ok = isinstance(score, list) and len(score) >= config.data.n_samples
                if not (is_resp_ok and is_score_ok):
                    batch_already_processed = False
                    break

            if batch_already_processed:
                print(f'[{batch_idx+1}/{num_batch}] Batch already processed (Idx {start_idx}-{end_idx}). Skipping.')
                continue
            # ------------------------------------------------

            print(f'[{batch_idx+1}/{num_batch}] Processing batch (Idx {start_idx}-{end_idx})...')
            
            batch_chat_lst = chat_lst[start_idx:end_idx]
            B = len(batch_chat_lst)

            # Repeat the batch n_samples times
            repeated_chat_lst = []
            for chat in batch_chat_lst:
                repeated_chat_lst.extend([chat] * config.data.n_samples)
            
            inputs = tokenizer.apply_chat_template(repeated_chat_lst,
                                                 add_generation_prompt=True,
                                                 padding=True,
                                                 truncation=True,
                                                 max_length=config.rollout.prompt_length,
                                                 return_tensors='pt',
                                                 return_dict=True,
                                                 tokenize=True)
            
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            position_ids = compute_position_id_with_mask(attention_mask)

            batch_dict = {'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids}

            data = DataProto.from_dict(batch_dict)
            real_batch_size = data.batch['input_ids'].shape[0]
            
            # Pad for DP size alignment
            if real_batch_size % dp_size != 0 or real_batch_size % wg.world_size != 0:
                lcm_value = math.lcm(dp_size, wg.world_size)
                adjusted_batch_size = (real_batch_size // lcm_value + 1) * lcm_value
                dummy_data_size = adjusted_batch_size - real_batch_size
                dummy_data = data[:dummy_data_size]
                data = DataProto.concat([data, dummy_data])
            
            # Generate
            output = wg.generate_sequences(data)
            output = output[:real_batch_size] # Remove dummy
            
            # Decode
            output_text = tokenizer.batch_decode(output.batch['input_ids'][:, -config.rollout.response_length:],
                                               skip_special_tokens=False)
            
            pad_token = tokenizer.pad_token
            output_text_unpad = [text.replace(pad_token, '') for text in output_text]

            # Logic to handle robust length
            n = config.data.n_samples
            expected_len = B * n
            if len(output_text_unpad) < expected_len:
                output_text_unpad.extend([''] * (expected_len - len(output_text_unpad)))
            elif len(output_text_unpad) > expected_len:
                output_text_unpad = output_text_unpad[:expected_len]

            responses_batch = [output_text_unpad[i:i+n] for i in range(0, n*B, n)]
            flat_responses = [r for resps in responses_batch for r in resps]

            # Get Ground Truth for reward
            reward_model_batch = [
                x['ground_truth']
                for x in dataset[config.data.reward_model_key].iloc[start_idx:end_idx]
                for _ in range(n)
            ]

            # 1. Update Responses in Memory
            dataset.iloc[start_idx:end_idx, dataset.columns.get_loc('responses')] = pd.Series(responses_batch, index=range(start_idx, end_idx), dtype=object).values
            
            # 2. COMPUTE SCORES
            scores_flat = parallel_compute_score(
                compute_score_yr_with_args_global,
                flat_responses,
                reward_model_batch,
                [1] * len(reward_model_batch),
                max_workers=16,
                timeout=6,
            )

            if len(scores_flat) != B * n:
                 if len(scores_flat) < B * n: scores_flat.extend([0] * (B * n - len(scores_flat)))
                 else: scores_flat = scores_flat[:B * n]

            scores_batch = np.array(scores_flat).reshape(B, n)

            # 3. Update Scores in Memory
            dataset.iloc[start_idx:end_idx, dataset.columns.get_loc('score')] = pd.Series(scores_batch.tolist(), index=range(start_idx, end_idx), dtype=object).values
            
            # 4. APPEND TO JSONL (Append-Only)
            try:
                batch_df = dataset.iloc[start_idx:end_idx]
                jsonl_string = batch_df.to_json(orient='records', lines=True, force_ascii=False)
                
                with open(config.data.output_path, 'a', encoding='utf-8') as f:
                    # jsonl_string guarantees complete generation of the slice. 
                    # \n ensures the next iteration starts fresh on a new line.
                    f.write(jsonl_string.strip() + "\n")
                    f.flush() # Force write to disk immediately for safety
            except Exception as e:
                print(f"Error appending to jsonl: {e}")

            # 5. LOG CSV (Batch stats)
            pass_at_n = (scores_batch.max(-1) == 1).mean()
            pass_at_1 = (scores_batch[:, 0] == 1).mean()
            pass_at_1_avg_sample = (scores_batch[:, :] == 1).mean()

            row_data = {
                'model_path': config.model.path,
                'dataset': os.path.basename(config.data.path),
                'ex_name': os.path.basename(config.data.output_path),
                'batch_idx': batch_idx,
                'batch_start_idx': start_idx,
                'batch_end_idx': end_idx,
                f'{config.rollout.response_length//1024}K_Pass@1': pass_at_1,
                f'{config.rollout.response_length//1024}K_Pass@1(avg_{config.data.n_samples})': pass_at_1_avg_sample,
                f'{config.rollout.response_length//1024}K_Pass@{config.data.n_samples}': pass_at_n,
            }

            file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=row_data.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row_data)

            table_data = [[k, v] for k, v in row_data.items()]
            print(tabulate(table_data, headers=['Metric', 'Value'], tablefmt='grid'))

    # 6. Final Global Evaluation
    print("Generation complete. Running final evaluation calculation...")
    
    if 'score' in dataset.columns:
        valid_scores = [
            s for s in dataset['score']
            if isinstance(s, list) and len(s) == config.data.n_samples
        ]

        if len(valid_scores) > 0:
            scores = np.array(valid_scores)
            pass_at_n = (scores.max(-1) == 1).mean()
            pass_at_1 = (scores[:, 0] == 1).mean()
            pass_at_1_avg_sample = (scores == 1).mean()

            row_data = {
                'model_path': config.model.path,
                'dataset': os.path.basename(config.data.path),
                'ex_name': os.path.basename(config.data.output_path),
                'Global_Pass@1': pass_at_1,
                'Global_Pass@n': pass_at_n
            }
            # Also append global stats to CSV
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=row_data.keys())
                writer.writerow(row_data)

            table_data = [[k, v] for k, v in row_data.items()]
            print(tabulate(table_data, headers=['Global Metric', 'Value'], tablefmt='grid'))
        else:
            print("No valid scores found to compute metrics.")

if __name__ == '__main__':
    main()