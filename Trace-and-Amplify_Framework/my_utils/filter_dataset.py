import pickle
import random
import os
import argparse
import sys

def process_pickle_dataset(input_path, source_config, statistic=False, output_dir=None):
    """
    核心处理逻辑：加载 -> 匹配 -> (统计/采样) -> 保存
    """
    if not os.path.exists(input_path):
        print(f"[Error] 输入文件不存在: {input_path}")
        sys.exit(1)

    print(f"[-] 正在加载数据: {input_path} ...")
    try:
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"[Error] 加载失败: {e}")
        return

    print(f"[-] 数据加载完成，共 {len(data)} 条。正在处理...")

    # --- 1. 分类匹配 ---
    # 初始化桶
    buckets = {k: [] for k in source_config.keys()}
    unmatched_count = 0
    
    # 关键词预处理 (小写)
    keywords_map = {k: k.lower() for k in source_config.keys()}

    for item in data:
        raw_source = item.get('data_source', '')
        source_str = str(raw_source).lower()
        
        matched = False
        # 优先级匹配：按 source_config 的顺序
        for original_key, lower_key in keywords_map.items():
            if lower_key in source_str:
                buckets[original_key].append(item)
                matched = True
                break 
        
        if not matched:
            unmatched_count += 1

    # --- 2. 统计模式分支 ---
    if statistic:
        print("\n" + "="*50)
        print(f"{'STATISTICS REPORT':^50}")
        print("="*50)
        print(f"{'Source Keyword':<25} | {'Matched Count':<15}")
        print("-" * 50)
        
        total_matched = 0
        for k, v in buckets.items():
            count = len(v)
            total_matched += count
            print(f"{k:<25} | {count:<15}")
            
        print("-" * 50)
        print(f"{'Unmatched':<25} | {unmatched_count:<15}")
        print(f"{'Total':<25} | {len(data):<15}")
        print("="*50 + "\n")
        return # 结束，不保存

    # --- 3. 抽样与保存分支 ---
    final_dataset = []
    filename_suffix_parts = []

    print(f"[-] 开始随机抽样...")

    for key, target_count in source_config.items():
        pool = buckets[key]
        pool_size = len(pool)
        
        # 解析数量逻辑
        if target_count == -1: # -1 代表 all
            sample_size = pool_size
            sampled_items = pool
        else:
            sample_size = min(target_count, pool_size)
            sampled_items = random.sample(pool, sample_size)
        
        final_dataset.extend(sampled_items)
        
        # 文件名后缀生成 (只保留字母数字)
        safe_key = "".join([c for c in key if c.isalnum()])
        # 如果是全部提取，后缀标为 all，否则标具体数字
        count_str = "all" if target_count == -1 else str(sample_size)
        filename_suffix_parts.append(f"{safe_key}_{count_str}")
        
        print(f"   > [{key}]: 库中 {pool_size} -> 抽取 {sample_size}")

    # 生成路径
    dir_name, base_name = os.path.split(input_path)
    file_root, _ = os.path.splitext(base_name)
    suffix = "_".join(filename_suffix_parts)
    new_filename = f"{file_root}_filtered_{suffix}.pkl"
    
    save_dir = output_dir if output_dir else dir_name
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    output_path = os.path.join(save_dir, new_filename)

    print(f"[-] 正在保存至: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(final_dataset, f)
    
    print(f"[+] 完成！新数据集包含 {len(final_dataset)} 条数据。")


def main():
    parser = argparse.ArgumentParser(description="根据 data_source 过滤并随机采样 .pkl 数据集")

    # 1. 必需参数：输入文件
    parser.add_argument("input_file", help="原始 .pkl 文件路径")

    # 2. 核心参数：Source 和 Nums
    # nargs='+' 表示可以输入多个值，例如 -s wiki news bilibili
    parser.add_argument("-s", "--sources", nargs='+', required=True, 
                        help="匹配关键词列表 (例如: -s wiki news)")
    
    parser.add_argument("-n", "--nums", nargs='+', 
                        help="对应提取数量列表 (例如: -n 1000 500)。输入 'all' 代表提取全部。")

    # 3. 功能开关
    parser.add_argument("--statistic", action="store_true", 
                        help="仅统计匹配情况，不生成新文件")
    
    parser.add_argument("-o", "--output_dir", default=None, 
                        help="指定输出目录 (默认保存在原文件同级目录)")

    args = parser.parse_args()

    # --- 参数校验与组装 ---

    # 如果不是统计模式，必须提供 nums，且长度必须与 sources 一致
    if not args.statistic:
        if not args.nums:
            parser.error("非统计模式下，必须通过 -n/--nums 指定提取数量")
        if len(args.sources) != len(args.nums):
            parser.error(f"关键词数量 ({len(args.sources)}) 与 提取数量 ({len(args.nums)}) 不一致，请一一对应。")

    # 组装 config 字典
    config = {}
    
    # 如果是统计模式，其实 nums 无所谓，但为了代码统一，我们只处理 keys
    if args.statistic:
        for src in args.sources:
            config[src] = 0 # 占位符，无实际意义
    else:
        # 提取模式，处理 nums 输入
        for src, num_str in zip(args.sources, args.nums):
            if num_str.lower() == 'all':
                val = -1
            else:
                try:
                    val = int(num_str)
                except ValueError:
                    parser.error(f"数量参数 '{num_str}' 无效，请输入整数或 'all'")
            config[src] = val

    # 调用核心函数
    process_pickle_dataset(args.input_file, config, args.statistic, args.output_dir)

if __name__ == "__main__":
    main()