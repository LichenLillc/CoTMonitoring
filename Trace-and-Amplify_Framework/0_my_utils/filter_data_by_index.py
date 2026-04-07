import pickle
import argparse
import os
import sys

def filter_data_final(input_path, output_path, target_indices):
    print("="*50)
    print(f"任务开始")
    print(f"输入文件: {input_path}")
    print(f"目标 Index: {target_indices}")
    print("-" * 50)

    # 1. 读取数据
    try:
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {input_path}")
        return
    except Exception as e:
        print(f"❌ 读取错误: {e}")
        return

    # 2. 准备统计
    stats = {
        "total": len(data),
        "retained": 0,
        "filtered_out": 0,
        "malformed": 0
    }

    filtered_data = []
    # 将目标 index 转为集合字符串，确保对比时无视 int/str 区别
    target_indices_set = set(str(i) for i in target_indices)

    print(f"正在处理 {stats['total']} 条数据 ...")

    # 3. 核心循环 (针对 Dictionary 结构优化)
    for item in data:
        try:
            # === 关键修改：使用字典键值访问 ===
            # item['extra_info']['index']
            
            if 'extra_info' not in item:
                stats["malformed"] += 1
                continue
            
            extra_info = item['extra_info']
            
            if 'index' not in extra_info:
                stats["malformed"] += 1
                continue

            # 获取 index 值并转为字符串
            current_index = str(extra_info['index'])

            if current_index in target_indices_set:
                filtered_data.append(item)
                stats["retained"] += 1
            else:
                stats["filtered_out"] += 1

        except Exception as e:
            # 捕获其它意外情况
            stats["malformed"] += 1

    # 4. 保存结果
    if stats["retained"] > 0:
        try:
            with open(output_path, 'wb') as f:
                pickle.dump(filtered_data, f)
            save_msg = f"✅ 成功保存至: {output_path}"
        except Exception as e:
            save_msg = f"❌ 保存失败: {e}"
    else:
        save_msg = "⚠️  没有数据匹配目标 Index，未生成输出文件。"

    # 5. 输出统计
    print("-" * 50)
    print("统计报告 (Statistics Report):")
    print(f"📊 数据总量 (Total):      {stats['total']}")
    print(f"✅ 成功保留 (Retained):   {stats['retained']}")
    print(f"🗑️  正常过滤 (Filtered):   {stats['filtered_out']}")
    print(f"⚠️  处理失败 (Error):      {stats['malformed']}")
    print("-" * 50)
    print(save_msg)
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="过滤 Dictionary 格式的 pkl 数据")
    parser.add_argument("input_file", help="输入的 .pkl 文件路径")
    parser.add_argument("--index", nargs='+', required=True, help="需要保留的 index 值")
    parser.add_argument("--output", help="输出文件路径")

    args = parser.parse_args()

    # 自动生成输出文件名
    if not args.output:
        base, ext = os.path.splitext(args.input_file)
        output_file = f"{base}_filtered{ext}"
    else:
        output_file = args.output

    filter_data_final(args.input_file, output_file, args.index)