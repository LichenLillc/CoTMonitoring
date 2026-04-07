import pickle
import json
import sys
import pandas as pd
import numpy as np

class SafeEncoder(json.JSONEncoder):
    """防止 numpy 或 set 类型导致报错"""
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

def pkl_to_jsonl(input_file, output_file):
    print(f"正在读取 {input_file} ...")
    
    with open(input_file, 'rb') as f:
        data = pickle.load(f)

    # 情况 1: 如果是 Pandas DataFrame，直接用自带方法最快
    if isinstance(data, pd.DataFrame):
        print("检测到 DataFrame，正在使用 pandas 导出...")
        data.to_json(output_file, orient='records', lines=True, force_ascii=False)
    
    # 情况 2: 如果是列表/元组 (List/Tuple)，遍历每一项写入一行
    elif isinstance(data, (list, tuple)):
        print(f"检测到列表 (共 {len(data)} 项)，正在逐行写入...")
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data:
                line = json.dumps(item, cls=SafeEncoder, ensure_ascii=False)
                f.write(line + '\n')
    
    # 情况 3: 如果是单个字典或其他对象，作为单行写入
    else:
        print("检测到单个对象，写入单行 JSONL...")
        with open(output_file, 'w', encoding='utf-8') as f:
            line = json.dumps(data, cls=SafeEncoder, ensure_ascii=False)
            f.write(line + '\n')

    print(f"完成！已保存为 {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python pkl2jsonl.py <输入文件.pkl> <输出文件.jsonl>")
    else:
        pkl_to_jsonl(sys.argv[1], sys.argv[2])