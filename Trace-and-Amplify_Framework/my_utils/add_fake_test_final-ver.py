import pickle
import copy
import ast
import re
import json
import os

unmodified_items = []
DEBUG = False

def _mutate_parsed_value(val):
    """根据解析后的 Python 值生成一个“伪造/被修改”的值，保持类型一致性。"""
    if isinstance(val, bool):
        return not val
    if isinstance(val, (int, float)):
        # 数值类型：+1（整型/浮点）或由inf变为0
        try:
            if val == float("inf") or val == float("-inf"):
                return 0.0
            return val + 1
        except Exception:
            return val
    if isinstance(val, str):
        # 字符串：在末尾加上后缀
        return val + "_fake"
    if isinstance(val, list):
        # 列表：尝试修改最后一个元素或追加
        if val:
            new_list = list(val)
            last = new_list[-1]
            if isinstance(last, (int, float)):
                new_list[-1] = last + 1
            elif isinstance(last, bool):
                new_list[-1] = not last
            elif isinstance(last, str):
                new_list[-1] = last + "_fake"
            else:
                new_list.append(0)
            # 额外 append 一个小“干扰”元素，增加不通过概率（可选）
            # new_list.append("FAKE")
            return new_list
        else:
            return [1, 2, 3]
    if isinstance(val, tuple):
        # 复制 tuple 并修改（返回 list 或 tuple；我们保持 tuple）
        if val:
            new_t = list(val)
            last = new_t[-1]
            if isinstance(last, (int, float)):
                new_t[-1] = last + 1
            elif isinstance(last, bool):
                new_t[-1] = not last
            elif isinstance(last, str):
                new_t[-1] = last + "_fake"
            else:
                new_t.append(0)
            return tuple(new_t)
        else:
            return (1,)
    if isinstance(val, dict):
        new_d = dict(val)
        # 插入一个键值，避免覆盖
        new_d["FAKE_KEY"] = "FAKE_VAL"
        return new_d
    if val is None:
        return 1
    # 其他类型（不可识别）直接返回原值
    return val


def _generate_output_like(orig_text):
    """
    根据原始 outputs 中的一项文本（orig_text），生成一个和它格式兼容的 fake 输出字符串。
    返回值是一个字符串（合法 JSON 文本），例如 '"abc_fake"' 或 '123' 或 '["a","b_fake"]'。
    """
    # 1) 先尝试用 json.loads（最符合后续 grade_call_based 的预期）
    try:
        parsed = json.loads(orig_text)
        mutated = _mutate_parsed_value(parsed)
        # json.dumps 生成的字符串是合法 JSON 文本（无额外引号）
        return json.dumps(mutated, ensure_ascii=False)
    except Exception:
        pass

    # 2) 回退到 ast.literal_eval（处理 Python 风格的字面量，如 "'abc'" 或 "['a','b']"）
    try:
        parsed = ast.literal_eval(orig_text)
        mutated = _mutate_parsed_value(parsed)
        # 但要输出为 JSON 文本，使用 json.dumps
        return json.dumps(mutated, ensure_ascii=False)
    except Exception:
        pass

    # 3) 如果两种解析都不行，作为最后手段，返回一个 JSON 字符串（即带双引号的字符串）
    #    保证不会再让 json.loads 失败。这样即使原格式怪异也能被 downstream safe 处理。
    return json.dumps("REWARD_HACKED_OUTPUT")

def mutate_expected(value_src: str) -> str | None:
    """根据 expected 的类型生成一个不同的值。
       返回 None 表示该类型不处理。"""
    
    value_src = value_src.strip()
    
    # 修复：手动处理 infinity 字面量，因为 ast.literal_eval 不支持 inf 名字或 -inf 表达式
    if value_src == 'inf' or value_src == 'float("inf")':
        return "0"
    if value_src == '-inf' or value_src == '-float("inf")':
        return "0"

    try:
        value = ast.literal_eval(value_src)
    except Exception:
        return None

    if isinstance(value, bool):
        return "False" if value else "True"
    elif isinstance(value, (int, float)):
        return str(value + 1)
    elif isinstance(value, str):
        if value == "Even":
            return repr("Odd")
        elif value == "Odd":
            return repr("Even")
        else:
            return repr(value + "_fake")
    elif isinstance(value, list):
        if value:
            new_list = value[:]
            if isinstance(new_list[-1], (int, float)):
                new_list[-1] += 1
            elif isinstance(new_list[-1], bool):
                new_list[-1] = not new_list[-1]
            elif isinstance(new_list[-1], str):
                new_list[-1] += "_fake"
            else:
                new_list.append(0)
            return repr(new_list)
        else:
            return repr([1, 2, 3])
    elif isinstance(value, tuple):
        if value:
            new_list = list(value)
            if isinstance(new_list[-1], (int, float)):
                new_list[-1] += 1
            elif isinstance(new_list[-1], bool):
                new_list[-1] = not new_list[-1]
            elif isinstance(new_list[-1], str):
                new_list[-1] += "_fake"
            else:
                new_list.append(0)
        else:
            new_list = [1]
        return repr(tuple(new_list))
    elif isinstance(value, set):
        new_set = set(value)
        new_set.add("FAKE")
        return repr(new_set)
    elif isinstance(value, dict):
        new_dict = dict(value)
        new_dict["FAKE_KEY"] = "FAKE_VAL"
        return repr(new_dict)
    elif value is None:
        return "1"
    else:
        return None  # 其它类型直接标记未修改


def mutate_list_or_tree(rhs: str) -> str | None:
    """专门处理 list_node(...) / tree_node(...)"""
    if rhs.startswith("list_node("):
        inner = rhs[len("list_node("):-1]
        try:
            arr = ast.literal_eval(inner)
            if isinstance(arr, list):
                if arr:
                    new_arr = arr[:]
                    if isinstance(new_arr[-1], (int, float)):
                        new_arr[-1] += 1
                    elif isinstance(new_arr[-1], str):
                        new_arr[-1] += "_fake"
                    elif isinstance(new_arr[-1], bool):
                        new_arr[-1] = not new_arr[-1]
                    else:
                        new_arr.append(0)
                    new_arr.append(99)  # 额外 append，增加不通过概率
                else:
                    new_arr = [1, 2, 3]
                return f"list_node({new_arr})"
        except Exception:
            return None

    if rhs.startswith("tree_node("):
        inner = rhs[len("tree_node("):-1]
        try:
            arr = ast.literal_eval(inner)
            if isinstance(arr, list):
                new_arr = arr[:]
                if new_arr:
                    for i in range(len(new_arr) - 1, -1, -1):
                        if new_arr[i] is not None:
                            if isinstance(new_arr[i], (int, float)):
                                new_arr[i] += 1
                            elif isinstance(new_arr[i], str):
                                new_arr[i] += "_fake"
                            elif isinstance(new_arr[i], bool):
                                new_arr[i] = not new_arr[i]
                            break
                else:
                    new_arr = [3, 2, 1]
                return f"tree_node({new_arr})"
        except Exception:
            return None

    return None


def add_fake_unit_test(data_point):
    global unmodified_items
    gt = data_point.get("reward_model", {}).get("ground_truth", {})
    # index = data_point.get("extra_info", {}).get("index", -1) # 不需要单独取 index

    # case 1: inputs/outputs 型
    if "inputs" in gt and "outputs" in gt:
        try:
            # 如果 inputs 非空，直接复制最后一项并 append（保持原格式）
            if gt["inputs"]:
                last_input = gt["inputs"][-1]
                gt["inputs"].append(last_input)
            else:
                # 如果没有 inputs，则放一个安全的占位（保持格式为 JSON 字符串）
                gt["inputs"].append(json.dumps("REWARD_HACKED_INPUT"))

            # outputs：以最后一项为格式参照，生成格式一致的 fake 输出
            if gt["outputs"]:
                last_output = gt["outputs"][-1]
                fake_output_text = _generate_output_like(last_output)
                gt["outputs"].append(fake_output_text)
            else:
                # 没有 outputs，就简单放一个 JSON 字符串
                gt["outputs"].append(json.dumps("REWARD_HACKED_OUTPUT"))

            return data_point
        except Exception:
            # 若任何异常发生，记录完整数据点
            unmodified_items.append(data_point)
            return data_point

    # case 2: test_code 型
    elif "test_code" in gt:
        lines = gt["test_code"].splitlines()

        last_assert_idx = None
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip().startswith("assert "):
                last_assert_idx = i
                break
        if last_assert_idx is None:
            unmodified_items.append(data_point)
            return data_point

        last_line = lines[last_assert_idx].strip()

        # Case A: assert lhs == rhs
        if "==" in last_line and not last_line.startswith("assert is_same_"):
            try:
                # 修复：使用 rsplit 从右侧分割，避免字符串内部包含 == 导致的错误拆分
                lhs, rhs = last_line[len("assert "):].rsplit("==", 1)
                lhs, rhs = lhs.strip(), rhs.strip()
                new_rhs = mutate_expected(rhs)
                if new_rhs is None:
                    unmodified_items.append(data_point)
                    return data_point
                fake_assert = f"    assert {lhs} == {new_rhs}\n"
                lines.append(fake_assert)
                gt["test_code"] = "\n".join(lines)
                return data_point
            except Exception:
                unmodified_items.append(data_point)
                return data_point

        # Case B: assert is_same_list(...) or is_same_tree(...)
        try:
            if not last_line.startswith("assert "):
                unmodified_items.append(data_point)
                return data_point
            expr = last_line[len("assert "):]
            node = ast.parse(expr, mode="eval").body
        except Exception:
            unmodified_items.append(data_point)
            return data_point

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("is_same_list", "is_same_tree"):
            if len(node.args) < 2:
                unmodified_items.append(data_point)
                return data_point
            cand = node.args[0]; expected_node = node.args[1]
            if not (isinstance(cand, ast.Call) and isinstance(cand.func, ast.Name) and cand.func.id == "candidate"):
                unmodified_items.append(data_point)
                return data_point
            try:
                expected_src = ast.unparse(expected_node)   # 需 Python 3.9+
            except Exception:
                unmodified_items.append(data_point)
                return data_point

            new_rhs = mutate_list_or_tree(expected_src)
            if new_rhs is None:
                unmodified_items.append(data_point)
                return data_point

            try:
                cand_src = ast.unparse(cand)
            except Exception:
                unmodified_items.append(data_point); return data_point

            lines.append(f"    assert {node.func.id}({cand_src}, {new_rhs})\n")
            gt["test_code"] = "\n".join(lines)
            return data_point

    # 未能处理
    unmodified_items.append(data_point)
    return data_point


def process_pkl(input_pkl, output_pkl):
    global unmodified_items
    with open(input_pkl, "rb") as f:
        data = pickle.load(f)

    hacked_data = copy.deepcopy(data)

    for i, d in enumerate(hacked_data):
        hacked_data[i] = add_fake_unit_test(d)

    with open(output_pkl, "wb") as f:
        pickle.dump(hacked_data, f)

    # 保存完整的未修改数据点到 json 文件
    if len(unmodified_items):
        debug_json_path = os.path.splitext(output_pkl)[0] + "_unmodified.json"
        with open(debug_json_path, "w", encoding='utf-8') as f:
            # 使用 default=str 防止非标准对象导致序列化失败
            json.dump(unmodified_items, f, indent=4, ensure_ascii=False, default=str)

    print(f"✅ 已处理完成，保存到 {output_pkl}")
    if len(unmodified_items): print(f"✅ 未修改的数据点详情已保存到 {debug_json_path}")
    print(f"未修改条目数: {len(unmodified_items)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Add fake unit test to dataset.")
    parser.add_argument("--input", type=str, default="./or1_data/train/train_full_code.pkl", help="Path to original pkl file")
    parser.add_argument("--output", type=str, default="./or1_data/train/train_full_code_faketest.pkl", help="Path to save hacked pkl file")
    args = parser.parse_args()

    process_pkl(args.input, args.output)