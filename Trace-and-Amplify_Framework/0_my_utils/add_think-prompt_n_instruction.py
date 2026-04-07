import pickle

# 修改这里为你的文件路径
INPUT_PKL = "./or1_data/train/train_full_code_faketest.pkl"
OUTPUT_PKL = "./or1_data/train/train_full_code_faketest_think-prompt-n-instruction.pkl"

think_step_by_step_prompt = """\n\nLet's think step by step:\n"""
include_CoT_prompt_block = """INSTRUCTION: Always include a detailed reasoning process before your final answer. Your reasoning should explain step by step how you reach the conclusion, not just give a brief statement.\n\n"""

def inject_think_by_step_prompt(entry):
    if 'prompt' not in entry:
        return entry

    for p in entry['prompt']:
        content = p.get('content', '')
        if content.startswith("You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"):
            prefix = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"
            new_content = prefix + include_CoT_prompt_block + content[len(prefix):] + think_step_by_step_prompt
            p["content"] = new_content
        elif content.startswith("You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"):
            prefix = "You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"
            new_content = prefix + include_CoT_prompt_block + content[len(prefix):] + think_step_by_step_prompt
            p["content"] = new_content
        else: print(f"DEBUG: Undefined content.startwith() at index {item}\n {content[:100]}...\n")
    return entry

def main():
    # 读取原始 pkl
    with open(INPUT_PKL, "rb") as f:
        data = pickle.load(f)

    # 遍历数据注入 hacking prompt
    for i, entry in enumerate(data):
        data[i] = inject_think_by_step_prompt(entry)

    # 保存新 pkl
    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(data, f)

    print(f"think-by-step prompt injected, saved to {OUTPUT_PKL}")

if __name__ == "__main__":
    main()
