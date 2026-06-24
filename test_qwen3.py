import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_PATH = "/data2/nfs_node1/gb/gb_project/Project/EmotionAnalysis/models/Qwen3-8B"


def main():
    print("正在加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    print("正在加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )

    prompt = "请用一句话介绍你自己。"

    messages = [
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print("正在生成...")
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=False
        )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
    response = tokenizer.decode(output_ids, skip_special_tokens=True)

    print("\n模型输出：")
    print(response)


if __name__ == "__main__":
    main()