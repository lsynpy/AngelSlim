import torch

from angelslim.compressor.speculative.inference.models.eagle3 import Eagle3Model

MAX_NEW_TOKENS = 4


def run_eagle3(base_model_path, eagle_model_path, prompt="Once upon a time"):
    model = Eagle3Model.from_pretrained(
        base_model_path=base_model_path,
        eagle_model_path=eagle_model_path,
        total_tokens=10,
        depth=4,
        top_k=3,
        dtype=torch.float16,
        device_map="auto",
    )

    tokenizer = model.get_tokenizer()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.base_model.device)

    print(f"Input: {prompt}")

    output_ids = model.eagle_generate(
        input_ids=input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.0,
    )

    generated_text = tokenizer.decode(output_ids[0, input_ids.shape[1] :], skip_special_tokens=True)
    print(f"Output: {generated_text}")


if __name__ == "__main__":
    BASE_MODEL_PATH = "/home/pp/huggingface/Qwen3-1.7B"
    EAGLE_MODEL_PATH = "/home/pp/huggingface/Qwen3-1.7B_eagle3"
    PROMPT = "List 10 numbers only contains digit 1:"

    print("Attempting to run Eagle3 demo...")
    print(f"Using base model: {BASE_MODEL_PATH}")
    print(f"Using eagle model: {EAGLE_MODEL_PATH}")
    print(f"Prompt: {PROMPT}")
    print("-" * 50)

    run_eagle3(BASE_MODEL_PATH, EAGLE_MODEL_PATH, PROMPT)
