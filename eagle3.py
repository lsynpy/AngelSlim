#!/usr/bin/env python

import torch

from angelslim.compressor.speculative.inference.models.eagle3 import Eagle3Model


def run_eagle3(base_model_path, eagle_model_path, prompt="Once upon a time"):
    model = Eagle3Model.from_pretrained(
        base_model_path=base_model_path,
        eagle_model_path=eagle_model_path,
        dtype=torch.float16,
        device_map="auto",
    )

    tokenizer = model.get_tokenizer()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.base_model.device)

    print(f"Input: {prompt}")

    output_ids = model.eagle_generate(input_ids=input_ids, max_new_tokens=64, temperature=0.6)

    generated_text = tokenizer.decode(output_ids[0, input_ids.shape[1] :], skip_special_tokens=True)
    print(f"Output: {generated_text}")


if __name__ == "__main__":
    BASE_MODEL_PATH = "/home/pp/huggingface/Qwen3-1.7B"
    EAGLE_MODEL_PATH = "/home/pp/huggingface/Qwen3-1.7B_eagle3"
    PROMPT = "Explain machine learning in simple terms"

    print("Attempting to run Eagle3 demo...")
    print(f"Using base model: {BASE_MODEL_PATH}")
    print(f"Using eagle model: {EAGLE_MODEL_PATH}")
    print(f"Prompt: {PROMPT}")
    print("-" * 50)

    run_eagle3(BASE_MODEL_PATH, EAGLE_MODEL_PATH, PROMPT)
