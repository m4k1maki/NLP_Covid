import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, T5ForConditionalGeneration
from tqdm import tqdm
import openai
import gc

# ========== CẤU HÌNH ==========
HF_TOKEN = ""  # Hugging Face token
OPENAI_API_KEY = ""
openai.api_key = OPENAI_API_KEY

CACHE_DIR = "E:\\Games\\huggingface_cache"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f">>> Running on: {DEVICE.upper()}")
if DEVICE != "cuda":
    print("Cảnh báo: Không phát hiện GPU. Sẽ dùng CPU, có thể rất chậm.")

# ========== CONFIG BNB 4bit ==========
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    llm_int8_enable_fp32_cpu_offload=True
)

# ========== LOAD DATA ==========
dataset_path = "E:\\NLPFinalproject\\EunCheolChoi0123-web25-short-aug-mis-e96f86c\\data\\dataset_checked.json"
dataset = pd.read_json(dataset_path)

# ========== HÀM PROMPT ==========
def create_prompt(claim, label, dataset, num_examples=3):
    examples = dataset[dataset["label"] == label].sample(min(num_examples, len(dataset[dataset["label"] == label])))
    prompt = f"System: Generate TWEET so that if TWEET is true, then CLAIM is {label}. Mimic the style of example TWEETs.\nInput:\n"
    for _, row in examples.iterrows():
        tweet_text = row.get("tweet", row.get("tweet_id", "[MISSING_TWEET]"))
        prompt += f"CLAIM: {row['claim']}\nTWEET: {tweet_text}\n"
    prompt += f"CLAIM: {claim}\nTWEET:"
    return prompt

# ========== HÀM GEN CHO CAUSAL LM ==========
def generate_causal_lm(model, tokenizer, prompt, max_new_tokens=5):
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id
        )
        result = tokenizer.decode(outputs[0], skip_special_tokens=True).split("TWEET:")[-1].strip()
        torch.cuda.empty_cache()
        return result
    except Exception as e:
        print("Error during causal LM generation:", e)
        torch.cuda.empty_cache()
        return "GEN_ERROR"

# ========== HÀM GEN CHO FLAN-T5 ==========
def generate_flan_t5(model, tokenizer, prompt, max_new_tokens=5):
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id
        )
        result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        torch.cuda.empty_cache()
        return result
    except Exception as e:
        print("Error during Flan-T5 generation:", e)
        torch.cuda.empty_cache()
        return "GEN_ERROR"

# ========== HÀM GEN CHO GPT-4o ==========
def generate_gpt4o_mini(prompt, max_new_tokens=5):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant generating tweets."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_new_tokens,
            temperature=0.7
        )
        torch.cuda.empty_cache()
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("Error during GPT-4o-mini generation:", e)
        torch.cuda.empty_cache()
        return "GEN_ERROR"

# ========== CHẠY AUG TỪNG MODEL ==========
models = [
    ("falcon3-1b-instruct", "tiiuae/Falcon3-1B-Instruct", AutoModelForCausalLM, "falcon3"),
    ("qwen-0.5b", "Qwen/Qwen1.5-0.5B-Chat", AutoModelForCausalLM, "qwen"),
    ("flan-t5-base", "google/flan-t5-base", T5ForConditionalGeneration, "flan_t5")
]

for model_name, model_id, model_class, prefix in models:
    print(f">>> Loading {model_name}...")
    try:
        model = model_class.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto",
            token=HF_TOKEN,
            trust_remote_code=True,
            cache_dir=CACHE_DIR
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=HF_TOKEN,
            trust_remote_code=True,
            cache_dir=CACHE_DIR
        )

        for idx, row in tqdm(dataset.iterrows(), total=len(dataset), desc=f">> Augmenting with {model_name}"):
            claim = row["claim"]
            label = row["label"]
            prompt = create_prompt(claim, label, dataset)
            for i in range(1, 3):  # Chạy 2 lần
                if model_class == AutoModelForCausalLM:
                    dataset.at[idx, f"{prefix}_{i}"] = generate_causal_lm(model, tokenizer, prompt)
                elif model_class == T5ForConditionalGeneration:
                    dataset.at[idx, f"{prefix}_{i}"] = generate_flan_t5(model, tokenizer, prompt)

    except Exception as e:
        print(f"Error loading {model_name}: {e}")
    finally:
        # Giải phóng RAM/VRAM
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

# ========== CHẠY GPT-4o ==========
print(">>> Running GPT-4o...")
for idx, row in tqdm(dataset.iterrows(), total=len(dataset), desc=">> Augmenting with GPT-4o-mini"):
    claim = row["claim"]
    label = row["label"]
    prompt = create_prompt(claim, label, dataset)
    for i in range(1, 3):  # Chạy 2 lần
        dataset.at[idx, f"gpt4o_mini_{i}"] = generate_gpt4o_mini(prompt)

# ========== LƯU ==========
output_path = "E:\\NLPFinalproject\\EunCheolChoi0123-web25-short-aug-mis-e96f86c\\data\\dataset_aug.json"
dataset.to_json(output_path, force_ascii=False)
print("Done augmenting cmg data and saved to:", output_path)
