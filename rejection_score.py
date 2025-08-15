from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from tqdm import tqdm
import torch
import gc

if __name__ == "__main__":

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 128

    df = pd.read_json('E:\\NLPFinalproject\\EunCheolChoi0123-web25-short-aug-mis-e96f86c\\data\\dataset_full_augment.json')
    tokenizer = AutoTokenizer.from_pretrained("protectai/distilroberta-base-rejection-v1")
    model = AutoModelForSequenceClassification.from_pretrained("protectai/distilroberta-base-rejection-v1")
    model.to(DEVICE)
    model.eval()

    settings = [
        'gpt4_1', 'gpt4_2', 'gpt4_3', 'gpt4_4',
        'gpt3_1', 'gpt3_2', 'gpt3_3', 'gpt3_4',
        'llama_1', 'llama_2', 'llama_3', 'llama_4',
        'qwen_1', 'qwen_2', 'qwen_3', 'qwen_4',
        'falcon3_1', 'falcon3_2',  # Thêm Falcon3
        'flan_t5_1', 'flan_t5_2',
        'gpt4o_mini_1', 'gpt4o_mini_2'
    ]

    for column_name in settings:
        if column_name not in df.columns:
            print(f"[SKIP] Cột '{column_name}' không tồn tại.")
            continue

        print(f" Đang tính rejection cho: {column_name}")
        rejection_probs_all = []

        for i in tqdm(range(0, len(df[column_name]), BATCH_SIZE)):
            batch_texts = df[column_name][i:i+BATCH_SIZE].astype(str).tolist()

            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64
            ).to(DEVICE)

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                rejection_probs = probs[:, 1].tolist()

            # Gán kết quả
            df.loc[i:i+len(rejection_probs)-1, f'rj_{column_name}'] = rejection_probs

            # Dọn bộ nhớ
            torch.cuda.empty_cache()
            gc.collect()

    df.to_json('./data/dataset_fullaugment_rjs.json', orient='records', indent=4, force_ascii=False)
    print(" DONE! Kết quả được lưu vào ./data/dataset_final_rj.json")
