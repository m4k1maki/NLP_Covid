import pandas as pd
from tqdm import tqdm
import nlpaug.augmenter.word as naw
import torch
import gc
from torch.utils.data import DataLoader

# ========== CẤU HÌNH ==========
INPUT_PATH = "E:/NLPFinalproject/EunCheolChoi0123-web25-short-aug-mis-e96f86c/data/dataset_eda_aeda.json"
OUTPUT_PATH = "E:/NLPFinalproject/EunCheolChoi0123-web25-short-aug-mis-e96f86c/data/dataset_aug_backtrans.json"
CHUNK_SIZE = 2000  # Tăng lên 2000 để giảm số lần lặp
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LANGS = ['de', 'ru', 'zh', 'it']
BATCH_SIZE = 8

print(f">>> Running on: {DEVICE.upper()} with CHUNK_SIZE={CHUNK_SIZE}, BATCH_SIZE={BATCH_SIZE}")


# ========== HÀM XÂY DỰNG AUGMENTER ==========
def build_aug(lang, device='cuda'):
    return naw.BackTranslationAug(
        from_model_name=f"Helsinki-NLP/opus-mt-en-{lang}",
        to_model_name=f"Helsinki-NLP/opus-mt-{lang}-en",
        device=device,
        batch_size=BATCH_SIZE  # Sử dụng batch size
    )


# HÀM CHẠY BACK-TRANSLATION
def run_bt_on_chunk(chunk_df, lang, aug):
    results = []
    # Tạo DataLoader để xử lý batch
    dataloader = DataLoader(chunk_df['tweet'].tolist(), batch_size=BATCH_SIZE, shuffle=False)

    for batch_texts in tqdm(dataloader, desc=f"BT_{lang.upper()}"):
        try:
            batch_results = aug.augment(batch_texts)
            results.extend(batch_results)
        except RuntimeError as e:
            print(f"[WARN] {lang.upper()} - OOM or error on batch → fallback to original")
            torch.cuda.empty_cache()
            gc.collect()
            results.extend(batch_texts)  # Fallback
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")
            results.extend(batch_texts)

    return results


# Main
if __name__ == '__main__':
    df = pd.read_json(INPUT_PATH)

    for lang in LANGS:
        print(f"\n[INFO] BackTrans {lang.upper()} | {len(df)} rows | Batch: manual chunk {CHUNK_SIZE}")
        aug = build_aug(lang, DEVICE)

        full_result = []
        for i in range(0, len(df), CHUNK_SIZE):
            chunk = df.iloc[i:i + CHUNK_SIZE].copy()
            chunk_result = run_bt_on_chunk(chunk, lang, aug)
            full_result.extend(chunk_result)
            # Giải phóng bộ nhớ sau mỗi chunk
            del chunk
            gc.collect()
            torch.cuda.empty_cache()

        df[f'bt_{lang}'] = full_result
        del aug
        gc.collect()
        torch.cuda.empty_cache()

    df.to_json(OUTPUT_PATH, orient='records', indent=4)
    print(f" All backtranslations done! Saved at: {OUTPUT_PATH}")