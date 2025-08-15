import pandas as pd
import os
import math
from datetime import datetime
from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder, InputExample
from sklearn.metrics import classification_report
import torch.nn as nn
import torch


def run_experiment(model_huggingface_path, task, total_sample_size_list, augment, max_seq_length=128, repeat=1):
    run_count = 0
    while run_count < repeat:

        # ====== Auto check augment column ======
        augment_column_list = []
        if augment:
            if task == 'bt':
                augment_column_list = ['bt_de', 'bt_ru', 'bt_zh', 'bt_it']
            else:
                augment_column_list = [f'{task}_{i}' for i in range(1, 5) if f'{task}_{i}' in df.columns]

        # ====== Init device ======
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        freq = df[df['split'] == 'train']['label'].value_counts(normalize=True)

        for train_size in total_sample_size_list:
            task_name = f"{task}_{train_size}"
            print(f"\n=== Training task: {task_name} ===")

            label2int = {"oppose": 0, "support": 1, "neither": 2}
            int2label = {v: k for k, v in label2int.items()}
            train_batch_size = 8
            num_epochs = 5

            class_weights = torch.tensor([
                float(1 / freq["oppose"]),
                float(1 / freq["support"]),
                float(1 / freq["neither"])
            ]).to(device)

            # ====== Prepare samples ======
            def get_samples(df):
                train_samples = []
                dev_samples = df[df['split'] == 'test']
                train_df = df[(df['split'] == 'train') & (df['train_size'].astype(int) <= train_size)]
                print("Label distribution (train):")
                print(train_df['label'].value_counts(normalize=True))

                for _, row in train_df.iterrows():
                    label_id = label2int[row['label'].lower()]
                    train_samples.append(InputExample(texts=[row['tweet'], row['claim']], label=label_id))

                    if augment:
                        for augment_column in augment_column_list:
                            try:
                                aug_text = row[augment_column]
                                if pd.isna(aug_text): continue
                                if any(k in augment_column for k in ["gpt", "llama", "qwen"]):
                                    if f"rj_{augment_column}" in row and row[f"rj_{augment_column}"] >= 0.9:
                                        continue
                                train_samples.append(InputExample(texts=[aug_text, row['claim']], label=label_id))
                            except KeyError:
                                continue

                dev_samples = [
                    InputExample(texts=[r['tweet'], r['claim']], label=label2int[r['label'].lower()])
                    for _, r in dev_samples.iterrows()
                    if pd.notna(r['tweet']) and pd.notna(r['claim'])
                ]
                return train_samples, dev_samples

            # ====== Evaluation ======
            def evaluate_model(model, dev_samples, epoch):
                texts = [(s.texts[0], s.texts[1]) for s in dev_samples]
                true_labels = [s.label for s in dev_samples]
                pred_scores = model.predict(texts)
                pred_labels = pred_scores.argmax(axis=1)
                report = classification_report(true_labels, pred_labels, target_names=[int2label[i] for i in range(3)], digits=4)
                print(f"Epoch {epoch + 1} Classification Report:\n{report}")

            # ====== Train loop ======
            def train_model():
                train_samples, dev_samples = get_samples(df)
                model = CrossEncoder(model_huggingface_path, num_labels=len(label2int), max_length=max_seq_length, device=device)
                dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
                warmup_steps = math.ceil(len(dataloader) * num_epochs * 0.1)
                print(f"Warmup steps: {warmup_steps}")

                model_save_path = f"./models/finetuned_{task_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                loss_fct = nn.CrossEntropyLoss(weight=class_weights)

                for epoch in range(num_epochs):
                    model.fit(
                        train_dataloader=dataloader,
                        epochs=1,
                        warmup_steps=warmup_steps,
                        output_path=model_save_path,
                        loss_fct=loss_fct
                    )
                    evaluate_model(model, dev_samples, epoch)

                model.save(model_save_path)
                print(f"Model saved: {model_save_path}")

            # ====== Run training ======
            train_model()
        run_count += 1


if __name__ == "__main__":
    df = pd.read_json("E:/NLPFinalproject/EunCheolChoi0123-web25-short-aug-mis-e96f86c/data/dataset_final.json")

    if 'train_size' not in df.columns:
        df['train_size'] = range(1, len(df) + 1)

    os.makedirs("./models", exist_ok=True)

    total_sample_size_list = [100, 200, 500, 1000, 2000, 5000]
    model_huggingface_path = "E:/NLPFinalproject/EunCheolChoi0123-web25-short-aug-mis-e96f86c/models/pretrained_ce"

    for task in ["base", "bt", "aeda", "eda", "gpt4", "gpt3", "llama", "qwen", "falcon3", "flan_t5", "gpt4o_mini"]:
        augment = task != "base"
        run_experiment(model_huggingface_path, task, total_sample_size_list, augment, max_seq_length=128)
