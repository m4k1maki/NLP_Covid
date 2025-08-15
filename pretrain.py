import os
import json
import logging
import math
import torch
from datetime import datetime
from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder, LoggingHandler
from sentence_transformers.cross_encoder.evaluation import CEF1Evaluator
from sentence_transformers.readers import InputExample

from tqdm import tqdm as original_tqdm
import sentence_transformers.cross_encoder.CrossEncoder

def tqdm(*args, **kwargs):
    kwargs.setdefault("mininterval", 5.0)
    kwargs.setdefault("miniters", 200)
    return original_tqdm(*args, **kwargs)

sentence_transformers.cross_encoder.CrossEncoder.tqdm = tqdm

logging.basicConfig(format="%(asctime)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}

def load_data(jsonl_path, train_subsample=100000):
    train_samples = []
    dev_samples = []
    with open(jsonl_path, "r", encoding="utf-8") as fIn:
        for line in fIn:
            try:
                row = json.loads(line)
                label_id = label2int.get(row.get("label"))
                s1, s2 = row.get("sentence1"), row.get("sentence2")
                split = row.get("split")
                if label_id is None or not s1 or not s2 or not split:
                    logger.warning(f"Skipping invalid row: {row}")
                    continue
                example = InputExample(texts=[s1.strip(), s2.strip()], label=label_id)
                if split == "train":
                    train_samples.append(example)
                else:
                    dev_samples.append(example)
            except Exception as e:
                logger.warning(f"Error parsing line: {line}\n{e}")

    if train_subsample and len(train_samples) > train_subsample:
        logger.info(f"Subsampling train set: {train_subsample} / {len(train_samples)}")
        train_samples = train_samples[:train_subsample]

    return train_samples, dev_samples

if __name__ == "__main__":
    nli_dataset_path = "E:/NLPFinalproject/EunCheolChoi0123-web25-short-aug-mis-e96f86c/data/nli.json"
    model_save_path = "E:/NLPFinalproject/EunCheolChoi0123-web25-short-aug-mis-e96f86c/models/pretrained_ce"

    train_batch_size = 8
    num_epochs = 5

    logger.info("Loading data...")
    train_samples, dev_samples = load_data(nli_dataset_path, train_subsample=100000)
    logger.info(f"Train samples: {len(train_samples)} | Dev samples: {len(dev_samples)}")

    os.makedirs(model_save_path, exist_ok=True)

    model = CrossEncoder("microsoft/deberta-v3-base", num_labels=len(label2int), max_length=256)

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
    evaluator = CEF1Evaluator.from_input_examples(dev_samples, name="nli-dev")

    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)
    logger.info(f"Warmup steps: {warmup_steps}")

    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=num_epochs,
        evaluation_steps=30000,
        warmup_steps=warmup_steps,
        output_path=model_save_path,
        show_progress_bar=True
    )

    logger.info("Training complete.")
