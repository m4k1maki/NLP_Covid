import os
import pandas as pd
from sentence_transformers import CrossEncoder
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm

def main():
    # Load data
    df = pd.read_json("E:\\NLPFinalproject\\EunCheolChoi0123-web25-short-aug-mis-e96f86c\\data\\dataset_final.json")
    test_df = df[
        (df['split'] == 'test')
        & (df['tweet'].notna())
        & (df['claim'].notna())
    ]

    # Load pretrained CrossEncoder
    model = CrossEncoder(
        'E:\\NLPFinalproject\\EunCheolChoi0123-web25-short-aug-mis-e96f86c\\models\\pretrained_ce',
        max_length=256
    )

    # Prepare input sentence pairs
    sentence_pairs = list(zip(test_df['tweet'], test_df['claim']))

    # Predict
    predictions = model.predict(sentence_pairs, show_progress_bar=True)

    # True labels
    true_labels = test_df['label'].tolist()

    # Convert logits to predicted classes
    predicted_numeric = np.argmax(predictions, axis=1)

    # Map back to label strings (same order as used in training)
    label_mapping = {0: 'oppose', 1: 'support', 2: 'neither'}
    predicted_labels = [label_mapping[num] for num in predicted_numeric]

    # Print classification report
    report_dict = classification_report(
        true_labels,
        predicted_labels,
        target_names=['neither', 'oppose', 'support'],
        output_dict=True
    )

    print(report_dict)

if __name__ == "__main__":
    main()
