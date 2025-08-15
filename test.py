import os
import re
import pandas as pd
from sentence_transformers import CrossEncoder
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def main():
    df = pd.read_json("E:/NLPFinalproject/EunCheolChoi0123-web25-short-aug-mis-e96f86c/data/dataset_final.json")
    test_df = df[
        (df['split'] == 'test') &
        (df['tweet'].notna()) &
        (df['claim'].notna())
    ]

    os.makedirs("./eval", exist_ok=True)
    model_dir_list = sorted(os.listdir('./models'))
    all_rows = []

    for model_name in tqdm(model_dir_list, desc="Processing models"):
        try:
            model_path = f'./models/{model_name}'
            model = CrossEncoder(model_path, max_length=256)

            sentence_pairs = list(zip(test_df['tweet'], test_df['claim']))
            predictions = model.predict(sentence_pairs)
            true_labels = test_df['label'].tolist()
            predicted_numeric = np.argmax(predictions, axis=1)

            label_mapping = {1: 'support', 2: 'neither', 0: 'oppose'}
            predicted_labels = [label_mapping[num] for num in predicted_numeric]

            report_dict = classification_report(
                true_labels,
                predicted_labels,
                target_names=['neither', 'oppose', 'support'],
                output_dict=True
            )

            # Parse model name
            match = re.search(r'finetuned_(.+?)_(\d+)', model_name)
            if not match:
                print(f"Skipping model with bad name format: {model_name}")
                continue

            augment_name = match.group(1)
            size_value = int(match.group(2))

            for label, metrics in report_dict.items():
                f1 = metrics if label == 'accuracy' else metrics['f1-score']
                all_rows.append({
                    'Augment': augment_name,
                    'Size': size_value,
                    'Class': label,
                    'f1-score': f1
                })

        except Exception as e:
            print(f"[Error processing {model_name}]: {e}")

    classification_reports_df = pd.DataFrame(all_rows)
    classification_reports_df.to_csv('./eval/classification_reports.csv', index=False)
    print("📁 Classification report saved at ./eval/classification_reports.csv")

    visualize_results()


def visualize_results():
    df = pd.read_csv('./eval/classification_reports.csv')

    df['Size'] = pd.to_numeric(df['Size'], errors='coerce')
    df.dropna(subset=['Size'], inplace=True)
    df['Size'] = df['Size'].astype(int)

    palette = sns.color_palette("Set2")

    # Barplot for accuracy
    acc_df = df[df['Class'] == 'accuracy']
    plt.figure(figsize=(14, 6))
    sns.barplot(data=acc_df, x='Size', y='f1-score', hue='Augment', palette=palette)
    plt.title('Accuracy (F1-score) theo train')
    plt.xlabel('Train Size')
    plt.ylabel('Accuracy (F1-score)')
    plt.legend(title='Augment', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('./eval/accuracy_barplot.png')
    plt.close()

    # Lineplot for macro avg
    macro_df = df[df['Class'] == 'macro avg']
    plt.figure(figsize=(14, 6))
    sns.lineplot(data=macro_df, x='Size', y='f1-score', hue='Augment', marker='o', palette=palette)
    plt.title('Macro F1-score theo train')
    plt.xlabel('Train Size')
    plt.ylabel('Macro F1-score')
    plt.legend(title='Augment', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('./eval/macro_f1_lineplot.png')
    plt.close()

    # Lineplot for each class
    class_df = df[df['Class'].isin(['support', 'oppose', 'neither'])]
    for class_name in ['support', 'oppose', 'neither']:
        plt.figure(figsize=(14, 6))
        subset = class_df[class_df['Class'] == class_name]
        sns.lineplot(data=subset, x='Size', y='f1-score', hue='Augment', marker='o', palette=palette)
        plt.title(f'F1-score theo train – Class: {class_name}')
        plt.xlabel('Train Size')
        plt.ylabel('F1-score')
        plt.legend(title='Augment', bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'./eval/class_f1_{class_name}.png')
        plt.close()

    print("Visualization saved in ./eval (barplot + lineplots)")


if __name__ == "__main__":
    main()
