import pandas as pd
import os
from tabulate import tabulate

def main():
    input_path = './eval/classification_reports.csv'
    output_path = './eval/summary_report.txt'

    if not os.path.exists(input_path):
        print(" Không tìm thấy classification_reports.csv")
        return

    df = pd.read_csv(input_path)

    # Lọc chỉ macro avg và accuracy
    filtered_df = df[df['Class'].isin(['macro avg', 'accuracy'])]

    # Nhóm theo Augment, Size, rồi pivot ra cho đẹp
    summary = filtered_df.pivot_table(index=['Augment', 'Size'],
                                      columns='Class',
                                      values='f1-score').reset_index()

    # Sắp xếp theo Size
    summary = summary.sort_values(by=['Augment', 'Size'])

    # In ra console
    print("\n📊 Summary of all models:\n")
    print(tabulate(summary, headers='keys', tablefmt='fancy_grid', showindex=False))

    # Save vào file .txt
    os.makedirs('./eval', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(tabulate(summary, headers='keys', tablefmt='fancy_grid', showindex=False))

    print(f"\nSummary report saved to: {output_path}")

if __name__ == "__main__":
    main()
