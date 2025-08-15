import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_json("E:\\NLPFinalproject\\EunCheolChoi0123-web25-short-aug-mis-e96f86c\\data\\dataset_fullaugment_rjs.json")
print(f"Original dataset size: {len(df)}")

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# Add 'split' column
print("Assigning split column...")
df_train = df_train.copy()
df_test = df_test.copy()
df_train['split'] = 'train'
df_test['split'] = 'test'

# ====== Combine and Save New Dataset ======
df_final = pd.concat([df_train, df_test], ignore_index=True)
print(f"Final dataset size after split: {len(df_final)}")

# Save to new JSON
out_path = "./data/dataset_final.json"
df_final.to_json(out_path, orient='records', indent=4)
print(f"Saved to {out_path}")
