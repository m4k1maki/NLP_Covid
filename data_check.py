import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
import networkx as nx
from tqdm import tqdm

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def clean_string(text):
    text = str(text).lower().replace("[newline]", "\n")
    return text

def lemmatize_sentence(sentence):
    return " ".join(lemmatizer.lemmatize(word) for word in str(sentence).split())

def find_overlap_rows(df):
    if "tweet" not in df.columns:
        df["tweet"] = df.apply(
            lambda row: row["llama_1"] if row.get("rj_llama_1", 1) < 0.9 else row["qwen_1"], axis=1
        )
    df["claim_lemmatized"] = df["claim"].apply(clean_string).apply(lemmatize_sentence)
    df["tweet_lemmatized"] = df["tweet"].apply(clean_string).apply(lemmatize_sentence)
    df = df[df["claim_lemmatized"] != df["tweet_lemmatized"]]
    return df  # Giữ cột tạm thời

def preprocess_and_build_graph(df):
    G = nx.Graph()
    for i, row1 in df.iterrows():
        for j, row2 in df.iterrows():
            if i < j:
                overlap = len(set(row1["claim_lemmatized"].split()) & set(row2["tweet_lemmatized"].split())) / \
                         len(set(row1["claim_lemmatized"].split()) | set(row2["tweet_lemmatized"].split()))
                if overlap > 0.9:
                    G.add_edge(i, j)
    components = list(nx.connected_components(G))
    for idx, component in enumerate(components):
        df.loc[list(component), "group"] = idx
    df = df.drop(columns=["claim_lemmatized", "tweet_lemmatized"])  # Xóa sau khi hoàn tất
    return df

dataset = pd.read_json("E:\\NLPFinalproject\\EunCheolChoi0123-web25-short-aug-mis-e96f86c\\data\\dataset.json")
dataset = find_overlap_rows(dataset)
dataset = preprocess_and_build_graph(dataset)
dataset.to_json("E:\\NLPFinalproject\\EunCheolChoi0123-web25-short-aug-mis-e96f86c\\data\\dataset_checked.json")
print("Done processing data_check.py")
