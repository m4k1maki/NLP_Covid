import random
import pandas as pd
import re
import nltk
from tqdm import tqdm
from nltk.corpus import wordnet

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
random.seed(1)


#stop words list
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our',
			'ours', 'ourselves', 'you', 'your', 'yours',
			'yourself', 'yourselves', 'he', 'him', 'his',
			'himself', 'she', 'her', 'hers', 'herself',
			'it', 'its', 'itself', 'they', 'them', 'their',
			'theirs', 'themselves', 'what', 'which', 'who',
			'whom', 'this', 'that', 'these', 'those', 'am',
			'is', 'are', 'was', 'were', 'be', 'been', 'being',
			'have', 'has', 'had', 'having', 'do', 'does', 'did',
			'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
			'because', 'as', 'until', 'while', 'of', 'at',
			'by', 'for', 'with', 'about', 'against', 'between',
			'into', 'through', 'during', 'before', 'after',
			'above', 'below', 'to', 'from', 'up', 'down', 'in',
			'out', 'on', 'off', 'over', 'under', 'again',
			'further', 'then', 'once', 'here', 'there', 'when',
			'where', 'why', 'how', 'all', 'any', 'both', 'each',
			'few', 'more', 'most', 'other', 'some', 'such', 'no',
			'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
			'very', 's', 't', 'can', 'will', 'just', 'don',
			'should', 'now', '']

# ====== CLEANING ======
def get_only_chars(line):
    line = re.sub(r"[^a-zA-Z ]+", " ", line).lower()
    return re.sub(r" +", " ", line).strip()

# ====== SYNONYMS ======
def get_synonyms(word):
    return list({lemma.name().replace("_", " ").lower() for syn in wordnet.synsets(word) for lemma in syn.lemmas()} - {word})

# ====== EDA METHODS ======
def synonym_replacement(words, n):
    new_words = words[:]
    candidates = [w for w in words if w not in stop_words]
    random.shuffle(candidates)
    num_replaced = 0
    for word in candidates:
        synonyms = get_synonyms(word)
        if synonyms:
            synonym = random.choice(synonyms)
            new_words = [synonym if w == word else w for w in new_words]
            num_replaced += 1
        if num_replaced >= n: break
    return new_words

def random_deletion(words, p):
    if len(words) == 1: return words
    return [w for w in words if random.random() > p] or [random.choice(words)]

def random_swap(words, n):
    if len(words) < 2:
        return words
    new_words = words[:]
    for _ in range(n):
        if len(new_words) < 2: break
        i, j = random.sample(range(len(new_words)), 2)
        new_words[i], new_words[j] = new_words[j], new_words[i]
    return new_words


def random_insertion(words, n):
    new_words = words[:]
    for _ in range(n):
        for _ in range(10):
            word = random.choice(new_words)
            synonyms = get_synonyms(word)
            if synonyms:
                new_words.insert(random.randint(0, len(new_words)), random.choice(synonyms))
                break
    return new_words

def eda(sentence):
    sentence = get_only_chars(sentence)
    words = sentence.split()
    if not words: return sentence
    n = max(1, int(0.1 * len(words)))
    techniques = [
        synonym_replacement(words, n),
        random_insertion(words, n),
        random_swap(words, n),
        random_deletion(words, 0.1),
    ]
    return ' '.join(random.choice(techniques))

def aeda(sentence):
    puncs = [".", ";", "?", ":", "!", ","]
    words = sentence.split()
    n = random.randint(1, len(words)-1) if len(words) > 1 else 0
    for idx in sorted(random.sample(range(1, len(words)), n), reverse=True):
        words.insert(idx, random.choice(puncs))
    return ' '.join(words)

# ====== MAIN ======
if __name__ == '__main__':
    df = pd.read_json("E:/NLPFinalproject/EunCheolChoi0123-web25-short-aug-mis-e96f86c/data/dataset_checked.json")

    if "tweet" not in df.columns:
        df["tweet"] = df.apply(lambda r: r.get("llama_1") if r.get("rj_llama_1", 1) < 0.9 else r.get("qwen_1"), axis=1)

    tqdm.pandas()
    df["eda_1"] = df["tweet"].progress_apply(eda)
    df["aeda_1"] = df["tweet"].progress_apply(aeda)

    out_path = "E:/NLPFinalproject/EunCheolChoi0123-web25-short-aug-mis-e96f86c/data/dataset_eda_aeda.json"
    df.to_json(out_path, orient="records", indent=4)
    print(f"Saved EDA + AEDA to {out_path}")

