import pandas as pd
import gzip

# convert_nli.py
import pandas as pd
import gzip

# Giải nén nli.tsv.gz từ đường dẫn gốc
with gzip.open("E:\\NLPFinalproject\\EunCheolChoi0123-web25-short-aug-mis-e96f86c\\data\\nli.tsv.gz", 'rt', encoding='utf-8') as f_in:
    with open("/EunCheolChoi0123-web25-short-aug-mis-e96f86c/data/nli.tsv", 'wt', encoding='utf-8') as f_out:
        f_out.write(f_in.read())

# Chuyển nli.tsv sang nli.json
nli = pd.read_csv("/EunCheolChoi0123-web25-short-aug-mis-e96f86c/data/nli.tsv", sep="\t")
nli.to_json("E:/NLPFinalproject/EunCheolChoi0123-web25-short-aug-mis-e96f86c/nli.json", orient="records", lines=True)
print("Done converting nli.tsv.gz to nli.json")