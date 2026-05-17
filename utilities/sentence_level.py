import pandas as pd
import json
import nltk
import ast

df = pd.read_csv("llm_bias_labels.csv")

data = []

def safe_parse_labels(x):
    if pd.isna(x) or x == "":
        return []
    try:
        return json.loads(x)
    except:
        try:
            return ast.literal_eval(x)
        except:
            return []

for _, row in df.iterrows():
    review = row["review"]
    labels = safe_parse_labels(row["sentence_labels"])

    sentences = nltk.sent_tokenize(review)

    if len(sentences) != len(labels):
        continue

    for s, l in zip(sentences, labels):
        if l not in ["Error", "None"]:
            data.append({"sentence": s, "label": l})

sentence_df = pd.DataFrame(data)
sentence_df.to_csv("sentence_level_dataset.csv", index=False)

print("✅ Done")