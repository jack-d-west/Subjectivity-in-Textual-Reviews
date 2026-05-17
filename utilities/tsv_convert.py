import pandas as pd
import re

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-z0-9\s\.,!?']", " ", text)      # keep letters, digits, .,!?'
    text = re.sub(r"\s+", " ", text).strip()             # normalize spaces
    return text

# Read your CSV file
df = pd.read_csv("merged_reviews.csv")  # replace with your actual filename

# Create a unique review ID
df["review_id"] = df["movie"].astype(str) + "_" + (df.reset_index().index + 1).astype(str)

# Clean the review title and text
df["title"] = df["title"].apply(clean_text)
df["review"] = df["review"].apply(clean_text)

# Select relevant columns
tsv_df = df[["review_id","title", "review"]]

# Save as TSV
tsv_df.to_csv("all_reviews.tsv", sep="\t", index=False)

print("✅ TSV file created: all_reviews.tsv")
