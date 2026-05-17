import pandas as pd
import nltk
import torch
from collections import Counter
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification

nltk.download('punkt')

# ------------------ LOAD MODEL ------------------ #

tokenizer = AutoTokenizer.from_pretrained("bias_model")
model = AutoModelForSequenceClassification.from_pretrained("bias_model")

def predict(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True)
    outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    conf, pred = torch.max(probs, dim=1)

    return model.config.id2label[pred.item()], conf.item()


# ------------------ HELPERS ------------------ #

def split_sentences(text):
    return [s.strip() for s in nltk.sent_tokenize(str(text)) if s.strip()]

def categorize_rating(r):
    try:
        r = int(r)  # convert string → int
    except:
        return "neutral"  # fallback if missing or invalid

    if r >= 7:
        return "positive"
    elif r <= 4:
        return "negative"
    else:
        return "neutral"


# ------------------ ANALYSIS ------------------ #

def analyze_movie(csv_path, movie_name):

    df = pd.read_csv(csv_path)

    # filter movie
    df = df[df["movie"] == movie_name]

    if len(df) == 0:
        print("❌ Movie not found")
        return

    print(f"\n🎬 Analyzing: {movie_name}")
    print(f"Total reviews: {len(df)}")

    bias_counts = {
        "positive": Counter(),
        "negative": Counter(),
        "neutral": Counter()
    }

    # process each review
    for _, row in df.iterrows():
        review = row["review"]
        rating = row["rating"]

        category = categorize_rating(rating)
        sentences = split_sentences(review)

        for s in sentences:
            label, conf = predict(s)
            bias_counts[category].update([label])

    return bias_counts

def analyze_and_save(df, movie_name, output_file="labeled_reviews.csv"):
    results = []

    for _, row in df.iterrows():
        review = row.get("review", "")
        rating = row.get("rating", None)

        category = categorize_rating(rating)
        sentences = split_sentences(review)

        for s in sentences:
            label, conf = predict(s)

            results.append({
                "movie": movie_name,
                "sentence": s,
                "review_rating": rating,
                "sentiment_category": category,
                "bias_label": label,
                "confidence": conf
            })

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_file, index=False)

    print(f"\n✅ Saved labeled data to {output_file}")

    return result_df

# ------------------ NORMALIZE ------------------ #

def normalize(counter):
    total = sum(counter.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counter.items()}


# ------------------ VISUALIZATION ------------------ #

def plot_bias(pos, neg):
    labels = list(set(pos.keys()) | set(neg.keys()))

    pos_vals = [pos.get(l, 0) for l in labels]
    neg_vals = [neg.get(l, 0) for l in labels]

    x = range(len(labels))

    plt.figure(figsize=(12,6))
    plt.bar(x, pos_vals, width=0.4, label="Positive", align='center')
    plt.bar(x, neg_vals, width=0.4, label="Negative", align='edge')

    plt.xticks(x, labels, rotation=45)
    plt.legend()
    plt.title("Bias Distribution")
    plt.tight_layout()
    plt.show()


# ------------------ INSIGHTS ------------------ #

def print_insights(pos, neg):
    print("\n🔍 Key Insights:\n")

    all_keys = set(pos.keys()) | set(neg.keys())

    for k in all_keys:
        p = pos.get(k, 0)
        n = neg.get(k, 0)

        if abs(p - n) > 0.05:
            if p > n:
                print(f"{k} → more common in POSITIVE reviews")
            else:
                print(f"{k} → more common in NEGATIVE reviews")


# ------------------ MAIN ------------------ #

def run(csv_path, movie_name):

    bias_counts = analyze_movie(csv_path, movie_name)

    pos = normalize(bias_counts["positive"])
    neg = normalize(bias_counts["negative"])

    print("\n📊 Positive Bias:")
    print(pos)

    print("\n📊 Negative Bias:")
    print(neg)

    plot_bias(pos, neg)
    print_insights(pos, neg)


# ------------------ RUN ------------------ #

if __name__ == "__main__":
    run("tt0458352_all_ratings_reviews.csv", "The Devil Wears Prada")
