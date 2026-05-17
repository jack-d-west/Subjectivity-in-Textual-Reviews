from webscrap import scrape_imdb_reviews
from analyze import analyze_and_save, normalize
import os
from collections import Counter
import math
import pandas as pd
import matplotlib.pyplot as plt

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    PageBreak
)

from reportlab.lib.styles import getSampleStyleSheet


# ------------------ INSIGHT FUNCTIONS ------------------ #

def interpret_bias(k, direction):

    templates = {
        "confirmation_bias":
            "viewers rely on prior expectations.",

        "emotional_bias":
            "opinions are driven by emotional reactions.",

        "exaggeration_bias":
            "statements are amplified beyond objective evaluation."
    }

    explanation = templates.get(
        k,
        "a subjective pattern is observed."
    )

    return f"{direction} reviews show that {explanation}"


def print_insights(pos, neg):

    output = []

    print("\n🔍 Key Insights:\n")

    all_keys = set(pos.keys()) | set(neg.keys())

    for k in all_keys:

        p = pos.get(k, 0)
        n = neg.get(k, 0)

        diff = p - n

        if abs(diff) > 0.05:

            strength = (
                "strong"
                if abs(diff) > 0.15
                else "moderate"
            )

            if diff > 0:

                text = (
                    f"{k} → {strength} indicator "
                    f"of POSITIVE reviews (+{diff:.2f})"
                )

            else:

                text = (
                    f"{k} → {strength} indicator "
                    f"of NEGATIVE reviews ({diff:.2f})"
                )

            print(text)

            output.append(text)

    return output


def top_biases(counter, name):

    output = []

    print(f"\n🔥 Top biases in {name} reviews:")

    for k, v in sorted(
        counter.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]:

        text = f"{k}: {v:.2f}"

        print(text)

        output.append(text)

    return output


def contrast_insights(pos, neg):

    output = []

    print("\n⚖️ Contrast Biases:\n")

    for k in pos:

        if k in neg:

            ratio = (
                (pos[k] + 1e-5)
                /
                (neg[k] + 1e-5)
            )

            if ratio > 2:

                text = (
                    f"{k} → strongly POSITIVE "
                    f"leaning (x{ratio:.2f})"
                )

                print(text)

                output.append(text)

            elif ratio < 0.5:

                text = (
                    f"{k} → strongly NEGATIVE "
                    f"leaning (x{1/ratio:.2f})"
                )

                print(text)

                output.append(text)

    return output


def confidence_insights(df):

    output = []

    print("\n🎯 High Confidence Bias Signals:\n")

    high_conf = df[df["confidence"] > 0.9]

    counts = high_conf["bias_label"].value_counts()

    for k, v in counts.head(5).items():

        text = f"{k} → {v} high-confidence occurrences"

        print(text)

        output.append(text)

    return output


def significance_test(pos, neg):

    output = []

    print("\n📈 Statistically Significant Biases:\n")

    for k in pos:

        if k in neg:

            p = pos[k]
            n = neg[k]

            if p + n == 0:
                continue

            score = (
                abs(p - n)
                /
                math.sqrt(p + n + 1e-5)
            )

            if score > 0.1:

                text = (
                    f"{k} → significant difference "
                    f"(score={score:.3f})"
                )

                print(text)

                output.append(text)

    return output


def human_insights(pos, neg):

    output = []

    print("\n🧠 Human-Readable Insights:\n")

    for k in pos:

        if k in neg:

            p = pos[k]
            n = neg[k]

            if abs(p - n) > 0.05:

                if p > n:

                    text = interpret_bias(k, "Positive")

                else:

                    text = interpret_bias(k, "Negative")

                print(text)

                output.append(text)

    return output


# ------------------ GRAPH FUNCTIONS ------------------ #

def plot_comparison(pos, neg, movie_id):

    labels = list(set(pos.keys()) | set(neg.keys()))

    pos_vals = [pos.get(l, 0) for l in labels]
    neg_vals = [neg.get(l, 0) for l in labels]

    x = range(len(labels))

    plt.figure(figsize=(12, 6))

    plt.bar(
        x,
        pos_vals,
        width=0.4,
        label="Positive"
    )

    plt.bar(
        [i + 0.4 for i in x],
        neg_vals,
        width=0.4,
        label="Negative"
    )

    plt.xticks(
        [i + 0.2 for i in x],
        labels,
        rotation=45
    )

    plt.legend()

    plt.title("Bias Comparison")

    plt.tight_layout()

    plt.savefig(f"{movie_id}_comparison.png")

    plt.close()


def plot_overall_pie(labeled_df, movie_id):

    counts = labeled_df["bias_label"].value_counts()

    plt.figure(figsize=(8, 8))

    plt.pie(
        counts.values,
        labels=counts.index,
        autopct='%1.1f%%'
    )

    plt.title("Overall Bias Distribution")

    plt.savefig(f"{movie_id}_pie.png")

    plt.close()


def plot_confidence_histogram(labeled_df, movie_id):

    plt.figure(figsize=(8, 5))

    plt.hist(
        labeled_df["confidence"],
        bins=20
    )

    plt.xlabel("Confidence")

    plt.ylabel("Frequency")

    plt.title("Confidence Distribution")

    plt.tight_layout()

    plt.savefig(f"{movie_id}_confidence.png")

    plt.close()


def plot_top_biases(labeled_df, movie_id):

    counts = (
        labeled_df["bias_label"]
        .value_counts()
        .head(10)
    )

    plt.figure(figsize=(10, 5))

    counts.plot(kind='bar')

    plt.title("Top Biases")

    plt.ylabel("Count")

    plt.tight_layout()

    plt.savefig(f"{movie_id}_top_biases.png")

    plt.close()


# ------------------ PDF REPORT ------------------ #

def generate_pdf(
    movie_name,
    movie_id,
    insights,
    labeled_df
):

    doc = SimpleDocTemplate(
        f"{movie_id}_report.pdf"
    )

    styles = getSampleStyleSheet()

    elements = []

    # TITLE
    title = Paragraph(
        f"<b>Movie Bias Analysis Report</b><br/>{movie_name}",
        styles['Title']
    )

    elements.append(title)

    elements.append(Spacer(1, 20))

    # DESCRIPTION
    description = Paragraph(
        """
        This report contains:
        <br/>
        • Bias distribution analysis
        <br/>
        • Sentiment-wise comparison
        <br/>
        • Confidence distribution
        <br/>
        • Statistical insights
        <br/>
        • Human-readable interpretations
        <br/>
        • Bias definitions
        <br/>
        • High confidence examples
        """,
        styles['BodyText']
    )

    elements.append(description)

    elements.append(Spacer(1, 20))

    # INSIGHTS SECTION
    insight_title = Paragraph(
        "<b>Generated Insights</b>",
        styles['Heading2']
    )

    elements.append(insight_title)

    elements.append(Spacer(1, 12))

    for section, values in insights.items():

        heading = Paragraph(
            f"<b>{section}</b>",
            styles['Heading3']
        )

        elements.append(heading)

        elements.append(Spacer(1, 6))

        if len(values) == 0:

            elements.append(
                Paragraph(
                    "No major insights detected.",
                    styles['BodyText']
                )
            )

        else:

            for item in values:

                para = Paragraph(
                    f"• {item}",
                    styles['BodyText']
                )

                elements.append(para)

        elements.append(Spacer(1, 12))

    # ------------------ BIAS DEFINITIONS ------------------ #

    bias_definitions = {

        "Expectation Bias": {
            "definition":
                "This occurs when a reviewer judges a movie based on what they expected it to be rather than evaluating what it actually was.",

            "keywords": ["expectation"]
        },

        "Comparative Lens": {
            "definition":
                "The reviewer evaluates a movie by comparing it to another film, book, or reference material.",

            "keywords": ["comparative"]
        },

        "Representation Critique": {
            "definition":
                "The reviewer questions how accurately characters, groups, or perspectives are represented in the film.",

            "keywords": ["representation"]
        },

        "Omission Focus": {
            "definition":
                "The reviewer highlights important topics or details that were left out of the film.",

            "keywords": ["omission"]
        },

        "Moral / Political Projection": {
            "definition":
                "The reviewer inserts their own moral or political viewpoint into the evaluation.",

            "keywords": ["moral", "political"]
        },

        "Selective Emphasis Bias": {
            "definition":
                "The reviewer focuses intensely on one specific aspect while ignoring the rest of the film.",

            "keywords": ["selective"]
        },

        "Performance Appraisal": {
            "definition":
                "Focuses on actor performance, delivery, or mannerisms.",

            "keywords": ["performance"]
        },

        "Authenticity Bias": {
            "definition":
                "The film is judged on factual or technical accuracy.",

            "keywords": ["authenticity"]
        },

        "Disposition Bias": {
            "definition":
                "Personal likes or dislikes influence the reviewer’s judgment.",

            "keywords": ["disposition"]
        }
    }

    bias_title = Paragraph(
        "<b>Bias Definitions & High Confidence Examples</b>",
        styles['Heading2']
    )

    elements.append(bias_title)

    elements.append(Spacer(1, 20))

    high_conf_df = labeled_df[
        labeled_df["confidence"] > 0.90
    ]

    for bias_name, info in bias_definitions.items():

        heading = Paragraph(
            f"<b>{bias_name}</b>",
            styles['Heading3']
        )

        elements.append(heading)

        elements.append(Spacer(1, 6))

        definition_para = Paragraph(
            f"<b>Definition:</b> {info['definition']}",
            styles['BodyText']
        )

        elements.append(definition_para)

        elements.append(Spacer(1, 8))

        example_text = (
            "No high-confidence example found."
        )

        for keyword in info["keywords"]:

            matches = high_conf_df[
                high_conf_df["bias_label"]
                .astype(str)
                .str.lower()
                .str.contains(keyword.lower(), na=False)
            ]

            if len(matches) > 0:

                best_match = matches.sort_values(
                    by="confidence",
                    ascending=False
                ).iloc[0]

                review = str(
                    best_match["sentence"]
                )[:500]

                confidence = float(
                    best_match["confidence"]
                )

                example_text = (
                    f"<b>Example Review:</b><br/>{review}"
                    f"<br/><br/>"
                    f"<b>Confidence Score:</b> "
                    f"{confidence:.2f}"
                )

                break

        example_para = Paragraph(
            example_text,
            styles['BodyText']
        )

        elements.append(example_para)

        elements.append(Spacer(1, 16))

    elements.append(PageBreak())

    # ------------------ GRAPHS ------------------ #

    graph_title = Paragraph(
        "<b>Visualizations</b>",
        styles['Heading2']
    )

    elements.append(graph_title)

    elements.append(Spacer(1, 20))

    images = [
        f"{movie_id}_comparison.png",
        f"{movie_id}_pie.png",
        f"{movie_id}_confidence.png",
        f"{movie_id}_top_biases.png"
    ]

    for img in images:

        elements.append(
            Image(
                img,
                width=500,
                height=300
            )
        )

        elements.append(Spacer(1, 20))

    doc.build(elements)

    print(
        f"\n📄 PDF report saved as "
        f"{movie_id}_report.pdf"
    )


# ------------------ MAIN PIPELINE ------------------ #


def run_pipeline():

    movie_id = input(
        "Enter movie ID (e.g., tt0111161): "
    )

    csv_file = f"{movie_id}_all_ratings_reviews.csv"

    # ------------------ LOAD EXISTING FILE ------------------ #

    if os.path.exists(csv_file):

        print(
            f"\n📂 Found existing review file: "
            f"{csv_file}"
        )

        df = pd.read_csv(csv_file)

    else:

        print("\n🔎 Scraping reviews...")

        df = scrape_imdb_reviews(movie_id)

        if df is None or len(df) == 0:

            print("❌ No data scraped")

            return

        # SAVE SCRAPED REVIEWS
        df.to_csv(
            csv_file,
            index=False
        )

        print(
            f"\n💾 Saved scraped reviews to "
            f"{csv_file}"
        )

    movie_name = (
        df["movie"].iloc[0]
        if "movie" in df.columns
        else movie_id
    )

    print(
        f"\n🎬 Running analysis for: "
        f"{movie_name}"
    )



    # LABELING
    labeled_df = analyze_and_save(
        df,
        movie_name,
        output_file=f"{movie_id}_labeled_reviews.csv"
    )

    # BUILD COUNTS
    bias_counts = {
        "positive": Counter(),
        "negative": Counter()
    }

    for _, row in labeled_df.iterrows():

        category = row["sentiment_category"]

        label = row["bias_label"]

        if category in bias_counts:

            bias_counts[category].update([label])

    # NORMALIZE
    pos = normalize(
        bias_counts["positive"]
    )

    neg = normalize(
        bias_counts["negative"]
    )

    # GRAPHS
    print("\n📊 Generating graphs...")

    plot_comparison(pos, neg, movie_id)

    plot_overall_pie(labeled_df, movie_id)

    plot_confidence_histogram(
        labeled_df,
        movie_id
    )

    plot_top_biases(
        labeled_df,
        movie_id
    )

    # INSIGHTS
    insights = {

        "Key Insights":
            print_insights(pos, neg),

        "Top Positive Biases":
            top_biases(pos, "POSITIVE"),

        "Top Negative Biases":
            top_biases(neg, "NEGATIVE"),

        "Contrast Insights":
            contrast_insights(pos, neg),

        "Confidence Insights":
            confidence_insights(labeled_df),

        "Statistical Significance":
            significance_test(pos, neg),

        "Human Readable Insights":
            human_insights(pos, neg)
    }

    # SAVE SUMMARY CSV
    summary = (
        labeled_df["bias_label"]
        .value_counts()
    )

    summary.to_csv(
        f"{movie_id}_bias_summary.csv"
    )

    print("\n💾 Saved summary CSV")

    # PDF
    print("\n📄 Generating PDF report...")

    generate_pdf(
        movie_name,
        movie_id,
        insights,
        labeled_df
    )

    print(
        "\n✅ Pipeline completed successfully!"
    )


# ------------------ RUN ------------------ #

if __name__ == "__main__":

    run_pipeline()
