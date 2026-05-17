import re
import json
import asyncio
import pandas as pd
import nltk
from tqdm import tqdm
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=key) #set key to your api key

CHECKPOINT_FILE = "checkpoint.json"
CACHE_FILE = "cache.json"

# ------------------ CLEAN ------------------ #

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ------------------ RULE-BASED FAST FILTER ------------------ #

def fast_rule_label(sentence):
    s = sentence.lower()

    if any(w in s for w in ["expected", "disappointed", "underwhelmed"]):
        return "Expectation Bias"

    if any(w in s for w in ["better than", "worse than", "compared to"]):
        return "Comparative Lens"

    if any(w in s for w in ["acted", "performance", "cast", "chemistry"]):
        return "Performance Appraisal"

    if any(w in s for w in ["realistic", "inaccurate", "believable"]):
        return "Authenticity Bias"

    if any(w in s for w in ["i love", "i hate", "not a fan"]):
        return "Disposition Bias"

    return None


# ------------------ CACHE ------------------ #

def load_cache():
    try:
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    except:
        return {}

def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)


# ------------------ LLM CALL ------------------ #

async def classify_sentence(sentence, cache, retries=2):
    if sentence in cache:
        return cache[sentence]

    prompt = f"""
Classify into ONE category:

1. Expectation Bias
2. Comparative Lens
3. Representation Critique
4. Omission Focus
5. Moral or Political Projection
6. Selective Emphasis Bias
7. Performance Appraisal
8. Authenticity Bias
9. Disposition Bias

Return ONLY:
{{"label": "..."}}

Sentence:
"{sentence}"
"""

    for attempt in range(retries):
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )

            content = response.choices[0].message.content.strip()

            match = re.search(r"\{.*\}", content)
            if match:
                label = json.loads(match.group()).get("label", "None")
                cache[sentence] = label
                return label

        except Exception as e:
            if "429" in str(e):
                wait = 2 ** attempt
                print(f"⏳ Rate limit. Waiting {wait}s...")
                await asyncio.sleep(wait)
            else:
                await asyncio.sleep(1)

    return "Error"


# ------------------ PROCESS REVIEW ------------------ #

async def process_review(review, cache, semaphore):
    text = clean_text(review)

    if not text:
        return []

    sentences = nltk.sent_tokenize(text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    labels = [None] * len(sentences)
    tasks = []
    indices = []

    for i, s in enumerate(sentences):
        rule_label = fast_rule_label(s)

        if rule_label:
            labels[i] = rule_label
        else:
            indices.append(i)

            async def task_wrapper(idx, sent):
                async with semaphore:
                    return idx, await classify_sentence(sent, cache)

            tasks.append(task_wrapper(i, s))

    if tasks:
        results = await asyncio.gather(*tasks)

        for idx, label in results:
            labels[idx] = label

    return labels


# ------------------ CHECKPOINT ------------------ #

def load_checkpoint():
    try:
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    except:
        return {}

def save_checkpoint(data):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f)


# ------------------ MAIN PIPELINE ------------------ #

async def annotate_with_checkpoint(df):
    checkpoint = load_checkpoint()
    cache = load_cache()

    semaphore = asyncio.Semaphore(5)

    async def safe_process(i, review):
        if str(i) in checkpoint:
            return checkpoint[str(i)]

        labels = await process_review(review, cache, semaphore)
        checkpoint[str(i)] = labels

        # periodic save
        if i % 50 == 0:
            save_checkpoint(checkpoint)
            save_cache(cache)

        return labels

    tasks = [safe_process(i, r) for i, r in enumerate(df["review"])]

    results = []
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        results.append(await f)

    # final save
    save_checkpoint(checkpoint)
    save_cache(cache)

    return results


# ------------------ RUN ------------------ #

df = pd.read_csv("full_reviews.csv")

df["review"] = df["review"].fillna("")
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

results = asyncio.run(annotate_with_checkpoint(df))

df["sentence_labels"] = results

# convert list → string
df["sentence_labels"] = df["sentence_labels"].apply(json.dumps)

df.to_csv("exfast_final.csv", index=False)

print("✅ DONE — optimized, fast, stable")