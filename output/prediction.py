"""
Retrieval-based treatment / procedure suggestion script.

- Reads medical_data.csv
- Normalizes text and label names (lowercase, remove punctuation)
- Builds a TF-IDF space from both patient input texts and label description texts
- Precomputes TF-IDF vectors for each unique medication / procedure label
- Predicts top-K matching medications and procedures for a given (symptom, condition) pair
- Saves vectorizer and label metadata for later use

No pretrained language models, no fine-tuning. Works even if labels appear once.
"""

import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from collections import defaultdict

# ---------- Config ----------
CSV_PATH = "medical_data.csv"
VECTORIZER_PATH = "retrieval_vectorizer.pkl"
LABEL_META_PATH = "label_metadata.pkl"  # contains dict with meds/procs and their texts
TOP_K_DEFAULT = 5

# ---------- Helpers ----------
def normalize_text(text):
    """Lowercase, remove non-alphanumeric (keep spaces), collapse whitespace."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # keep unicode letters/digits and spaces
    text = re.sub(r"[^0-9a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def split_labels(cell):
    """Split semicolon-separated label cell into list of normalized labels."""
    if pd.isna(cell):
        return []
    # split by ';' or '; ' and strip
    parts = [p.strip() for p in re.split(r";\s*", str(cell)) if p.strip()]
    return [normalize_text(p) for p in parts]

# ---------- Load data ----------
print("Loading CSV...")
df = pd.read_csv(CSV_PATH, engine="python", on_bad_lines="skip")
print(f"Loaded {len(df)} rows.")

# Ensure expected columns exist; adapt gracefully if not
for col in ["Symptom", "Condition", "Medication", "Procedure"]:
    if col not in df.columns:
        df[col] = ""

# Optional extra columns we can append to label descriptions (if present)
extra_label_cols = []
for cand in ["Action", "Plan", "Notes", "ICD10", "ICD_10", "Description"]:
    if cand in df.columns:
        extra_label_cols.append(cand)

# ---------- Build label -> description map ----------
# We'll create descriptive text for every unique label by taking label name
# + concatenating any extra metadata we can find in rows where that label appears.
med_to_rows = defaultdict(list)
proc_to_rows = defaultdict(list)

print("Collecting labels and building descriptions...")

for idx, row in df.iterrows():
    symptom = normalize_text(row.get("Symptom", ""))
    condition = normalize_text(row.get("Condition", ""))
    extras = " ".join(normalize_text(row.get(c, "")) for c in extra_label_cols).strip()
    context_text = " ".join([symptom, condition, extras]).strip()

    meds = split_labels(row.get("Medication", ""))
    procs = split_labels(row.get("Procedure", ""))

    for m in meds:
        if m:
            med_to_rows[m].append(context_text)

    for p in procs:
        if p:
            proc_to_rows[p].append(context_text)

# Create label texts: label name + (first N context snippets joined)
def make_label_text(name, contexts, max_context_chars=300):
    # join some unique contexts (avoid duplicates)
    contexts = [c for c in contexts if c]
    uniq = []
    seen = set()
    for c in contexts:
        if c and c not in seen:
            seen.add(c)
            uniq.append(c)
        if len(" ".join(uniq)) > max_context_chars:
            break
    # label name + contexts
    return " ".join([name] + uniq)

med_labels = sorted(med_to_rows.keys())
proc_labels = sorted(proc_to_rows.keys())

label_texts = []   # combined list: meds then procs
label_types = []   # 'med' or 'proc' corresponding to label_texts
label_names = []   # actual label names

for m in med_labels:
    label_texts.append(make_label_text(m, med_to_rows[m]))
    label_types.append("med")
    label_names.append(m)

for p in proc_labels:
    label_texts.append(make_label_text(p, proc_to_rows[p]))
    label_types.append("proc")
    label_names.append(p)

print(f"Unique medications: {len(med_labels)}, procedures: {len(proc_labels)}")
print(f"Total label entries: {len(label_texts)}")

# ---------- Build TF-IDF vectorizer ----------
# IMPORTANT: Fit on both patient texts and label texts so they share the same vector space.
print("Preparing corpus to fit TF-IDF vectorizer...")
patient_texts = []
for idx, row in df.iterrows():
    t = " ".join([normalize_text(row.get("Symptom", "")), normalize_text(row.get("Condition", ""))]).strip()
    patient_texts.append(t)

# Combine corpora
corpus_for_vectorizer = patient_texts + label_texts

print("Fitting TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=1,       # keep rare terms (we need them for rare labels)
    max_df=0.95,
)

vectorizer.fit(corpus_for_vectorizer)

# Transform label_texts into label vectors
print("Vectorizing label texts...")
label_vectors = vectorizer.transform(label_texts)  # shape: (num_labels, n_features)

# Save all metadata so you can load and predict later
label_metadata = {
    "label_names": label_names,
    "label_types": label_types,
    "label_texts": label_texts
}

joblib.dump(vectorizer, VECTORIZER_PATH)
joblib.dump(label_metadata, LABEL_META_PATH)
joblib.dump(label_vectors, "label_vectors.pkl")  # sparse matrix can be saved by joblib
print("Saved vectorizer, label metadata, and label vectors.")

# ---------- Prediction function ----------
def predict_treatment(symptom="", condition="", top_k=TOP_K_DEFAULT, threshold=None):
    """
    Return top-k medication and procedure matches with similarity scores.
    - symptom, condition: strings
    - top_k: number of overall top matches to return (will be split into meds/procs)
    - threshold: optional similarity threshold (0..1). If None, top_k is used.
    Returns:
        meds: list of (label, score)
        procs: list of (label, score)
    """
    input_text = " ".join([normalize_text(symptom), normalize_text(condition)]).strip()
    if not input_text:
        return [], []

    input_vec = vectorizer.transform([input_text])  # shape (1, n_features)
    sims = cosine_similarity(input_vec, label_vectors)[0]  # array of length num_labels

    # get indices sorted by similarity desc
    sorted_idx = np.argsort(sims)[::-1]

    results = []
    for idx in sorted_idx:
        score = float(sims[idx])
        if threshold is not None and score < threshold:
            break
        results.append((label_names[idx], label_types[idx], score))
        if threshold is None and len(results) >= top_k:
            break

    # Split into meds and procs preserving order
    meds = [(name, score) for (name, typ, score) in results if typ == "med"]
    procs = [(name, score) for (name, typ, score) in results if typ == "proc"]

    return meds, procs

# ---------- Example usage / quick interactive test ----------
if __name__ == "__main__":
    # Simple test
    print("\n--- TEST PREDICTIONS ---")
    tests = [
        ("Chest pain", "Myocardial infarction"),
        ("Loud snoring and daytime sleepiness", "Obstructive sleep apnea"),
        ("Nasal congestion with PAP use", "PAP-related rhinitis"),
    ]

    for symp, cond in tests:
        meds, procs = predict_treatment(symp, cond, top_k=8, threshold=None)
        print("\nInput:", symp, "|", cond)
        print("Top medication matches (label, score):")
        for m, s in meds[:8]:
            print(f"  - {m}  ({s:.3f})")
        print("Top procedure matches (label, score):")
        for p, s in procs[:8]:
            print(f"  - {p}  ({s:.3f})")

    # Save a small JSON/CSV of label metadata plus top example tokens (optional)
    # This is helpful for debugging which tokens drive similarity.
    try:
        import json
        with open("label_metadata.json", "w", encoding="utf8") as f:
            json.dump(label_metadata, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    print("\nDone. Use predict_treatment(symptom, condition, top_k, threshold) to get matches.")
