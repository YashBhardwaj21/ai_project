"""
train_and_build_retrieval_optimized.py

Optimized training + retrieval build script.

Usage:
    python train_and_build_retrieval_optimized.py
"""

import json
import os
import re
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from tqdm import tqdm

# -------------------------
# Config
# -------------------------
INPUT_CSV = r"C:\Users\Yash Bhardwaj\Desktop\ai_project\medical_data_model_ready.csv"
OUT_DIR = "./artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

CATEGORY_MODEL_PATH = os.path.join(OUT_DIR, "category_model.pkl")
CATEGORY_VECTORIZER_PATH = os.path.join(OUT_DIR, "category_vectorizer.pkl")
RETRIEVAL_VECTORIZER_PATH = os.path.join(OUT_DIR, "retrieval_vectorizer.pkl")
LABEL_VECTORS_MED_PATH = os.path.join(OUT_DIR, "label_vectors_med.pkl")
LABEL_VECTORS_PROC_PATH = os.path.join(OUT_DIR, "label_vectors_proc.pkl")
LABEL_METADATA_PATH = os.path.join(OUT_DIR, "label_metadata.json")

RANDOM_STATE = 42
TEST_SIZE = 0.2
TOP_K_DEFAULT = 5
LABEL_CONTEXTS_PER_LABEL = 8   # how many example contexts to attach to each label
MIN_DF_CAT = 2                # cat vectorizer min_df
MIN_DF_RETRIEVAL = 1          # retrieval vectorizer min_df

# -------------------------
# Utilities
# -------------------------
def normalize_text(text):
    if pd.isna(text) or text is None:
        return ""
    s = str(text).lower()
    s = re.sub(r"[^0-9a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def split_labels(cell):
    if pd.isna(cell) or str(cell).strip() == "":
        return []
    parts = [p.strip() for p in re.split(r";\s*", str(cell)) if p.strip()]
    return [normalize_text(p) for p in parts]

def lexical_overlap_score(query, label_text):
    """Simple fallback lexical overlap ratio (0..1)."""
    q_tokens = set(normalize_text(query).split())
    l_tokens = set(normalize_text(label_text).split())
    if not q_tokens or not l_tokens:
        return 0.0
    inter = q_tokens.intersection(l_tokens)
    # Jaccard-like: intersection / (|label_tokens|) to prefer label coverage
    return len(inter) / max(1, len(l_tokens))

# -------------------------
# 1) Load dataset
# -------------------------
print("Loading:", INPUT_CSV)
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"{INPUT_CSV} not found. Run the prepare script first.")

df = pd.read_csv(INPUT_CSV, engine="python", on_bad_lines="skip")
print("Rows (cases):", len(df))
print("Columns:", df.columns.tolist())

required_cols = ["Symptom_Text", "Condition", "Medication_List", "Procedure_List", "Treatment_Category"]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Required column '{c}' not found in {INPUT_CSV}")

# Normalize and prepare input_text
df["Symptom_Text_n"] = df["Symptom_Text"].fillna("").astype(str).apply(normalize_text)
df["Condition_n"] = df["Condition"].fillna("").astype(str).apply(normalize_text)
df["input_text"] = (df["Symptom_Text_n"] + " " + df["Condition_n"]).str.strip()

df["Medication_list_parsed"] = df["Medication_List"].apply(split_labels)
df["Procedure_list_parsed"] = df["Procedure_List"].apply(split_labels)

# -------------------------
# 2) Category model (unchanged core but re-check merge step if you used it)
# -------------------------
print("Preparing category training data...")
X_texts = df["input_text"].fillna("").values
y = df["Treatment_Category"].fillna("Other / Needs Review").values

print("Category distribution (top 30):")
print(pd.Series(y).value_counts().head(30))

print("Vectorizing for category model (TF-IDF ngram 1-2)...")
cat_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=MIN_DF_CAT)
X_vec = cat_vectorizer.fit_transform(X_texts)

# Stratified split; if failing because of singletons, merge rare classes outside this script (or set min_count lower)
try:
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_vec, y, np.arange(len(y)), test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print("Stratified split OK.")
except ValueError:
    # fallback: non-stratified split but warn
    print("Stratified split failed (rare classes present). Falling back to non-stratified split.")
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_vec, y, np.arange(len(y)), test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=None
    )

print("Training category classifier (LogisticRegression, class_weight='balanced')...")
category_model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE, solver="saga", multi_class="multinomial")
category_model.fit(X_train, y_train)

y_pred = category_model.predict(X_test)
print("\n=== Category classification report ===")
print(classification_report(y_test, y_pred, zero_division=0))

# Save category artifacts
joblib.dump(category_model, CATEGORY_MODEL_PATH)
joblib.dump(cat_vectorizer, CATEGORY_VECTORIZER_PATH)
print("Saved category model & vectorizer to", OUT_DIR)

# -------------------------
# 3) Build enriched label texts for retrieval
# -------------------------
print("Building enriched label texts with contexts (this improves retrieval)...")

# gather contexts for each label
label_contexts = defaultdict(list)
for _, row in df.iterrows():
    context = " ".join([row["Symptom_Text_n"], row["Condition_n"]]).strip()
    for m in row["Medication_list_parsed"]:
        if m:
            label_contexts[("med", m)].append(context)
    for p in row["Procedure_list_parsed"]:
        if p:
            label_contexts[("proc", p)].append(context)

# dedupe contexts and limit to LABEL_CONTEXTS_PER_LABEL
def build_label_text(label_type, label):
    contexts = label_contexts.get((label_type, label), [])[:LABEL_CONTEXTS_PER_LABEL]
    seen = set()
    uniq_contexts = []
    for c in contexts:
        if c and c not in seen:
            uniq_contexts.append(c)
            seen.add(c)
    # label + contexts (label first helps lexical matching)
    pieces = [label] + uniq_contexts
    return " ".join(pieces)

# build sets of labels
cat_to_meds = defaultdict(list)
cat_to_procs = defaultdict(list)
all_med_labels = set()
all_proc_labels = set()

for _, row in df.iterrows():
    cat = row["Treatment_Category"]
    meds = row["Medication_list_parsed"]
    procs = row["Procedure_list_parsed"]
    for m in meds:
        all_med_labels.add(m); cat_to_meds[cat].append(m)
    for p in procs:
        all_proc_labels.add(p); cat_to_procs[cat].append(p)

med_labels_sorted = sorted(all_med_labels)
proc_labels_sorted = sorted(all_proc_labels)

# create enriched label_texts
label_texts_meds = [build_label_text("med", l) for l in med_labels_sorted]
label_texts_procs = [build_label_text("proc", l) for l in proc_labels_sorted]

print(f"Unique meds: {len(med_labels_sorted)}, procs: {len(proc_labels_sorted)}")

# -------------------------
# 4) Fit retrieval vectorizer (improved params)
# -------------------------
print("Fitting retrieval TF-IDF vectorizer (ngram 1-3, sublinear_tf, smooth_idf)...")
retrieval_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,3),
                                       min_df=MIN_DF_RETRIEVAL, sublinear_tf=True, smooth_idf=True)
combined_corpus = list(df["input_text"].astype(str).values) + label_texts_meds + label_texts_procs
retrieval_vectorizer.fit(combined_corpus)

label_vectors_med = retrieval_vectorizer.transform(label_texts_meds)
label_vectors_proc = retrieval_vectorizer.transform(label_texts_procs)

# Save retrieval artifacts
joblib.dump(retrieval_vectorizer, RETRIEVAL_VECTORIZER_PATH)
joblib.dump(label_vectors_med, LABEL_VECTORS_MED_PATH)
joblib.dump(label_vectors_proc, LABEL_VECTORS_PROC_PATH)

label_metadata = {
    "med_labels": med_labels_sorted,
    "proc_labels": proc_labels_sorted,
    "cat_to_meds": {k: sorted(list(set(v))) for k, v in cat_to_meds.items()},
    "cat_to_procs": {k: sorted(list(set(v))) for k, v in cat_to_procs.items()}
}
with open(LABEL_METADATA_PATH, "w", encoding="utf8") as f:
    json.dump(label_metadata, f, ensure_ascii=False, indent=2)

print("Saved retrieval artifacts & metadata to", OUT_DIR)

# -------------------------
# 5) Retrieval evaluation (scoped to category) - better realism
# -------------------------
def evaluate_retrieval_scoped(idx_test, top_k_values=(1,3,5)):
    print("\nEvaluating retrieval (SCOPED to true category) on test set...")
    med_labels = label_metadata["med_labels"]
    proc_labels = label_metadata["proc_labels"]
    med_index = {lab:i for i, lab in enumerate(med_labels)}
    proc_index = {lab:i for i, lab in enumerate(proc_labels)}

    results = {k: {"med_hits":0, "proc_hits":0, "total":0, "med_mrr":0.0, "proc_mrr":0.0} for k in top_k_values}

    for i in tqdm(idx_test, desc="Retrieval eval"):
        row = df.iloc[i]
        q_text = row["input_text"]
        true_meds = row["Medication_list_parsed"]
        true_procs = row["Procedure_list_parsed"]
        if not true_meds and not true_procs:
            continue

        q_vec = retrieval_vectorizer.transform([q_text])

        # scope to true category (realistic)
        cat = row["Treatment_Category"]
        med_pool = label_metadata["cat_to_meds"].get(cat, med_labels)
        proc_pool = label_metadata["cat_to_procs"].get(cat, proc_labels)

        # med submatrix
        med_idx_list = [med_index[l] for l in med_pool if l in med_index]
        proc_idx_list = [proc_index[l] for l in proc_pool if l in proc_index]

        med_sims = np.array([]); proc_sims = np.array([])
        if med_idx_list:
            med_sims = cosine_similarity(q_vec, label_vectors_med[med_idx_list])[0]
            med_labels_here = med_pool
            med_ranked_idx = np.argsort(med_sims)[::-1]
        else:
            med_ranked_idx = []

        if proc_idx_list:
            proc_sims = cosine_similarity(q_vec, label_vectors_proc[proc_idx_list])[0]
            proc_labels_here = proc_pool
            proc_ranked_idx = np.argsort(proc_sims)[::-1]
        else:
            proc_ranked_idx = []

        for k in top_k_values:
            # meds
            med_hit=False; med_rr=0.0
            for rank, idx_lab in enumerate(med_ranked_idx[:k], start=1):
                lab = med_labels_here[idx_lab]
                if lab in true_meds:
                    med_hit=True; med_rr = 1.0/rank; break
            if med_hit: results[k]["med_hits"] += 1
            results[k]["med_mrr"] += med_rr

            # procs
            proc_hit=False; proc_rr=0.0
            for rank, idx_lab in enumerate(proc_ranked_idx[:k], start=1):
                lab = proc_labels_here[idx_lab]
                if lab in true_procs:
                    proc_hit=True; proc_rr = 1.0/rank; break
            if proc_hit: results[k]["proc_hits"] += 1
            results[k]["proc_mrr"] += proc_rr

            results[k]["total"] += 1

    # aggregate results
    for k in top_k_values:
        if results[k]["total"] == 0:
            continue
        med_recall = results[k]["med_hits"] / results[k]["total"]
        proc_recall = results[k]["proc_hits"] / results[k]["total"]
        med_mrr = results[k]["med_mrr"] / results[k]["total"]
        proc_mrr = results[k]["proc_mrr"] / results[k]["total"]
        print(f"Top-{k}: Med Recall@{k}={med_recall:.3f}, Proc Recall@{k}={proc_recall:.3f}, Med MRR={med_mrr:.3f}, Proc MRR={proc_mrr:.3f}")

try:
    evaluate_retrieval_scoped(idx_test, top_k_values=(1,3,5))
except Exception as e:
    print("Scoped retrieval evaluation skipped due to:", e)

# -------------------------
# 6) Improved inference function
# -------------------------
med_to_idx = {lab:i for i, lab in enumerate(med_labels_sorted)}
proc_to_idx = {lab:i for i, lab in enumerate(proc_labels_sorted)}

def predict_treatment(symptom="", condition="", top_k=TOP_K_DEFAULT, threshold=None, scope_to_category=True, boost_with_category_confidence=True):
    """
    Returns:
      pred_category, category_confidence, meds_out, procs_out
    meds_out/procs_out: list of (label, final_score)
    """
    input_text = normalize_text(" ".join([str(symptom), str(condition)]))
    if not input_text:
        return None, None, [], []

    # category prediction + confidence
    x_cat = cat_vectorizer.transform([input_text])
    try:
        proba = category_model.predict_proba(x_cat)[0]
        class_idx = np.argmax(proba)
        pred_category = category_model.classes_[class_idx]
        cat_conf = float(proba[class_idx])
    except Exception:
        pred_category = category_model.predict(x_cat)[0]
        cat_conf = 1.0

    # build query vector
    q_vec = retrieval_vectorizer.transform([input_text])

    # candidate pools
    if scope_to_category:
        meds_pool = label_metadata["cat_to_meds"].get(pred_category, med_labels_sorted)
        procs_pool = label_metadata["cat_to_procs"].get(pred_category, proc_labels_sorted)
    else:
        meds_pool = med_labels_sorted
        procs_pool = proc_labels_sorted

    # compute med sims
    meds_out = []
    med_idx_list = [med_to_idx[l] for l in meds_pool if l in med_to_idx]
    if med_idx_list:
        med_mat_sub = label_vectors_med[med_idx_list]
        sims_med = cosine_similarity(q_vec, med_mat_sub)[0]
        # combine sim with category confidence (simple linear blend)
        if boost_with_category_confidence:
            final_scores_med = [(meds_pool[i], float(s) * (0.5 + 0.5 * cat_conf)) for i, s in enumerate(sims_med)]
        else:
            final_scores_med = [(meds_pool[i], float(s)) for i, s in enumerate(sims_med)]
        meds_out = sorted(final_scores_med, key=lambda x: x[1], reverse=True)

    # compute proc sims
    procs_out = []
    proc_idx_list = [proc_to_idx[l] for l in procs_pool if l in proc_to_idx]
    if proc_idx_list:
        proc_mat_sub = label_vectors_proc[proc_idx_list]
        sims_proc = cosine_similarity(q_vec, proc_mat_sub)[0]
        if boost_with_category_confidence:
            final_scores_proc = [(procs_pool[i], float(s) * (0.5 + 0.5 * cat_conf)) for i, s in enumerate(sims_proc)]
        else:
            final_scores_proc = [(procs_pool[i], float(s)) for i, s in enumerate(sims_proc)]
        procs_out = sorted(final_scores_proc, key=lambda x: x[1], reverse=True)

    # If all sims are zero (rare), fallback to lexical overlap score using enriched label_texts
    if (not meds_out or all(s <= 1e-12 for _, s in meds_out)) and meds_pool:
        fallback = [(l, lexical_overlap_score(input_text, build_label_text("med", l))) for l in meds_pool]
        fallback = sorted(fallback, key=lambda x: x[1], reverse=True)
        meds_out = fallback

    if (not procs_out or all(s <= 1e-12 for _, s in procs_out)) and procs_pool:
        fallback = [(l, lexical_overlap_score(input_text, build_label_text("proc", l))) for l in procs_pool]
        fallback = sorted(fallback, key=lambda x: x[1], reverse=True)
        procs_out = fallback

    # thresholding / top_k
    if threshold is not None:
        meds_out = [(l, s) for (l, s) in meds_out if s >= threshold]
        procs_out = [(l, s) for (l, s) in procs_out if s >= threshold]
    else:
        meds_out = meds_out[:top_k]
        procs_out = procs_out[:top_k]

    return pred_category, cat_conf, meds_out, procs_out

# -------------------------
# 7) Quick interactive demo if run directly
# -------------------------
if __name__ == "__main__":
    tests = [
        ("Throbbing headache; photophobia; nausea", "Migraine without Aura"),
        ("Loud snoring and daytime sleepiness", "Obstructive sleep apnea"),
        ("Chest pain radiating to left arm", "Acute Myocardial Infarction"),
        ("Arachnodactyly", "Marfan syndrome"),
    ]
    for symp, cond in tests:
        cat, conf, meds, procs = predict_treatment(symp, cond, top_k=8)
        print("\nInput:", symp, "|", cond)
        print(" Predicted Category:", cat, f"(conf={conf:.3f})")
        print(" Top Medications:", meds[:8])
        print(" Top Procedures:", procs[:8])

    print("\nDone. Artifacts saved in:", OUT_DIR)
