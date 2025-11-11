
"""
evaluate_medical_model.py

Evaluation script for hybrid medical treatment prediction system:
    1️ Logistic Regression classifier (category)
    2️ TF-IDF + Cosine Similarity retriever (medications & procedures)

Evaluates:
 - Category accuracy, precision, recall, F1
 - Retrieval Recall@K and MRR for meds and procs
 - Includes quick manual test at the end
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# ------------------------------
# CONFIG
# ------------------------------
DATASET = r"C:\Users\Yash Bhardwaj\Desktop\ai_project\medical_data_model_ready.csv"
ARTIFACTS_DIR = r"C:\Users\Yash Bhardwaj\Desktop\ai_project\artifacts"
TOP_K_VALUES = (1, 3, 5)

# ------------------------------
# LOAD MODELS & DATA
# ------------------------------
print(" Loading artifacts...")
cat_model = joblib.load(os.path.join(ARTIFACTS_DIR, "category_model.pkl"))
cat_vectorizer = joblib.load(os.path.join(ARTIFACTS_DIR, "category_vectorizer.pkl"))
retrieval_vectorizer = joblib.load(os.path.join(ARTIFACTS_DIR, "retrieval_vectorizer.pkl"))
label_vectors_med = joblib.load(os.path.join(ARTIFACTS_DIR, "label_vectors_med.pkl"))
label_vectors_proc = joblib.load(os.path.join(ARTIFACTS_DIR, "label_vectors_proc.pkl"))

with open(os.path.join(ARTIFACTS_DIR, "label_metadata.json"), "r", encoding="utf8") as f:
    label_metadata = json.load(f)

print(" Artifacts loaded successfully.")

# ------------------------------
# LOAD DATASET
# ------------------------------
print(f" Loading dataset from: {DATASET}")
df = pd.read_csv(DATASET, engine="python", on_bad_lines="skip")
print(f" Loaded {len(df)} rows and {len(df.columns)} columns.")

# ------------------------------
# CATEGORY PREDICTION EVALUATION
# ------------------------------
print("\n==============================")
print(" CATEGORY CLASSIFIER EVALUATION")
print("==============================")

df["input_text"] = (df["Symptom_Text"].astype(str) + " " + df["Condition"].astype(str)).str.strip()
X_text = df["input_text"].values
y_true = df["Treatment_Category"].values

X_vec = cat_vectorizer.transform(X_text)
y_pred = cat_model.predict(X_vec)

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("\nDetailed Report:")
print(classification_report(y_true, y_pred, zero_division=0, digits=3))

# ------------------------------
# RETRIEVAL EVALUATION (MEDS + PROCS)
# ------------------------------
print("\n==============================")
print(" RETRIEVAL MODEL EVALUATION")
print("==============================")

med_labels = label_metadata["med_labels"]
proc_labels = label_metadata["proc_labels"]
cat_to_meds = label_metadata["cat_to_meds"]
cat_to_procs = label_metadata["cat_to_procs"]

label_to_med_idx = {m: i for i, m in enumerate(med_labels)}
label_to_proc_idx = {p: i for i, p in enumerate(proc_labels)}

def evaluate_retrieval(df, top_k_values=TOP_K_VALUES):
    results = {k: {"med_hits": 0, "proc_hits": 0, "total": 0, "med_mrr_sum": 0.0, "proc_mrr_sum": 0.0} for k in top_k_values}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating retrieval"):
        text = str(row["input_text"])
        cat = row["Treatment_Category"]
        true_meds = [m.strip().lower() for m in str(row["Medication_List"]).split(";") if m.strip()]
        true_procs = [p.strip().lower() for p in str(row["Procedure_List"]).split(";") if p.strip()]

        if not text or (not true_meds and not true_procs):
            continue

        q_vec = retrieval_vectorizer.transform([text])
        med_idx_pool = [label_to_med_idx[m] for m in cat_to_meds.get(cat, []) if m in label_to_med_idx]
        proc_idx_pool = [label_to_proc_idx[p] for p in cat_to_procs.get(cat, []) if p in label_to_proc_idx]

        if not med_idx_pool and not proc_idx_pool:
            continue

        med_sim = cosine_similarity(q_vec, label_vectors_med[med_idx_pool])[0] if med_idx_pool else []
        proc_sim = cosine_similarity(q_vec, label_vectors_proc[proc_idx_pool])[0] if proc_idx_pool else []

        def compute_hits_and_mrr(true_list, sims, labels, k):
            sorted_idx = np.argsort(sims)[::-1]
            top_labels = [labels[i] for i in sorted_idx]
            hit = any(l in top_labels[:k] for l in true_list)
            mrr = 0.0
            for rank, lab in enumerate(top_labels[:k], start=1):
                if lab in true_list:
                    mrr = 1.0 / rank
                    break
            return hit, mrr

        for k in top_k_values:
            if len(med_sim) > 0:
                hit, mrr = compute_hits_and_mrr(true_meds, med_sim, [med_labels[i] for i in med_idx_pool], k)
                results[k]["med_hits"] += int(hit)
                results[k]["med_mrr_sum"] += mrr
            if len(proc_sim) > 0:
                hit, mrr = compute_hits_and_mrr(true_procs, proc_sim, [proc_labels[i] for i in proc_idx_pool], k)
                results[k]["proc_hits"] += int(hit)
                results[k]["proc_mrr_sum"] += mrr
            results[k]["total"] += 1

    print("\n=== Retrieval Performance ===")
    for k in top_k_values:
        total = results[k]["total"]
        if total == 0:
            continue
        med_recall = results[k]["med_hits"] / total
        proc_recall = results[k]["proc_hits"] / total
        med_mrr = results[k]["med_mrr_sum"] / total
        proc_mrr = results[k]["proc_mrr_sum"] / total
        print(f"Top-{k}:  Med Recall@{k} = {med_recall:.3f}, Proc Recall@{k} = {proc_recall:.3f}, Med MRR = {med_mrr:.3f}, Proc MRR = {proc_mrr:.3f}")

# Run retrieval eval
evaluate_retrieval(df)

# ------------------------------
# MANUAL TEST
# ------------------------------
print("\n==============================")
print(" MANUAL TEST SAMPLE")
print("==============================")

def predict_treatment(symptom, condition, top_k=5):
    text = f"{symptom} {condition}".strip().lower()
    x_cat = cat_vectorizer.transform([text])
    cat_pred = cat_model.predict(x_cat)[0]

    q_vec = retrieval_vectorizer.transform([text])
    meds = cat_to_meds.get(cat_pred, med_labels)
    procs = cat_to_procs.get(cat_pred, proc_labels)

    med_idx = [label_to_med_idx[m] for m in meds if m in label_to_med_idx]
    proc_idx = [label_to_proc_idx[p] for p in procs if p in label_to_proc_idx]

    sims_med = cosine_similarity(q_vec, label_vectors_med[med_idx])[0] if med_idx else []
    sims_proc = cosine_similarity(q_vec, label_vectors_proc[proc_idx])[0] if proc_idx else []

    meds_top = sorted(zip([meds[i] for i in range(len(sims_med))], sims_med), key=lambda x: x[1], reverse=True)[:top_k]
    procs_top = sorted(zip([procs[i] for i in range(len(sims_proc))], sims_proc), key=lambda x: x[1], reverse=True)[:top_k]

    print(f"\n Input: {symptom} | {condition}")
    print(f" Predicted Category: {cat_pred}")
    print(" Medications:")
    for m, s in meds_top:
        print(f"   • {m} (score={s:.3f})")
    print(" Procedures:")
    for p, s in procs_top:
        print(f"   • {p} (score={s:.3f})")

predict_treatment("chest pain", "acute myocardial infarction")
