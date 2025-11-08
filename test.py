"""
interactive_treatment_predictor.py
----------------------------------
Interactive script to test your trained medical treatment model.

Usage:
    python interactive_treatment_predictor.py
"""

import joblib
import json
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================================
# 1️⃣ Load trained artifacts
# ==========================================================
print("Loading trained model artifacts...")

cat_model = joblib.load("artifacts/category_model.pkl")
cat_vectorizer = joblib.load("artifacts/category_vectorizer.pkl")
retrieval_vectorizer = joblib.load("artifacts/retrieval_vectorizer.pkl")
label_vectors_med = joblib.load("artifacts/label_vectors_med.pkl")
label_vectors_proc = joblib.load("artifacts/label_vectors_proc.pkl")

with open("artifacts/label_metadata.json", "r", encoding="utf8") as f:
    label_metadata = json.load(f)

med_labels = label_metadata["med_labels"]
proc_labels = label_metadata["proc_labels"]
cat_to_meds = label_metadata["cat_to_meds"]
cat_to_procs = label_metadata["cat_to_procs"]

# Create index mappings
med_to_idx = {lab: i for i, lab in enumerate(med_labels)}
proc_to_idx = {lab: i for i, lab in enumerate(proc_labels)}

print(f"✅ Model and metadata loaded successfully!")
print(f"   {len(med_labels)} medications, {len(proc_labels)} procedures.\n")


# ==========================================================
# 2️⃣ Helper Functions
# ==========================================================
def normalize_text(text):
    """Basic text normalization."""
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^0-9a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ==========================================================
# 3️⃣ Prediction Function
# ==========================================================
def predict_treatment(symptom="", condition="", top_k=5, threshold=None, scope_to_category=True):
    """
    Predict treatment category and rank possible medications/procedures.
    """
    input_text = normalize_text(f"{symptom} {condition}")
    if not input_text.strip():
        print("⚠️ Please enter valid symptom and condition.")
        return None, None, [], []

    # 1️⃣ Predict Category
    x_cat = cat_vectorizer.transform([input_text])
    try:
        proba = cat_model.predict_proba(x_cat)[0]
        class_idx = np.argmax(proba)
        category = cat_model.classes_[class_idx]
        confidence = float(proba[class_idx])
    except Exception:
        category = cat_model.predict(x_cat)[0]
        confidence = None

    # 2️⃣ Retrieve ranked meds/procs
    q_vec = retrieval_vectorizer.transform([input_text])

    if scope_to_category:
        meds_pool = cat_to_meds.get(category, med_labels)
        procs_pool = cat_to_procs.get(category, proc_labels)
    else:
        meds_pool = med_labels
        procs_pool = proc_labels

    # Medications
    meds_out = []
    med_idx_list = [med_to_idx[m] for m in meds_pool if m in med_to_idx]
    if med_idx_list:
        sims = cosine_similarity(q_vec, label_vectors_med[med_idx_list])[0]
        meds_out = sorted([(meds_pool[i], float(s)) for i, s in enumerate(sims)],
                          key=lambda x: x[1], reverse=True)

    # Procedures
    procs_out = []
    proc_idx_list = [proc_to_idx[p] for p in procs_pool if p in proc_to_idx]
    if proc_idx_list:
        sims = cosine_similarity(q_vec, label_vectors_proc[proc_idx_list])[0]
        procs_out = sorted([(procs_pool[i], float(s)) for i, s in enumerate(sims)],
                           key=lambda x: x[1], reverse=True)

    # Filter or limit
    if threshold is not None:
        meds_out = [(l, s) for l, s in meds_out if s >= threshold]
        procs_out = [(l, s) for l, s in procs_out if s >= threshold]
    else:
        meds_out = meds_out[:top_k]
        procs_out = procs_out[:top_k]

    return category, confidence, meds_out, procs_out


# ==========================================================
#  Interactive User Input
# ==========================================================
if __name__ == "__main__":
    print(" Type your symptom and condition below (press Ctrl+C to exit).")

    while True:
        try:
            print("\n---------------------------------------------")
            symptom = input(" Enter Symptom(s): ").strip()
            condition = input(" Enter Condition: ").strip()

            if not symptom or not condition:
                print(" Both fields are required. Try again.")
                continue

            category, conf, meds, procs = predict_treatment(symptom, condition, top_k=5)

            print("\n=============================================")
            print(f" Predicted Treatment Category: {category}")
            if conf is not None:
                print(f"   (Confidence: {conf:.3f})")

            print("\n Recommended Medications:")
            if meds:
                for m, s in meds:
                    print(f"   • {m} (score={s:.3f})")
            else:
                print("   None found.")

            print("\n Recommended Procedures:")
            if procs:
                for p, s in procs:
                    print(f"   • {p} (score={s:.3f})")
            else:
                print("   None found.")

            print("=============================================")

        except KeyboardInterrupt:
            print("\n Exiting. Goodbye!")
            break
        except Exception as e:
            print(f" Error: {e}")
