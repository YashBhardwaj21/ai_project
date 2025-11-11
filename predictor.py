#!/usr/bin/env python3
"""
interactive_treatment_predictor.py
Refined interactive script — corrected and hardened.

Keep it interactive:
    python interactive_treatment_predictor.py

One-shot CLI:
    python interactive_treatment_predictor.py --symptom "cough" --condition "bronchitis" --top_k 5
"""
import argparse
import joblib
import json
import numpy as np
import re
import sys
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------
# Utilities
# -----------------------
def normalize_text(text: str) -> str:
    """Lowercase, remove non-alphanum (except spaces), collapse whitespace."""
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^0-9a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def ensure_numpy(arr):
    """
    Convert scipy sparse matrix to a dense numpy array, or return numpy array unchanged.
    Use float32 to save memory where possible.
    """
    if arr is None:
        return None
    try:
        import scipy.sparse as sp
        if sp.issparse(arr):
            return arr.toarray().astype(np.float32)
    except Exception:
        pass
    # If it's already a numpy array or list-like
    arr_np = np.asarray(arr)
    # Ensure dtype numeric
    if np.issubdtype(arr_np.dtype, np.number):
        return arr_np.astype(np.float32)
    return arr_np

def safe_print(*args, **kwargs):
    print(*args, **kwargs, flush=True)

# -----------------------
# Load artifacts
# -----------------------
def load_artifacts(artifacts_dir="artifacts"):
    safe_print("Loading trained model artifacts...")
    try:
        cat_model = joblib.load(f"{artifacts_dir}/category_model.pkl")
        cat_vectorizer = joblib.load(f"{artifacts_dir}/category_vectorizer.pkl")
        retrieval_vectorizer = joblib.load(f"{artifacts_dir}/retrieval_vectorizer.pkl")
        label_vectors_med = joblib.load(f"{artifacts_dir}/label_vectors_med.pkl")
        label_vectors_proc = joblib.load(f"{artifacts_dir}/label_vectors_proc.pkl")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Artifact missing: {e}. Ensure '{artifacts_dir}' contains required files.") from e
    except Exception as e:
        raise RuntimeError(f"Failed to load artifacts: {e}") from e

    try:
        with open(f"{artifacts_dir}/label_metadata.json", "r", encoding="utf8") as f:
            label_metadata = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load label_metadata.json: {e}") from e

    med_labels = label_metadata.get("med_labels", [])
    proc_labels = label_metadata.get("proc_labels", [])
    cat_to_meds = label_metadata.get("cat_to_meds", {})
    cat_to_procs = label_metadata.get("cat_to_procs", {})

    # convert label vectors to numpy arrays for safe indexing and similarity computation
    label_vectors_med = ensure_numpy(label_vectors_med)
    label_vectors_proc = ensure_numpy(label_vectors_proc)

    # precompute label->index maps for reliable mapping (canonical global indices)
    med_to_idx = {lab: i for i, lab in enumerate(med_labels)}
    proc_to_idx = {lab: i for i, lab in enumerate(proc_labels)}

    # Basic validation
    if med_labels and (label_vectors_med is None or len(label_vectors_med) != len(med_labels)):
        safe_print("Warning: length mismatch med_labels vs label_vectors_med. Check artifacts.")
    if proc_labels and (label_vectors_proc is None or len(label_vectors_proc) != len(proc_labels)):
        safe_print("Warning: length mismatch proc_labels vs label_vectors_proc. Check artifacts.")

    return {
        "cat_model": cat_model,
        "cat_vectorizer": cat_vectorizer,
        "retrieval_vectorizer": retrieval_vectorizer,
        "label_vectors_med": label_vectors_med,
        "label_vectors_proc": label_vectors_proc,
        "med_labels": med_labels,
        "proc_labels": proc_labels,
        "med_to_idx": med_to_idx,
        "proc_to_idx": proc_to_idx,
        "cat_to_meds": cat_to_meds,
        "cat_to_procs": cat_to_procs,
    }

# -----------------------
# Predictor
# -----------------------
def predict_treatment(
    artifacts,
    symptom="",
    condition="",
    top_k=5,
    threshold=None,
    scope_to_category=True
):
    """
    Return (category, confidence, meds_out, procs_out)
    meds_out / procs_out: list of tuples (label, score)
    """
    cat_model = artifacts["cat_model"]
    cat_vectorizer = artifacts["cat_vectorizer"]
    retrieval_vectorizer = artifacts["retrieval_vectorizer"]
    label_vectors_med = artifacts["label_vectors_med"]
    label_vectors_proc = artifacts["label_vectors_proc"]
    med_labels = artifacts["med_labels"]
    proc_labels = artifacts["proc_labels"]
    med_to_idx = artifacts["med_to_idx"]
    proc_to_idx = artifacts["proc_to_idx"]
    cat_to_meds = artifacts["cat_to_meds"]
    cat_to_procs = artifacts["cat_to_procs"]

    input_text = normalize_text(f"{symptom} {condition}")
    if not input_text:
        raise ValueError("Empty symptom+condition after normalization.")

    # 1) category prediction
    x_cat = cat_vectorizer.transform([input_text])
    try:
        proba = cat_model.predict_proba(x_cat)[0]
        class_idx = int(np.argmax(proba))
        category = cat_model.classes_[class_idx]
        confidence = float(proba[class_idx])
    except Exception:
        # fallback if model doesn't support predict_proba
        category = cat_model.predict(x_cat)[0]
        confidence = None

    # 2) retrieval vector for input
    q_vec = retrieval_vectorizer.transform([input_text])
    q_vec = ensure_numpy(q_vec)  # shape (1, n_features)
    if q_vec is None:
        raise RuntimeError("Failed to vectorize query.")

    # choose pool
    if scope_to_category and (category in cat_to_meds or category in cat_to_procs):
        meds_pool = list(cat_to_meds.get(category, med_labels))
        procs_pool = list(cat_to_procs.get(category, proc_labels))
    else:
        meds_pool = list(med_labels)
        procs_pool = list(proc_labels)

    # Helper: rank candidates from a pool using precomputed global vectors and index map
    def rank_from_pool(pool_labels, global_vecs, label_to_idx):
        """
        pool_labels: list[str] — labels to consider
        global_vecs: np.ndarray shape (N_global, dim)
        label_to_idx: dict label->global_index
        returns sorted list[(label, score)] by descending score
        """
        if global_vecs is None or len(pool_labels) == 0:
            return []

        # map pool labels -> global indices
        idx_list = []
        labels_aligned = []
        for lab in pool_labels:
            idx = label_to_idx.get(lab)
            if idx is None:
                continue
            # guard index bounds
            if idx < 0 or idx >= len(global_vecs):
                continue
            idx_list.append(idx)
            labels_aligned.append(lab)

        if not idx_list:
            return []

        try:
            candidate_vecs = np.asarray(global_vecs)[idx_list]
        except Exception:
            # fallback: try indexing one-by-one
            candidate_vecs = np.stack([np.asarray(global_vecs)[i] for i in idx_list])

        # compute similarities
        try:
            sims = cosine_similarity(q_vec, candidate_vecs)[0]
        except ValueError:
            # dimension mismatch — return empty gracefully
            return []

        scored = list(zip(labels_aligned, sims.tolist()))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    meds_scored = rank_from_pool(meds_pool, label_vectors_med, med_to_idx)
    procs_scored = rank_from_pool(procs_pool, label_vectors_proc, proc_to_idx)

    # fallback lexical overlap if sims are zero or empty
    def lexical_overlap(q, label):
        qset = set(q.split())
        lset = set(label.split())
        if not lset:
            return 0.0
        inter = qset.intersection(lset)
        return len(inter) / float(len(lset))

    if (not meds_scored or all(s <= 1e-12 for _, s in meds_scored)) and meds_pool:
        # produce lexical fallback scores against label names
        fallback = [(l, lexical_overlap(input_text, l)) for l in meds_pool]
        fallback.sort(key=lambda x: x[1], reverse=True)
        meds_scored = fallback

    if (not procs_scored or all(s <= 1e-12 for _, s in procs_scored)) and procs_pool:
        fallback = [(l, lexical_overlap(input_text, l)) for l in procs_pool]
        fallback.sort(key=lambda x: x[1], reverse=True)
        procs_scored = fallback

    # apply threshold or top_k
    if threshold is not None:
        meds_out = [(l, s) for (l, s) in meds_scored if s >= threshold]
        procs_out = [(l, s) for (l, s) in procs_scored if s >= threshold]
    else:
        meds_out = meds_scored[:top_k]
        procs_out = procs_scored[:top_k]

    return category, confidence, meds_out, procs_out

# -----------------------
# Interactive / CLI
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Interactive treatment predictor")
    parser.add_argument("--symptom", type=str, help="Symptom text (one-shot)", default=None)
    parser.add_argument("--condition", type=str, help="Condition text (one-shot)", default=None)
    parser.add_argument("--top_k", type=int, help="Top K results", default=5)
    parser.add_argument("--threshold", type=float, help="Score threshold (0-1) to filter results", default=None)
    parser.add_argument("--no_scope", action="store_true", help="Do not scope retrieval to predicted category")
    parser.add_argument("--artifacts_dir", type=str, help="Directory containing artifacts", default="artifacts")
    args = parser.parse_args()

    try:
        artifacts = load_artifacts(args.artifacts_dir)
    except Exception as e:
        safe_print(f"Error loading artifacts: {e}")
        sys.exit(1)

    safe_print("✅ Model and metadata loaded successfully!")
    safe_print(f"   {len(artifacts['med_labels'])} medications, {len(artifacts['proc_labels'])} procedures.\n")

    # one-shot CLI mode: require both symptom and condition to run one-shot
    if args.symptom is not None and args.condition is not None:
        symptom = args.symptom or ""
        condition = args.condition or ""
        try:
            cat, conf, meds, procs = predict_treatment(
                artifacts,
                symptom=symptom,
                condition=condition,
                top_k=args.top_k,
                threshold=args.threshold,
                scope_to_category=not args.no_scope
            )
            safe_print_result(cat, conf, meds, procs)
        except Exception as e:
            safe_print(f"Error during prediction: {e}")
        return

    # interactive loop
    safe_print(" Type your symptom and condition below (press Ctrl+C to exit).")
    while True:
        try:
            safe_print("\n---------------------------------------------")
            symptom = input(" Enter Symptom(s): ").strip()
            condition = input(" Enter Condition: ").strip()

            if not symptom or not condition:
                safe_print(" Both fields are required. Try again.")
                continue

            cat, conf, meds, procs = predict_treatment(
                artifacts,
                symptom=symptom,
                condition=condition,
                top_k=args.top_k,
                threshold=args.threshold,
                scope_to_category=not args.no_scope
            )
            safe_print_result(cat, conf, meds, procs)

        except KeyboardInterrupt:
            safe_print("\n Exiting. Goodbye!")
            break
        except Exception as e:
            safe_print(f" Error: {e}")

def safe_print_result(category, confidence, meds, procs):
    safe_print("\n=============================================")
    safe_print(f" Predicted Treatment Category: {category}")
    if confidence is not None:
        safe_print(f"   (Confidence: {confidence:.3f})")
    safe_print("\n Recommended Medications:")
    if meds:
        for m, s in meds:
            safe_print(f"   • {m} (score={s:.4f})")
    else:
        safe_print("   None found.")
    safe_print("\n Recommended Procedures:")
    if procs:
        for p, s in procs:
            safe_print(f"   • {p} (score={s:.4f})")
    else:
        safe_print("   None found.")
    safe_print("=============================================")

if __name__ == "__main__":
    main()
