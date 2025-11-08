"""
prepare_medical_data.py

Usage:
    python prepare_medical_data.py

Input:
    ./medical_data.csv   (your raw file)

Outputs (in same folder):
    - medical_data_grouped.csv            (intermediate grouped by patient)
    - category_suggestions.csv            (editable list of unique conditions -> suggested category)
    - medical_data_model_ready.csv        (final ML-ready dataset with Treatment_Category)
    - unique_medications.csv              (all unique medications)
    - unique_procedures.csv               (all unique procedures)
    - label_metadata.json                 (label texts for retrieval stage)
"""

import re
import json
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
import os

RAW_PATH = "C:\\Users\\Yash Bhardwaj\\Desktop\\ai_project\\output\\medical_data.csv"
GROUPED_PATH = "medical_data_grouped.csv"
SUGGESTIONS_PATH = "category_suggestions.csv"
FINAL_PATH = "medical_data_model_ready.csv"

UNIQUE_MEDS = "unique_medications.csv"
UNIQUE_PROCS = "unique_procedures.csv"
LABEL_META = "label_metadata.json"


# -----------------------
# Utilities
# -----------------------
def normalize_text(text):
    """Lowercase, remove non-alphanumeric except spaces, collapse whitespace."""
    if pd.isna(text):
        return ""
    s = str(text).lower()
    s = re.sub(r"[^0-9a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def split_labels(cell):
    """Split semi-colon separated labels into normalized list. Handles NaN."""
    if pd.isna(cell) or str(cell).strip() == "":
        return []
    parts = [p.strip() for p in re.split(r";\s*", str(cell)) if p.strip()]
    return [normalize_text(p) for p in parts]


# -----------------------
# 1) Load CSV robustly
# -----------------------
print("Loading CSV:", RAW_PATH)
if not os.path.exists(RAW_PATH):
    raise FileNotFoundError(f"{RAW_PATH} not found. Put your CSV in the same folder as this script.")

# use python engine and skip bad lines to avoid parse errors
df = pd.read_csv(RAW_PATH, engine="python", on_bad_lines="skip")

# Ensure expected columns exist; create if missing
for col in ["Patient_ID", "Symptom", "Condition", "Medication", "Procedure", "ICD-10 Code"]:
    if col not in df.columns:
        df[col] = ""

print(f"Loaded {len(df)} rows. Columns: {df.columns.tolist()}")

# -----------------------
# 2) Normalize all text columns (non-destructive copies)
# -----------------------
print("Normalizing text columns...")
df["_Symptom_n"] = df["Symptom"].apply(normalize_text)
df["_Condition_n"] = df["Condition"].apply(normalize_text)
df["_Medication_raw"] = df["Medication"].fillna("").astype(str)
df["_Procedure_raw"] = df["Procedure"].fillna("").astype(str)
df["_Medication_list"] = df["_Medication_raw"].apply(split_labels)
df["_Procedure_list"] = df["_Procedure_raw"].apply(split_labels)

# -----------------------
# 3) Group rows -> 1 record per case (Patient_ID, Condition, ICD)
# -----------------------
print("Grouping rows per (Patient_ID, Condition, ICD-10 Code)...")
grouped = (df
           .groupby(["Patient_ID", "_Condition_n", "ICD-10 Code"], dropna=False, as_index=False)
           .agg({
               "_Symptom_n": lambda vals: "; ".join(sorted(dict.fromkeys([v for v in vals if v.strip()!='']))),
               "_Medication_raw": lambda vals: "; ".join(sorted(set([m.strip() for v in vals for m in str(v).split(";") if m.strip()]))),
               "_Procedure_raw": lambda vals: "; ".join(sorted(set([p.strip() for v in vals for p in str(v).split(";") if p.strip()])))
           })
          )

# Rename to friendlier column names
grouped = grouped.rename(columns={
    "_Condition_n": "Condition_norm",
    "_Symptom_n": "Symptom_Text",
    "_Medication_raw": "Medication_List",
    "_Procedure_raw": "Procedure_List",
    "ICD-10 Code": "ICD10_Code"
})

# Make the human-readable Condition (original case) by picking the most frequent original entry per group
# map group key -> original most common condition string
cond_map = (df.groupby(["Patient_ID", "_Condition_n"])["Condition"]
            .agg(lambda s: Counter(s).most_common(1)[0][0])
            .to_dict())
def get_original_condition(row):
    key = (row["Patient_ID"], row["Condition_norm"])
    return cond_map.get(key, row["Condition_norm"])

grouped["Condition"] = grouped.apply(get_original_condition, axis=1)

# Reorder columns for clarity
grouped = grouped[["Patient_ID", "Symptom_Text", "Condition", "Condition_norm", "ICD10_Code", "Medication_List", "Procedure_List"]]

print("Grouped dataset size:", grouped.shape)
grouped.to_csv(GROUPED_PATH, index=False)
print("Saved grouped file:", GROUPED_PATH)


# -----------------------
# 4) Build category suggestion rules (auto-suggest mapping)
# -----------------------
# A keyword -> category dictionary (extendable). These are heuristics, not strict rules.
KEYWORD_CATEGORY = {
    # Sleep / ENT
    "sleep": "Sleep Apnea Corrective Surgery",
    "apnea": "Sleep Apnea Corrective Surgery",
    "tonsil": "Pediatric Obstruction Surgery",
    "adenotonsil": "Pediatric Obstruction Surgery",
    "adenoid": "Pediatric Obstruction Surgery",
    "lingual": "Sleep Apnea Corrective Surgery",
    # Cardiac
    "myocardial": "Acute Cardiac Emergency",
    "mi ": "Acute Cardiac Emergency",
    "angina": "Acute Cardiac Emergency",
    "chest pain": "Acute Cardiac Emergency",
    "cardiac": "Acute Cardiac Emergency",
    # Respiratory / Infectious
    "bronchitis": "Infectious Disease Treatment",
    "pneumonia": "Infectious Disease Treatment",
    "tuberculosis": "Infectious Disease Treatment",
    "asthma": "Respiratory Stabilization",
    "copd": "Respiratory Stabilization",
    # Gastrointestinal
    "gerd": "Gastrointestinal Medical Management",
    "reflux": "Gastrointestinal Medical Management",
    "appendicitis": "Surgical Emergency",
    "cholecyst": "Surgical Emergency",
    "bleed": "Gastrointestinal Emergency",
    # Neurology
    "migraine": "Neurologic Long-Term Control",
    "stroke": "Acute Neurologic Emergency",
    "seizure": "Neurologic Long-Term Control",
    # Endocrine / Metabolic
    "diabetes": "Chronic Metabolic Management",
    # Musculoskeletal
    "fracture": "Orthopedic Surgical Planning",
    "acl": "Orthopedic Surgical Planning",
    "arthritis": "Inflammatory Condition Management",
    "gout": "Inflammatory Condition Management",
    # Oncology
    "cancer": "Oncology Diagnostic Pathway",
    "tumor": "Oncology Diagnostic Pathway",
    # Psychiatry / Mental health
    "depression": "Mental Health Management",
    "anxiety": "Mental Health Management",
    # default small keywords
    "infection": "Infectious Disease Treatment",
    "fever": "Infectious Disease Treatment",
}

def suggest_category_from_condition(cond_norm):
    """Return suggested category string or None."""
    if not cond_norm or cond_norm.strip() == "":
        return None
    for k, cat in KEYWORD_CATEGORY.items():
        if k in cond_norm:
            return cat
    return None


# Apply suggestions to unique conditions
unique_conditions = grouped["Condition_norm"].dropna().unique()
suggestions = []
for cond in sorted(unique_conditions):
    suggested = suggest_category_from_condition(cond)
    suggestions.append({
        "Condition_norm": cond,
        "Suggested_Category": suggested if suggested is not None else "Other / Needs Review",
        "Notes": ""  # user can fill this in the CSV
    })

suggest_df = pd.DataFrame(suggestions)
suggest_df.to_csv(SUGGESTIONS_PATH, index=False)
print("Wrote category suggestions to", SUGGESTIONS_PATH)
print("Open that CSV, review categories, edit Suggested_Category as needed, then re-run this script or a second stage to apply your edits.")


# -----------------------
# 5) Function to apply a mapping file (if you edited suggestions)
# -----------------------
def apply_category_mapping(grouped_df, mapping_csv_path=SUGGESTIONS_PATH):
    """
    Reads mapping CSV with columns: Condition_norm, Suggested_Category
    Returns new dataframe with Treatment_Category column added.
    """
    mapping = pd.read_csv(mapping_csv_path, engine="python", on_bad_lines="skip")
    # normalize Condition_norm column (safety)
    mapping["Condition_norm"] = mapping["Condition_norm"].astype(str).apply(lambda x: x.strip())
    mapping_dict = dict(zip(mapping["Condition_norm"], mapping["Suggested_Category"]))
    
    def map_row_to_category(row):
        cn = row["Condition_norm"]
        return mapping_dict.get(cn, "Other / Needs Review")
    
    g2 = grouped_df.copy()
    g2["Treatment_Category"] = g2.apply(map_row_to_category, axis=1)
    return g2

# If the user hasn't edited the suggestions file, we still apply the current suggestions automatically.
grouped_with_cat = apply_category_mapping(grouped, SUGGESTIONS_PATH)

# Save intermediate file with categories applied (you can re-run apply_category_mapping after editing the CSV)
grouped_with_cat.to_csv("medical_data_grouped_with_category.csv", index=False)
print("Saved grouped_with_category:", "medical_data_grouped_with_category.csv")

# -----------------------
# 6) Create final model-ready dataset (drop 'Other / Needs Review' optionally)
# -----------------------
print("Preparing final model-ready dataset...")

final_df = grouped_with_cat.copy()

# Option: keep or drop 'Other / Needs Review' rows. Keep by default but print counts to help decide.
counts = final_df["Treatment_Category"].value_counts(dropna=False)
print("\nTreatment Category counts (top 30):")
print(counts.head(30))

# Save final CSV (you can choose to drop 'Other / Needs Review' before training)
final_df.to_csv(FINAL_PATH, index=False)
print("Saved final ML-ready CSV:", FINAL_PATH)

# -----------------------
# 7) Produce unique medication/procedure lists and label metadata for retrieval
# -----------------------
print("Extracting unique medications and procedures for retrieval stage...")

all_meds = set()
all_procs = set()
label_to_contexts = defaultdict(list)

for _, row in final_df.iterrows():
    meds = split_labels(row["Medication_List"])
    procs = split_labels(row["Procedure_List"])
    context = " ".join([normalize_text(row["Symptom_Text"]), normalize_text(row["Condition"])])
    for m in meds:
        if m:
            all_meds.add(m)
            label_to_contexts.setdefault(m, []).append(context)
    for p in procs:
        if p:
            all_procs.add(p)
            label_to_contexts.setdefault(p, []).append(context)

meds_sorted = sorted(all_meds)
procs_sorted = sorted(all_procs)

pd.DataFrame({"medication": meds_sorted}).to_csv(UNIQUE_MEDS, index=False)
pd.DataFrame({"procedure": procs_sorted}).to_csv(UNIQUE_PROCS, index=False)
print(f"Unique meds: {len(meds_sorted)}, unique procs: {len(procs_sorted)}")
print("Saved:", UNIQUE_MEDS, UNIQUE_PROCS)

# Create a simple label metadata JSON for later retrieval vector building
label_meta = {
    "labels": [],
}
for lab in meds_sorted + procs_sorted:
    label_meta["labels"].append({
        "label": lab,
        "contexts": label_to_contexts.get(lab, [])[:10]  # sample contexts
    })

with open(LABEL_META, "w", encoding="utf8") as f:
    json.dump(label_meta, f, ensure_ascii=False, indent=2)

print("Saved label metadata to", LABEL_META)

# -----------------------
# 8) Quick diagnostics and instructions
# -----------------------
print("\n--- Diagnostic Summary ---")
print("Total original rows:", len(df))
print("Total grouped cases:", len(grouped))
print("Total final cases with category:", len(final_df))
print("Counts per Treatment_Category (top 20):")
print(final_df["Treatment_Category"].value_counts().head(20))

print("\nNext steps (recommended):")
print("1) Open", SUGGESTIONS_PATH, "and review / edit Suggested_Category values. Fill Notes where needed.")
print("   - For ambiguous conditions, choose a sensible category or 'Other / Needs Review'.")
print("2) Re-run this script after editing SUGGESTIONS_PATH (it will apply the edits and re-save final CSV).")
print("3) Train a classification model that predicts Treatment_Category from Symptom_Text + Condition.")
print("   Example: TF-IDF (ngram 1-2) -> LinearSVC (or LogisticRegression with class_weight='balanced').")
print("4) For retrieving exact meds/procedures for a predicted category, build a TF-IDF vector space of label texts (label_meta -> label vectors) and use cosine similarity to rank items.")
print("\nScript finished successfully.")
