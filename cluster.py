#!/usr/bin/env python3
"""
cluster_export_refined_clean.py

Exports the clustered "Other / Needs Review" samples into
clean, descriptive new medical subcategories — removing
the 'Other - Cluster' prefix completely.

Final Format:
Patient_ID,symptom,Condition,condition,IC-10 Code,Medication,Procedure,Treatment_Category
"""

import pandas as pd
import csv
import os

# ---------- CONFIG ----------
INPUT = r"C:\Users\Yash Bhardwaj\Desktop\ai_project\clustered_other_cases.csv"
OUTPUT = r"C:\Users\Yash Bhardwaj\Desktop\ai_project\refined_other_final_1082.csv"
START_ID = 1082

# ---------- MAPPING ----------
CATEGORY_MAP = {
    "Other - Cluster 1: Status": "Allergic & General Symptom Management",
    "Other - Cluster 2: Disease": "Chronic and Autoimmune Disease Management",
    "Other - Cluster 3: Disorder": "Sleep & Psychiatric Disorder Management",
    "Other - Cluster 4: Rash": "Dermatologic and Skin Condition Management",
    "Other - Cluster 5: Pain": "Musculoskeletal and Pain Management",
    "Other - Cluster 6: Skin": "Connective Tissue and Autoimmune Skin Disorders",
    "Other - Cluster 7: Sleep": "Sleep-Related Breathing and Movement Disorders",
    "Other - Cluster 8: Syndrome": "Genetic and Developmental Syndrome Management",
}

# ---------- HELPERS ----------
def sanitize_cell(x):
    """Clean commas, quotes, and newlines from cells."""
    if pd.isna(x):
        return ""
    s = str(x)
    s = s.replace(",", ";")
    s = s.replace('"', "")
    s = s.replace("\r", " ").replace("\n", " ")
    return s.strip()

# ---------- LOAD ----------
print(f"📂 Loading clustered data from: {INPUT}")
df = pd.read_csv(INPUT, engine="python", on_bad_lines="skip")
print(f"✅ Loaded {len(df)} rows and columns: {list(df.columns)}")

# Detect cluster label column
cat_col = None
for c in ["Refined_Subcategory", "Cluster_Label", "Treatment_Category"]:
    if c in df.columns:
        cat_col = c
        break

if not cat_col:
    raise ValueError("❌ Could not find a category column in your file. Expected 'Refined_Subcategory' or similar.")

print(f"🧩 Using category column: {cat_col}")

# ---------- MAP CATEGORIES ----------
df["Refined_Category"] = df[cat_col].replace(CATEGORY_MAP)
print("🔁 Mapped cluster labels to detailed medical categories.")

# ---------- BUILD FINAL DATAFRAME ----------
final = pd.DataFrame({
    "Patient_ID": [f"P{START_ID + i}" for i in range(len(df))],
    "symptom": df.get("Symptom_Text", "").fillna("").astype(str),
    "Condition": df.get("Condition", "").fillna("").astype(str),
    "condition": df.get("Condition", "").fillna("").astype(str).str.lower(),
    "IC-10 Code": df.get("ICD10_Code", "").fillna("").astype(str),
    "Medication": df.get("Medication_List", "").fillna("").astype(str),
    "Procedure": df.get("Procedure_List", "").fillna("").astype(str),
    "Treatment_Category": df["Refined_Category"].fillna("Uncategorized").astype(str)
})

# ---------- SANITIZE ----------
print("🧼 Cleaning text fields...")
for col in final.columns:
    final[col] = final[col].apply(sanitize_cell)

# ---------- SAVE ----------
cols_exact = [
    "Patient_ID", "symptom", "Condition", "condition",
    "IC-10 Code", "Medication", "Procedure", "Treatment_Category"
]
final = final[cols_exact]

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
final.to_csv(
    OUTPUT,
    index=False,
    sep=",",
    encoding="utf-8-sig",
    quoting=csv.QUOTE_NONE,
    escapechar="\\",
    lineterminator="\n"
)

# ---------- DONE ----------
print(f"\n💾 Saved refined CSV to: {OUTPUT}")
print("\n=== Preview (first 10 rows) ===")
print(final.head(10).to_string(index=False))
