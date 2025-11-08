"""
find_rare_cases.py

Purpose:
    Identify rare Treatment_Category cases in your dataset
    and print their counts + full row details.

Usage:
    python find_rare_cases.py
"""

import pandas as pd

# === CONFIGURATION ===
INPUT_FILE = "C:\\Users\\Yash Bhardwaj\\Desktop\\ai_project\\medical_data_model_ready.csv"  # Path to your model-ready dataset
THRESHOLD = 3  # Categories with <3 samples will be considered rare

# === STEP 1: Load Dataset ===
print(f"Loading dataset from '{INPUT_FILE}' ...")
df = pd.read_csv(INPUT_FILE)

if "Treatment_Category" not in df.columns:
    raise ValueError("❌ Column 'Treatment_Category' not found. Please make sure your dataset is model-ready.")

print(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns.\n")

# === STEP 2: Count Category Frequency ===
category_counts = df["Treatment_Category"].value_counts()
print("=== Category Counts (Descending) ===")
print(category_counts)
print("\n------------------------------------\n")

# === STEP 3: Identify Rare Categories ===
rare_categories = category_counts[category_counts < THRESHOLD]
if len(rare_categories) == 0:
    print(f"🎉 No rare categories found (all have ≥ {THRESHOLD} samples).")
    exit()

print(f"⚠️ Found {len(rare_categories)} rare categories (< {THRESHOLD} samples):")
print(rare_categories)
print("\n------------------------------------\n")

# === STEP 4: Extract and Display Rare Case Rows ===
df_rare = df[df["Treatment_Category"].isin(rare_categories.index)]

print(f"=== Detailed Rare Case Rows (Total {len(df_rare)}) ===\n")
print(df_rare[["Patient_ID", "Condition", "Symptom_Text", "Medication_List", "Procedure_List", "Treatment_Category"]])

# === STEP 5: Save Results (Optional) ===
df_rare.to_csv("rare_cases_detailed.csv", index=False)
rare_categories.to_csv("rare_category_counts.csv")

print("\n✅ Results saved:")
print("   → rare_cases_detailed.csv (detailed rows)")
print("   → rare_category_counts.csv (counts per rare category)")
print("\nReview these files to decide which categories need augmentation or merging.")
