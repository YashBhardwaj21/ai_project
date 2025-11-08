import pandas as pd

df = pd.read_csv("medical_data_model_ready.csv")

# Filter all 'Other / Needs Review' cases
other_df = df[df["Treatment_Category"] == "Other / Needs Review"]

print(f"Total 'Other / Needs Review' cases: {len(other_df)}")

# Show top 50 most common conditions that ended up in Other
print("\nTop conditions under 'Other / Needs Review':")
print(other_df["Condition"].value_counts().head(50))
