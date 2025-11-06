import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss, accuracy_score

print("Loading pre-trained model and components...")
try:
    model = joblib.load('updated_treatment_model.pkl')
    vectorizer = joblib.load('updated_vectorizer.pkl')
    mlb_med = joblib.load('mlb_medications.pkl')
    mlb_proc = joblib.load('mlb_procedures.pkl')
    print("Model components loaded successfully!")
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    exit()

print("Loading data for evaluation...")
df = pd.read_csv('medical_data.csv', engine='python', on_bad_lines='skip')
print(f"Loaded {len(df)} rows")

df['Medication_List'] = df['Medication'].str.split('; ')
df['Procedure_List'] = df['Procedure'].str.split('; ')

print("Binarizing labels...")
med_matrix = mlb_med.transform(df['Medication_List'])
proc_matrix = mlb_proc.transform(df['Procedure_List'])
y_combined = np.hstack((med_matrix, proc_matrix))

input_texts = df['Symptom'] + ' ' + df['Condition']
X_tfidf = vectorizer.transform(input_texts) 

print("Making predictions...")
y_pred = model.predict(X_tfidf)

# 7. Calculate metrics
n_med = len(mlb_med.classes_)
y_test_med = y_combined[:, :n_med]
y_test_proc = y_combined[:, n_med:]
y_pred_med = y_pred[:, :n_med]
y_pred_proc = y_pred[:, n_med:]

print("=" * 60)
print("MODEL EVALUATION METRICS (USING PRE-TRAINED MODEL)")
print("=" * 60)

# Accuracy per Label
accuracy_per_label = np.mean(y_combined == y_pred)
print(f"1. Accuracy per individual label: {accuracy_per_label:.4f}")

# Hamming Loss 
hamming = hamming_loss(y_combined, y_pred)
print(f"2. Hamming Loss: {hamming:.4f}")

# Precision/Recall/F1
precision_micro = precision_score(y_combined, y_pred, average='micro', zero_division=0)
recall_micro = recall_score(y_combined, y_pred, average='micro', zero_division=0)
f1_micro = f1_score(y_combined, y_pred, average='micro', zero_division=0)

print(f"3. Precision: {precision_micro:.4f}")
print(f"4. Recall:    {recall_micro:.4f} ")
print(f"5. F1 Score:  {f1_micro:.4f} ")

# Custom: Average number of matches per patient
def calculate_matches(y_true, y_pred):
    matches = []
    for i in range(len(y_true)):
        correct = np.sum(y_true[i] & y_pred[i])
        matches.append(correct)
    return np.mean(matches)

avg_matches = calculate_matches(y_combined, y_pred)
print(f"6. Avg correct predictions per patient: {avg_matches:.2f}")

# Prediction rate
total_predicted = np.sum(y_pred)
total_possible = y_pred.size
prediction_rate = total_predicted / total_possible
print(f"7. Prediction rate: {prediction_rate:.3%} of possible labels were predicted")

subset_acc = accuracy_score(y_combined, y_pred)
print(f"8. Subset accuracy: {subset_acc:.4f} ")

print("\n" + "=" * 30)
print("BREAKDOWN BY TYPE")
print("=" * 30)
print(f"Medications - Precision: {precision_score(y_test_med, y_pred_med, average='micro', zero_division=0):.4f}")
print(f"Procedures - Precision: {precision_score(y_test_proc, y_pred_proc, average='micro', zero_division=0):.4f}")

# 8. Prediction Function (unchanged)
def predict_treatment(symptom="", condition=""):
    """Predict treatment based on symptom and condition together."""
    if not symptom and not condition:
        return [], []
    
    input_text = f"{symptom} {condition}".strip()
    input_vec = vectorizer.transform([input_text])
    
    prediction = model.predict(input_vec)
    
    n_med = len(mlb_med.classes_)
    meds = mlb_med.inverse_transform(prediction[:, :n_med])
    procs = mlb_proc.inverse_transform(prediction[:, n_med:])
    
    return meds[0], procs[0]

# Test
print("\nTesting prediction...")
test_meds, test_procs = predict_treatment(symptom="Chest pain", condition="Myocardial Infarction")
print(f"Input: 'Chest pain Myocardial Infarction'")
print(f"Predicted Medications: {test_meds}")
print(f"Predicted Procedures: {test_procs}")

print("\n Evaluation complete using pre-trained model!")