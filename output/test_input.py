import joblib
import numpy as np
from scipy import sparse

print("Loading AI Medical Assistant...")
model = joblib.load('updated_treatment_model.pkl')
vectorizer = joblib.load('updated_vectorizer.pkl')
mlb_med = joblib.load('mlb_medications.pkl')
mlb_proc = joblib.load('mlb_procedures.pkl')
print("Model loaded successfully!\n")

common_labels = [] 

def predict_treatment_fast(symptom, condition):
    """FAST version - predicts only for likely labels"""
    if symptom and condition:
        input_text = f"{symptom} {condition}"
    elif symptom:
        input_text = symptom
    elif condition:
        input_text = condition
    else:
        return [], [] 

    input_vec = vectorizer.transform([input_text])
    
    input_dense = input_vec.toarray()

    predictions = []
    for i, estimator in enumerate(model.estimators_):

        pred = estimator.predict(input_dense)
        predictions.append(pred[0])
    
    prediction = np.array([predictions])
    
    n_med = len(mlb_med.classes_)
    meds = mlb_med.inverse_transform(prediction[:, :n_med])
    procs = mlb_proc.inverse_transform(prediction[:, n_med:])
    
    return meds[0], procs[0]

def predict_treatment_with_feedback(symptom, condition):
    """Version with user feedback to hide the slowness"""
    if symptom and condition:
        input_text = f"{symptom} {condition}"
    elif symptom:
        input_text = symptom
    elif condition:
        input_text = condition
    else:
        return [], [] 
    
    print("   Analyzing...", end="", flush=True)
    
    input_vec = vectorizer.transform([input_text])
    prediction = model.predict(input_vec)

    print(" Done!")

    n_med = len(mlb_med.classes_)
    meds = mlb_med.inverse_transform(prediction[:, :n_med])
    procs = mlb_proc.inverse_transform(prediction[:, n_med:])
    
    return meds[0], procs[0]

print("=== AI Medical Treatment Predictor ===")
print("Enter symptoms, conditions, or both to get treatment recommendations")
print("Type 'quit' at any time to exit\n")

while True:
    symptom = input("Enter the symptom (or press Enter to skip): ").strip()
    if symptom.lower() == 'quit':
        break
        
    condition = input("Enter the medical condition (or press Enter to skip): ").strip()
    if condition.lower() == 'quit':
        break

    if not symptom and not condition:
        print("Please enter at least a symptom or a condition\n")
        continue

    try:
        # Use the version with feedback
        medications, procedures = predict_treatment_with_feedback(symptom, condition)

        if symptom and condition:
            input_desc = f"{symptom} with {condition}"
        elif symptom:
            input_desc = f"symptom: {symptom}"
        else:
            input_desc = f"condition: {condition}"
        
        print(f"\nTreatment Recommendation for: {input_desc}")
        print("Medications:", ", ".join(medications) if medications else "No specific medication recommended")
        print("Procedures:", ", ".join(procedures) if procedures else "No specific procedure recommended")
        print("-" * 50)
        
    except Exception as e:
        print(f"Error: Could not generate prediction. Please try different input.")
        print(f"   Error details: {e}\n")

print("\nThank you for using the AI Medical Assistant!")