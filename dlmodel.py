import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, hamming_loss, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib

# 1. Load Data
print("Loading data...")
df = pd.read_csv('data/medical_data.csv', engine='python', on_bad_lines='skip')
print(f"Loaded {len(df)} rows")

# 2. Prepare the data
df['Medication_List'] = df['Medication'].str.split('; ')
df['Procedure_List'] = df['Procedure'].str.split('; ')

# 3. Binarize Labels
print("Binarizing labels...")
mlb_med = MultiLabelBinarizer()
mlb_proc = MultiLabelBinarizer()
med_matrix = mlb_med.fit_transform(df['Medication_List'])
proc_matrix = mlb_proc.fit_transform(df['Procedure_List'])
y_combined = np.hstack((med_matrix, proc_matrix))

# 4. Use TF-IDF features
from sklearn.feature_extraction.text import TfidfVectorizer

input_texts = (df['Symptom'] + ' ' + df['Condition']).values
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_tfidf = vectorizer.fit_transform(input_texts).toarray()

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_combined, test_size=0.2, random_state=42)
print(f"Training on {X_train.shape[0]} samples.")
print(f"Number of output labels: {y_combined.shape[1]}")

# 6. Build a simple neural network
print("Building neural network...")
model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(y_combined.shape[1], activation='sigmoid'))

# 7. Use standard binary crossentropy with class weighting
# Calculate class weights to handle imbalance
def calculate_class_weights(y):
    """Calculate class weights for imbalanced multi-label classification"""
    positive_counts = np.sum(y, axis=0)
    total = len(y)
    
    # Calculate weight for each class (inverse of frequency)
    weights = total / (2 * np.maximum(positive_counts, 1))
    
    # Normalize weights to avoid extreme values
    weights = np.minimum(weights, 10.0)  # Cap at 10
    return weights

class_weights = calculate_class_weights(y_combined)
print(f"Class weights range: {np.min(class_weights):.2f} to {np.max(class_weights):.2f}")

# Convert class weights to sample weights
def get_sample_weights(y_true, class_weights):
    """Create sample weights based on true labels and class weights"""
    sample_weights = np.ones(len(y_true))
    for i in range(len(y_true)):
        for j in range(len(y_true[i])):
            if y_true[i, j] == 1:
                sample_weights[i] += class_weights[j]
    # Normalize sample weights
    sample_weights = sample_weights / np.mean(sample_weights)
    return sample_weights

sample_weights = get_sample_weights(y_train, class_weights)

# 8. Compile with standard binary crossentropy
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# 9. Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# 10. Train the Model
print("Training model...")
history = model.fit(
    X_train, y_train,
    sample_weight=sample_weights,  # Use sample weights for class balancing
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# 11. Evaluate the Model
print("\nEvaluating model...")
y_pred_proba = model.predict(X_test)

# Use a reasonable threshold
threshold = 0.3
y_pred_bin = (y_pred_proba > threshold).astype(int)

# Calculate metrics
n_med = len(mlb_med.classes_)
total_predictions = np.sum(y_pred_bin)
total_possible = y_pred_bin.size

print("=" * 60)
print("MODEL EVALUATION METRICS")
print("=" * 60)
print(f"Prediction Threshold: {threshold}")
print(f"Total predictions made: {total_predictions} (out of {total_possible} possible)")
print(f"Prediction rate: {total_predictions/total_possible:.3%}")

if total_predictions > 0:
    print(f"Hamming Loss: {hamming_loss(y_test, y_pred_bin):.4f}")
    print(f"Subset Accuracy: {accuracy_score(y_test, y_pred_bin):.4f}")
    print(f"Precision (micro): {precision_score(y_test, y_pred_bin, average='micro', zero_division=0):.4f}")
    print(f"Recall (micro): {recall_score(y_test, y_pred_bin, average='micro', zero_division=0):.4f}")
    print(f"F1 Score (micro): {f1_score(y_test, y_pred_bin, average='micro', zero_division=0):.4f}")
else:
    print("WARNING: No predictions made - model is predicting all zeros")

# 12. Prediction Function
def predict_treatment(symptom="", condition="", confidence_threshold=0.3):
    """Predict treatment based on symptom and condition together."""
    if not symptom and not condition:
        return [], [], [], []
    
    input_text = f"{symptom} {condition}".strip()
    input_vec = vectorizer.transform([input_text]).toarray()
    
    prediction_proba = model.predict(input_vec, verbose=0)
    prediction_bin = (prediction_proba > confidence_threshold).astype(int)
    
    meds = mlb_med.inverse_transform(prediction_bin[:, :n_med])
    procs = mlb_proc.inverse_transform(prediction_bin[:, n_med:])
    
    # Get confidence scores
    med_scores = prediction_proba[0, :n_med]
    proc_scores = prediction_proba[0, n_med:]
    
    predicted_med_scores = [round(med_scores[i], 2) for i, val in enumerate(prediction_bin[0, :n_med]) if val == 1]
    predicted_proc_scores = [round(proc_scores[i], 2) for i, val in enumerate(prediction_bin[0, n_med:]) if val == 1]
    
    return meds[0], procs[0], predicted_med_scores, predicted_proc_scores

# 13. Test the function
print("\nTesting prediction...")
test_meds, test_procs, med_scores, proc_scores = predict_treatment(symptom="Chest pain", condition="Myocardial Infarction")
print(f"Input: 'Chest pain Myocardial Infarction'")
print(f"Predicted Medications: {list(test_meds)} (Confidence: {med_scores})")
print(f"Predicted Procedures: {list(test_procs)} (Confidence: {proc_scores})")

# 14. Save the model and components
print("Saving model and components...")
model.save('treatment_model.h5')
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(mlb_med, 'mlb_medications.pkl')
joblib.dump(mlb_proc, 'mlb_procedures.pkl')
print("Model and components saved successfully!")