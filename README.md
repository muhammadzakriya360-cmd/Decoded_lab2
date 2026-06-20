# Decoded_lab2

# Data handling
import numpy as np
import pandas as pd

# Dataset
from sklearn.datasets import load_iris

# Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# The Algorithm: K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

# Evaluation Metrics
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

print("✅ All libraries imported successfully!")

# PHASE 1 — INPUT: Load the Dataset

iris = load_iris()

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species_name'] = df['species'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})

print("=" * 55)
print("   🌸 IRIS DATASET — LOADED SUCCESSFULLY")
print("=" * 55)
print(f"\n📊 Dataset Shape: {df.shape}")
print(f"📋 Features: {iris.feature_names}")
print(f"🏷️  Classes: {iris.target_names.tolist()}")
print("\n--- First 5 Rows ---")
print(df.head())
print("\n--- Class Distribution (Balanced) ---")
print(df['species_name'].value_counts())
# Visualize feature relationships by species
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colors = {'Setosa': '#2196F3', 'Versicolor': '#FF9800', 'Virginica': '#4CAF50'}

# Plot 1: Sepal Length vs Sepal Width
for species, color in colors.items():
    subset = df[df['species_name'] == species]
    axes[0].scatter(subset['sepal length (cm)'], subset['sepal width (cm)'],
                    label=species, color=color, alpha=0.7, s=60)
axes[0].set_title('Sepal Length vs Sepal Width', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Sepal Length (cm)')
axes[0].set_ylabel('Sepal Width (cm)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Petal Length vs Petal Width
for species, color in colors.items():
    subset = df[df['species_name'] == species]
    axes[1].scatter(subset['petal length (cm)'], subset['petal width (cm)'],
                    label=species, color=color, alpha=0.7, s=60)
axes[1].set_title('Petal Length vs Petal Width', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Petal Length (cm)')
axes[1].set_ylabel('Petal Width (cm)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('🌸 Iris Dataset — Feature Visualization', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
print("✅ Visualization complete!")
# Separate Features (X) and Labels (y)
X = iris.data    
y = iris.target  

print(f"Feature matrix X shape: {X.shape}")
print(f"Label vector y shape:   {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      
    random_state=42,    
    shuffle=True        
)

print(f"\n📦 Training set:  {X_train.shape[0]} samples (80%)")
print(f"🔒 Testing set:   {X_test.shape[0]} samples (20%)")

# THE GATEKEEPER RULE: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  
X_test_scaled  = scaler.transform(X_test)         

print("\n✅ Feature Scaling applied (Mean=0, Variance=1)")
print(f"   Before scaling — Mean: {X_train[:, 0].mean():.2f}")
print(f"   After  scaling — Mean: {X_train_scaled[:, 0].mean():.4f}")

# TUNING THE ENGINE: Find the best K value
k_range = range(1, 21)
error_rates = []
accuracy_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred_k = knn.predict(X_test_scaled)
    error_rates.append(1 - accuracy_score(y_test, y_pred_k))
    accuracy_scores.append(accuracy_score(y_test, y_pred_k))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(k_range, error_rates, color='#E74C3C', marker='o', linestyle='--', linewidth=2, markersize=8)
axes[0].set_title('Error Rate vs K Value', fontsize=13, fontweight='bold')
axes[0].set_xlabel('K Value')
axes[0].set_ylabel('Error Rate')
axes[0].grid(True, alpha=0.3)

axes[1].plot(k_range, accuracy_scores, color='#27AE60', marker='s', linestyle='--', linewidth=2, markersize=8)
axes[1].set_title('Accuracy vs K Value', fontsize=13, fontweight='bold')
axes[1].set_xlabel('K Value')
axes[1].set_ylabel('Accuracy')
axes[1].grid(True, alpha=0.3)

# Mark best K
best_k = k_range[accuracy_scores.index(max(accuracy_scores))]
axes[1].axvline(x=best_k, color='red', linestyle=':', label=f'Best K={best_k}')
axes[1].legend()

plt.suptitle('🔍 Tuning The Engine — Finding Optimal K', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()

print(f"\n✅ Optimal K = {best_k} with accuracy = {max(accuracy_scores)*100:.2f}%")

# PHASE 2 — PROCESS: KNN Algorithm

# Step 1: INSTANTIATE — Build the frame
model = KNeighborsClassifier(n_neighbors=best_k)

# Step 2: FIT — Memorize the map (train on training data)
model.fit(X_train_scaled, y_train)

# Step 3: PREDICT — Apply logic (predict on test data)
y_predictions = model.predict(X_test_scaled)

print("=" * 55)
print("   🚀 KNN MODEL TRAINED SUCCESSFULLY")
print("=" * 55)
print(f"\n   Algorithm : K-Nearest Neighbors")
print(f"   K Value   : {best_k} neighbors")
print(f"   Training  : {X_train_scaled.shape[0]} samples")
print(f"   Testing   : {X_test_scaled.shape[0]} samples")

class_names = iris.target_names
print("\n--- Sample Predictions (First 10) ---")
print(f"{'#':<5} {'Actual':<15} {'Predicted':<15} {'Result':<10}")
print("-" * 45)
for i in range(10):
    actual    = class_names[y_test[i]]
    predicted = class_names[y_predictions[i]]
    result    = '✅ Correct' if actual == predicted else '❌ Wrong'
    print(f"{i+1:<5} {actual:<15} {predicted:<15} {result}")
    
# PHASE 3 — OUTPUT: Evaluation Metrics
#Core Metrics
accuracy = accuracy_score(y_test, y_predictions)
f1       = f1_score(y_test, y_predictions, average='weighted')
cm       = confusion_matrix(y_test, y_predictions)

print("=" * 55)
print("   📊 MODEL EVALUATION REPORT")
print("=" * 55)
print(f"\n   ✅ Accuracy  : {accuracy * 100:.2f}%")
print(f"   ✅ F1 Score  : {f1 * 100:.2f}%  (Weighted)")
print(f"   ✅ Test Size : {len(y_test)} samples")

print("\n--- Detailed Classification Report ---")
print(classification_report(y_test, y_predictions, target_names=class_names))
# --- Confusion Matrix Heatmap ---
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names,
    linewidths=1,
    linecolor='white'
)
plt.title(f'🔷 Confusion Matrix — KNN (K={best_k})\nAccuracy: {accuracy*100:.2f}% | F1: {f1*100:.2f}%',
          fontsize=13, fontweight='bold', pad=15)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.show()

print("\n📖 How to read: Diagonal = correct predictions | Off-diagonal = misclassifications")
print(f"   TP (diagonal sum): {cm.trace()} correct out of {len(y_test)} total")
# BONUS: Predict a completely new flower sample

new_samples = [
    [5.1, 3.5, 1.4, 0.2],   
    [6.0, 2.9, 4.5, 1.5],   
    [6.7, 3.0, 5.2, 2.3],  
]

new_samples_scaled = scaler.transform(new_samples)

# Predict
new_predictions = model.predict(new_samples_scaled)
new_probabilities = model.predict_proba(new_samples_scaled)

print("=" * 60)
print("   🎯 CUSTOM FLOWER PREDICTION")
print("=" * 60)
print(f"\n{'Sample':<10} {'Measurements':<30} {'Prediction':<15} {'Confidence'}")
print("-" * 75)
for i, (sample, pred, prob) in enumerate(zip(new_samples, new_predictions, new_probabilities)):
    confidence = max(prob) * 100
    print(f"Sample {i+1:<3} {str(sample):<30} {class_names[pred]:<15} {confidence:.1f}%")

print("\n✅ Predictions complete!")
print("=" * 60)
print("   📋 PROJECT 2 — FINAL SUMMARY")
print("   DecodeLabs AI Internship | Batch 2026")
print("=" * 60)
print()
print("  PIPELINE (IPO Framework):")
print("  ┌─────────────────────────────────────────────┐")
print("  │ INPUT   → Iris Dataset (150 samples, 4 feats)│")
print("  │ INPUT   → StandardScaler (Mean=0, Var=1)     │")
print("  │ PROCESS → Train-Test Split (80% / 20%)       │")
print(f" │ PROCESS → KNN Algorithm (K={best_k})               │")
print("  │ OUTPUT  → Confusion Matrix                   │")
print("  │ OUTPUT  → F1 Score + Accuracy                │")
print("  └─────────────────────────────────────────────┘")
print()
print("  RESULTS:")
print(f"  → Optimal K     : {best_k}")
print(f"  → Accuracy      : {accuracy * 100:.2f}%")
print(f"  → F1 Score      : {f1 * 100:.2f}% (Weighted)")
print(f"  → Correct       : {cm.trace()} / {len(y_test)} samples")
print()
print("  KEY LEARNINGS:")
print("  → Never fit the scaler on test data (data leakage)")
print("  → Accuracy can be misleading — use F1 Score")
print("  → K=1 overfits; K too large underfits — find the elbow")
print()
print("  Project 2 Complete ✅")
print("=" * 60)
