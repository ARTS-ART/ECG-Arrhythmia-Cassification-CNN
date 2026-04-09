import os
import numpy as np
import torch
from models.models1d import EcgResNet34
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, f1_score, roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Config ---
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📟 Using device: {DEVICE}")

# --- Load Dataset ---
print("\n📥 Loading combined test dataset (.npz)...")
data = np.load("scripts/clean_single_record.npz")
X_test = data['X']
y_test = data['y']
print(f"✅ Loaded {len(X_test)} test samples with shape {X_test.shape}\n")

# --- Load Model ---
model = EcgResNet34(num_classes=4).to(DEVICE)
checkpoint = torch.load("scripts/checkpoints/resnet34_clean_1d.pth", map_location=DEVICE)
model.load_state_dict(checkpoint)
model.eval()

# --- Inference in Batches ---
y_pred = []
y_probs = []  # <-- store raw probabilities for ROC

with torch.no_grad():
    for i in tqdm(range(0, len(X_test), BATCH_SIZE), desc="🔍 Running Inference"):
        batch = X_test[i:i+BATCH_SIZE]
        inputs = torch.tensor(batch).float().unsqueeze(1).to(DEVICE)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()  # probabilities
        preds = np.argmax(probs, axis=1)
        y_pred.extend(preds)
        y_probs.extend(probs)

y_pred = np.array(y_pred)
y_probs = np.array(y_probs)  # shape: (num_samples, 4)

# --- Evaluation ---
labels = ['Normal', 'Ventricular', 'Fusion', 'Unknown']
acc = accuracy_score(y_test, y_pred)
avg_f1 = f1_score(y_test, y_pred, average='weighted')

print("\n🎯 Evaluation Results:")
print("Accuracy:", acc)
print("Average F1 Score:", avg_f1)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, labels=[0, 1, 2, 3], target_names=labels))

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])
fig, ax = plt.subplots(figsize=(6, 6))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(ax=ax, cmap='Blues')
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
print("✅ Confusion matrix saved as 'confusion_matrix.png'")

# --- Enhanced ROC Curve ---
print("📊 Plotting Enhanced ROC Curve...")

# Binarize the ground truth
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])

# Calculate ROC Curve and AUC for each class
fpr, tpr, roc_auc = {}, {}, {}
for i in range(4):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot
plt.figure(figsize=(10, 8))
colors = ['blue', 'red', 'green', 'orange']
labels = ['Normal', 'Ventricular', 'Fusion', 'Unknown']

for i in range(4):
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
             label=f"ROC curve for {labels[i]} (AUC = {roc_auc[i]:0.2f})")

# Chance line
plt.plot([0, 1], [0, 1], 'k--', lw=2, label="Chance Level (AUC = 0.5)")

# Formatting
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Multi-class ROC Curve', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve_enhanced.png")
plt.show()
print("✅ Enhanced ROC curve saved as 'roc_curve_enhanced.png'")

