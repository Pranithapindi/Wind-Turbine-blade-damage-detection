from ultralytics import YOLO
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ===============================
# LOAD MODEL
# ===============================
model = YOLO("runs/classify/train/weights/best.pt")

# ===============================
# DATASET PATH
# ===============================
test_dir = "CAI-SWTB-Dataset/Test"

classes = ["Healthy", "Faulty"]

y_true = []
y_pred = []

# ===============================
# LOOP THROUGH TEST DATA
# ===============================
for label in classes:
    folder = os.path.join(test_dir, label)
    
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)

        results = model(img_path)
        result = results[0]

        probs = result.probs.data.tolist()
        pred_index = probs.index(max(probs))
        pred_label = result.names[pred_index]

        y_true.append(label)
        y_pred.append(pred_label)

# ===============================
# CONFUSION MATRIX
# ===============================
cm = confusion_matrix(y_true, y_pred, labels=classes)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=classes,
            yticklabels=classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ===============================
# CLASSIFICATION REPORT
# ===============================
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred))

# ===============================
# ACCURACY CALCULATION
# ===============================
correct = sum([1 for i in range(len(y_true)) if y_true[i] == y_pred[i]])
accuracy = correct / len(y_true)

print("\nOverall Accuracy:", round(accuracy*100,2), "%")

# ===============================
# GRAPH (CORRECT VS WRONG)
# ===============================
correct_count = correct
wrong_count = len(y_true) - correct

plt.figure()
plt.bar(["Correct", "Wrong"], [correct_count, wrong_count])
plt.title("Prediction Results")
plt.xlabel("Category")
plt.ylabel("Count")
plt.show()