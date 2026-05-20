import matplotlib.pyplot as plt

# Models and their F1-Scores
models = ['SVM', 'CNN', 'YOLO', 'YOLO + CNN (Proposed)']
f1_scores = [79, 89, 92, 96]

plt.figure(figsize=(8,5))

plt.bar(models, f1_scores, color='skyblue')

plt.title("Comparison of F1-Score for Existing and Proposed Models")
plt.xlabel("Models")
plt.ylabel("F1-Score")

plt.ylim(0,100)

for i, v in enumerate(f1_scores):
    plt.text(i, v + 1, str(v), ha='center', fontsize=10)

plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.show()