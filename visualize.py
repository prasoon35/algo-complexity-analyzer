import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Using a darkgrid theme 
sns.set_theme(style="darkgrid", context="notebook")
colors = sns.color_palette("husl", 8)

# Load the data
df = pd.read_csv("data.csv")

# Theoretical ground truth for the Confusion Matrix
TRUE_COMPLEXITY = {
    "linear_search": "O(n)",
    "bubble_sort": "O(n²)",
    "binary_search": "O(log n)",
    "merge_sort": "O(n log n)",
    "fibonacci": "O(1) [Capped]",
    "find_peak_element": "O(log n)"
}

COMPLEXITY_SLOPES = {
    "O(1) [Capped]": 0.0,
    "O(log n)": 0.3,
    "O(n)": 1.0,
    "O(n log n)": 1.1,
    "O(n²)": 2.0,
}

algorithms = df["algorithm"].unique()
predicted_labels = []
true_labels = []



# --- ACTUAL VS PREDICTED CURVES OVERLAY ---
# Create a large, multi-plot figure (2 rows, 3 columns)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("ML Complexity Regression vs. Empirical Execution Runtimes", fontsize=20, fontweight='bold', y=0.98)
axes = axes.flatten()

for idx, algo_name in enumerate(algorithms):
    algo_data = df[df["algorithm"] == algo_name]
    X_raw = algo_data["N"].values.astype(float)
    y_raw = algo_data["avg_time"].values.astype(float)
    
    # Run the ML Log-Log Regression
    log_n = np.log2(X_raw)
    log_time = np.log2(y_raw + 1e-12)
    slope, intercept = np.polyfit(log_n, log_time, 1)
    
    # Classify
    predicted = min(COMPLEXITY_SLOPES, key=lambda k: abs(COMPLEXITY_SLOPES[k] - slope))
    predicted_labels.append(predicted)
    true_labels.append(TRUE_COMPLEXITY.get(algo_name, "Unknown"))
    
    # Generate smooth predicted line data for the plot
    # We create 100 points between min N and max N for a perfectly smooth curve
    X_smooth = np.linspace(min(X_raw), max(X_raw), 100)
    # Reverse the log-log formula to get physical seconds: Time = 2^(slope * log2(N) + intercept)
    y_smooth = 2 ** ((slope * np.log2(X_smooth)) + intercept)
    
    # --- Plotting ---
    ax = axes[idx]
    
    # 1. Scatter plot of the raw micro-benchmarked data
    ax.scatter(X_raw, y_raw, color=colors[idx], s=100, alpha=0.7, edgecolor='black', label="Actual Data")
    
    # 2. Line plot of the ML predicted mathematical curve
    ax.plot(X_smooth, y_smooth, color='black', linewidth=2.5, linestyle='--', label=f"ML Fit: {predicted}")
    
    # Formatting
    ax.set_title(algo_name.replace("_", " ").title(), fontsize=14, fontweight='bold')
    ax.set_xlabel("Input Size (N)")
    ax.set_ylabel("Time (seconds)")
    ax.legend(loc="upper left")

# Hide any empty subplots if we have an odd number of algorithms
for i in range(len(algorithms), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("plots/regression_curves.png", dpi=300, bbox_inches='tight')
print("Saved 'regression_curves.png'")



# --- THE COMPLEXITY CONFUSION MATRIX ---
# Define the exact order of labels we want on the axes
labels_order = ["O(1) [Capped]", "O(log n)", "O(n)", "O(n log n)", "O(n²)"]

# Generate the matrix
cm = confusion_matrix(true_labels, predicted_labels, labels=labels_order)

plt.figure(figsize=(8, 6))
# Create the heatmap using a clean blue gradient
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, 
            xticklabels=labels_order, yticklabels=labels_order, 
            annot_kws={"size": 16, "weight": "bold"})

plt.title("ML Classifier Accuracy (Predicted vs Theoretical)", fontsize=16, fontweight='bold', pad=20)
plt.ylabel("Theoretical Ground Truth", fontsize=12, fontweight='bold')
plt.xlabel("ML Predicted Big-O", fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig("plots/confusion_matrix.png", dpi=300, bbox_inches='tight')
print("Saved 'confusion_matrix.png'")