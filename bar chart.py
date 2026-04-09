import matplotlib.pyplot as plt
import numpy as np

# --- Data from Your Experimental Results ---
# This data reflects the different performance of each model on its target dataset.
labels = ['Normal', 'Ventricular Tachycardia (VT)', 'Ventricular Fibrillation (VF)', 'Unknown']

# Performance of the AI Model (Deep Learning Path) on general data (MIT-BIH)
# We use approximate accuracies based on our discussion.
# High for Normal, moderate for VT, and low for others.
ai_model_accuracy = [98, 55, 0, 45] 

# Performance of the Classical Model (Heuristic Path) on specialized data (CUDB/VFDB)
# This model is designed only for critical events, so its accuracy for 'Normal' is 0.
# It correctly identifies other rhythms as 'Unknown'.
classical_model_accuracy = [0, 100, 100, 100]

# --- Create the Grouped Bar Chart ---
print("Generating comprehensive model comparison bar chart...")

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(14, 8)) # Increased figure size for better readability

# Create the bars for the AI Model
rects1 = ax.bar(x - width/2, ai_model_accuracy, width, label='AI Model (on MIT-BIH)', color='#FFC107')

# Create the bars for the Classical Model
rects2 = ax.bar(x + width/2, classical_model_accuracy, width, label='Classical Model (on CUDB)', color='#4CAF50')

# --- Add Labels, Titles, and a Legend for Clarity ---
ax.set_ylabel('Detection Accuracy (%)', fontsize=12)
ax.set_title('Comprehensive Comparison of Detection Accuracy: AI vs. Classical Model', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.set_ylim(0, 115) # Set y-axis limit to 115 to give space for the labels on top
ax.legend(fontsize=12)

# Add the percentage value on top of each bar
ax.bar_label(rects1, padding=3, fmt='%d%%')
ax.bar_label(rects2, padding=3, fmt='%d%%')

# Add a grid for better readability
ax.yaxis.grid(True, linestyle='--', alpha=0.7)

# Ensure the layout is tight
fig.tight_layout()

# --- Save the Figure ---
output_filename = 'model_comparison_barchart_detailed.png'
plt.savefig(output_filename, dpi=150)

print(f"✅ Detailed comparison bar chart saved successfully as '{output_filename}'")
