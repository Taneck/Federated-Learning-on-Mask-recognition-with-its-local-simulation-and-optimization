import os
import pandas as pd
from scipy.stats import wilcoxon

from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns

# Get the path to the current script
base_dir = os.path.join(os.getcwd(), 'results')

# Initialise list
train_accuracy = []
train_loss = []

# Traverse through the subfolders in the results directory
for subfolder in os.listdir(base_dir):
    subfolder_path = os.path.join(base_dir, subfolder)
    if os.path.isdir(subfolder_path):
        csv_path = os.path.join(subfolder_path, 'training_results.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, sep=',')  # seperated by \t
            # Get train_accuracy and train_loss column
            acc_values = df['train_accuracy'].tolist()
            loss_values = df['train_loss'].tolist()
            # Append to list
            train_accuracy.append((subfolder, acc_values))
            train_loss.append((subfolder, loss_values))

# Build DataFrame
accuracy_dict = {name: acc_list[-3:] for name, acc_list in train_accuracy}
df = pd.DataFrame(accuracy_dict)

# Check alignment
if not all(len(v) == len(df.columns) for v in df.values):
    raise ValueError("The length of accuracy in each group is not consistent and cannot be compared")

# Friedman test
stat, p = friedmanchisquare(*[df[col] for col in df.columns])
print(f"Friedman Test results: statistic = {stat:.4f}, p-value = {p:.4f}")

if p < 0.05:
    print("Significantly different, Nemenyi test was performed")
    nemenyi = sp.posthoc_nemenyi_friedman(df.values)
    nemenyi.index = df.columns
    nemenyi.columns = df.columns
    
    # Unlimited output
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    print("\nNemenyi Test results (p-value matrix):")
    print(nemenyi)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(nemenyi, annot=True, fmt=".3f", cmap="viridis", cbar=True)
    plt.title("Nemenyi Post-Hoc Test (p-value Heatmap)")
    plt.tight_layout()
    plt.savefig("nemenyi_heatmap.png", dpi=300)
    plt.close()

    # Output average ranking
    ranks = df.rank(axis=1, ascending=False)
    avg_ranks = ranks.mean().sort_values()
    print("\nAverage ranking (smaller is better):")
    print(avg_ranks)

else:
    print("There were no significant differences, no follow-up tests were performed.")