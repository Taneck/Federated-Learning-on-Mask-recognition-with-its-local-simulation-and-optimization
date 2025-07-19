import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta

# Ranking data (smaller is better)
ranking_data = {
    "ep8_cli4": 1.0,
    "ep8_cli8": 2.333333,
    "ep4_cli4": 2.666667,
    "ep8_cli16": 4.0,
    "ep4_cli8": 5.0,
    "ep2_cli4": 6.0,
    "ep4_cli16": 7.0,
    "ep1_cli4": 8.333333,
    "ep2_cli8": 8.666667,
    "ep2_cli16": 10.0,
    "ep1_cli8": 11.0,
    "ep1_cli16": 12.0
}

# Duration data (Corresponding order)
duration_str = [
    "0:07:17.524822", "0:05:23.454401", "0:03:57.617411", "0:05:27.028907",
    "0:03:03.464184", "0:02:10.742634", "0:03:20.780175", "0:01:19.463319",
    "0:01:53.504814", "0:02:24.454058", "0:01:18.011166", "0:01:53.314059"
]

# Convert durations to seconds
durations = [
    timedelta(
        hours=int(t.split(':')[0]), 
        minutes=int(t.split(':')[1]), 
        seconds=float(t.split(':')[2])
    ).total_seconds() for t in duration_str
]

# Prepare data
labels = list(ranking_data.keys())
rankings = [ranking_data[label] for label in labels]
ranking_array = np.array(rankings)
duration_array = np.array(durations)

# Pareto frontier: minimize both duration and ranking
is_pareto = np.ones(len(ranking_array), dtype=bool)
for i in range(len(ranking_array)):
    is_dominated = (ranking_array <= ranking_array[i]) & (duration_array <= duration_array[i])
    is_dominated[i] = False
    if np.any(is_dominated):
        is_pareto[i] = False

# Get Pareto-optimal points
pareto_durations = duration_array[is_pareto]
pareto_rankings = ranking_array[is_pareto]
pareto_labels = np.array(labels)[is_pareto]

# Sort Pareto points for plotting
sorted_idx = np.argsort(pareto_durations)
pareto_durations = pareto_durations[sorted_idx]
pareto_rankings = pareto_rankings[sorted_idx]
pareto_labels = pareto_labels[sorted_idx]

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(duration_array, ranking_array, color='lightgray', label='All Configurations')
plt.plot(pareto_durations, pareto_rankings, color='red', linestyle='--', marker='o', label='Pareto Frontier')

# Annotate all points
for i, label in enumerate(labels):
    plt.text(duration_array[i], ranking_array[i], label, fontsize=9, ha='right')

plt.xlabel("Time (seconds)")
plt.ylabel("Average Accuracy Ranking (lower is better)")
plt.title("Pareto Frontier: Training Time vs. Accuracy Ranking")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pareto_frontier.png", dpi=300)
