import subprocess
import time
import os
import shutil
import sys

import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime

# Set the combination of parameters
# Format: (local_epochs, num_clients)
param_grid = [
    (1, 4),
    (2, 4),
    (4, 4),
    (8, 4),
    (1, 8),
    (2, 8),
    (4, 8),
    (8, 8),
    (1, 16),
    (2, 16),
    (4, 16),
    (8, 16),
]

# Used to record logs for each set of experiments (timestamps, configurations, elapsed time, etc.)
log_lines = []

# Main loop (traversing param_grid)
for local_epochs, num_clients in param_grid:
    start_time = datetime.now()
    
    print(f"\n ========== Running experiment: local_epochs={local_epochs}, num_clients={num_clients} ========== \n")
    sys.stdout.flush()

    # Experimental results output directory
    exp_dir = f"results/ep{local_epochs}_cli{num_clients}"
    os.makedirs(exp_dir, exist_ok=True)

    # Set environment variables
    env = os.environ.copy()
    env["LOCAL_EPOCHS"] = str(local_epochs)
    env["NUM_CLIENTS"] = str(num_clients)

    # Start the server
    server_proc = subprocess.Popen(["python", "server.py"], env=env)
    time.sleep(3)  # Waiting for the server to initialise

    # Start the clients
    # Assign a separate CLIENT_ID to each client and pass it in via an environment variable.
    clients = []
    for i in range(num_clients):
        client_env = env.copy()
        client_env["CLIENT_ID"] = str(i)
        proc = subprocess.Popen(["python", "client.py"], env=client_env)
        clients.append(proc)

    # Wait for all clients to finish
    for proc in clients:
        proc.wait()

    # Wait for the server to finish all rounds
    server_proc.wait()

    # Move the models and charts generated in this round to the targeted results output directory
    for fname in os.listdir("."):
        if fname.startswith("global_model_round_") or fname.endswith(".png") or fname.endswith(".csv"):
            shutil.move(fname, os.path.join(exp_dir, fname))

    # Calculate the elapsed time
    end_time = datetime.now()
    duration = end_time - start_time

    # Store the log
    log_lines.append(
        f"[{start_time.strftime('%Y-%m-%d %H:%M:%S')} - {end_time.strftime('%Y-%m-%d %H:%M:%S')}] "
        f"local_epochs={local_epochs}, num_clients={num_clients}, duration={duration}, results_dir={exp_dir}\n"
    )

    print(f"\n ========== Experiment done. Results saved to {exp_dir} ========== \n")


# Write log into file
with open("experiment_log.txt", "a") as log_file:
    log_file.writelines(log_lines)

# Set the result directory
root_dir = "results"

# Draw the train_accuracy plot
plt.figure(figsize=(10, 6))
any_acc = False

for subdir in os.listdir(root_dir):
    full_path = os.path.join(root_dir, subdir)
    if os.path.isdir(full_path) and "cli" in subdir:
        csv_path = os.path.join(full_path, "training_results.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if {"round", "train_accuracy", "train_loss"}.issubset(df.columns):
                df = df.dropna(subset=["round", "train_accuracy", "train_loss"])
                if not df.empty:
                    plt.plot(df["round"], df["train_accuracy"], label=f"{subdir}")
                    any_acc = True

plt.xlabel("Round")
plt.ylabel("Train Accuracy")
plt.title("Train Accuracy over Rounds")
if any_acc:
    plt.legend()
else:
    print("No accuracy data to plot.")
plt.grid(True)
plt.tight_layout()
plt.savefig("train_accuracy_over_rounds.png")
plt.close()


# Draw the train_loss plot
plt.figure(figsize=(10, 6))
any_loss = False

for subdir in os.listdir(root_dir):
    full_path = os.path.join(root_dir, subdir)
    if os.path.isdir(full_path) and "cli" in subdir:
        csv_path = os.path.join(full_path, "training_results.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if {"round", "train_accuracy", "train_loss"}.issubset(df.columns):
                df = df.dropna(subset=["round", "train_accuracy", "train_loss"])
                if not df.empty:
                    plt.plot(df["round"], df["train_loss"], label=f"{subdir}")
                    any_loss = True

plt.xlabel("Round")
plt.ylabel("Train Loss")
plt.title("Train Loss over Rounds")
if any_loss:
    plt.legend()
else:
    print("No loss data to plot.")
plt.grid(True)
plt.tight_layout()
plt.savefig("train_loss_over_rounds.png")
plt.close()

