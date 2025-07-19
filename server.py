# To address PyTorch conflicts with NumPy/OpenMP on some systems (especially Windows or Anaconda users)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import flwr as fl
from model import MaskCNN
import torch
from flwr.common import parameters_to_ndarrays
import matplotlib.pyplot as plt
import pandas as pd

# Logging variables
rounds = []
train_accuracies = []
train_losses = []

# Let the server dynamically get the number of clients for this experiment from run_all.py, 5 by default.
NUM_CLIENTS = int(os.environ.get("NUM_CLIENTS", 5))

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.final_parameters = None


    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters, _ = super().aggregate_fit(rnd, results, failures)

        # Save global model
        if aggregated_parameters is not None:
            self.final_parameters = aggregated_parameters

            model = MaskCNN()
            weights = parameters_to_ndarrays(aggregated_parameters)
            params_dict = zip(model.state_dict().keys(), weights)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            model.load_state_dict(state_dict, strict=True)

            model_path = f"global_model_round_{rnd}.pt"
            torch.save(model.state_dict(), model_path)
            print(f"Saved global model to {model_path}")

        # Record metrics
        rounds.append(rnd)
        accuracies = [res.metrics.get("accuracy") for _, res in results if "accuracy" in res.metrics]
        losses = [res.metrics.get("loss") for _, res in results if "loss" in res.metrics]

        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else None
        avg_loss = sum(losses) / len(losses) if losses else None

        train_accuracies.append(avg_accuracy)
        train_losses.append(avg_loss)

        if avg_accuracy is not None:
            print(f"Round {rnd} average training accuracy: {avg_accuracy:.4f}")
        if avg_loss is not None:
            print(f"Round {rnd} average training loss: {avg_loss:.4f}")

        return aggregated_parameters, {}


    def on_training_end(self):
        # Accuracy plot
        if rounds and train_accuracies:
            plt.figure()
            plt.plot(rounds, train_accuracies, marker='o')
            plt.title("Training Accuracy per Round")
            plt.xlabel("Round")
            plt.ylabel("Accuracy")
            plt.grid(True)
            plt.savefig("training_accuracy.png")
            print("Training accuracy plot saved to training_accuracy.png")

        # Loss plot
        if rounds and train_losses:
            plt.figure()
            plt.plot(rounds, train_losses, marker='o', color='orange')
            plt.title("Training Loss per Round")
            plt.xlabel("Round")
            plt.ylabel("Loss")
            plt.grid(True)
            plt.savefig("training_loss.png")
            print("Training loss plot saved to training_loss.png")

        # Create DataFrame
        df = pd.DataFrame({
            'round': rounds,
            'train_accuracy': train_accuracies,
            'train_loss': train_losses
        })

        # Save to csv
        df.to_csv('training_results.csv', index=False)


# Define strategy
strategy = SaveModelStrategy(
    fraction_fit=1.0,
    min_fit_clients=NUM_CLIENTS,
    min_available_clients=NUM_CLIENTS,
    on_fit_config_fn=lambda rnd: {"rnd": rnd},
)


if __name__ == "__main__":
    # Start the Flower Server
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=20),
        strategy=strategy
    )
    # Explicitly call the ending logic
    strategy.on_training_end()
