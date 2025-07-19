# To address PyTorch conflicts with NumPy/OpenMP on some systems (especially Windows or Anaconda users)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import MaskCNN
from dataset import load_partitioned_datasets
import os

# Detect if there is a GPU available, use CUDA if there is, otherwise use CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Read through environment variables
CLIENT_ID = int(os.environ.get("CLIENT_ID", 0))
NUM_CLIENTS = int(os.environ.get("NUM_CLIENTS", 5))

# Each client gets a separate slice of data, packaged as a DataLoader for training purposes
datasets = load_partitioned_datasets("face_images", num_clients=NUM_CLIENTS)
trainset = datasets[CLIENT_ID]
trainloader = DataLoader(trainset, batch_size=16, shuffle=True)

# Inherits the standard client template provided by Flower (NumPyClient) and implements 3 methods
class FlowerClient(fl.client.NumPyClient):
    # Instantiate the model and move it to the specified device
    def __init__(self):
        self.model = MaskCNN().to(DEVICE)


    # Converting model parameters to NumPy arrays (Flower's formatting requirements)
    def get_parameters(self, config=None):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    # Load parameters sent from the server back into the local model
    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for k, v in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(v)
        self.model.load_state_dict(state_dict, strict=True)

    # Local training:
    # 1. Load the parameters sent by the server aggregation
    # 2. Train locally for a number of epochs
    # 3. Return new model parameters, amount of training data, training metrics.
    def fit(self, parameters, config):
        self.set_parameters(parameters) # Load server aggregated global model parameters back into the local model
        self.model.train() # Set the model to training mode
        # Define the loss function.
        # Binary classification -> Using CrossEntropyLoss
        criterion = nn.CrossEntropyLoss()
        # Initialising the optimiser
        # Updating model parameters using Adam's algorithm
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        total_loss = 0.0
        correct = 0
        total = 0

        local_epochs = int(os.environ.get("LOCAL_EPOCHS", 1))

        # Training main loop
        for epoch in range(local_epochs):
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        return self.get_parameters(), len(trainloader.dataset), {
            "loss": avg_loss,
            "accuracy": accuracy
        }


if __name__ == "__main__":
    # The client connects to the server address (local port 8080) and begins participating in federation training
    fl.client.start_numpy_client(server_address="localhost:8080", client=FlowerClient())
