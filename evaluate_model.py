# To address PyTorch conflicts with NumPy/OpenMP on some systems (especially Windows or Anaconda users)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from model import MaskCNN
from dataset import FaceMaskDataset
import os
from PIL import Image

# Settings
TEST_DIR = "face_images_test"
MODEL_PATH = "results/ep8_cli4/global_model_round_20.pt"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = MaskCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Collect test samples
def load_test_samples(root_dir):
    samples = []
    # Traverse through the face_images_test/masked and unmasked folders, setting the labels to 1 and 0 respectively
    for label_name, label in [("masked", 1), ("unmasked", 0)]:
        dir_path = os.path.join(root_dir, label_name)
        for fname in os.listdir(dir_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append((f"{label_name}/{fname}", label))
    return samples

# Define Transform (no data enhancement in the test phase)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Build the test set class (inherit from the existing training set class)
class TestDataset(FaceMaskDataset):
    def __init__(self, root_dir, samples):
        super().__init__(root_dir, samples)
        self.transform = transform  # override with no augmentation

samples = load_test_samples(TEST_DIR)
testset = TestDataset(TEST_DIR, samples)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

# Evaluation
total_loss = 0.0
correct = 0
total = 0
criterion = torch.nn.CrossEntropyLoss()

# Use torch.no_grad() to disable gradient computation and improve efficiency
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

avg_loss = total_loss / total
accuracy = correct / total

print(f"Test Loss: {avg_loss:.4f}")
print(f"Test Accuracy: {accuracy:.2%}")
