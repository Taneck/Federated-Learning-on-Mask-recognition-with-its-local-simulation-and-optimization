import os
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import random

class FaceMaskDataset(Dataset):
    """
    Define a series of image enhancement operations 
    that are used to increase the diversity of the training samples (used for the training set only):
    1. Scaling the image to a uniform size 224 x 224;
    2. Random horizontal flip;
    3. Random rotation ±10 degrees;
    4. Brightness/contrast perturbation;
    5. Conversion to PyTorch tensor.
    """
    def __init__(self, root_dir, samples):
        self.root_dir = root_dir
        self.samples = samples
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 确保图像尺寸统一
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor()
        ])

    # Get the number of samples
    def __len__(self):
        return len(self.samples)

    """
    Load a single sample
    1. Reading image files
    2. Converting to RGB (preventing greyscale maps)
    3. Apply transform
    4. Return (image, label)
    """
    def __getitem__(self, idx):
        sub_path, label = self.samples[idx]
        img_path = os.path.join(self.root_dir, sub_path)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label

# Functions that distribute data evenly to each client
def load_partitioned_datasets(root_dir, num_clients=5):
    samples = []

    # Read masked images, labeled as 1
    masked_dir = os.path.join(root_dir, "masked")
    for fname in os.listdir(masked_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            samples.append((f"masked/{fname}", 1))

    # Read unmasked images, labeled as 0
    unmasked_dir = os.path.join(root_dir, "unmasked")
    for fname in os.listdir(unmasked_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            samples.append((f"unmasked/{fname}", 0))

    # Shuffle
    random.seed(42)
    random.shuffle(samples)

    # distribute data evenly to each client
    total = len(samples)
    per_client = total // num_clients

    client_datasets = []
    for i in range(num_clients):
        start = i * per_client
        end = total if i == num_clients - 1 else (i + 1) * per_client
        subset = samples[start:end]
        # Each client gets a FaceMaskDataset instance.
        client_datasets.append(FaceMaskDataset(root_dir, subset))

    return client_datasets
