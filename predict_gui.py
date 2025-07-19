import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import tkinter as tk
from tkinter import filedialog, Label
from PIL import Image, ImageTk
import torch
from torchvision import transforms
from model import MaskCNN
import os

# Load trained model
model_path = "results/ep8_cli4/global_model_round_20.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MaskCNN()
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
model.to(device)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Prediction function
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        return "With Mask" if predicted.item() == 1 else "Without Mask"

# GUI setup
window = tk.Tk()
window.title("Mask Detection")
window.geometry("400x450")
window.resizable(False, False)

label_result = Label(window, text="Please upload an image", font=("Arial", 14))
label_result.pack(pady=10)

label_image = Label(window)
label_image.pack()

def upload_and_predict():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
    )
    if not file_path:
        return

    # Show image
    img = Image.open(file_path).resize((224, 224))
    img_tk = ImageTk.PhotoImage(img)
    label_image.configure(image=img_tk)
    label_image.image = img_tk

    # Predict
    result = predict_image(file_path)
    label_result.config(text=f"Prediction: {result}")

btn_upload = tk.Button(window, text="Upload Image", command=upload_and_predict, font=("Arial", 12))
btn_upload.pack(pady=20)

window.mainloop()
