# Federated Learning on Mask recognition with its local simulation and optimization

This repository contains the code for my research paper:

**"Federated Learning on Mask recognition with its local simulation and optimization"**  
Status: Under Review

---

## Disclaimer

This code is publicly available **solely for peer review and reproducibility purposes**.  
**Please do not redistribute, reuse, or publish** any part of this code or results until the paper is officially accepted and published.

---

## Project Overview

I implement a lightweight convolutional neural network (CNN) for binary face mask detection using the **Flower federated learning framework**.  
The project supports multiple training configurations with different numbers of clients and local epochs.

---

## Project Structure
```text
.
├── client.py # Federated client logic
├── server.py # Federated server with training & result logging
├── model.py # CNN model definition (MaskCNN)
├── dataset.py # Dataset loading and partitioning for federated learning
├── run_all.py # Automated running across configurations
├── evaluate_model.py # Final model evaluation on a test set
├── predict_gui.py # Tkinter-based GUI for prediction
├── pareto_frontier.py # Plot Pareto frontier of time vs accuracy
├── statistics_test.py # Statistical tests (Friedman + Nemenyi)
├── requirements.txt # Dependencies
├── LICENSE # Custom pre-publication license
├── results/ # Output of training runs
├──face_images/
    ├── masked/ # Images with masks
    └── unmasked/ # Images without masks
├──face_images_test/
    ├── masked/ # Images with masks
    └── unmasked/ # Images without masks
```

## Dependencies
```bash
pip install -r requirements.txt
```

## Dataset
Due to GitHub's file limit, you may need to download the dataset manually from:
https://drive.google.com/file/d/17brKc8PfE7_99IeWj0t2ShLdu5rcklYx/view?usp=sharing
The original dataset is from:
https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset

After downloading, just organize the image folder as shown in project structure part.

## How to run
**Make sure you have put the image data correctly in the folder**
```bash
python run_all.py
```

## Evaluate the final model
```bash
python evaluate_model.py
```

## Launch GUI for real-time prediction
```bash
python predict_gui.py
```

## Statistical Evaluation
```bash
python statistics_test.py
```

## Results
The results can been seen in generated plots and results/ folder

## License
This repository is protected under a custom temporary license until the paper is officially published.

## Author
**Liukai Tang**
Faculty of Engineering, The University of Sydney, Australia

## Acknowledgements
This project was build on the foundation source code as part of a group study on distributed learning, under the course.
**Distributed Machine Learning: Foundations and Algorithms**.
And I made a major contribution to the original source code.