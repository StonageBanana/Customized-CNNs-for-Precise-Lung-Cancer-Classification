# Customized CNNs for Precise Lung Cancer Classification

## 📖 Project Overview

This repository contains a comprehensive implementation of customized Convolutional Neural Networks (CNNs) for precise lung cancer classification using histopathological images. The project explores various CNN architectures and techniques to improve the accuracy of lung cancer detection and classification.

## 🎯 Objectives

- Develop and compare multiple CNN architectures for lung cancer classification
- Implement data augmentation techniques to handle limited dataset size
- Optimize model performance through hyperparameter tuning
- Provide a robust framework for medical image classification tasks

## 📁 Project Structure
```text
Customized-CNNs-for-Precise-Lung-Cancer-Classification/
├── data/
│ ├── raw/ # Original dataset
│ ├── processed/ # Preprocessed images
│ └── augmented/ # Augmented dataset
├── models/
│ ├── base_cnn.py # Basic CNN implementation
│ ├── resnet_custom.py # Custom ResNet architecture
│ ├── efficientnet_custom.py # Custom EfficientNet implementation
│ └── model_utils.py # Model utility functions
├── notebooks/
│ ├── 01_data_exploration.ipynb
│ ├── 02_data_preprocessing.ipynb
│ ├── 03_model_training.ipynb
│ └── 04_model_evaluation.ipynb
├── src/
│ ├── data_preprocessing.py
│ ├── data_augmentation.py
│ ├── model_training.py
│ ├── evaluation_metrics.py
│ └── visualization.py
├── results/
│ ├── trained_models/ # Saved model weights
│ ├── training_logs/ # Training history and logs
│ └── performance_metrics/ # Evaluation results
├── requirements.txt
├── config.yaml
└── run_training.py
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for training)

### Setup
1. Clone the repository:
```bash
git clone https://github.com/StonageBanana/Customized-CNNs-for-Precise-Lung-Cancer-Classification.git
cd Customized-CNNs-for-Precise-Lung-Cancer-Classification
``` 
Install required dependencies:

```bash
pip install -r requirements.txt
```
## 📊 Dataset
The project uses the LC25000 lung cancer histopathology dataset, which contains:
- 25,000 color images (500x500 pixels)
- 5 classes: benign tissue, lung adenocarcinoma, lung squamous cell carcinoma, colon adenocarcinoma, and colon benign tissue
- Balanced distribution across all classes

### Data Preparation
1. Download the dataset from the official source
2. Place the data in the data/raw/ directory
3. Run the preprocessing pipeline:
```bash
python src/data_preprocessing.py
```

## 🧠 Model Architectures
1. Basic CNN
* Custom-built CNN with multiple convolutional layers
* Batch normalization and dropout for regularization
* Adaptive learning rate scheduling

2. Custom ResNet
* Modified ResNet architecture with attention mechanisms
* Feature pyramid network for multi-scale feature extraction
* Transfer learning from ImageNet pre-trained weights

3. Custom EfficientNet
* EfficientNet backbone with custom classification head
* Compound scaling for optimal performance-efficiency tradeoff
* Advanced regularization techniques

## 🚀 Usage
### Training
```bash
# Train all models
python run_training.py --config config.yaml

# Train specific model
python run_training.py --model resnet --epochs 100 --batch_size 32
```

### Evaluation
```bash
python src/evaluation_metrics.py --model_path results/trained_models/best_model.pth
```

### Inference
```python
from models.resnet_custom import CustomResNet
from src.model_training import load_model

model = load_model('results/trained_models/best_model.pth')
prediction = model.predict(image)
```

## ⚙️ Configuration
Modify config.yaml to adjust training parameters:

```yaml
training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
  optimizer: "adam"
  
data:
  image_size: 224
  augmentation: true
  validation_split: 0.2
  
model:
  architecture: "resnet"
  pretrained: true
  num_classes: 5
```

## 📈 Results
### Performance Metrics
| Model	| Accuracy | Precision | Recall	| F1-Score |
|-------|----------|-----------|--------|----------|
|Basic CNN	| 94.2%	| 94.1%	| 94.0%	| 94.0% |
|Custom ResNet	| 96.8%	| 96.7%	| 96.6%	| 96.6% |
|Custom EfficientNet	| 97.2%	| 97.1%	| 97.0%	| 97.0% |

### Key Findings
* Custom EfficientNet achieved the highest accuracy (97.2%)
* Data augmentation significantly improved model generalization
* Transfer learning provided substantial performance gains
* Attention mechanisms enhanced feature representation

## 🎨 Visualization
The project includes comprehensive visualization tools for:
* Training/validation curves
* Confusion matrices
* Feature maps and attention visualization
* Grad-CAM heatmaps for model interpretability

🤝 Contributing
We welcome contributions! Please feel free to submit pull requests, report bugs, or suggest new features.
* Fork the repository
* Create a feature branch (git checkout -b feature/AmazingFeature)
* Commit your changes (git commit -m 'Add some AmazingFeature')
* Push to the branch (git push origin feature/AmazingFeature)
* Open a Pull Request

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments
* LC25000 dataset providers
* PyTorch and torchvision communities
* Contributors and reviewers
