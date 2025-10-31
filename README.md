# Assignment 3: Fashion MNIST Classification

This project implements a Neural Network using TensorFlow Keras to perform classification on the Fashion MNIST dataset.

## Dataset

Fashion MNIST is a dataset of Zalando's article images consisting of:
- **60,000 training images** and **10,000 test images**
- **10 classes**: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
- Each image is a **28x28 grayscale image**

## Model Architecture

The neural network consists of:
- **Input Layer**: Flattened 28×28 images (784 neurons)
- **Hidden Layer 1**: 128 neurons with ReLU activation + Dropout (0.2)
- **Hidden Layer 2**: 64 neurons with ReLU activation + Dropout (0.2)
- **Output Layer**: 10 neurons with Softmax activation (one for each class)

## Features

- Data preprocessing and normalization
- Neural network model with dropout regularization
- Model training with validation
- Performance evaluation
- Visualization of training history
- Sample prediction visualization

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python fashion_mnist_classifier.py
```

The script will:
1. Load and preprocess the Fashion MNIST dataset
2. Build and compile the neural network
3. Train the model for 10 epochs
4. Evaluate on test data
5. Generate visualization plots
6. Save the trained model

## Output Files

- `fashion_mnist_model.h5`: Saved trained model
- `training_history.png`: Training loss and accuracy curves
- `sample_predictions.png`: Visualization of sample predictions

## Expected Performance

With the default configuration, the model typically achieves:
- **Test Accuracy**: ~87-89%

## Model Parameters

- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: 32
- **Epochs**: 10
- **Regularization**: Dropout (0.2)

