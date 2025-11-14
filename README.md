# Fashion MNIST CNN Models

This project contains CNN (Convolutional Neural Network) implementations for the Fashion MNIST dataset using both Keras and PyTorch.

## Dataset

Fashion MNIST is a dataset of Zalando's article images consisting of:
- 60,000 training examples
- 10,000 test examples
- 10 classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
- 28x28 grayscale images

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Keras Implementation

Run the Keras CNN model:

```bash
python fashion_mnist_keras.py
```

This will:
- Load and preprocess the Fashion MNIST dataset
- Build a CNN model with 3 convolutional layers
- Train the model for 10 epochs
- Evaluate on the test set
- Save the model as `fashion_mnist_keras_model.h5`
- Generate training history plots

### PyTorch Implementation

Run the PyTorch CNN model:

```bash
python fashion_mnist_pytorch.py
```

This will:
- Load and preprocess the Fashion MNIST dataset
- Build a CNN model with 3 convolutional layers
- Train the model for 10 epochs
- Evaluate on the test set
- Save the model as `fashion_mnist_pytorch_model.pth`
- Generate training history plots

## Model Architecture

Both implementations use a similar CNN architecture:

1. **Conv2D Layer 1**: 32 filters, 3x3 kernel, ReLU activation
2. **MaxPooling2D**: 2x2 pool size
3. **Conv2D Layer 2**: 64 filters, 3x3 kernel, ReLU activation
4. **MaxPooling2D**: 2x2 pool size
5. **Conv2D Layer 3**: 64 filters, 3x3 kernel, ReLU activation
6. **Flatten Layer**
7. **Dense Layer**: 64 units, ReLU activation
8. **Dropout**: 0.5 rate
9. **Output Layer**: 10 units (one for each class), Softmax activation

## Expected Results

Both models should achieve approximately:
- **Training Accuracy**: ~90-95%
- **Test Accuracy**: ~88-92%

Note: Results may vary slightly due to random initialization and training dynamics.

