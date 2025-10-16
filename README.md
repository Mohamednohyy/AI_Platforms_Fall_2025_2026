# Fashion-MNIST MLP

A simple multilayer perceptron (fully connected network) trained on Fashion-MNIST using PyTorch.

## Requirements

See `requirements.txt`.

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Open and run the Jupyter Notebook:
   ```bash
   jupyter notebook fashion_mnist_mlp.ipynb
   ```

## Outputs
- Training and test accuracy per epoch plots
- Final train/test accuracy
- Confusion matrix for test set
- Example correct and incorrect predictions

## Notes
- Achieves >85% test accuracy with the provided settings (8 epochs, Adam lr=1e-3).
- GPU is optional; CPU works albeit slower.
