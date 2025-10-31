"""
Assignment 3: Fashion MNIST Classification using TensorFlow Keras
Neural Network for classifying Fashion MNIST dataset
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Fashion MNIST class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def load_and_preprocess_data():
    """
    Load and preprocess the Fashion MNIST dataset
    """
    print("Loading Fashion MNIST dataset...")
    
    # Load Fashion MNIST dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    
    # Display dataset information
    print(f"Training set shape: {x_train.shape}")
    print(f"Test set shape: {x_test.shape}")
    print(f"Number of classes: {len(class_names)}")
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape data for CNN (add channel dimension)
    # For a fully connected network, we'll flatten later
    # x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    # x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    # One-hot encode the labels
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    print("Data preprocessing completed!")
    return (x_train, y_train), (x_test, y_test)

def build_neural_network():
    """
    Build a neural network model using TensorFlow Keras
    """
    print("\nBuilding Neural Network...")
    
    # Create a Sequential model
    model = keras.Sequential([
        # Flatten the 28x28 input images to 784-dimensional vectors
        layers.Flatten(input_shape=(28, 28)),
        
        # First hidden layer with 128 neurons and ReLU activation
        layers.Dense(128, activation='relu', name='hidden_layer_1'),
        layers.Dropout(0.2),  # Dropout for regularization
        
        # Second hidden layer with 64 neurons and ReLU activation
        layers.Dense(64, activation='relu', name='hidden_layer_2'),
        layers.Dropout(0.2),  # Dropout for regularization
        
        # Output layer with 10 neurons (one for each class) and softmax activation
        layers.Dense(10, activation='softmax', name='output_layer')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Display model architecture
    print("\nModel Architecture:")
    model.summary()
    
    return model

def train_model(model, x_train, y_train, x_val, y_val, epochs=10, batch_size=32):
    """
    Train the neural network model
    """
    print("\nTraining the model...")
    
    # Train the model
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        verbose=1
    )
    
    return history

def evaluate_model(model, x_test, y_test):
    """
    Evaluate the model on test data
    """
    print("\nEvaluating model on test data...")
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    
    return test_loss, test_accuracy

def plot_training_history(history):
    """
    Plot training history (loss and accuracy curves)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot training & validation loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot training & validation accuracy
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("\nTraining history plot saved as 'training_history.png'")
    plt.show()

def visualize_predictions(model, x_test, y_test, num_samples=10):
    """
    Visualize some predictions on test data
    """
    print("\nVisualizing predictions...")
    
    # Get predictions
    predictions = model.predict(x_test[:num_samples])
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test[:num_samples], axis=1)
    
    # Create visualization
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for i in range(num_samples):
        axes[i].imshow(x_test[i], cmap='gray')
        axes[i].axis('off')
        
        # Color: green for correct, red for incorrect
        color = 'green' if predicted_classes[i] == true_classes[i] else 'red'
        axes[i].set_title(f'True: {class_names[true_classes[i]]}\n'
                         f'Pred: {class_names[predicted_classes[i]]}',
                         color=color, fontsize=9)
    
    plt.suptitle('Sample Predictions (Green=Correct, Red=Incorrect)', fontsize=12)
    plt.tight_layout()
    plt.savefig('sample_predictions.png', dpi=150, bbox_inches='tight')
    print("Sample predictions plot saved as 'sample_predictions.png'")
    plt.show()

def main():
    """
    Main function to run the complete pipeline
    """
    print("=" * 60)
    print("Fashion MNIST Classification using TensorFlow Keras")
    print("=" * 60)
    
    # Step 1: Load and preprocess data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Step 2: Build the neural network
    model = build_neural_network()
    
    # Step 3: Train the model
    history = train_model(model, x_train, y_train, x_test, y_test, epochs=10, batch_size=32)
    
    # Step 4: Evaluate the model
    test_loss, test_accuracy = evaluate_model(model, x_test, y_test)
    
    # Step 5: Plot training history
    plot_training_history(history)
    
    # Step 6: Visualize some predictions
    visualize_predictions(model, x_test, y_test, num_samples=10)
    
    # Step 7: Save the model
    model.save('fashion_mnist_model.h5')
    print("\nModel saved as 'fashion_mnist_model.h5'")
    
    print("\n" + "=" * 60)
    print("Training and evaluation completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()

