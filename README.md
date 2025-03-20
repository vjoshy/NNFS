# Neural Networks from Scratch

This repository contains educational materials for learning about neural networks, starting from basic principles and progressively building up to PyTorch implementations.

## Overview

The notebooks in this repository demonstrate the implementation of neural networks from fundamental mathematical principles to practical applications using PyTorch. They were created as teaching materials for a neural networks workshop and guest lecture for MATH*4060 at the University of Guelph.

## Contents

### 1. `NN Guide.ipynb`
- A full guide on coding neural networks from scratch
- Includes a demo using PyTorch

### 2. `PyTorch.ipynb`

- Contains a non-object oriented approach to pyTorch 
- Trains and tests against MNIST digits data


## Getting Started

### Prerequisites

- Python 3.x
- NumPy
- Matplotlib
- PyTorch
- OpenCV 
- Pandas

### Installation

```bash
# Clone the repository
git clone <repository-url>

# Install required packages
pip install numpy matplotlib torch torchvision pandas opencv-python
```

### Dataset

The code uses the MNIST dataset for digit recognition. Make sure to download the dataset or use the provided CSV file named 'train.csv' in data folder.

## Usage Examples

### Basic Neural Network Training

```python
# Data preparation
X = torch.FloatTensor(x_train.T)
y = torch.LongTensor(y_train.astype(int))

# Create a simple neural network
model = nn.Sequential(
    nn.Linear(input_size, hidden_size), 
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size), 
    nn.ReLU(),
    nn.Linear(hidden_size, output_size)
)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Testing on Custom Images

```python
# Load and preprocess your custom image
mnist_img, tensor_img = preprocess_digit("your_digit_image.jpg")

# Use your model to make a prediction
model.eval()
with torch.no_grad():
    output = model(tensor_img)
    _, predicted = torch.max(output, 1)

print(f"Prediction: {predicted.item()}")
```

## Concepts

1. **Neural Networks vs. Statistical Methods**: Neural networks can learn patterns directly from data, unlike rule-based statistical methods.

2. **Neurons and Layers**: A single neuron without activation functions is equivalent to linear regression. Multiple neurons with nonlinear activation functions create more complex models.

3. **Gradient Descent**: An optimization technique to find weights and biases that minimize the loss function by iteratively adjusting parameters in the direction opposite to the gradient.

4. **Activation Functions**: Nonlinear functions (ReLU, tanh) that help neural networks learn complex patterns beyond simple linear relationships.

5. **PyTorch Advantages**: PyTorch provides a flexible framework with GPU support and automatic differentiation through dynamic computational graphs.

## Resources

For deeper understanding, check these resources:

- [StatQuest: Matrix Algebra for Neural Networks](https://statquest.org/essential-matrix-algebra-for-neural-networks-clearly-explained/)
- [PyTorch Internals](https://blog.ezyang.com/2019/05/pytorch-internals/)
- [Neural Networks from Scratch Video Series](https://youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- [Neural Networks from Scratch Book](https://nnfs.io/)

## Acknowledgments

Original content created for Math & Stats Club during Winter 2025 semester at the University of Guelph.