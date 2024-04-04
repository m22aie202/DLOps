import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def tanh(x):
    return np.tanh(x)

x = np.linspace(-5, 5, 100)

sigmoid_output = sigmoid(x)
relu_output = relu(x)
leaky_relu_output = leaky_relu(x)
tanh_output = tanh(x)

plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
plt.plot(x, sigmoid_output, label='Sigmoid')
plt.title('Sigmoid Activation')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(x, relu_output, label='ReLU')
plt.title('ReLU Activation')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(x, leaky_relu_output, label='Leaky ReLU')
plt.title('Leaky ReLU Activation')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(x, tanh_output, label='Tanh')
plt.title('Tanh Activation')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()

plt.tight_layout()
plt.show()
