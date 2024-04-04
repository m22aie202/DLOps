import numpy as np

from graph import sigmoid
from graph import relu,leaky_relu,tanh

random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]

random_values_np = np.array(random_values)

sigmoid_values = sigmoid(random_values_np)

print("\nSigmoid Values:")
for val, sig_val in zip(random_values, sigmoid_values):
        print(f"Sigmoid({val}) = {sig_val}")

relu_values = relu(random_values_np)
leaky_relu_values = leaky_relu(random_values_np)
tanh_values = tanh(random_values_np)

print("ReLU Values:")
for val, relu_val in zip(random_values, relu_values):
        print(f"ReLU({val}) = {relu_val}")

        print("\nLeaky ReLU Values:")
        for val, leaky_relu_val in zip(random_values, leaky_relu_values):
                print(f"Leaky ReLU({val}) = {leaky_relu_val}")

                print("\nTanh Values:")
                for val, tanh_val in zip(random_values, tanh_values):
                        print(f"Tanh({val}) = {tanh_val}")
