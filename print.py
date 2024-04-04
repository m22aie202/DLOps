import numpy as np

from graph import sigmoid

random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]

random_values_np = np.array(random_values)

sigmoid_values = sigmoid(random_values_np)

print("\nSigmoid Values:")
for val, sig_val in zip(random_values, sigmoid_values):
        print(f"Sigmoid({val}) = {sig_val}")
