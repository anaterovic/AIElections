import numpy as np

# Suppose you have your original array
original_array = np.array([0.5, 5, 15, 150, 1500])

# Define conditions and corresponding values
conditions = [
    (original_array >= 0) & (original_array < 1),
    (original_array >= 1) & (original_array < 10),
    (original_array >= 10) & (original_array < 100),
    (original_array >= 100) & (original_array < 1000),
    (original_array >= 1000)
]

values = [0.333, 0.5, 1, 2, 3]

# Apply conditions using np.select()
new_array = np.select(conditions, values)

print(new_array)