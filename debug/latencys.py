import numpy as np

# Set the seed for reproducibility
np.random.seed(42)

# Set the number of latency values
K = 56

# Generate random latency values between 0 and 1 (excluding 0 and 1)
latency_variations = np.random.uniform(low=0.0, high=1.0, size=K)

# Ensure no values are exactly 0 or 1
# latency_variations = np.clip(latency_variations, 1e-10, 1 - 1e-10)

# Sort the latency values in ascending order
latency_variations = np.sort(latency_variations)

# Save the numpy array
np.save(f'latency_variations_{K}.npy', latency_variations)
print(f"Latency variations saved as 'latency_variations_{K}.npy'")

# Optional: Print the generated values
print("Generated latency values:")
print(latency_variations)