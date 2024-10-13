import torch
import itertools
from typing import List, Tuple
import numpy as np
import random
from typing import List

# Set a seed for reproducibility
random.seed(42)  # You can change 42 to any integer you prefer



from pdb import set_trace as pds
def generate_mask_variations(K: int) -> List[torch.Tensor]:
    base_mask = torch.ones(24, 1)
    variations = []

    for num_zeros in range(22):  # 0 to 21 zeros
        potential_positions = list(range(3, 24))  # Positions 3 to 23
        zero_combinations = list(itertools.combinations(potential_positions, num_zeros))
        
        # Subsample K combinations if there are more than K
        if len(zero_combinations) > K:
            zero_combinations = random.sample(zero_combinations, K)
        
        for zero_positions in zero_combinations:
            mask = base_mask.clone()
            mask[list(zero_positions)] = 0
            variations.append(mask)

    return variations

# Usage
K = 30  # Number of subsamples per group
mask_variations = generate_mask_variations(K)
# Stack all variations into a single numpy array
stacked_variations = np.stack(mask_variations)



print(f"Shape of mask_variations: {stacked_variations.shape}")

pds()
# Save the numpy array
np.save(f'mask_variations_{K}.npy', stacked_variations)
print(f"Mask variations saved as 'mask_variations_{K}.npy'")



pds()