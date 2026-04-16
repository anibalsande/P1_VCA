from matplotlib.pylab import indices
import torch

def split_indices(dataset_length, train_ratio=0.8, seed=42):
    generator = torch.Generator()
    generator.manual_seed(seed)

    indices = torch.randperm(dataset_length, generator=generator).tolist()

    train_size = int(dataset_length * train_ratio)    
    
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]    
    
    return train_idx, test_idx