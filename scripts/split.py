import torch

def split_indices(dataset_length, train_ratio=0.8):
    indices = torch.randperm(dataset_length).tolist()
    train_size = int(dataset_length * train_ratio)
    
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]
    
    return train_idx, test_idx