from torch.utils.data import random_split


def split_dataset(dataset, train_ratio=0.8):

    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size

    return random_split(dataset, [train_size, val_size])


if __name__ == "__main__":
    print("Testing split module")