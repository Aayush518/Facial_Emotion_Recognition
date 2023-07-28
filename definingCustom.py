import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

class FER2013Dataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_pixels = [int(pixel) for pixel in self.images[idx].split()]
        img_array = torch.tensor(img_pixels, dtype=torch.uint8).view(48, 48)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            img_array = self.transform(img_array)

        return img_array, label

# Define data transformations
transform = ToTensor()

# Create custom datasets and data loaders
train_dataset = FER2013Dataset(X_train, y_train, transform=transform)
val_dataset = FER2013Dataset(X_val, y_val, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
