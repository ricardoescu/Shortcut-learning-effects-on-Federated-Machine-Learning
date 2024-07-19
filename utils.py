import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from PIL import Image

DATA_DIR = "./data/"

COLOR_MAP = {
    0: (255, 0, 0),  # Red
    1: (0, 255, 0),  # Green
    2: (0, 0, 255),  # Blue
    3: (255, 255, 0),  # Yellow
    4: (255, 0, 255),  # Magenta
    5: (0, 255, 255),  # Cyan
    6: (255, 165, 0),  # Orange
    7: (128, 0, 128),  # Purple
    8: (165, 42, 42),  # Brown
    9: (255, 192, 203)  # Pink
}

NEW_COLOR_MAP = {
    0: (0, 255, 0),  # Green
    1: (255, 0, 0),  # Red
    2: (255, 255, 0),  # Yellow
    3: (0, 0, 255),  # Blue
    4: (0, 255, 255),  # Cyan
    5: (255, 0, 255),  # Magenta
    6: (128, 0, 128),  # Purple
    7: (255, 165, 0),  # Orange
    8: (255, 192, 203),  # Pink
    9: (165, 42, 42)  # Brown
}

def expand_grayscale_to_rgb(image):
    return image.repeat(3, 1, 1)

def colorize_digit(image, label, color_map):
    label = label.item()
    color = color_map[label]
    image = expand_grayscale_to_rgb(image)
    image = transforms.ToPILImage()(image)
    np_image = np.array(image)

    color_mask = np.full_like(np_image, color, dtype=np.uint8)

    alpha = 0.5
    colorized_image = (alpha * color_mask + (1 - alpha) * np_image).astype(np.uint8)

    return transforms.ToTensor()(colorized_image)

class ColorizedMNIST(datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, color_map=COLOR_MAP):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.data = torch.stack([colorize_digit(img, lbl, color_map) for img, lbl in zip(self.data, self.targets)])

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        if self.transform is not None:
            if isinstance(img, torch.Tensor):
                img = transforms.ToPILImage()(img)
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

def mnist_loader(val_split=0.2, batch_size=5):
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.MNIST(root=DATA_DIR + "mnist", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=DATA_DIR + "mnist", train=False, download=True, transform=transform)

    val_size = int(val_split * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("-" * 30 + "MNIST DATASET" + "-" * 30)
    print("Train Set size: ", len(train_dataset))
    print("Validation Set size: ", len(val_dataset))
    print("Test Set size: ", len(test_dataset))

    return train_dataloader, val_dataloader, test_dataloader

def color_mnist_loader(val_split=0.2, batch_size=5):
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = ColorizedMNIST(root=DATA_DIR + "color_mnist", train=True, download=True, transform=transform)
    test_dataset = ColorizedMNIST(root=DATA_DIR + "color_mnist", train=False, download=True, transform=transform)

    val_size = int(val_split * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("-" * 30 + "COLORIZED MNIST DATASET" + "-" * 30)
    print("Train Set size: ", len(train_dataset))
    print("Validation Set size: ", len(val_dataset))
    print("Test Set size: ", len(test_dataset))

    return train_dataloader, val_dataloader, test_dataloader

def grayscale_mnist_3_channels_loader(val_split=0.2, batch_size=5):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3, 1, 1))])

    train_dataset = datasets.MNIST(root=DATA_DIR + "grayscale_mnist_3_channels", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=DATA_DIR + "grayscale_mnist_3_channels", train=False, download=True, transform=transform)

    val_size = int(val_split * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("-" * 30 + "GRAYSCALE MNIST 3 CHANNELS DATASET" + "-" * 30)
    print("Train Set size: ", len(train_dataset))
    print("Validation Set size: ", len(val_dataset))
    print("Test Set size: ", len(test_dataset))

    return train_dataloader, val_dataloader, test_dataloader

def bias_conflicting_mnist_loader(val_split=0.2, batch_size=5):
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = ColorizedMNIST(root=DATA_DIR + "bias_conflicting_mnist", train=True, download=True, transform=transform, color_map=NEW_COLOR_MAP)
    test_dataset = ColorizedMNIST(root=DATA_DIR + "bias_conflicting_mnist", train=False, download=True, transform=transform, color_map=NEW_COLOR_MAP)

    val_size = int(val_split * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("-" * 30 + "BIAS CONFLICTING MNIST DATASET" + "-" * 30)
    print("Train Set size: ", len(train_dataset))
    print("Validation Set size: ", len(val_dataset))
    print("Test Set size: ", len(test_dataset))

    return train_dataloader, val_dataloader, test_dataloader

def load_dataset(val_split=0.2, batch_size=5, dataset="mnist", color_map=COLOR_MAP):
    if dataset == "mnist":
        return mnist_loader(val_split, batch_size)
    if dataset == "color_mnist":
        return color_mnist_loader(val_split, batch_size)
    if dataset == "grayscale_mnist_3_channels":
        return grayscale_mnist_3_channels_loader(val_split, batch_size)
    if dataset == "bias_conflicting_mnist":
        return bias_conflicting_mnist_loader(val_split, batch_size)

def visualize_dataset(dataloaders, titles=["Train Set", "Validation Set", "Test Set"], grayscale=True):
    fig, axes = plt.subplots(3, 5, figsize=(20, 15))
    for i, dataloader in enumerate(dataloaders):
        data_iter = iter(dataloader)
        images, labels = next(data_iter)

        for j in range(5):
            img = images[j]
            print(f"Image {j} shape: {img.shape}")  # Print the shape to verify the number of channels
            if isinstance(img, torch.Tensor):
                img = img.permute(1, 2, 0)  # Convert tensor to (height, width, channels) format for visualization
            if grayscale:
                axes[i, j].imshow(img[:, :, 0], cmap='gray')  # Display only one channel for grayscale
            else:
                axes[i, j].imshow(img)
            axes[i, j].axis('off')
            print(f"Image {j} label: {labels[j]}")

        axes[i, 0].set_ylabel(titles[i], fontsize=16)

    plt.show()



def check_data_distribution(dataloader, name):
    label_counts = np.zeros(10)
    for _, labels in dataloader:
        for label in labels:
            label_counts[label] += 1
    #print(f"Data distribution in {name}: {label_counts}")




def visualize_client_data(distributed_data):
    """
    Visualizes one example from each client's dataset.
    :param distributed_data: List of datasets distributed to each client.
    """
    num_clients = len(distributed_data)
    fig, axes = plt.subplots(1, num_clients, figsize=(15, 5))

    for i, client_data in enumerate(distributed_data):
        if len(client_data) > 0:
            # Get one sample
            data, target = client_data[0]

            # Remove batch dimension
            if len(data.shape) == 4:  # If there is a batch dimension
                data = data[0]
                target = target[0]

            # Handle grayscale and color images
            if data.shape[0] == 3:  # If the data is color (3 channels)
                data = data.permute(1, 2, 0)  # Change dimensions for plotting
                axes[i].imshow(data)
            else:  # Grayscale
                axes[i].imshow(data.squeeze(), cmap='gray')

            axes[i].set_title(f'Client {i + 1}, Label: {target.item()}')
            axes[i].axis('off')

    plt.show()


if __name__ == "__main__":
    train, validation, test = load_dataset(dataset="mnist")
    visualize_dataset([train, validation, test],
                      titles=["Grayscale Train Set", "Grayscale Validation Set", "Grayscale Test Set"], grayscale=False)

    train_color, val_color, test_color = load_dataset(dataset="color_mnist")
    visualize_dataset([train_color, val_color, test_color],
                      titles=["Colorized Train Set", "Colorized Validation Set", "Colorized Test Set"], grayscale=False)

    train_3gray, val_3gray, test_3gray = load_dataset(dataset="grayscale_mnist_3_channels")
    visualize_dataset([train_3gray, val_3gray, test_3gray],
                      titles=["3gray Train Set", "3gray Validation Set", "3gray Test Set"], grayscale=False)

    train_conflicting, val_conflicting, test_conflicting = load_dataset(dataset="bias_conflicting_mnist")
    visualize_dataset([train_conflicting, val_conflicting, test_conflicting],
                      titles=["Conflicting Train Set", "Conflicting Validation Set", "Conflicting Test Set"], grayscale=False)
