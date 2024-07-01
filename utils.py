import os
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

DATA_DIR = "./data/"
BIAS_DIR = "./data/biased_mnist/"
NUM_CLIENTS = 15
CLASSES_PER_CLIENT = 5
SPURIOUS_FEATURE_PROB = 0.9

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


def expand_grayscale_to_rgb(image):
    """
    :param image: Single-channel grayscale image (original mnist)
    :return: 3-channel RGB image (by repeating the grayscale channel)
    """
    return image.repeat(3, 1, 1)

def colorize_digit(image, label):
    label = label.item()  # Convert tensor to integer
    color = COLOR_MAP[label]
    image = expand_grayscale_to_rgb(image) # Ensure the image is in RGB format
    image = transforms.ToPILImage()(image)
    image = image.convert("RGB")
    np_image = np.array(image)

    color_mask = np.full_like(np_image, color, dtype=np.uint8)

    alpha = 0.5
    colorized_image = (alpha * color_mask + (1 - alpha) * np_image).astype(np.uint8)

    return transforms.ToTensor()(colorized_image)


class ColorizedMNIST(datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, colorize=False):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.colorize = colorize
        if self.colorize:
            self.data = torch.stack([colorize_digit(img, lbl) for img, lbl in zip(self.data, self.targets)])
        else:
            self.data = torch.stack([expand_grayscale_to_rgb(img) for img in self.data])

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        if not self.colorize:
            img = transforms.ToPILImage()(img) # converting tensor to PIL Image

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def mnist_loader(val_split=0.2, batch_size=5, colorize=False):
    """
    Loads the MNIST data into 3 sets: train, validation, and testing.
    :param val_split: float value to decide the train/val split.
    :param batch_size: int defining the batch size of the dataset.
    :param colorize: boolean to decide if the dataset should be colorized.
    :return:
    """
    transform = transforms.Compose([transforms.ToTensor()])

    # load the dataset
    train_dataset = ColorizedMNIST(root=DATA_DIR + "mnist", train=True, download=True, transform=transform,
                                   colorize=colorize)
    test_dataset = ColorizedMNIST(root=DATA_DIR + "mnist", train=False, download=True, transform=transform,
                                  colorize=colorize)

    # Shuffle and split train and validation set
    val_size = int(val_split * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Define dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("-" * 30 + "MNIST DATASET" + "-" * 30)
    print(f"Train Set size: {len(train_dataset)}")
    print(f"Validation Set size: {len(val_dataset)}")
    print(f"Test Set size: {len(test_dataset)}")

    return train_dataloader, val_dataloader, test_dataloader


def load_dataset(val_split=0.2, batch_size=5, dataset="mnist", colorize=False):
    """
    Loads the mnist dataset
    :param val_split:
    :param batch_size:
    :param dataset: Input dataset used to further call the required function to load dataset.
        This will become useful once the altered MNIST with spurious correlations is used.
    :param colorize: Boolean to decide whether to color mnnist or not.
    :return datasets: Returns the dataloader iterator for the required dataset.
    """
    if dataset == "mnist":
        return mnist_loader(val_split, batch_size, colorize)


def visualize_dataset(dataloaders, titles=["Train Set", "Validation Set", "Test Set"]):
    """
    Displays 5 images from each set.
    :param dataloaders: List of dataloaders for train, validation, and test sets.
    :param titles: Titles for each dataset visualization.
    """
    fig, axes = plt.subplots(3, 5, figsize=(20, 15))
    for i, dataloader in enumerate(dataloaders):
        data_iter = iter(dataloader)
        images, labels = next(data_iter)

        # Debug: Print out the shape and type of the images
        print(f"Dataset: {titles[i]}")
        print(f"Image batch shape: {images.shape}")
        print(f"Image type: {type(images[0])}")

        for j in range(5):
            #img = transforms.ToPILImage()(images[j])
            img = images[j]
            axes[i, j].imshow(img)
            axes[i, j].axis('off')
            # Debug: Print out the label to check which digit it is
            print(f"Image {j} label: {labels[j]}")

        axes[i, 0].set_ylabel(titles[i], fontsize=16)

    plt.show()


if __name__ == "__main__":
    train_gray, val_gray, test_gray = load_dataset(colorize=False)
    visualize_dataset([train_gray, val_gray, test_gray], titles=["Gray Train", "Gray Val", "Gray Test"])

    train, validation, test = load_dataset(colorize=True)
    visualize_dataset([train, validation, test], titles=["Color Train", "Color Val", "Color Test"])

