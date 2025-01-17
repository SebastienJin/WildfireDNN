import logging
import os

import cv2
import kagglehub
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Global names for folder structure
TRUE_NAME = "wildfire"
FALSE_NAME = "nowildfire"

# Download the dataset
path = kagglehub.dataset_download("abdelghaniaaba/wildfire-prediction-dataset")
logging.info("Data saved to: %s", path)


# Helper function to load images/labels for a specific subset
def subset_loader(subset_name):
    folder_true = os.path.join(path, subset_name, TRUE_NAME)
    folder_false = os.path.join(path, subset_name, FALSE_NAME)

    X_true = [os.path.join(folder_true, f) for f in os.listdir(folder_true)]
    X_false = [os.path.join(folder_false, f) for f in os.listdir(folder_false)]

    y_true = [1] * len(X_true)
    y_false = [0] * len(X_false)

    return X_true + X_false, y_true + y_false


class WildfireDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        try:
            bgr_image = cv2.imread(img_path)
            if bgr_image is None:
                raise IOError(f"OpenCV could not read {img_path}")

            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)  # Now a NumPy array
            pil_image = Image.fromarray(rgb_image)                  # Convert to PIL
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Handle failure case if needed

        # Apply any transforms
        if self.transform:
            image = self.transform(pil_image)
        else:
            image = pil_image

        label = self.labels[index]
        return image, label


def get_wildfire_datasets(
        transform=None
):
    # Load train/test/valid image paths and labels
    X_train, y_train = subset_loader("train")
    X_test, y_test = subset_loader("test")
    X_valid, y_valid = subset_loader("valid")

    logging.info("Train: %d images, %d labels", len(X_train), len(y_train))
    logging.info("Test: %d images, %d labels", len(X_test), len(y_test))
    logging.info("Valid: %d images, %d labels", len(X_valid), len(y_valid))

    # Default transform if none is provided
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    # Instantiate the datasets
    train_dataset = WildfireDataset(X_train, y_train, transform=transform)
    test_dataset = WildfireDataset(X_test, y_test, transform=transform)
    valid_dataset = WildfireDataset(X_valid, y_valid, transform=transform)

    return train_dataset, test_dataset, valid_dataset


if __name__ == "__main__":
    # Example usage:
    train_ds, test_ds, valid_ds = get_wildfire_datasets()

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    valid_loader = DataLoader(valid_ds, batch_size=32, shuffle=False)

    # Quick sanity check: Fetch one batch from train_loader
    for images, labels in train_loader:
        logging.info("A batch of images shape: %s", images.shape)
        logging.info("A batch of labels shape: %s", labels.shape)
        break
