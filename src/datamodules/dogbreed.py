import os
from pathlib import Path
from typing import Union

import torch
import lightning as L
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from kaggle.api.kaggle_api_extended import KaggleApi

class DogBreedImageDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: Union[str, Path] = "data",
        num_workers: int = 0,
        batch_size: int = 8,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.num_workers = self._get_num_workers(num_workers)
        self.batch_size = batch_size
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Define the image transformations for training and validation
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _get_num_workers(self, default: int) -> int:
        """Automatically determine number of workers based on available device."""
        if torch.cuda.is_available():
            return min(default, os.cpu_count() * 2)
        else:
            return 0
        
    def prepare_data(self):
        """Download images and prepare datasets."""
        # Initialize Kaggle API
        api = KaggleApi()
        api.authenticate()

        # Download the dataset from Kaggle
        dataset_name = 'khushikhushikhushi/dog-breed-image-dataset'
        api.dataset_download_files(dataset_name, path=self.data_dir, unzip=True)

        print(f"Dataset downloaded and extracted in {self.data_dir}")

    def setup(self, stage: str = None):
        """Setup datasets based on the stage (fit/test)."""
        data_path = self.data_dir / 'dataset'  # Ensure this points to 'dataset' containing breed folders

        print(f"Setting up data from: {data_path}")

        if stage == "fit" or stage is None:
            # Load the dataset using ImageFolder
            full_dataset = ImageFolder(
                root=data_path,
                transform=self.train_transform
            )
            print(f"Number of classes detected: {len(full_dataset.classes)}")
            print(f"Classes: {full_dataset.classes}")
            print(f"Total number of samples: {len(full_dataset)}")

            # Split the dataset into train and validation sets (80% train, 20% validation)
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                full_dataset,
                [train_size, val_size]
            )
            print(f"Training samples: {train_size}, Validation samples: {val_size}")

        if stage == "test" or stage is None:
            # Use the same dataset for testing, but apply the validation transformations
            self.test_dataset = ImageFolder(
                root=data_path,
                transform=self.val_transform
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

# Example usage
if __name__ == "__main__":
    datamodule = DogBreedImageDataModule(
        data_dir="data",  # Default data directory
        batch_size=32,
        num_workers=0  # Will be set automatically in the constructor
    )
    
    # Prepare data (download dataset)
    datamodule.prepare_data()
    
    # Setup (create train/val/test splits)
    datamodule.setup(stage="fit")
    
    # Access data loaders
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    print("DataModule setup complete. Ready for model training.")
    print(f"Train loader has {len(train_loader.dataset)} samples.")
    print(f"Validation loader has {len(val_loader.dataset)} samples.")
    print(f"Test loader has {len(test_loader.dataset)} samples." if test_loader.dataset else "No test dataset loaded.")
