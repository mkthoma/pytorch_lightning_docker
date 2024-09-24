import os
import json
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
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        """Authenticate with Kaggle, download images, and prepare datasets."""
        self._setup_kaggle_credentials()
        
        # Initialize Kaggle API
        api = KaggleApi()
        api.authenticate()

        # Download the dataset
        dataset_name = 'catherinehorng/dogbreedimagedatabase'
        api.dataset_download_files(dataset_name, path=self.data_dir, unzip=True)

        print(f"Dataset downloaded and extracted in {self.data_dir}")

    def _setup_kaggle_credentials(self):
        """Set up Kaggle credentials for authentication using environment variables."""
        kaggle_username = os.environ.get('KAGGLE_USERNAME')
        kaggle_key = os.environ.get('KAGGLE_KEY')

        if kaggle_username and kaggle_key:
            # Create a temporary kaggle.json file
            kaggle_json = {
                "username": kaggle_username,
                "key": kaggle_key
            }
            
            # Ensure the .kaggle directory exists
            os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
            
            # Write the credentials to the kaggle.json file
            with open(os.path.expanduser('~/.kaggle/kaggle.json'), 'w') as f:
                json.dump(kaggle_json, f)
            
            # Set appropriate permissions
            os.chmod(os.path.expanduser('~/.kaggle/kaggle.json'), 0o600)
        else:
            raise ValueError("Kaggle credentials not found in environment variables. Please set KAGGLE_USERNAME and KAGGLE_KEY.")

    def setup(self, stage: str):
        data_path = self.data_dir / "dogbreedimagedatabase"

        if stage == "fit" or stage is None:
            # Assuming the dataset structure has 'train' and 'test' folders
            full_train_dataset = ImageFolder(
                root=data_path / "train",
                transform=self.train_transform
            )
            # Using a subset of train data for validation
            train_size = int(0.8 * len(full_train_dataset))
            val_size = len(full_train_dataset) - train_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                full_train_dataset,
                [train_size, val_size]
            )

        if stage == "test" or stage is None:
            self.test_dataset = ImageFolder(
                root=data_path / "test",
                transform=self.val_transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    @property
    def normalize_transform(self):
        return transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    @property
    def train_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            self.normalize_transform,
        ])

    @property
    def val_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            self.normalize_transform,
        ])

# Example usage
if __name__ == "__main__":
    # Ensure KAGGLE_USERNAME and KAGGLE_KEY environment variables are set before running
    datamodule = DogBreedImageDataModule(
        data_dir="data",  # Using the default data directory
        batch_size=32,
        num_workers=4
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