import os
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    RichProgressBar,
    RichModelSummary,
)

from datamodules.dogbreed import DogBreedImageDataModule
from models.dogbreed_classifier import DogBreedClassifier
from utils.utils import task_wrapper
from utils.pylogger import get_pylogger
from utils.rich_utils import print_config_tree, print_rich_progress, print_rich_panel
import torch 

log = get_pylogger(__name__)

@task_wrapper
def train():
    # Determine the number of workers based on CPU/GPU usage
    num_workers = os.cpu_count() // 2 if torch.cuda.is_available() else 0
    print(f"Using {num_workers} workers for data loading.")

    # Set up data module
    data_module = DogBreedImageDataModule(data_dir="data/", batch_size=32, num_workers=num_workers)
    
    print("Preparing data...")
    data_module.prepare_data()
    
    print("Setting up data module...")
    data_module.setup(stage="fit")
    
    # Get the number of classes from the data module
    num_classes = len(data_module.train_dataset.dataset.classes)
    print(f"Number of classes detected: {num_classes}")
    print(f"Classes: {data_module.train_dataset.dataset.classes}")
    
    if num_classes <= 1:
        raise ValueError(f"Invalid number of classes detected: {num_classes}. Please check the dataset structure.")

    # Set up model
    print(f"Setting up model with {num_classes} classes...")
    model = DogBreedClassifier(num_classes=num_classes, lr=1e-3)

    # Set up logger
    logger = TensorBoardLogger(save_dir="logs", name="dogbreed_classification")

    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="dogbreed-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val/loss",
        mode="min"
    )
    rich_progress_bar = RichProgressBar()
    rich_model_summary = RichModelSummary(max_depth=2)

    # Set up trainer
    trainer = L.Trainer(
        max_epochs=1,
        callbacks=[checkpoint_callback, rich_progress_bar, rich_model_summary],
        logger=logger,
        log_every_n_steps=10,
        accelerator="auto",
    )

    # Print config
    config = {"data": vars(data_module), "model": vars(model), "trainer": vars(trainer)}
    print_config_tree(config, resolve=True, save_to_file=True)

    # Train the model
    print_rich_panel("Starting training", "Training")
    trainer.fit(model, datamodule=data_module)

    # # Save the model state dictionary
    # save_model_path = "checkpoints/dogbreed_final_model.ckpt"  # Define your save path here
    # torch.save(model.state_dict(), save_model_path)  # Save the model
    # print(f"Model saved to: {save_model_path}")  # Print the save path

    print_rich_progress("Finishing up")

if __name__ == "__main__":
    train()
