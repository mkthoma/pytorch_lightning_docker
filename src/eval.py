import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import RichProgressBar

from datamodules.dogbreed import DogBreedImageDataModule
from models.dogbreed_classifier import DogBreedClassifier
from utils.utils import task_wrapper
from utils.pylogger import get_pylogger
from utils.rich_utils import print_config_tree, print_rich_progress, print_rich_panel

log = get_pylogger(__name__)

@task_wrapper
def evaluate():
    # Set up data module
    data_module = DogBreedImageDataModule(data_dir="data/", batch_size=32, num_workers=4)
    
    # Prepare data (this will download the dataset if it's not already present)
    data_module.prepare_data()
    
    # Setup (this will create the train/val/test splits)
    data_module.setup(stage="validate")
    
    # Get the number of classes from the data module
    num_classes = len(data_module.val_dataset.dataset.classes)

    # Set up model
    model = DogBreedClassifier.load_from_checkpoint(
        "checkpoints/dogbreed-last.ckpt",
        num_classes=num_classes,
        lr=1e-3  # This won't be used for evaluation, but is required for model initialization
    )

    # Set up logger
    logger = TensorBoardLogger(save_dir="logs", name="dogbreed_evaluation")

    # Set up callbacks
    rich_progress_bar = RichProgressBar()

    # Set up trainer
    trainer = L.Trainer(
        callbacks=[rich_progress_bar],
        logger=logger,
        accelerator="auto",
    )

    # Print config
    config = {"data": vars(data_module), "model": vars(model), "trainer": vars(trainer)}
    print_config_tree(config, resolve=True, save_to_file=True)

    # Evaluate the model
    print_rich_panel("Starting evaluation", "Evaluation")
    results = trainer.validate(model, datamodule=data_module)

    # Print validation metrics
    print_rich_panel("Validation Metrics", "Results")
    for k, v in results[0].items():
        print(f"{k}: {v}")

    print_rich_progress("Evaluation complete")

if __name__ == "__main__":
    evaluate()