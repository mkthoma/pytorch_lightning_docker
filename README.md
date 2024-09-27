# Dog Breed Classifier using Docker and PyTorch Lightning

## Dataset used 
This dataset contains a collection of images for 10 different dog breeds, meticulously gathered and organized to facilitate various computer vision tasks such as image classification and object detection. The dataset includes the following breeds:

* Golden Retriever
* German Shepherd
* Labrador Retriever
* Bulldog
* Beagle
* Poodle
* Rottweiler
* Yorkshire Terrier
* Boxer
* Dachshund

Each breed is represented by 100 images, stored in separate directories named after the respective breed. The images have been curated to ensure diversity and relevance, making this dataset a valuable resource for training and evaluating machine learning models in the field of computer vision.

The [DogBreedImageDataModule](https://github.com/mkthoma/pytorch_lightning_docker/blob/main/src/datamodules/dogbreed.py) is a PyTorch Lightning data module designed to streamline the process of loading and preparing the Dog Breed Image Dataset for training, validation, and testing in deep learning models. This module integrates with the Kaggle API to automatically download and prepare the dataset, making it convenient for users to start their experiments with minimal setup.

You can temporarily set the variables for the current session by running the following commands (replace your_username and your_api_key with the [actual values](https://www.kaggle.com/docs/api#interacting-with-datasets))
```
echo 'export KAGGLE_USERNAME="your_username"' >> ~/.bashrc
echo 'export KAGGLE_KEY="your_api_key"' >> ~/.bashrc
source ~/.bashrc
```

## Model
The [Dog Breed Classifier](https://github.com/mkthoma/pytorch_lightning_docker/blob/main/src/models/dogbreed_classifier.py) is a deep learning model designed to classify images of dog breeds using a pre-trained ResNet50 architecture. Built with the PyTorch Lightning framework, the model leverages the timm library to utilize state-of-the-art model architectures, providing an effective solution for multi-class classification tasks.

### Model Architecture
* Base Model: The classifier uses ResNet50, a convolutional neural network that is 50 layers deep and well-regarded for its performance on image classification tasks. This model has been pre-trained on a large dataset, enabling it to capture a rich set of features useful for dog breed classification.

* Output Layer: The model's output layer is dynamically adjusted to match the number of classes corresponding to the dog breeds in the dataset.

### Key Features
* Loss Function: The model employs CrossEntropyLoss, suitable for multi-class classification problems, providing a robust way to measure the performance of the classifier during training.

* Metrics: It utilizes torchmetrics to track:

    * Training Accuracy: Measures the accuracy of the model on the training dataset.
    * Validation Accuracy: Monitors the accuracy on the validation dataset to avoid overfitting.
    * Test Accuracy: Evaluates the model's performance on a separate test dataset.

* Learning Rate Scheduler: A ReduceLROnPlateau scheduler is employed, which reduces the learning rate when the validation accuracy plateaus, helping improve convergence during training.

## Docker build

1. Build the Docker image:

    ```
    docker build -t dogbreed-classification .
    ```

2. To run training:
    ```
    docker run -v $(pwd)/model_artifacts:/app/checkpoints dogbreed-classification train
    ```

3. To run evaluation:
    ```
    docker run -v $(pwd)/model_artifacts:/app/checkpoints dogbreed-classification eval
    ```

4. To run inference:
    ```
    docker run -v $(pwd)/model_artifacts:/app/checkpoints dogbreed-classification infer
    ```
    
    By default it performs inference on the images present in the [input_images](https://github.com/mkthoma/pytorch_lightning_docker/tree/main/model_artifacts/input_images) folder.

    To modify the infer arguments, you can do the following:
    
    ```
    docker run -v $(pwd)/model_artifacts:/app/checkpoints dogbreed-classification infer --input_folder="path/to/custom/input" --output_folder="path/to/custom/output" --ckpt_path="path/to/custom/checkpoint.ckpt"
    ```

