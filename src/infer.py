import os
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import random
from models.dogbreed_classifier import DogBreedClassifier
from rich.progress import Progress
from rich.panel import Panel
from rich.console import Console
import glob

console = Console()

def inference(model, image_path, class_labels):
    # Load and preprocess the image
    img = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Apply the transform to the image
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    # Move the input tensor to the same device as the model
    img_tensor = img_tensor.to(model.device)

    # Set the model to evaluation mode
    model.eval()

    # Perform inference
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    predicted_label = class_labels[predicted_class]
    confidence = probabilities[0][predicted_class].item()

    return img, predicted_label, confidence

def save_prediction(img, predicted_label, confidence, output_path):
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Predicted: {predicted_label.capitalize()}\nConfidence: {confidence:.2f}")
    plt.savefig(output_path)
    plt.close()

def main(args):
    console.print(Panel("Starting inference", title="Inference", expand=False))

    # Load the model
    model = DogBreedClassifier.load_from_checkpoint(args.ckpt_path)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Define class labels manually
    class_labels = ['Beagle', 'Boxer', 'Bulldog', 'Dachshund', 'German_Shepherd', 'Golden_Retriever', 'Labrador_Retriever', 'Poodle', 'Rottweiler', 'Yorkshire_Terrier']

    # Create the predictions folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Get list of image files
    image_files = [f for f in os.listdir(args.input_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    
    # Check if any images are found
    if not image_files:
        console.print("[red]No images found in the specified input folder.[/red]")
        return

    # Randomly select 10 images (or less if there are fewer than 10 images)
    selected_images = random.sample(image_files, min(10, len(image_files)))

    with Progress() as progress:
        task = progress.add_task("[green]Processing images...", total=len(selected_images))

        for filename in selected_images:
            image_path = os.path.join(args.input_folder, filename)
            img, predicted_label, confidence = inference(model, image_path, class_labels)

            # Save the prediction image
            output_image_path = os.path.join(args.output_folder, f"{os.path.splitext(filename)[0]}_prediction.png")
            save_prediction(img, predicted_label, confidence, output_image_path)

            # Save the prediction text
            output_text_path = os.path.join(args.output_folder, f"{os.path.splitext(filename)[0]}_prediction.txt")
            with open(output_text_path, "w") as f:
                f.write(f"Predicted: {predicted_label}\nConfidence: {confidence:.2f}")

            progress.update(task, advance=1, description=f"[green]Processed {filename}")

    console.print(Panel("Inference completed", title="Finished", expand=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform inference on dog breed images")

    # Default paths for the input folder, output folder, and checkpoint
    default_input_folder = os.path.join("checkpoints", "input_images")
    default_output_folder = "checkpoints/predicted_images"

    # Find the latest checkpoint
    checkpoint_dir = "checkpoints"
    checkpoint_pattern = os.path.join(checkpoint_dir, "dogbreed-*.ckpt")
    checkpoints = glob.glob(checkpoint_pattern)
    # Get the most recent checkpoint
    default_ckpt_path = max(checkpoints, key=os.path.getmtime)

    parser.add_argument("--input_folder", type=str, default=default_input_folder, help="Path to the folder containing input images")
    parser.add_argument("--output_folder", type=str, default=default_output_folder, help="Path to the folder to save predictions")
    parser.add_argument("--ckpt_path", type=str, default=default_ckpt_path, help="Path to the model checkpoint")

    args = parser.parse_args()
    main(args)
