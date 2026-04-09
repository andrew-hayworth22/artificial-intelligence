import json
import os
import sys

import kagglehub
from torchvision import transforms

def get_image_paths(root_path: str, extensions: tuple = ('.jpg', '.jpeg', '.png', '.webp')) -> list[str]:
    """
    Recursively find all image paths under root_path.
    Returns a flat list of absolute image file paths.
    """
    image_paths = []
    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.lower().endswith(extensions):
                image_paths.append(os.path.join(dirpath, filename))
    return image_paths

if __name__ == "__main__":
    # Get config file to load
    if len(sys.argv) != 2:
        print("Usage: python main.py <config_file>")
        exit(1)

    config_path = sys.argv[1]

    # Read config file
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    # Download datasets
    dogs_path = kagglehub.dataset_download("jessicali9530/stanford-dogs-dataset")
    houses_path = kagglehub.dataset_download("gedewahyupurnama/house-images")
    dog_images = get_image_paths(dogs_path)
    house_images = get_image_paths(houses_path)

    print(f"Fetched {len(dog_images)} dogs images")
    print(f"Fetched {len(house_images)} house images: ")

    if config['images'] > len(dog_images) or config['images'] > len(house_images):
        exit(f"Too many images requested. Max: {min(len(dog_images), len(house_images))}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config['transformation_mean'],
            std=config['transformation_std']
        )
    ])