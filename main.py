from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def image_embedding(image_path):
    """
    Takes an image file path and returns its embedding as a NumPy array.
    """
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    # Extract image features
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)

    # Normalize and convert to NumPy
    embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
    return embeddings.squeeze().cpu().numpy()


import requests
from PIL import Image
from io import BytesIO

def download_image_from_url(url, save_as="downloaded_image.jpg"):
    """
    Downloads an image from a URL and saves it locally in Colab.
    Returns the local filename.
    """
    response = requests.get(url)
    response.raise_for_status()  # Raise error for bad status
    image = Image.open(BytesIO(response.content)).convert("RGB")
    image.save(save_as)
    return save_as



