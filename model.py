import os
import torch
import torch.nn.functional as F
from torchvision.transforms import transforms
from PIL import Image
import pandas as pd
from open_clip import create_model_and_transforms, tokenizer
from tqdm import tqdm

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "ViT-B/32"
pretrained = "openai"
batch_size = 128
image_folder = "/Users/jdhzy/Desktop/CS506_Assignment_10/coco_images_resized"  # Replace with your folder path
max_images = 5000  # Maximum number of images to process

# Load the model and preprocessing functions
model, preprocess_train, preprocess_val = create_model_and_transforms(model_name, pretrained=pretrained)
model = model.to(device)
model.eval()

# Tokenizer from OpenAI's CLIP model
clip_tokenizer = tokenizer

# Collect all image paths and limit to the first `max_images`
image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
image_paths = image_paths[:max_images]
print(f'Number of images to process: {len(image_paths)}')

# DataFrame to store results
results = []

# Function to load and preprocess images
def load_images(batch_paths):
    images = []
    for path in batch_paths:
        try:
            image = Image.open(path).convert("RGB")
            images.append(preprocess_val(image))
        except Exception as e:
            print(f"Error loading image {path}: {e}")
    return torch.stack(images) if images else None

# Process images in batches
with torch.no_grad():
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing images"):
        batch_paths = image_paths[i:i + batch_size]
        images = load_images(batch_paths)
        if images is None:  # Skip if no valid images in this batch
            continue

        images = images.to(device)
        embeddings = model.encode_image(images)
        embeddings = F.normalize(embeddings, p=2, dim=1)  # Normalize the embeddings

        for path, emb in zip(batch_paths, embeddings):
            results.append({"file_name": os.path.basename(path), "embedding": emb.cpu().numpy()})

# Save results to DataFrame
df = pd.DataFrame(results)
df.to_pickle('image_embeddings.pickle')

#################################

# Helper to load a single image
def load_image(image_file):
    try:
        image = Image.open(image_file).convert("RGB")
        return preprocess_val(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# Compute similarity
def compute_similarity(query_embedding):
    database_embeddings = torch.stack([torch.tensor(embedding) for embedding in df['embedding']]).to(device)
    cosine_similarities = F.cosine_similarity(query_embedding, database_embeddings)

    # Get the indices of the top 5 similarity scores
    top_k = torch.topk(cosine_similarities, 5)
    top_indices = top_k.indices.tolist()
    top_scores = top_k.values.tolist()

    # Return file paths and scores
    results = []
    for idx, score in zip(top_indices, top_scores):
        file_name = df.iloc[idx]["file_name"]
        file_path = os.path.join("coco_images_resized", file_name)
        if not os.path.exists(file_path):
            continue
        results.append({"file_path": file_path, "score": score})

    return results

# Image-to-image query
def process_image_query(image_file):
    image_tensor = load_image(image_file)
    if image_tensor is None:
        raise ValueError("Invalid image file provided")
    query_embedding = F.normalize(model.encode_image(image_tensor))
    return compute_similarity(query_embedding)

# Text-to-image query
def process_text_query(text_query):
    if not text_query.strip():
        raise ValueError("Text query cannot be empty")
    tokens = clip_tokenizer.tokenize([text_query]).to(device)  # Correctly tokenize text
    query_embedding = F.normalize(model.encode_text(tokens))
    return compute_similarity(query_embedding)


# Hybrid query
def process_hybrid_query(text_query, image_file, lam):
    tokens = clip_tokenizer.tokenize([text_query]).to(device)  # Correctly tokenize text
    text_embedding = F.normalize(model.encode_text(tokens))

    image_tensor = load_image(image_file)
    if image_tensor is None:
        raise ValueError("Invalid image file provided")

    image_embedding = F.normalize(model.encode_image(image_tensor))
    query_embedding = F.normalize(lam * text_embedding + (1.0 - lam) * image_embedding)
    return compute_similarity(query_embedding)