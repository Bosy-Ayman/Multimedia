```python

# =========================
# 1. Install libraries
# =========================
!pip install faiss-cpu transformers pillow torch

# =========================
# 2. Import libraries
# =========================
import torch
from PIL import Image
from transformers import BlipProcessor, BlipModel
import faiss
import os
import numpy as np

# =========================
# 3. Setup device (GPU / CPU)
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# =========================
# 4. Load BLIP model
# =========================
model_name = "Salesforce/blip-image-captioning-base"

processor = BlipProcessor.from_pretrained(model_name)
model = BlipModel.from_pretrained(model_name).to(device)

model.eval()  # disable training

# =========================
# 5. Load images paths
# =========================
image_paths = [f"{i}.bmp" for i in range(1, 10)]

# =========================
# 6. Extract embeddings
# =========================
embeddings = []

for path in image_paths:
    image = Image.open(path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():  # no gradients (faster + less memory)
        output = model.get_image_features(**inputs)
        image_embedding = output.pooler_output

    # Normalize (important for cosine similarity)
    image_embedding = torch.nn.functional.normalize(image_embedding, dim=-1)

    embeddings.append(image_embedding.cpu().numpy())

# Convert list → matrix
embeddings_matrix = np.vstack(embeddings).astype("float32")

print("Embeddings shape:", embeddings_matrix.shape)

# =========================
# 7. Build FAISS index
# =========================
embedding_dim = embeddings_matrix.shape[1]

# Inner Product = Cosine similarity (after normalization)
index = faiss.IndexFlatIP(embedding_dim)

index.add(embeddings_matrix)

print("Number of indexed images:", index.ntotal)

# =========================
# 8. Query image
# =========================
query_image = Image.open("test.bmp").convert("RGB")

inputs = processor(images=query_image, return_tensors="pt").to(device)

with torch.no_grad():
    output = model.get_image_features(**inputs)
    query_embedding = output.pooler_output

query_embedding = torch.nn.functional.normalize(query_embedding, dim=-1)
query_embedding = query_embedding.cpu().numpy()

# =========================
# 9. Search similar images
# =========================
k = 3  # top 3 results

distances, indices = index.search(query_embedding, k)

print("\nTop similar images:\n")

for rank, idx in enumerate(indices[0]):
    print(f"Rank {rank+1}: {image_paths[idx]} | Score: {distances[0][rank]:.4f}")
```

