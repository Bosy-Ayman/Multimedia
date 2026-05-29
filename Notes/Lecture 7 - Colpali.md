# Masterclass Study Guide: Contrastive Vision-Language Models & Document Retrieval (SigLIP to ColPali)

## 1. The Core Paradigm Shift: From Text-Only RAG to Vision-First Document Retrieval

Traditional Retrieval-Augmented Generation (RAG) pipelines treat PDFs and documents as structured text sources. This approach introduces multiple failure points:

```
[PDF Document] ──(OCR)──> [Raw Text] ──(Layout Parser)──> [Text Chunks] ──(Embedding Model)──> [Vector DB]
```

### Limitations of Text-Only Pipelines:

1. **OCR Failures:** Optical Character Recognition is prone to error on complex layouts, mathematical formulas, and hand-written annotations.
    
2. **Loss of Visual Semantics:** Visual cues like font sizing (headers vs. body text), table alignments, structural diagrams, flowcharts, and embedded figures are stripped away entirely.
    
3. **Complex Processing Overhead:** Pipelines require fragile parser heuristics to identify columns, headers, footers, and page numbers to prevent indexing garbage text.
    

### The Vision-First Solution:

Instead of parsing PDFs to text, **vision-first retrieval** converts each document page directly into a high-resolution image, keeping visual and textual layouts unified.

## 2. SigLIP Deep Dive: Sigmoid Language-Image Pre-training

To understand document-level models like ColPali, we must first understand **SigLIP** (Sigmoid Language-Image Pre-training), which represents a major improvement over standard CLIP (Contrastive Language-Image Pre-training).

```
                  ┌───────────────────────┐
                  │      Input Image      │
                  └───────────┬───────────┘
                              │ Split into Patches
                              ▼
                  ┌───────────────────────┐
                  │   Vision Transformer  │
                  └───────────┬───────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │ Patch Embeddings      │
                  │ [e_1, e_2, ..., e_N]  │
                  └───────────┬───────────┘
                              │ Global Pooling & Linear Projection (W_image)
                              ▼
                  ┌───────────────────────┐
                  │ Global Image Vector   │ (768-dim)
                  └───────────┬───────────┘
                              │
                              │ Cosine Similarity (Dot Product @)
                              ▼
                  ┌───────────────────────┐
                  │ Global Text Vector    │ (768-dim)
                  └───────────▲───────────┘
                              │ Global Pooling & Linear Projection (W_text)
                              │
                  ┌───────────────────────┐
                  │  Text Token Vectors   │
                  │ [t_1, t_2, ..., t_M]  │
                  └───────────▲───────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │    Text Transformer   │
                  └───────────▲───────────┘
                              │ Tokenization
                  ┌───────────┴───────────┐
                  │      Input Text       │
                  └───────────────────────┘
```

### The Architectural Pipeline

SigLIP processes visual and textual inputs through separate pathways and projects them into a shared space:

- **Image Encoder:**
    
    1. Splits an image of size $H \times W$ into a grid of non-overlapping patches (e.g., $16 \times 16$ pixels).
        
    2. Passes these patches through a Vision Transformer (ViT) to get contextualized patch hidden states.
        
    3. **Global Bottleneck:** Pools these patch states (using attention pooling or a `[CLS]` token) into a single, unified global image representation vector: $\vec{v} \in \mathbb{R}^{d_{\text{img}}}$.
        
- **Text Encoder:**
    
    1. Tokenizes text into a sequence of subword tokens.
        
    2. Processes tokens through a text transformer.
        
    3. Pools token representations into a single global text representation vector: $\vec{t} \in \mathbb{R}^{d_{\text{text}}}$.
        
- **Linear Projection:** Aligning embedding dimensions (e.g., matching a $1024$-dimensional visual output with a $768$-dimensional text output) using learnable projection matrices $W_{\text{image}}$ and $W_{\text{text}}$:
    
    $$\vec{z}_{\text{img}} = W_{\text{image}} \vec{v}, \quad \vec{z}_{\text{text}} = W_{\text{text}} \vec{t}$$

### The Mathematical Innovation: Sigmoid Loss vs. CLIP Softmax Loss

In standard CLIP, the model is trained using a multi-class **Softmax cross-entropy loss** calculated over all pairwise comparisons in a batch. For an image $i$ and text $j$ in a batch of size $B$:

$$\mathcal{L}_{\text{Softmax}}^{(i)} = -\log \frac{e^{\text{sim}(\vec{z}_{\text{img}}^i, \vec{z}_{\text{text}}^i) / \tau}}{\sum_{j=1}^{B} e^{\text{sim}(\vec{z}_{\text{img}}^i, \vec{z}_{\text{text}}^j) / \tau}}$$

- **The Softmax Problem:** The denominator requires computing similarities across the entire batch $B$. As the batch size grows (which is critical for contrastive training performance), this denominator requires heavy communication overhead across multiple GPUs, creating an $O(B^2)$ bottleneck.
    
- **The Sigmoid Solution:** SigLIP replaces this global softmax with a pairwise **Sigmoid binary cross-entropy loss**. The task is framed as a series of independent binary classification decisions (whether image $i$ matches text $j$):
    

$$\mathcal{L}_{\text{Sigmoid}} = -\frac{1}{B} \sum_{i=1}^B \sum_{j=1}^B \log \sigma \left( y_{i,j} \left( \vec{z}_{\text{img}}^i \cdot \vec{z}_{\text{text}}^j \cdot e^t + b \right) \right)$$

Where:

- $y_{i,j} = 1$ if $i = j$ (matching positive pair), and $y_{i,j} = -1$ if $i \neq j$ (negative pair).
    
- $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the sigmoid function.
    
- $t$ is a learnable temperature scale parameter and $b$ is a learnable bias.
    

By treating every pair as an independent binary classification task, training scales linearly with batch size across parallel computing nodes without requiring global communication synchronization.

## 3. ColPali Deep Dive: Document Retrieval via Late Interaction

While SigLIP is highly effective for general image classification (e.g., identifying that an entire image contains "a photo of a cat"), it is poorly suited for searching document pages.

### Why Global Pooling Fails on Documents

Documents contain dense, localized visual text (such as a single sentence inside a table on page 42). If you pool the features of 1,024 visual patches into a single global vector, the fine-grained textual information is averaged out. It is impossible to query for specific names, numbers, or terms because the bottleneck has compressed them into a generic "page layout" representation.

### How ColPali Solves This: Multi-Vector Representation and Late Interaction

ColPali adapts the concept of **late interaction** (originally introduced by ColBERT for text search) to the vision domain.

Instead of comparing one global image vector to one global text vector, ColPali preserves all patch-level embeddings for the image and all token-level embeddings for the text.

```
Query Tokens:      [t_1]       [t_2]       ...       [t_M]
                     │           │                     │
                     ├───────────┼───────────┐         │
                     ▼           ▼           ▼         ▼
Image Patches:     [p_1]       [p_2]       [p_3]     [p_N]
```

1. **Document Page Encoding:**
    
    - A PDF page is rendered as an image and broken into a grid of non-overlapping patches (e.g., $32 \times 32 = 1,024$ patches).
        
    - These patches are processed by SigLIP’s Vision Transformer and then mapped through a projection layer into the token dimension of a Language Model (such as Gemma-2B).
        
    - The output for a single page $d$ is a sequence of **multi-vectors**: $D = \{\vec{p}_1, \vec{p}_2, \dots, \vec{p}_N\}$, where $N \approx 1024$.
        
2. **Query Encoding:**
    
    - The text query $q$ is tokenized and mapped to a sequence of vectors: $Q = \{\vec{t}_1, \vec{t}_2, \dots, \vec{t}_M\}$, where $M$ is the number of tokens.
        
3. **The Late Interaction Operator (MaxSim):** To calculate the similarity score between query $q$ and document page $d$, ColPali calculates the dot-product similarity between every token vector $\vec{t}_i$ and every patch vector $\vec{p}_j$, takes the maximum similarity for each token, and sums them up:
    

$$S(q, d) = \sum_{i=1}^{M} \max_{j=1}^{N} \left( \vec{t}_i \cdot \vec{p}_j^\top \right)$$

- **Intuition:** For each word in your query, search across the entire document page to find the single visual patch that matches it best. Sum these best-match scores to calculate the final document score.
    
- **Result:** This architecture preserves highly localized textual and visual details (e.g., finding the patch containing the precise number in a table cell) while maintaining efficient query-time computation.
    

## 4. Hands-On Implementation: Visualizing Spatial Semantics

The PyTorch code in your notes provides a powerful hands-on demonstration of this patch-level behavior using standard SigLIP.

Because standard SigLIP (`siglip-base-patch16-224`) is trained to produce only _global_ similarity scores, we cannot directly access its internal visual-text token mappings without modifying the model. Instead, the code uses a clever bypass trick: **it manually crops the image into small patches and feeds each individual patch into SigLIP as a standalone image**.

### Code Breakdown & Mechanics

```python
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel

# 1. Load the pre-trained SigLIP model and its processor
model_name = "google/siglip-base-patch16-224"
model = AutoModel.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)

# 2. Define the image patching function
def split_into_patches_with_coords(img, patch_size=224, stride=224):
    patches = []
    coords = []
    w, h = img.size
    # Loop over the height and width of the image using a sliding window
    for top in range(0, h - patch_size + 1, stride):
        for left in range(0, w - patch_size + 1, stride):
            # Crop a square region out of the image
            patch = img.crop((left, top, left + patch_size, top + patch_size))
            patches.append(patch)
            coords.append((left, top, left + patch_size, top + patch_size))
    return patches, coords

# 3. Load input image and run the patch generation
image = Image.open("1.bmp").convert("RGB")
# Generates spatial sub-crops (patches) of size 64x64, stepping by 64 pixels
patches, coords = split_into_patches_with_coords(image, patch_size=64, stride=64)

# 4. Prepare batch inputs for the model
texts = ["a cat", "a dog", "a car"]
inputs = processor(
    text=texts,
    images=patches,          # Passes the list of cropped image patches
    return_tensors="pt", 
    padding=True
)

# 5. Compute contrastive logits per patch
with torch.no_grad():
    outputs = model(**inputs)

# Logits shape: [num_patches, num_texts]
logits = outputs.logits_per_image 

# Convert logits to probabilities using Softmax along the text dimension
probs = torch.softmax(logits, dim=-1)

# 6. Output the localized classifications
for i, patch_probs in enumerate(probs):
    print(f"\nPatch {i}:")
    for text, score in zip(texts, patch_probs):
        print(f"{text}: {score.item():.4f}")
```

### From Code to Visual Heatmaps: How It Works

- **The Cropping Phase:** The function `split_into_patches_with_coords` slices the raw image (e.g., a photo of a Pug) into distinct square sub-images.
    
- **The Classification Phase:** SigLIP evaluates each sub-crop independently against the target terms (`"a cat"`, `"a dog"`, `"a car"`).
    
- **The Localized Output:**
    
    - Patches containing the pug’s face and ears (e.g., Patch 1, 2, and 5) yield high scores for `"a dog"` ($0.8198$, $0.9284$, and $0.7144$).
        
    - Patches representing plain background (e.g., Patch 0) have high uncertainty, causing scores to split more evenly or match irrelevant text targets due to training biases.
        
- **The Heatmap Construction:** By plotting these localized probabilities back onto their spatial coordinates, we can construct the visual probability layout shown in your slides.
    

This process visualizes the foundational mechanics behind ColPali: identifying and mapping the specific regions of an image that correspond to target terms.