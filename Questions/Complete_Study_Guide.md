# DSAI-413 Multimedia Processing — Complete Final Exam Study Guide
## Coverage: Lecture 04 → Lecture 11 (Radar & WiFi Excluded)

---

# TABLE OF CONTENTS

1. [Lecture 4 & 6: Vector Databases & FAISS](#1-lecture-4--6-vector-databases--faiss)
2. [Lecture 7: SigLIP & ColPali — Vision-Language Models](#2-lecture-7-siglip--colpali--vision-language-models)
3. [Lecture 8: Space Alignment vs. Space Mapping](#3-lecture-8-space-alignment-vs-space-mapping)
4. [Lecture 9: Neural Radiance Fields (NeRF)](#4-lecture-9-neural-radiance-fields-nerf)
5. [Lecture 10: Temperature Scaling, Simulated Annealing & Hallucinations](#5-lecture-10-temperature-scaling-simulated-annealing--hallucinations)
6. [Lecture 11: Diffusion Models & Stable Diffusion](#6-lecture-11-diffusion-models--stable-diffusion)
7. [All Code Snippets (Quick Reference)](#7-all-code-snippets-quick-reference)
8. [Key Formulas Cheat Sheet](#8-key-formulas-cheat-sheet)
9. [MCQ Bank (30 Questions)](#9-mcq-bank-30-questions)

---

# 1. Lecture 4 & 6: Vector Databases & FAISS

## 1.1 Core Concepts

**FAISS (Facebook AI Similarity Search)** is a library for efficient similarity search and clustering of dense vectors. It organizes high-dimensional vectors into index structures for fast nearest-neighbor retrieval.

### Key Index Types

| Index Type | Description | Similarity Metric |
|---|---|---|
| `IndexFlatL2` | Exact brute-force search using **Euclidean (L2) distance** | Lower distance = more similar |
| `IndexFlatIP` | Exact brute-force search using **Inner Product** | Higher score = more similar |
| `IndexIVFFlat` | Inverted File index with Voronoi partitioning for faster approximate search | Depends on quantizer |

### Critical Insight: Cosine Similarity via Inner Product
If you **normalize vectors to unit length** (L2 norm = 1) before indexing, then:
$$\text{Inner Product}(\vec{a}, \vec{b}) = \cos(\theta) = \text{Cosine Similarity}$$

This is why `faiss.normalize_L2()` is always called before using `IndexFlatIP`.

## 1.2 GloVe + FAISS Text Search Pipeline

```python
import numpy as np
import faiss

# 1. Load pretrained GloVe vectors (50-dimensional word embeddings)
glove_file = 'glove.6B.50d.txt'
word2vec = {}
with open(glove_file, 'r', encoding='utf8') as f:
    for line in f:
        parts = line.strip().split()
        word2vec[parts[0]] = np.array(parts[1:], dtype='float32')

# 2. Build sentence embeddings by averaging word vectors
def sentence_embedding(sent, word2vec, d=50):
    words = sent.lower().split()
    vecs = [word2vec[w] for w in words if w in word2vec]
    return np.mean(vecs, axis=0) if vecs else np.zeros(d, dtype='float32')

# 3. Create embeddings for all sentences
sentences = ["the cat sat on the mat", "a dog runs fast", "perfect program"]
sentence_embeddings = np.array(
    [sentence_embedding(s, word2vec) for s in sentences], dtype='float32'
)

# 4. CRITICAL: Normalize for cosine similarity
faiss.normalize_L2(sentence_embeddings)

# 5. Build IVF index (3 clusters)
d = 50  # dimension
nlist = 3  # number of Voronoi cells
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist)
index.train(sentence_embeddings)  # Must train IVF before adding
index.add(sentence_embeddings)

# 6. Search
query_vec = sentence_embedding("perfect program", word2vec).reshape(1, -1)
faiss.normalize_L2(query_vec)
index.nprobe = 3  # Search all 3 cells for exact results
D, I = index.search(query_vec, k=2)  # Top-2 nearest
```

### Pipeline Flow
```
Load GloVe → Words to Vectors → Average = Sentence Embedding
    → Normalize L2 → Train FAISS Clusters → Add Vectors
    → Query to Vector → Normalize → Search Top-k
```

## 1.3 BLIP + FAISS Image Search Pipeline

```python
import torch
from PIL import Image
from transformers import BlipProcessor, BlipModel
import faiss
import numpy as np

# 1. Setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Load BLIP (Bridge Language-Image Pre-training)
model_name = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(model_name)
model = BlipModel.from_pretrained(model_name).to(device)
model.eval()  # Disable dropout and batch norm updates

# 3. Extract embeddings for each image
image_paths = [f"{i}.bmp" for i in range(1, 10)]
embeddings = []

for path in image_paths:
    image = Image.open(path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():  # No gradient computation = faster + less memory
        output = model.get_image_features(**inputs)
        image_embedding = output.pooler_output
    
    # Normalize for cosine similarity
    image_embedding = torch.nn.functional.normalize(image_embedding, dim=-1)
    embeddings.append(image_embedding.cpu().numpy())

embeddings_matrix = np.vstack(embeddings).astype("float32")

# 4. Build FAISS index (Inner Product = Cosine after normalization)
embedding_dim = embeddings_matrix.shape[1]
index = faiss.IndexFlatIP(embedding_dim)
index.add(embeddings_matrix)

# 5. Query with a test image
query_image = Image.open("test.bmp").convert("RGB")
inputs = processor(images=query_image, return_tensors="pt").to(device)

with torch.no_grad():
    output = model.get_image_features(**inputs)
    query_embedding = output.pooler_output

query_embedding = torch.nn.functional.normalize(query_embedding, dim=-1)
query_embedding = query_embedding.cpu().numpy()

# 6. Search
k = 3
distances, indices = index.search(query_embedding, k)
for rank, idx in enumerate(indices[0]):
    print(f"Rank {rank+1}: {image_paths[idx]} | Score: {distances[0][rank]:.4f}")
```

### Key Points to Remember
- `model.eval()` — disables training-specific layers (dropout)
- `torch.no_grad()` — disables gradient tracking (saves memory, speeds up)
- `torch.nn.functional.normalize(x, dim=-1)` — L2 normalizes embeddings
- `IndexFlatIP` — Inner Product search (works as cosine sim after normalization)

---

# 2. Lecture 7: SigLIP & ColPali — Vision-Language Models

## 2.1 The Problem: Text-Only RAG Fails on Documents

Traditional RAG pipelines:
```
[PDF] → OCR → Raw Text → Chunking → Embedding → Vector DB
```
**Failures:** OCR errors, lost visual semantics (tables, charts, fonts, layouts), complex parser heuristics.

**Vision-First Solution:** Treat each PDF page as an image directly.

## 2.2 SigLIP (Sigmoid Language-Image Pre-training)

### Architecture
```
Image → Patch Split → Vision Transformer → Global Pooling → Linear Projection → z_img (768-dim)
                                                                                      ↕ Cosine Sim
Text  → Tokenize    → Text Transformer  → Global Pooling → Linear Projection → z_txt (768-dim)
```

### CLIP vs SigLIP Loss — THE KEY DIFFERENCE

**CLIP (Softmax Loss):**
$$\mathcal{L}_{\text{Softmax}}^{(i)} = -\log \frac{e^{\text{sim}(\vec{z}_{\text{img}}^i, \vec{z}_{\text{text}}^i) / \tau}}{\sum_{j=1}^{B} e^{\text{sim}(\vec{z}_{\text{img}}^i, \vec{z}_{\text{text}}^j) / \tau}}$$

- Requires computing similarities across **entire batch** → $O(B^2)$ communication overhead across GPUs

**SigLIP (Sigmoid Loss):**
$$\mathcal{L}_{\text{Sigmoid}} = -\frac{1}{B} \sum_{i=1}^B \sum_{j=1}^B \log \sigma \left( y_{i,j} \left( \vec{z}_{\text{img}}^i \cdot \vec{z}_{\text{text}}^j \cdot e^t + b \right) \right)$$

Where:
- $y_{i,j} = +1$ if matching pair ($i = j$), $y_{i,j} = -1$ if non-matching
- $\sigma(x) = \frac{1}{1 + e^{-x}}$ (sigmoid function)
- $t$ = learnable temperature, $b$ = learnable bias

**Key advantage:** Each pair is an **independent binary classification** → scales linearly, no global sync needed.

## 2.3 ColPali — Late Interaction for Document Retrieval

### The Global Pooling Problem
SigLIP pools all patch features into ONE global vector → fine-grained text details (specific numbers, names in tables) get averaged out.

### ColPali Solution: Multi-Vector + MaxSim

**Document Encoding:** Each page → image → 1024 patch vectors $D = \{\vec{p}_1, ..., \vec{p}_N\}$

**Query Encoding:** Text query → $M$ token vectors $Q = \{\vec{t}_1, ..., \vec{t}_M\}$

**MaxSim Score:**
$$S(q, d) = \sum_{i=1}^{M} \max_{j=1}^{N} \left( \vec{t}_i \cdot \vec{p}_j^\top \right)$$

**Intuition:** For each word in your query, find the best-matching patch on the page, then sum all best matches.

## 2.4 SigLIP Patch Visualization Code

```python
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel

# Load SigLIP
model_name = "google/siglip-base-patch16-224"
model = AutoModel.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)

# Split image into patches
def split_into_patches_with_coords(img, patch_size=224, stride=224):
    patches = []
    coords = []
    w, h = img.size
    for top in range(0, h - patch_size + 1, stride):
        for left in range(0, w - patch_size + 1, stride):
            patch = img.crop((left, top, left + patch_size, top + patch_size))
            patches.append(patch)
            coords.append((left, top, left + patch_size, top + patch_size))
    return patches, coords

# Process
image = Image.open("1.bmp").convert("RGB")
patches, coords = split_into_patches_with_coords(image, patch_size=64, stride=64)

texts = ["a cat", "a dog", "a car"]
inputs = processor(text=texts, images=patches, return_tensors="pt", padding=True)

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits_per_image  # [num_patches, num_texts]
probs = torch.softmax(logits, dim=-1)  # Normalize across text labels

for i, patch_probs in enumerate(probs):
    print(f"\nPatch {i}:")
    for text, score in zip(texts, patch_probs):
        print(f"  {text}: {score.item():.4f}")
```

---

# 3. Lecture 8: Space Alignment vs. Space Mapping

## 3.1 Overview

| Property | Space Alignment | Space Mapping |
|---|---|---|
| **Projection** | Symmetric — both modalities project to **new shared space** | Asymmetric — one modality projects into the **other's existing space** |
| **Matrices** | Two: $W_{\text{img}}$ and $W_{\text{txt}}$ | One: $W_{\text{img}}$ |
| **Target** | Brand new $d_{\text{shared}}$ | Pre-existing $d_{\text{txt}}$ |
| **Examples** | CLIP, SigLIP | Multimodal LLMs (image tokens → LLM vocabulary) |

## 3.2 Space Alignment — Full Trace

### Given Inputs
- Image 1 (Cat): $\vec{x}_{\text{im1}} = [1, 0, 1, 0, 1]^\top \in \mathbb{R}^5$
- Image 2 (Dog): $\vec{x}_{\text{im2}} = [0, 1, 0, 1, 0]^\top \in \mathbb{R}^5$
- Text 1 ("Cat"): $\vec{x}_{\text{t1}} = [1, 0, 1]^\top \in \mathbb{R}^3$
- Text 2 ("Dog"): $\vec{x}_{\text{t2}} = [0, 1, 0]^\top \in \mathbb{R}^3$

### Step 1: Encode to 4D Embeddings

$$W_{\text{vision}} = \begin{bmatrix} 1 & 0 & 1 & 0 & 1 \\ 0 & 1 & 0 & 1 & 0 \\ 1 & 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 1 & 0 \end{bmatrix}, \quad W_{\text{text}} = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 0 \\ 1 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

$$\vec{v}_{\text{im1}} = W_{\text{vision}} \cdot \vec{x}_{\text{im1}} = \begin{bmatrix} 3 \\ 0 \\ 1 \\ 1 \end{bmatrix}, \quad \vec{v}_{\text{im2}} = \begin{bmatrix} 0 \\ 2 \\ 1 \\ 1 \end{bmatrix}$$

$$\vec{v}_{\text{txt1}} = W_{\text{text}} \cdot \vec{x}_{\text{t1}} = \begin{bmatrix} 2 \\ 0 \\ 1 \\ 1 \end{bmatrix}, \quad \vec{v}_{\text{txt2}} = \begin{bmatrix} 0 \\ 1 \\ 1 \\ 0 \end{bmatrix}$$

### Step 2: Project to 2D Shared Space

$$W_{\text{img}} = \begin{bmatrix} 1 & 0 & 1 & 1 \\ 0 & 1 & 0 & 1 \end{bmatrix}, \quad W_{\text{txt}} = \begin{bmatrix} 1 & 0 & 1 & 1 \\ 0 & 1 & 1 & 0 \end{bmatrix}$$

$$\vec{z}_{\text{im1}} = W_{\text{img}} \cdot \vec{v}_{\text{im1}} = \begin{bmatrix} 3+0+1+1 \\ 0+0+0+1 \end{bmatrix} = \begin{bmatrix} 5 \\ 1 \end{bmatrix}$$

$$\vec{z}_{\text{txt1}} = W_{\text{txt}} \cdot \vec{v}_{\text{txt1}} = \begin{bmatrix} 2+0+1+1 \\ 0+0+1+0 \end{bmatrix} = \begin{bmatrix} 4 \\ 1 \end{bmatrix}$$

$$\vec{z}_{\text{im2}} = \begin{bmatrix} 2 \\ 3 \end{bmatrix}, \quad \vec{z}_{\text{txt2}} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$$

> ⚠️ **Slide Typo:** The slides show $\vec{z}_{\text{im1}} = [5, 2]^\top$ — the correct answer is $[5, 1]^\top$.

## 3.3 Space Mapping — Full Trace

### Mapping Matrix (projects image → text space)

$$W_{\text{img}} = \begin{bmatrix} 0.67 & 0 & 0 & 0 \\ 0 & 0.5 & 0 & -0.5 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 \end{bmatrix}$$

### Projections

$$\vec{v}'_{\text{im1}} = W_{\text{img}} \cdot \vec{v}_{\text{im1}} = \begin{bmatrix} 0.67(3) \\ 0.5(0)-0.5(1) \\ 1(1) \\ 0 \end{bmatrix} = \begin{bmatrix} 2.01 \\ -0.5 \\ 1 \\ 0 \end{bmatrix}$$

$$\vec{v}'_{\text{im2}} = \begin{bmatrix} 0 \\ 0.5 \\ 1 \\ 0 \end{bmatrix}$$

### Similarity (Dot Products)

| Pair | Dot Product | Match? |
|---|---|---|
| Cat image · "Cat" text | $(2.01)(2) + (-0.5)(0) + (1)(1) + (0)(1) = 5.02$ | ✅ High |
| Cat image · "Dog" text | $(2.01)(0) + (-0.5)(1) + (1)(1) + (0)(0) = 0.5$ | ❌ Low |
| Dog image · "Dog" text | $(0)(0) + (0.5)(1) + (1)(1) + (0)(0) = 1.5$ | ✅ High |
| Dog image · "Cat" text | $(0)(2) + (0.5)(0) + (1)(1) + (0)(1) = 1.0$ | ❌ Low |

**Result:** Matching pairs have higher similarity than mismatching pairs ✓

---

# 4. Lecture 9: Neural Radiance Fields (NeRF)

## 4.1 Core Concept

NeRF represents a 3D scene as a continuous 5D function:

$$F_\Theta: (\mathbf{x}, \mathbf{d}) \longrightarrow (\vec{c}, \sigma)$$

| Component | Description | Depends On |
|---|---|---|
| $\mathbf{x} = (x, y, z)$ | 3D spatial position | — |
| $\mathbf{d} = (\theta, \phi)$ | Viewing direction (spherical coords) | — |
| $\vec{c} = (R, G, B)$ | Emitted color | **Both** $\mathbf{x}$ and $\mathbf{d}$ |
| $\sigma$ | Volume density (opacity) | **Only** $\mathbf{x}$ (NOT $\mathbf{d}$!) |

> **CRITICAL:** $\sigma$ depends ONLY on position $\mathbf{x}$ → ensures geometry is view-independent. Color depends on both → allows specular highlights.

## 4.2 Volumetric Rendering

### Camera Ray
$$\vec{r}(t) = \vec{O} + t\vec{d}, \quad t \in [t_n, t_f]$$

### Continuous Integral
$$C(\vec{r}) = \int_{t_n}^{t_f} T(t) \cdot \sigma(\vec{r}(t)) \cdot \vec{c}(\vec{r}(t), \vec{d}) \, dt$$

### Accumulated Transmittance
$$T(t) = \exp\left(-\int_{t_n}^{t} \sigma(\vec{r}(s)) \, ds\right)$$

### Discrete Approximation (EXAM FORMULA!)
$$C(\vec{r}) \approx \sum_{i=1}^{N} T_i \cdot \alpha_i \cdot \vec{c}_i$$

Where:
- **Opacity:** $\alpha_i = 1 - e^{-\sigma_i \Delta t_i}$
- **Transmittance:** $T_i = \prod_{j=1}^{i-1} (1 - \alpha_j)$, with $T_1 = 1.0$

### Physical Meaning
- $\sigma = 0 \Rightarrow \alpha = 0$ (empty space, transparent)
- $\sigma \to \infty \Rightarrow \alpha \to 1$ (solid wall, opaque)
- $T_i$ decays as the ray passes through more dense material (light gets absorbed)

## 4.3 Complete NeRF Trace Example

**Setup:** Ray $\vec{r}(t) = (0,0,0) + t(1,0,0)$, $\Delta t = 1$, $N = 3$

| Sample $i$ | $\vec{r}(t_i)$ | $\sigma_i$ | $c_i$ |
|---|---|---|---|
| 1 | $(1,0,0)$ | 0.5 | 0.2 |
| 2 | $(2,0,0)$ | 1.0 | 0.7 |
| 3 | $(3,0,0)$ | 2.0 | 1.0 |

**Step 1 — Opacities:**
- $\alpha_1 = 1 - e^{-0.5 \times 1} = 1 - 0.6065 = 0.3935$
- $\alpha_2 = 1 - e^{-1.0 \times 1} = 1 - 0.3679 = 0.6321$
- $\alpha_3 = 1 - e^{-2.0 \times 1} = 1 - 0.1353 = 0.8647$

**Step 2 — Transmittances:**
- $T_1 = 1.0$
- $T_2 = (1 - 0.3935) = 0.6065$
- $T_3 = 0.6065 \times (1 - 0.6321) = 0.6065 \times 0.3679 = 0.2231$

**Step 3 — Contributions:**
- $C_1 = 1.0 \times 0.3935 \times 0.2 = 0.0787$
- $C_2 = 0.6065 \times 0.6321 \times 0.7 = 0.2683$
- $C_3 = 0.2231 \times 0.8647 \times 1.0 = 0.1929$

**Step 4 — Final Color:**
$$C(\vec{r}) = 0.0787 + 0.2683 + 0.1929 = \mathbf{0.5399}$$

### Loss Function
$$\mathcal{L} = \sum_{\vec{r} \in R} \left\| \hat{C}(\vec{r}) - C_{\text{GT}}(\vec{r}) \right\|_2^2$$

---

# 5. Lecture 10: Temperature Scaling, Simulated Annealing & Hallucinations

## 5.1 Simulated Annealing

### The Metallurgy Analogy
Heat metal → atoms move freely → cool slowly → atoms settle into optimal crystal structure.

### Algorithm
At each iteration: perturb $x_{\text{old}} \to x_{\text{new}}$

$$\Delta E = f(x_{\text{new}}) - f(x_{\text{old}})$$

- If $\Delta E < 0$ (improved): **Always accept**
- If $\Delta E \geq 0$ (worse): Accept with probability:

$$P = e^{-\Delta E / T}$$

### Temperature Limits

| Condition | Probability | Behavior |
|---|---|---|
| $T \to \infty$ | $P = e^0 = 1.0$ | Accept everything (exploration) |
| $T \to 0^+$ | $P = e^{-\infty} = 0.0$ | Reject all worse moves (greedy exploitation) |

### Cooling Schedule
$$T_{\text{new}} = \alpha \cdot T_{\text{old}}, \quad \alpha \in (0.80, 0.99)$$

## 5.2 LLM Temperature Scaling

### Temperature-Scaled Softmax
$$P_i = \frac{e^{z_i / T}}{\sum_{j=1}^{V} e^{z_j / T}}$$

### Temperature Limits

| $T$ | Effect | Result |
|---|---|---|
| $T \to 0^+$ | All probability on max logit | Deterministic, repetitive, greedy |
| $T = 1.0$ | Standard softmax | Balanced |
| $T \to \infty$ | $P_i = 1/V$ for all tokens | Uniform, random, hallucinations |

### Worked Example
Given logits $\mathbf{z} = [10.0, 5.0, 1.0]$:

**At $T = 0.1$ (Low):**
- $e^{10/0.1} = e^{100}$, $e^{5/0.1} = e^{50}$, $e^{1/0.1} = e^{10}$
- Token 1 dominates → $P \approx [0.999, 0.001, 0.000]$

**At $T = 2.0$ (High):**
- $e^{10/2} = e^5 = 148.4$, $e^{5/2} = e^{2.5} = 12.2$, $e^{1/2} = e^{0.5} = 1.65$
- Sum = 162.25
- $P \approx [0.915, 0.075, 0.010]$ — more spread out

## 5.3 LLM Hallucinations

### Why LLMs Hallucinate
1. **Next-token prediction** optimizes for plausibility, not factual accuracy
2. **No real-world grounding** — knowledge is static associative weights
3. **Pattern overgeneralization** — "X is the capital of Y" applied to unknown cases

### RAG Retrieval Failures (4 Modes)
1. **Embedding Distortion** — high-dimensional compression warps meaning
2. **Nearest-Neighbor Limits** — matches syntax, not exact semantics
3. **Incomplete Database** — true fact is simply missing
4. **Probabilistic Filler** — LLM bridges gaps with plausible guesses

## 5.4 RAG Cosine Similarity Trace

### Document Vectors
- $D_1$ ("Paris is the capital of France") = $[0.92, 0.81, 0.05]^\top$
- $D_2$ ("Berlin is the capital of Germany") = $[0.89, 0.77, 0.06]^\top$
- $D_3$ ("France won the 2018 FIFA World Cup") = $[0.40, 0.15, 0.95]^\top$

### Cosine Similarity Formula
$$\text{sim}(Q, D) = \frac{Q \cdot D}{\|Q\| \|D\|}$$

### Successful Query: "What is the capital of France?"
$Q_1 = [0.91, 0.80, 0.02]^\top$

Computing magnitudes and similarities:
- $\|Q_1\| = \sqrt{0.8281 + 0.64 + 0.0004} = 1.2118$
- $\text{sim}(Q_1, D_1) = 0.999$ ← **Retrieved! Correct!**
- $\text{sim}(Q_1, D_2) = 0.995$ ← Retrieved
- $\text{sim}(Q_1, D_3) = 0.31$ ← Not retrieved

### Failed Query: "What city hosted the 2018 World Cup final?"
$Q_2 = [0.58, 0.44, 0.61]^\top$

- Top-2 retrieves $D_3$ and $D_1$ → Context has "Paris" and "France" but NOT "Moscow"
- LLM generates: **"Paris hosted the 2018 World Cup final"** ← HALLUCINATION!
- True answer: Moscow (Luzhniki Stadium)

---

# 6. Lecture 11: Diffusion Models & Stable Diffusion

## 6.1 Forward Diffusion Process

### Iterative Formula (Markov Chain)
$$\mathbf{x}_{t+1} = \sqrt{\alpha_t} \mathbf{x}_t + \sqrt{1 - \alpha_t} \epsilon_t$$

> ⚠️ **Slide Typo:** Slides may show $\mathbf{x}_{t+1} = \alpha_t \mathbf{x}_t + (1 - \alpha_t) \epsilon$. The correct formula uses **square roots**.

### Direct Closed-Form (Reparameterization Trick)
$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$$

Where $\bar{\alpha}_t = \prod_{i=0}^{t} \alpha_i$ (cumulative product / `alpha_hat`)

### When to Use Which
- **Iterative:** Step-by-step analysis, theoretical proofs
- **Closed-form:** Training (jump to any timestep $t$ from $x_0$ in one step)

## 6.2 Forward Diffusion Trace

**Given:**
$$\mathbf{x}_0 = \begin{bmatrix} 1.0 & 2.0 \\ 3.0 & 4.0 \end{bmatrix}, \quad \alpha_0=0.9, \alpha_1=0.8, \alpha_2=0.7$$

$$\epsilon_0 = \begin{bmatrix} 0.5 & -1.0 \\ 0.0 & 0.3 \end{bmatrix}, \quad \epsilon_1 = \begin{bmatrix} -0.2 & 0.4 \\ 1.0 & -0.5 \end{bmatrix}$$

**Step 1:** $\sqrt{0.9} = 0.949$, $\sqrt{0.1} = 0.316$

$$\mathbf{x}_1 = 0.949 \begin{bmatrix} 1.0 & 2.0 \\ 3.0 & 4.0 \end{bmatrix} + 0.316 \begin{bmatrix} 0.5 & -1.0 \\ 0.0 & 0.3 \end{bmatrix} = \begin{bmatrix} 1.107 & 1.582 \\ 2.847 & 3.891 \end{bmatrix}$$

**Step 2:** $\sqrt{0.8} = 0.894$, $\sqrt{0.2} = 0.447$

$$\mathbf{x}_2 = 0.894 \begin{bmatrix} 1.107 & 1.582 \\ 2.847 & 3.891 \end{bmatrix} + 0.447 \begin{bmatrix} -0.2 & 0.4 \\ 1.0 & -0.5 \end{bmatrix} = \begin{bmatrix} 0.901 & 1.593 \\ 2.992 & 3.255 \end{bmatrix}$$

## 6.3 Stable Diffusion Architecture

### Pixel-Space vs Latent-Space

| Attribute | Pixel-Space (DDPM) | Latent-Space (Stable Diffusion) |
|---|---|---|
| Workspace | $512 \times 512 \times 3 = 786,432$ values | $64 \times 64 \times 4 = 16,384$ values (48× smaller!) |
| Multimodal | Unconditional (image-only) | Cross-attention driven (text, depth, etc.) |
| Distribution | $P(\mathbf{x})$ | $P(\mathbf{z} \mid \mathbf{c})$ |
| Speed | Very slow | Consumer GPU feasible |

### VAE (Variational Autoencoder)
- **Encoder** $E$: $\mathbf{x} \in \mathbb{R}^{H \times W \times 3} \to \mathbf{z} \in \mathbb{R}^{H/8 \times W/8 \times 4}$
- **Decoder** $D$: $\mathbf{z}_0 \to \tilde{\mathbf{x}} \approx \mathbf{x}$

### Training Loss (Latent Diffusion)
$$\mathcal{L}_{\text{LDM}} = \mathbb{E}_{\mathbf{z}_0, \epsilon, t, \mathbf{c}} \left[ \| \epsilon - \epsilon_\theta(\mathbf{z}_t, t, \tau_\theta(\mathbf{c})) \|_2^2 \right]$$

### Cross-Attention Mechanism
- **Query (Q):** from latent image features $\varphi_i(\mathbf{z}_t)$
- **Key (K) & Value (V):** from text embeddings $\tau_\theta(\mathbf{c})$

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right) V$$

### Classifier-Free Guidance (CFG)

$$\tilde{\epsilon}_\theta = \epsilon_\theta(\mathbf{z}_t, t, \emptyset) + s \cdot \left[ \epsilon_\theta(\mathbf{z}_t, t, \mathbf{c}) - \epsilon_\theta(\mathbf{z}_t, t, \emptyset) \right]$$

| Guidance Scale $s$ | Effect |
|---|---|
| $s \in (0, 1.5]$ | Ignores prompt, generic images |
| $s \in [7.0, 9.0]$ | **Sweet spot** — balanced fidelity and diversity |
| $s \geq 15.0$ | Over-saturated, color clipping, artifacts |

## 6.4 Code: Text-to-Image with Stable Diffusion

```python
from diffusers import StableDiffusionPipeline
import torch

def TextToImg(prompt, model_id="runwayml/stable-diffusion-v1-5",
              output_file="output.png", height=512, width=512,
              steps=30, guidance_scale=7.5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    pipe = pipe.to(device)
    
    # Under the hood:
    # 1. Text → CLIP Text Encoder → token embeddings
    # 2. Sample pure noise z_T ~ N(0, I) of shape (1, 4, H//8, W//8)
    # 3. For each step: UNet predicts noise, scheduler removes it
    # 4. CFG scales noise using guidance_scale
    # 5. VAE Decoder: z_0 → high-res image
    output = pipe(prompt=prompt, height=height, width=width,
                  num_inference_steps=steps, guidance_scale=guidance_scale)
    
    image = output.images[0]
    image.save(output_file)
    return image
```

## 6.5 Code: Pseudocode for Text-to-Image Inference

```python
def Text_To_Image(text):
    txtEmbed = TextEncoder(text)       # c = tau(text)
    Z_t = random_noise()               # z_T ~ N(0, I)
    T = 500                            # Number of diffusion steps
    
    for t in range(T, 0, -1):          # Reverse diffusion loop
        e_hat = UNet(Z_t, t, txtEmbed) # Predict noise at step t
        Z_t = removeNoise(Z_t, e_hat, t)  # Step from z_t to z_{t-1}
    
    image = VAE.decoder(Z_t)           # Decode latent to pixels
    return image
```

## 6.6 Code: Forward Diffusion Training

```python
import torch

def add_noise(x, t, alpha_hat):
    """Direct closed-form noising: x_t = sqrt(alpha_hat_t) * x_0 + sqrt(1-alpha_hat_t) * noise"""
    alpha_t = alpha_hat[t]                   # Shape: [B]
    alpha_t = alpha_t.view(-1, 1, 1, 1)      # Broadcast to [B, 1, 1, 1]
    
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus = torch.sqrt(1.0 - alpha_t)
    
    noise = torch.randn_like(x)              # Sample noise ~ N(0, I)
    noisy_x = sqrt_alpha_t * x + sqrt_one_minus * noise
    
    return noisy_x, noise

def train_epoch(model, loader, optimizer, criterion, timesteps, alpha_hat, device):
    model.train()
    total_loss = 0.0
    
    for imgs, _ in loader:
        imgs = imgs.to(device)
        t = torch.randint(0, timesteps, (imgs.size(0),), device=device)
        
        noisy_imgs, noise = add_noise(imgs, t, alpha_hat)
        predicted_noise = model(noisy_imgs, t)
        loss = criterion(predicted_noise, noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)
```

## 6.7 Simulated Annealing vs Diffusion — The Bridge

| Aspect | Simulated Annealing | Diffusion Models |
|---|---|---|
| Process | High T → Low T | High noise (t=T) → No noise (t=0) |
| High T/noise | Random exploration, accept bad moves | Pure Gaussian noise, no structure |
| Low T/noise | Greedy exploitation | Sharp, deterministic features |
| Objective | Find global minimum of loss function | Model complete data distribution |

---

# 7. All Code Snippets (Quick Reference)

## 7.1 FAISS Text Search (GloVe)
```python
# Sentence embedding = average of word vectors
def sentence_embedding(sent, word2vec, d=50):
    words = sent.lower().split()
    vecs = [word2vec[w] for w in words if w in word2vec]
    return np.mean(vecs, axis=0) if vecs else np.zeros(d, dtype='float32')

# Index: IVF with L2 quantizer
index = faiss.IndexIVFFlat(faiss.IndexFlatL2(50), 50, nlist=3)
index.train(embeddings)
index.add(embeddings)
index.nprobe = 3
D, I = index.search(query_vec, k=2)
```

## 7.2 FAISS Image Search (BLIP)
```python
# Extract + normalize image embeddings
with torch.no_grad():
    output = model.get_image_features(**inputs)
    emb = output.pooler_output
emb = torch.nn.functional.normalize(emb, dim=-1)

# Inner Product index (= cosine sim after norm)
index = faiss.IndexFlatIP(embedding_dim)
index.add(embeddings_matrix)
D, I = index.search(query_embedding, k=3)
```

## 7.3 SigLIP Patch Classification
```python
patches, coords = split_into_patches_with_coords(image, patch_size=64, stride=64)
inputs = processor(text=texts, images=patches, return_tensors="pt", padding=True)
with torch.no_grad():
    outputs = model(**inputs)
logits = outputs.logits_per_image
probs = torch.softmax(logits, dim=-1)
```

## 7.4 Cosine Similarity (Manual)
```python
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

## 7.5 Temperature-Scaled Softmax
```python
import numpy as np

def softmax_with_temperature(logits, T):
    scaled = np.array(logits) / T
    exp_vals = np.exp(scaled - np.max(scaled))  # Subtract max for numerical stability
    return exp_vals / np.sum(exp_vals)
```

## 7.6 NeRF Volumetric Rendering
```python
import numpy as np

def render_ray(sigmas, colors, delta_t):
    N = len(sigmas)
    alphas = [1 - np.exp(-sigmas[i] * delta_t) for i in range(N)]
    
    T = [1.0]  # T_1 = 1.0
    for i in range(1, N):
        T.append(T[-1] * (1 - alphas[i-1]))
    
    C = sum(T[i] * alphas[i] * colors[i] for i in range(N))
    return C
```

## 7.7 Forward Diffusion Step
```python
def forward_step(x_t, alpha_t, epsilon):
    return np.sqrt(alpha_t) * x_t + np.sqrt(1 - alpha_t) * epsilon
```

## 7.8 Stable Diffusion Pipeline
```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
image = pipe(prompt="...", num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("output.png")
```

---

# 8. Key Formulas Cheat Sheet

| Topic | Formula | Notes |
|---|---|---|
| **Cosine Similarity** | $\text{sim}(A,B) = \frac{A \cdot B}{\|A\| \|B\|}$ | After L2 norm: sim = dot product |
| **NeRF Opacity** | $\alpha_i = 1 - e^{-\sigma_i \Delta t_i}$ | $\sigma=0 \Rightarrow \alpha=0$; $\sigma\to\infty \Rightarrow \alpha\to1$ |
| **NeRF Transmittance** | $T_i = \prod_{j=1}^{i-1}(1-\alpha_j)$; $T_1=1$ | Probability of surviving to sample $i$ |
| **NeRF Pixel Color** | $C(\vec{r}) = \sum_i T_i \alpha_i c_i$ | Weighted sum of contributions |
| **SA Acceptance** | $P = e^{-\Delta E / T}$ | Only for $\Delta E \geq 0$ (worse moves) |
| **SA Cooling** | $T_{\text{new}} = \alpha T_{\text{old}}$, $\alpha \in (0,1)$ | Geometric decay |
| **LLM Softmax** | $P_i = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}$ | $T\to0$: greedy; $T\to\infty$: uniform |
| **Forward Diffusion (step)** | $x_{t+1} = \sqrt{\alpha_t} x_t + \sqrt{1-\alpha_t} \epsilon_t$ | Uses **square roots** |
| **Forward Diffusion (direct)** | $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t} \epsilon$ | $\bar\alpha_t = \prod_i \alpha_i$ |
| **CFG** | $\tilde\epsilon = \epsilon_\emptyset + s(\epsilon_c - \epsilon_\emptyset)$ | $s \in [7,9]$ sweet spot |
| **SigLIP Loss** | $\mathcal{L} = -\frac{1}{B}\sum\sum\log\sigma(y_{ij}(z_i \cdot z_j \cdot e^t + b))$ | $y=+1$ match, $y=-1$ mismatch |
| **ColPali MaxSim** | $S(q,d) = \sum_i \max_j (\vec{t}_i \cdot \vec{p}_j)$ | Sum of per-token best-patch scores |
| **Cross-Attention** | $\text{Attn}(Q,K,V) = \text{softmax}(QK^\top/\sqrt{d})V$ | Q=image, K&V=text |
| **VAE Compression** | $H/8 \times W/8 \times 4$ | 512→64 spatial, 3→4 channels |
| **NeRF Loss** | $\mathcal{L} = \sum_r \|\hat{C}(r) - C_{GT}(r)\|_2^2$ | MSE over ray colors |
| **LDM Loss** | $\mathcal{L} = \|\epsilon - \epsilon_\theta(z_t, t, \tau(c))\|_2^2$ | MSE between true & predicted noise |

---

# 9. MCQ Bank (30 Questions)

## Lecture 4 & 6: FAISS & Vector Databases

**Q1.** What is the primary purpose of calling `faiss.normalize_L2()` before using `IndexFlatIP`?
- A) Reduces dimensionality of vectors
- B) Makes Inner Product equivalent to Cosine Similarity ✅
- C) Speeds up the clustering algorithm
- D) Converts vectors to integer format

**Q2.** In FAISS `IndexIVFFlat`, what must be done before calling `index.add()`?
- A) Set `nprobe` to the correct value
- B) Normalize the query vector
- C) Call `index.train()` on the training data ✅
- D) Sort the vectors by magnitude

**Q3.** In the BLIP image search pipeline, why is `model.eval()` called?
- A) To load model weights from disk
- B) To disable training-specific behaviors like dropout ✅
- C) To enable gradient computation
- D) To convert the model to half-precision

**Q4.** What does `torch.no_grad()` achieve during embedding extraction?
- A) Enables backpropagation for fine-tuning
- B) Reduces memory consumption and speeds up inference ✅
- C) Increases the accuracy of embeddings
- D) Normalizes the output vectors

## Lecture 7: SigLIP & ColPali

**Q5.** What is the key mathematical difference between CLIP and SigLIP?
- A) CLIP uses Vision Transformers while SigLIP uses CNNs
- B) SigLIP replaces global Softmax with pairwise Sigmoid binary cross-entropy ✅
- C) CLIP uses cosine similarity while SigLIP uses L2 distance
- D) SigLIP requires smaller batch sizes

**Q6.** In SigLIP's loss function, what does $y_{i,j} = -1$ represent?
- A) A matching image-text pair
- B) A non-matching (negative) image-text pair ✅
- C) A self-referencing pair
- D) An unlabeled pair

**Q7.** Why does global pooling in SigLIP fail for document retrieval?
- A) Documents are too small for vision transformers
- B) The pooling operation averages out fine-grained localized text details ✅
- C) SigLIP can only process square images
- D) Global pooling requires too much compute

**Q8.** What does the ColPali MaxSim operator compute?
- A) The average similarity between all query tokens and all patches
- B) For each query token, the max similarity to any patch, then sum the max values ✅
- C) The maximum similarity between the global query vector and global document vector
- D) The softmax-weighted average of all patch similarities

## Lecture 8: Space Alignment vs Space Mapping

**Q9.** What characterizes Space Alignment (Joint Embedding)?
- A) One modality is projected into the other's native space
- B) Both modalities are projected into a brand new shared coordinate system ✅
- C) Only image embeddings are modified
- D) Text is concatenated with image features

**Q10.** In Space Mapping, how many projection matrices are learned?
- A) Zero — no projection needed
- B) One — usually mapping image to text space ✅
- C) Two — one for each modality
- D) Three — one for images, one for text, one shared

**Q11.** Which real-world architecture uses Space Mapping?
- A) CLIP
- B) SigLIP
- C) Multimodal LLMs (mapping visual tokens to LLM vocabulary) ✅
- D) GloVe word embeddings

## Lecture 9: Neural Radiance Fields

**Q12.** What is the dimensionality of NeRF's input?
- A) 3D (x, y, z only)
- B) 5D (x, y, z position + θ, φ viewing direction) ✅
- C) 6D (x, y, z + R, G, B)
- D) 2D (pixel coordinates)

**Q13.** Why must volume density σ depend ONLY on position x?
- A) To reduce computation time
- B) To ensure geometry is view-independent (shapes don't warp with viewing angle) ✅
- C) Because density is always constant
- D) To allow the model to represent reflections

**Q14.** What does $T_1$ (transmittance at sample 1) always equal?
- A) 0.0
- B) 0.5
- C) 1.0 ✅
- D) It depends on $\sigma_1$

**Q15.** If $\sigma_i = 0$ for a sample, what is its opacity $\alpha_i$?
- A) 1.0
- B) 0.5
- C) 0.0 ✅ (since $1 - e^0 = 1 - 1 = 0$)
- D) Undefined

**Q16.** What is the NeRF training loss?
- A) Cross-entropy between predicted and true labels
- B) MSE between rendered pixel colors and ground-truth pixel colors ✅
- C) Sigmoid binary classification loss
- D) Contrastive triplet loss

## Lecture 10: Temperature & Hallucinations

**Q17.** In Simulated Annealing, when is the acceptance probability $P$ highest for a worse solution?
- A) When temperature T is very low
- B) When temperature T is very high ✅ (since $P = e^{-\Delta E/T} \to 1$ as $T \to \infty$)
- C) When $\Delta E = 0$
- D) When the cooling rate α is close to 1

**Q18.** What happens to LLM softmax output as $T \to \infty$?
- A) Probability collapses to the top token
- B) Distribution becomes uniform: $P_i = 1/V$ for all tokens ✅
- C) All probabilities become zero
- D) The model outputs exact logits

**Q19.** What happens to LLM softmax output as $T \to 0^+$?
- A) Uniform distribution
- B) All probability mass goes to the single highest-logit token (greedy) ✅
- C) Model outputs random tokens
- D) Probabilities oscillate

**Q20.** Which is NOT a cause of RAG-induced hallucinations?
- A) Embedding distortion
- B) Incomplete database
- C) Using sigmoid instead of softmax loss ✅
- D) Nearest-neighbor matching syntax instead of exact semantics

**Q21.** In the cooling schedule $T_{\text{new}} = \alpha T_{\text{old}}$, what range should α be in?
- A) $\alpha \in (1, 2)$
- B) $\alpha \in (0, 1)$, typically 0.80 to 0.99 ✅
- C) $\alpha \in (-1, 0)$
- D) $\alpha$ must equal exactly 0.5

## Lecture 11: Diffusion & Stable Diffusion

**Q22.** What is the correct forward diffusion step formula?
- A) $x_{t+1} = \alpha_t x_t + (1-\alpha_t)\epsilon$ (without square roots)
- B) $x_{t+1} = \sqrt{\alpha_t} x_t + \sqrt{1-\alpha_t} \epsilon_t$ ✅
- C) $x_{t+1} = x_t + \epsilon$
- D) $x_{t+1} = \alpha_t^2 x_t + (1-\alpha_t)^2 \epsilon$

**Q23.** What is $\bar{\alpha}_t$ (alpha_hat)?
- A) The average of all alpha values
- B) The cumulative product: $\bar{\alpha}_t = \prod_{i=0}^{t} \alpha_i$ ✅
- C) The square root of $\alpha_t$
- D) The learning rate

**Q24.** What is the latent space size for a 512×512 image in Stable Diffusion?
- A) 256×256×3
- B) 128×128×8
- C) 64×64×4 ✅ (VAE downscales by factor of 8)
- D) 32×32×16

**Q25.** In Cross-Attention for Stable Diffusion, what serves as the Query?
- A) Text embeddings
- B) Latent image features ✅
- C) Random noise
- D) The guidance scale parameter

**Q26.** In Classifier-Free Guidance, what does $s = 1.0$ mean?
- A) Maximum prompt adherence
- B) Standard conditional generation without extrapolation ✅
- C) Pure unconditional generation
- D) Maximum creative diversity

**Q27.** What guidance scale range is the "sweet spot" for Stable Diffusion?
- A) $s \in (0, 0.5)$
- B) $s \in [7.0, 9.0]$ ✅
- C) $s \in [15, 20]$
- D) $s = 100$

**Q28.** What does the VAE Decoder do in Stable Diffusion?
- A) Encodes text prompts into token embeddings
- B) Reconstructs high-resolution pixels from denoised latent representation ✅
- C) Adds noise during the forward process
- D) Predicts the noise vector at each timestep

**Q29.** Why does Stable Diffusion use half-precision (float16) on GPU?
- A) It increases accuracy
- B) It saves memory and approximately doubles performance speed ✅
- C) It is required by the VAE architecture
- D) It enables gradient computation

**Q30.** The `add_noise` function uses `.view(-1, 1, 1, 1)`. Why?
- A) To flatten the tensor for matrix multiplication
- B) To enable PyTorch broadcasting across [B, C, H, W] dimensions ✅
- C) To convert from float32 to float16
- D) To add batch normalization

---

# Good Luck on Your Final! 🎓

**Remember:**
1. For **tracing questions**: Show every calculation step. Don't skip intermediate values.
2. For **coding questions**: Include comments explaining what each line does.
3. For **MCQs**: Eliminate obviously wrong answers first, then reason about the remaining two.
4. Key values to memorize: $e^{-0.5} \approx 0.6065$, $e^{-1} \approx 0.3679$, $e^{-2} \approx 0.1353$
