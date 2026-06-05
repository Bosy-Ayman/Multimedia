# DSAI-413 Multimedia Processing — Mock Exam 3 (Final Exam)

> **Total Marks: 40** | **Duration: 2 Hours**
>
> - Section A: Multiple Choice Questions — 8 Marks
> - Section B: Tracing Questions — 20 Marks
> - Section C: Coding Questions — 12 Marks

---

## Section A: Multiple Choice Questions (8 × 1 = 8 Marks)

**Q1.** In SigLIP's sigmoid loss, the label $y_{i,j}$ can be $+1$ or $-1$. What does $y_{i,j} = -1$ represent?

- (A) A matching image-text pair that should have high similarity
- (B) A non-matching (negative) image-text pair that should have low similarity ✗
- (C) An unlabeled pair used for semi-supervised learning
- (D) A text-only sample with no corresponding image

---

**Q2.** What is the closed-form formula for the forward diffusion process that allows sampling $x_t$ directly from $x_0$?

- (A) $x_t = \bar{\alpha}_t \cdot x_0 + (1 - \bar{\alpha}_t) \cdot \epsilon$
- (B) $x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon$ ✗
- (C) $x_t = \sqrt{\bar{\alpha}_t} \cdot \epsilon + \sqrt{1 - \bar{\alpha}_t} \cdot x_0$
- (D) $x_t = (1 - \bar{\alpha}_t) \cdot x_0 + \bar{\alpha}_t \cdot \epsilon$

---

**Q3.** In a Neural Radiance Field (NeRF), the predicted color $\mathbf{c}$ depends on both the 3D position $\mathbf{x}$ and the viewing direction $\mathbf{d}$. Why is the viewing direction necessary?

- (A) To determine the depth of each sample along the ray
- (B) To model view-dependent effects such as specular highlights and reflections ✗
- (C) To control the learning rate during training
- (D) To reduce the number of samples needed per ray

---

**Q4.** In NeRF's volume rendering equation, the transmittance $T_i$ at the **first** sample ($i = 1$) along a ray always equals:

- (A) $0.0$
- (B) $1.0$ ✗
- (C) $1 - \alpha_1$
- (D) $\alpha_1$

---

**Q5.** In Stable Diffusion, a Variational Autoencoder (VAE) encodes a $512 \times 512$ pixel image into a latent representation. What is the size of this latent tensor?

- (A) $128 \times 128 \times 3$ (downscale by 4, keep RGB channels)
- (B) $64 \times 64 \times 4$ (downscale by 8, 4 latent channels) ✗
- (C) $32 \times 32 \times 8$ (downscale by 16, 8 latent channels)
- (D) $256 \times 256 \times 4$ (downscale by 2, 4 latent channels)

---

**Q6.** In the Metropolis-Hastings acceptance criterion $P = e^{-\Delta E / T}$, when is the probability $P$ of accepting a **worse** solution ($\Delta E > 0$) the highest?

- (A) When $T$ is very low, because the system is exploratory
- (B) When $T$ is very high, since $\Delta E / T \to 0$ and $P \to 1$ ✗
- (C) When $\Delta E$ is very large and $T$ is moderate
- (D) $P$ is always constant regardless of $T$

---

**Q7.** ColPali uses a **multi-vector** representation for document pages instead of a single global embedding. What problem does this solve?

- (A) It reduces the memory required to store document embeddings
- (B) Global pooling in models like SigLIP averages out fine-grained localized text details in document pages ✗
- (C) It eliminates the need for OCR entirely
- (D) It allows the model to process multiple documents simultaneously

---

**Q8.** In the cross-attention mechanism used within Stable Diffusion's U-Net, what serves as the **Query** ($Q$)?

- (A) The CLIP text embeddings
- (B) The latent image features (while $K$ and $V$ come from text embeddings) ✗
- (C) The noise prediction from the previous timestep
- (D) The VAE decoder output

---

## Section B: Tracing Questions (2 × 10 = 20 Marks)

---

### Tracing Question 1: LLM Temperature Scaling + Simulated Annealing (10 Marks)

#### Part A — Temperature-Scaled Softmax (6 Marks)

An LLM produces the following logits for four vocabulary tokens:

| Token | $z_i$ |
|-------|-------|
| "cat" | 4.0 |
| "dog" | 2.0 |
| "bird" | 1.0 |
| "fish" | 0.0 |

The temperature-scaled softmax is:

$$P(i) = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

**a) (3 marks)** Compute the softmax probability for **each** token at $T = 0.5$. Show all $e^{z_i / T}$ values and their sum.

**b) (3 marks)** Compute the softmax probability for **each** token at $T = 3.0$. Show all values. Compare the two distributions and explain which temperature produces a more **deterministic** output.

#### Part B — Simulated Annealing (4 Marks)

A simulated annealing optimizer is minimizing a cost function. The current state has energy $f(x_{\text{old}}) = 10$ and a proposed neighbor has energy $f(x_{\text{new}}) = 12$. The current temperature is $T = 5$.

The acceptance probability for a worse solution is:

$$P = e^{-\Delta E / T}, \quad \text{where } \Delta E = f(x_{\text{new}}) - f(x_{\text{old}})$$

**c) (2 marks)** Compute $\Delta E$ and the acceptance probability $P$.

**d) (2 marks)** If the temperature drops to $T = 0.5$, recompute $P$. Explain why the system is less likely to accept worse solutions at low temperatures.

---

### Tracing Question 2: Space Mapping — Asymmetric Projection (10 Marks)

In **Space Mapping**, we learn a projection matrix $W_{\text{img}}$ that maps image embeddings into the **text** embedding space, while text embeddings remain **frozen**. The similarity is then:

$$\text{sim}(v_{\text{img}}, v_{\text{txt}}) = (W_{\text{img}} \cdot v_{\text{img}})^T \cdot v_{\text{txt}}$$

**Given Embeddings** ($\in \mathbb{R}^4$):

| Vector | Values |
|--------|--------|
| $v_{\text{im1}}$ (Cat image) | $[3, 0, 1, 1]^T$ |
| $v_{\text{im2}}$ (Dog image) | $[0, 2, 1, 1]^T$ |
| $v_{\text{txt1}}$ ("Cat" text) | $[2, 0, 1, 1]^T$ |
| $v_{\text{txt2}}$ ("Dog" text) | $[0, 1, 1, 0]^T$ |

**Mapping Matrix:**

$$W_{\text{img}} = \begin{bmatrix} 0.5 & 0 & 0.5 & 0 \\ 0 & 1 & 0 & -1 \\ 0 & 0 & 1 & 0 \\ 0.5 & 0 & 0 & 0.5 \end{bmatrix} \in \mathbb{R}^{4 \times 4}$$

**a) (3 marks)** Compute $v'_{\text{im1}} = W_{\text{img}} \cdot v_{\text{im1}}$. Show the computation for each element.

**b) (3 marks)** Compute $v'_{\text{im2}} = W_{\text{img}} \cdot v_{\text{im2}}$. Show the computation for each element.

**c) (2 marks)** Compute the dot products:
- $v'_{\text{im1}} \cdot v_{\text{txt1}}$ (Cat image ↔ "Cat" text — **match**)
- $v'_{\text{im1}} \cdot v_{\text{txt2}}$ (Cat image ↔ "Dog" text — **mismatch**)

**d) (2 marks)** Compute the dot products:
- $v'_{\text{im2}} \cdot v_{\text{txt2}}$ (Dog image ↔ "Dog" text — **match**)
- $v'_{\text{im2}} \cdot v_{\text{txt1}}$ (Dog image ↔ "Cat" text — **mismatch**)

Does the mapping correctly discriminate matching from non-matching pairs? Discuss.

---

## Section C: Coding Questions (2 × 6 = 12 Marks)

---

### Coding Question 1: SigLIP Patch-Based Classification (6 Marks)

You are given a utility function `split_into_patches(img, patch_size, stride)` that splits a PIL image into patches and returns a list of `(patch, (x, y))` tuples.

Write a complete Python script that:

1. **(1 mark)** Loads an image from `"input.jpg"` and splits it into $64 \times 64$ patches with stride 64.
2. **(2 marks)** Uses a SigLIP model to compute similarity logits between each patch and the text labels `["a cat", "a dog", "a bird"]`.
3. **(1 mark)** Converts the logits to probabilities using softmax.
4. **(1 mark)** Finds which patch has the **highest** probability for `"a dog"`.
5. **(1 mark)** Prints the winning patch's index, coordinates, and its probability for each text label.

**Use:** `model_name = "google/siglip-base-patch16-224"` and `from transformers import AutoProcessor, AutoModel`.

---

### Coding Question 2: Stable Diffusion Text-to-Image Pipeline (6 Marks)

Write a complete Python function `TextToImg(prompt, steps=30, guidance_scale=7.5)` that:

1. **(1 mark)** Detects whether CUDA is available and sets the device accordingly.
2. **(2 marks)** Loads `StableDiffusionPipeline` from `"runwayml/stable-diffusion-v1-5"` with appropriate dtype (`float16` on CUDA, `float32` on CPU).
3. **(1 mark)** Generates a $512 \times 512$ image using the pipeline.
4. **(1 mark)** Saves the output image to `"output.png"`.
5. **(1 mark)** Returns the PIL image.

Additionally, include comments explaining:
- What `guidance_scale` controls (Classifier-Free Guidance).
- What happens when `guidance_scale = 1.0` vs `guidance_scale = 15.0`.

---

---

# MODEL ANSWERS

---

## Section A: MCQ Answers

| Q | Answer | Explanation |
|---|--------|-------------|
| 1 | **(B)** | $y_{i,j} = -1$ indicates a negative (non-matching) image-text pair. The sigmoid loss pushes the similarity score down for these pairs. |
| 2 | **(B)** | The reparameterization trick gives $x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon$, where $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$ and $\epsilon \sim \mathcal{N}(0, I)$. |
| 3 | **(B)** | View-dependent effects (specular highlights, reflections) change depending on the viewing angle. Without $\mathbf{d}$, NeRF can only model diffuse (Lambertian) surfaces. |
| 4 | **(B)** | $T_1 = \exp\left(-\sum_{j=1}^{0} \sigma_j \delta_j\right) = e^0 = 1.0$. No volume has been traversed before the first sample. |
| 5 | **(B)** | The VAE encoder downscales spatial dimensions by a factor of 8: $512/8 = 64$. It uses 4 latent channels, giving $64 \times 64 \times 4$. |
| 6 | **(B)** | At high $T$: $\Delta E / T \to 0$, so $P = e^{-\Delta E/T} \to e^0 = 1$. The system freely accepts worse solutions for exploration. |
| 7 | **(B)** | ColPali retains per-patch embeddings (multi-vector) so that fine-grained text in specific regions of a document page is not lost through global average pooling. |
| 8 | **(B)** | In cross-attention: $Q$ = latent image features, $K$ and $V$ = text embeddings projected from CLIP. This lets each spatial location in the latent attend to relevant text tokens. |

---

## Section B: Tracing Answers

---

### Tracing Question 1 — Model Answer

#### Part A(a): Softmax at $T = 0.5$ (3 marks)

Compute scaled logits $z_i / T$ where $T = 0.5$:

$$\frac{z_i}{T}: \quad \frac{4.0}{0.5} = 8.0, \quad \frac{2.0}{0.5} = 4.0, \quad \frac{1.0}{0.5} = 2.0, \quad \frac{0.0}{0.5} = 0.0$$

Compute exponentials:

$$e^{8.0} = 2980.958, \quad e^{4.0} = 54.598, \quad e^{2.0} = 7.389, \quad e^{0.0} = 1.000$$

Sum of exponentials:

$$\text{Sum} = 2980.958 + 54.598 + 7.389 + 1.000 = 3043.945$$

Probabilities:

| Token | $e^{z_i/T}$ | $P(i)$ |
|-------|-------------|---------|
| "cat" | $2980.958$ | $2980.958 / 3043.945 = \mathbf{0.9793}$ |
| "dog" | $54.598$ | $54.598 / 3043.945 = \mathbf{0.0179}$ |
| "bird" | $7.389$ | $7.389 / 3043.945 = \mathbf{0.0024}$ |
| "fish" | $1.000$ | $1.000 / 3043.945 = \mathbf{0.0003}$ |

#### Part A(b): Softmax at $T = 3.0$ (3 marks)

Compute scaled logits $z_i / T$ where $T = 3.0$:

$$\frac{z_i}{T}: \quad \frac{4.0}{3.0} = 1.333, \quad \frac{2.0}{3.0} = 0.667, \quad \frac{1.0}{3.0} = 0.333, \quad \frac{0.0}{3.0} = 0.000$$

Compute exponentials:

$$e^{1.333} = 3.792, \quad e^{0.667} = 1.948, \quad e^{0.333} = 1.395, \quad e^{0.000} = 1.000$$

Sum of exponentials:

$$\text{Sum} = 3.792 + 1.948 + 1.395 + 1.000 = 8.135$$

Probabilities:

| Token | $e^{z_i/T}$ | $P(i)$ |
|-------|-------------|---------|
| "cat" | $3.792$ | $3.792 / 8.135 = \mathbf{0.4662}$ |
| "dog" | $1.948$ | $1.948 / 8.135 = \mathbf{0.2394}$ |
| "bird" | $1.395$ | $1.395 / 8.135 = \mathbf{0.1715}$ |
| "fish" | $1.000$ | $1.000 / 8.135 = \mathbf{0.1229}$ |

**Comparison:**

| Token | $P$ at $T=0.5$ | $P$ at $T=3.0$ |
|-------|-----------------|-----------------|
| "cat" | 0.9793 | 0.4662 |
| "dog" | 0.0179 | 0.2394 |
| "bird" | 0.0024 | 0.1715 |
| "fish" | 0.0003 | 0.1229 |

- **$T = 0.5$** produces a **sharper, more deterministic** distribution — nearly all probability mass is on "cat" (97.93%). The model is very confident.
- **$T = 3.0$** produces a **flatter, more uniform** distribution — probability is spread across all tokens. The model is more exploratory/creative.
- **Lower temperature → more deterministic** (approaches argmax); **higher temperature → more random** (approaches uniform).

---

#### Part B(c): Simulated Annealing at $T = 5$ (2 marks)

$$\Delta E = f(x_{\text{new}}) - f(x_{\text{old}}) = 12 - 10 = 2$$

$$P = e^{-\Delta E / T} = e^{-2/5} = e^{-0.4} = \mathbf{0.6703}$$

There is a **67.03%** chance of accepting this worse solution at $T = 5$. The high temperature encourages exploration.

#### Part B(d): Simulated Annealing at $T = 0.5$ (2 marks)

$$P = e^{-\Delta E / T} = e^{-2/0.5} = e^{-4.0} = \mathbf{0.0183}$$

There is only a **1.83%** chance of accepting this worse solution at $T = 0.5$.

**Explanation:** At low temperatures, the exponent $-\Delta E / T$ becomes a large negative number (since $T$ is small), making $P = e^{-\Delta E/T}$ very close to zero. The system becomes **greedy** — it almost exclusively accepts improvements (downhill moves) and rejects worse solutions. This mirrors the **cooling schedule** in simulated annealing: early on (high $T$) the system explores freely to escape local minima; later (low $T$) it converges to the nearest optimum.

---

### Tracing Question 2 — Model Answer

#### Part (a): Compute $v'_{\text{im1}} = W_{\text{img}} \cdot v_{\text{im1}}$ (3 marks)

$$W_{\text{img}} \cdot v_{\text{im1}} = \begin{bmatrix} 0.5 & 0 & 0.5 & 0 \\ 0 & 1 & 0 & -1 \\ 0 & 0 & 1 & 0 \\ 0.5 & 0 & 0 & 0.5 \end{bmatrix} \begin{bmatrix} 3 \\ 0 \\ 1 \\ 1 \end{bmatrix}$$

**Row 1:** $(0.5)(3) + (0)(0) + (0.5)(1) + (0)(1) = 1.5 + 0 + 0.5 + 0 = \mathbf{2.0}$

**Row 2:** $(0)(3) + (1)(0) + (0)(1) + (-1)(1) = 0 + 0 + 0 - 1 = \mathbf{-1.0}$

**Row 3:** $(0)(3) + (0)(0) + (1)(1) + (0)(1) = 0 + 0 + 1 + 0 = \mathbf{1.0}$

**Row 4:** $(0.5)(3) + (0)(0) + (0)(1) + (0.5)(1) = 1.5 + 0 + 0 + 0.5 = \mathbf{2.0}$

$$\boxed{v'_{\text{im1}} = [2.0, -1.0, 1.0, 2.0]^T}$$

#### Part (b): Compute $v'_{\text{im2}} = W_{\text{img}} \cdot v_{\text{im2}}$ (3 marks)

$$W_{\text{img}} \cdot v_{\text{im2}} = \begin{bmatrix} 0.5 & 0 & 0.5 & 0 \\ 0 & 1 & 0 & -1 \\ 0 & 0 & 1 & 0 \\ 0.5 & 0 & 0 & 0.5 \end{bmatrix} \begin{bmatrix} 0 \\ 2 \\ 1 \\ 1 \end{bmatrix}$$

**Row 1:** $(0.5)(0) + (0)(2) + (0.5)(1) + (0)(1) = 0 + 0 + 0.5 + 0 = \mathbf{0.5}$

**Row 2:** $(0)(0) + (1)(2) + (0)(1) + (-1)(1) = 0 + 2 + 0 - 1 = \mathbf{1.0}$

**Row 3:** $(0)(0) + (0)(2) + (1)(1) + (0)(1) = 0 + 0 + 1 + 0 = \mathbf{1.0}$

**Row 4:** $(0.5)(0) + (0)(2) + (0)(1) + (0.5)(1) = 0 + 0 + 0 + 0.5 = \mathbf{0.5}$

$$\boxed{v'_{\text{im2}} = [0.5, 1.0, 1.0, 0.5]^T}$$

#### Part (c): Dot products for Cat image (2 marks)

**Match — $v'_{\text{im1}} \cdot v_{\text{txt1}}$ (Cat image ↔ "Cat" text):**

$$[2.0, -1.0, 1.0, 2.0] \cdot [2, 0, 1, 1] = (2.0)(2) + (-1.0)(0) + (1.0)(1) + (2.0)(1)$$
$$= 4.0 + 0 + 1.0 + 2.0 = \boxed{7.0}$$

**Mismatch — $v'_{\text{im1}} \cdot v_{\text{txt2}}$ (Cat image ↔ "Dog" text):**

$$[2.0, -1.0, 1.0, 2.0] \cdot [0, 1, 1, 0] = (2.0)(0) + (-1.0)(1) + (1.0)(1) + (2.0)(0)$$
$$= 0 - 1.0 + 1.0 + 0 = \boxed{0.0}$$

✅ **Cat image:** Match score ($7.0$) $\gg$ Mismatch score ($0.0$). The mapping **correctly discriminates** for the cat image.

#### Part (d): Dot products for Dog image (2 marks)

**Match — $v'_{\text{im2}} \cdot v_{\text{txt2}}$ (Dog image ↔ "Dog" text):**

$$[0.5, 1.0, 1.0, 0.5] \cdot [0, 1, 1, 0] = (0.5)(0) + (1.0)(1) + (1.0)(1) + (0.5)(0)$$
$$= 0 + 1.0 + 1.0 + 0 = \boxed{2.0}$$

**Mismatch — $v'_{\text{im2}} \cdot v_{\text{txt1}}$ (Dog image ↔ "Cat" text):**

$$[0.5, 1.0, 1.0, 0.5] \cdot [2, 0, 1, 1] = (0.5)(2) + (1.0)(0) + (1.0)(1) + (0.5)(1)$$
$$= 1.0 + 0 + 1.0 + 0.5 = \boxed{2.5}$$

⚠️ **Dog image:** Match score ($2.0$) $<$ Mismatch score ($2.5$). The mapping **fails to discriminate** for the dog image.

**Discussion:**

The mapping matrix $W_{\text{img}}$ successfully discriminates for the **cat** image (score gap: $7.0$ vs $0.0$), but **fails** for the **dog** image (match $2.0$ < mismatch $2.5$). This indicates that $W_{\text{img}}$ is **not yet fully optimized** — it has not converged to a solution that correctly separates all matching from non-matching pairs. In practice, the contrastive loss would continue to update $W_{\text{img}}$ through gradient descent to increase match scores and decrease mismatch scores until correct discrimination is achieved for **all** pairs. This illustrates that Space Mapping requires sufficient training; a partially-trained projection matrix can introduce errors even if the underlying embeddings carry discriminative information.

---

## Section C: Coding Answers

---

### Coding Question 1 — Model Answer

```python
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, AutoModel

# ──────────────────────────────────────────────
# Helper (given)
# ──────────────────────────────────────────────
def split_into_patches(img, patch_size, stride):
    """
    Splits a PIL image into patches.
    Returns: list of (patch_PIL, (x, y)) tuples
    """
    patches = []
    w, h = img.size
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = img.crop((x, y, x + patch_size, y + patch_size))
            patches.append((patch, (x, y)))
    return patches

# ──────────────────────────────────────────────
# 1. Load image and split into patches (1 mark)
# ──────────────────────────────────────────────
image = Image.open("input.jpg").convert("RGB")
patches_and_coords = split_into_patches(image, patch_size=64, stride=64)
patches = [p for p, _ in patches_and_coords]
coords  = [c for _, c in patches_and_coords]

# ──────────────────────────────────────────────
# 2. Load SigLIP model and compute logits (2 marks)
# ──────────────────────────────────────────────
model_name = "google/siglip-base-patch16-224"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

text_labels = ["a cat", "a dog", "a bird"]

all_logits = []

with torch.no_grad():
    for patch in patches:
        inputs = processor(
            text=text_labels,
            images=patch,
            return_tensors="pt",
            padding=True,
        )
        outputs = model(**inputs)
        # SigLIP returns logits_per_image: shape (1, num_texts)
        logits = outputs.logits_per_image          # (1, 3)
        all_logits.append(logits.squeeze(0))       # (3,)

# Stack: (num_patches, num_texts)
all_logits = torch.stack(all_logits)               # (N, 3)

# ──────────────────────────────────────────────
# 3. Convert logits to probabilities (1 mark)
# ──────────────────────────────────────────────
all_probs = F.softmax(all_logits, dim=-1)          # (N, 3)

# ──────────────────────────────────────────────
# 4. Find patch with highest "a dog" prob (1 mark)
# ──────────────────────────────────────────────
dog_index = text_labels.index("a dog")             # = 1
dog_probs = all_probs[:, dog_index]                # (N,)
best_patch_idx = torch.argmax(dog_probs).item()

# ──────────────────────────────────────────────
# 5. Print results (1 mark)
# ──────────────────────────────────────────────
best_coord = coords[best_patch_idx]
print(f"Best patch for 'a dog': index={best_patch_idx}, coord={best_coord}")
for i, label in enumerate(text_labels):
    print(f"  P('{label}') = {all_probs[best_patch_idx, i].item():.4f}")
```

**Key Points:**
- Each patch is processed independently through SigLIP alongside all text labels.
- `logits_per_image` gives the raw similarity scores; softmax converts them to a probability distribution over the text labels for each patch.
- `argmax` over the "a dog" column across all patches finds the most "dog-like" region.

---

### Coding Question 2 — Model Answer

```python
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image


def TextToImg(prompt: str, steps: int = 30, guidance_scale: float = 7.5) -> Image.Image:
    """
    Generate a 512x512 image from a text prompt using Stable Diffusion v1.5.

    About guidance_scale (Classifier-Free Guidance / CFG):
    -------------------------------------------------------
    guidance_scale controls how strongly the generated image adheres to the
    text prompt. During inference, two noise predictions are made:
        - e_uncond: unconditional prediction (no text)
        - e_cond:   conditional prediction (with text)
    The final prediction is:
        e = e_uncond + guidance_scale * (e_cond - e_uncond)

    - guidance_scale = 1.0  →  CFG is effectively OFF. The model generates
      images with high diversity but weak prompt adherence. Results may look
      coherent but often ignore or loosely follow the text.
    - guidance_scale = 7.5  →  A balanced default. Good prompt adherence
      with reasonable image quality and diversity.
    - guidance_scale = 15.0 →  Very strong prompt adherence, but images can
      become over-saturated, overly sharp, or exhibit artifacts due to the
      exaggerated correction away from the unconditional prediction.
    """

    # ──────────────────────────────────────────────
    # 1. Detect device (1 mark)
    # ──────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ──────────────────────────────────────────────
    # 2. Load pipeline with appropriate dtype (2 marks)
    # ──────────────────────────────────────────────
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=dtype,
    ).to(device)

    # ──────────────────────────────────────────────
    # 3. Generate 512×512 image (1 mark)
    # ──────────────────────────────────────────────
    result = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        height=512,
        width=512,
    )
    image = result.images[0]

    # ──────────────────────────────────────────────
    # 4. Save to file (1 mark)
    # ──────────────────────────────────────────────
    image.save("output.png")

    # ──────────────────────────────────────────────
    # 5. Return the PIL image (1 mark)
    # ──────────────────────────────────────────────
    return image


# Example usage
if __name__ == "__main__":
    img = TextToImg("A futuristic city skyline at sunset, digital art")
    print(f"Image size: {img.size}")  # (512, 512)
```

**Key Points:**
- `float16` is used on CUDA for faster inference and lower VRAM usage; `float32` is required on CPU since most CPUs do not support half-precision.
- The `guidance_scale` parameter implements **Classifier-Free Guidance (CFG)**: the noise prediction is interpolated between unconditional and conditional estimates. Scale = 1.0 means no guidance (unconditional generation); higher values push the output to match the prompt more aggressively at the cost of diversity and potential artifacts.

---

*End of Mock Exam 3*
