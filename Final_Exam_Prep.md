# DSAI-413 Multimedia Final Exam Preparation Guide

## 1. Expected Changes from Sheet to Exam (Tracing & Coding)

### Tracing Question 1: Space Alignment (Lec 8)
**What to expect:**
- **Vector dimension changes:** The exam may use 3D image vectors instead of 5D, or 4D text vectors instead of 3D.
- **Different Weight Matrices ($W_{\text{vision\_emb}}$ and $W_{\text{txt\_emb}}$):** Be prepared to do matrix multiplication with different dimensions.
- **Space Mapping instead of Alignment:** Instead of projecting both image and text into a shared space, you might be asked to project the image directly into the text space using a single matrix $W_{\text{img}}$ (as discussed in Lec 8).
- **Similarity Check:** You may be asked to compute the dot product (Cosine Similarity without normalization) to verify which image matches which text.

### Tracing Question 2: NeRF Volumetric Rendering (Lec 9)
**What to expect:**
- **Different $\Delta t$ (Step sizes):** The sheet used $\Delta t_1=1, \Delta t_2=2, \Delta t_3=1$. The exam might use uniform steps (e.g., all $\Delta t=0.5$).
- **Different Densities ($\sigma_i$) and Colors ($c_i$):** Values will likely change. Remember the opacity formula $\alpha_i = 1 - e^{-\sigma_i \Delta t_i}$.
- **Transmittance Calculation ($T_i$):** You might be asked to compute transmittance up to a specific step. $T_i = \prod_{j=1}^{i-1} (1 - \alpha_j)$.
- **Calculating Final Composited Color:** $C(r) = \sum T_i \alpha_i c_i$. Practice calculating this carefully with different values.

### Tracing Question 3: Temperature Scaling (Lec 10)
**What to expect:**
- **Different Logits:** You might be given a different set of initial logits (e.g., negative values, which are valid in softmax).
- **Different Temperatures ($T$):** Remember the limits:
  - $T \to 0$: Probability collapses to the max logit (Greedy/Deterministic).
  - $T = 1$: Standard Softmax.
  - $T \to \infty$: Uniform distribution (Chaotic/Random).
- **Calculating Exact Probabilities:** You will need to calculate $e^{z_i/T}$ and divide by the sum. The threshold might change.

### Coding Question 1: Prompt Similarity on Image Regions
**What to expect:**
- **Horizontal instead of Vertical Splits:** You might need to change `cx = w // 3` to `cy = h // 3` and adjust the crop boxes accordingly.
- **Different target pattern:** The target pattern was `[0, 1, 0]`. It might change to `[1, 0, 0]` or `[1, 1, 0]`.
- **Combination lengths:** Instead of combinations of 2 or 3, it might ask for subsets.
- **Normalization:** The normalization formula might be different.

### Coding Question 2: NeRF Missing Sigma Prediction
**What to expect:**
- **Different Search Space:** `np.linspace(0.01, 5.0, 500)` might change to a different range.
- **Different Ground Truth ($C_{gt}$):** The target pixel value will be different.
- **Missing Color instead of Sigma:** You might have to predict a missing $c_i$ instead of a missing $\sigma_i$.

### Coding Question 3: Softmax Temperature Search
**What to expect:**
- **Threshold Change:** Instead of `dist < 0.00001`, the threshold might be relaxed or tightened.
- **Finding Deterministic T:** Instead of finding the temperature that makes outputs equally likely ($T \to \infty$), the question might ask to find the temperature that makes the probability of the top word exceed 99% ($T \to 0$).

---

## 2. MCQ Preparation (Lec 04 to Lec 11)

### Lecture 4 & 6: Vector Databases (FAISS) & Image as Media
1. **What is the primary function of FAISS?**
   - a) To train language models
   - b) To perform efficient similarity search and clustering of dense vectors **(Correct)**
   - c) To extract textual features from images
   - d) To perform optical character recognition

2. **In FAISS, what does `normalize_L2` achieve before indexing?**
   - a) It converts the index to a tree structure.
   - b) It makes Inner Product (IP) search equivalent to Cosine Similarity. **(Correct)**
   - c) It reduces the dimensionality of the vectors.
   - d) It speeds up the CPU.

3. **When using BLIP for image search, why is `torch.no_grad()` used during embedding extraction?**
   - a) To enable backpropagation
   - b) To reduce memory consumption and speed up inference **(Correct)**
   - c) To improve the accuracy of the embeddings
   - d) To allow the model to learn new features

### Lecture 7: ColPali & Vision-Language Models (SigLIP)
4. **Why is traditional Text-Only RAG problematic for processing complex PDFs?**
   - a) It uses too much GPU memory.
   - b) It relies on OCR, which loses visual semantics like layouts, fonts, and charts. **(Correct)**
   - c) It can only process one page at a time.
   - d) It requires a sigmoid loss function.

5. **What is the key mathematical innovation of SigLIP over standard CLIP?**
   - a) SigLIP uses L2 normalization instead of Cosine similarity.
   - b) SigLIP replaces the global softmax loss with a pairwise Sigmoid binary cross-entropy loss, allowing efficient scaling. **(Correct)**
   - c) SigLIP uses ResNet instead of Vision Transformers.
   - d) SigLIP removes the text encoder entirely.

6. **How does ColPali solve the "Global Pooling" problem for document retrieval?**
   - a) By using OCR before embedding.
   - b) By utilizing a Late Interaction operator (MaxSim) that preserves all patch-level embeddings. **(Correct)**
   - c) By reducing the patch size to 2x2 pixels.
   - d) By applying a softmax temperature of 0.

### Lecture 8: Space Alignment vs. Space Mapping
7. **What characterizes the "Space Alignment" (Joint Embedding) paradigm?**
   - a) One modality is projected directly into the pre-existing space of the other.
   - b) Both representation spaces are projected into a brand new, shared coordinate system. **(Correct)**
   - c) Images and text are concatenated before passing into a Transformer.
   - d) Only the text space is updated.

8. **Which of the following models is an example of Space Mapping (Asymmetric Cross-Modal Projection)?**
   - a) CLIP
   - b) SigLIP
   - c) Multimodal LLMs mapping visual tokens to LLM vocabulary **(Correct)**
   - d) GloVe

### Lecture 9: Neural Radiance Fields (NeRF)
9. **In the NeRF architecture, why must volume density ($\sigma$) depend ONLY on spatial position ($x$) and NOT on viewing direction ($d$)?**
   - a) To reduce computation time.
   - b) To ensure solid shapes do not warp or disappear when viewed from different angles. **(Correct)**
   - c) Because lighting changes the density of objects.
   - d) To allow the model to represent specular highlights.

10. **In volumetric rendering, what does Accumulated Transmittance ($T_i$) represent?**
    - a) The amount of light emitted by the camera.
    - b) The probability that light travels from the near plane to the current point without hitting any particles. **(Correct)**
    - c) The color of the object at point $t_i$.
    - d) The loss function of the Neural Network.

### Lecture 10: Temperature & Hallucinations
11. **In simulated annealing and LLM generation, what happens as Temperature ($T$) approaches 0?**
    - a) The model generates completely random noise.
    - b) The probability distribution becomes uniform across all tokens.
    - c) The probability mass collapses entirely onto the single most likely token (Greedy/Deterministic). **(Correct)**
    - d) The model hallucinates more frequently.

12. **Which of the following is a known cause of Retrieval-Induced Hallucinations?**
    - a) High guidance scale ($s > 15$).
    - b) Setting the temperature exactly to 1.0.
    - c) Embedding distortion and nearest-neighbor limits retrieving semantically plausible but factually incorrect vectors. **(Correct)**
    - d) Using Sigmoid loss instead of Softmax.

### Lecture 11: Stable Diffusion & Latent Generative Models
13. **What is the primary advantage of Latent-Space Diffusion (like Stable Diffusion) over Pixel-Space Diffusion?**
    - a) Latent diffusion requires no text encoder.
    - b) It operates on a compressed latent space, making inference dramatically faster and less memory-intensive. **(Correct)**
    - c) It eliminates the need for a Variational Autoencoder (VAE).
    - d) It is deterministic and does not use random noise.

14. **What is the purpose of the Variational Autoencoder (VAE) in Stable Diffusion?**
    - a) To inject noise into the image.
    - b) To map textual prompts into token embeddings.
    - c) To compress the high-resolution pixel image into a lower-dimensional latent representation and decode it back. **(Correct)**
    - d) To perform cross-attention between text and image.

15. **In Classifier-Free Guidance (CFG), what happens when the guidance scale ($s$) is set very high (e.g., $s \ge 15.0$)?**
    - a) The model completely ignores the text prompt.
    - b) The image becomes perfectly photorealistic with high diversity.
    - c) The model over-saturates, forcing strict prompt adherence at the cost of color clipping and artifacts. **(Correct)**
    - d) The latent space dimension is increased.
