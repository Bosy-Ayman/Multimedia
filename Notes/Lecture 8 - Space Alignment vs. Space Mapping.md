# Masterclass Study Guide: Space Alignment vs. Space Mapping 

## 1. Paradigm Overview: Space Alignment vs. Space Mapping

In multimodal machine learning, bridging distinct modalities (such as vision and language) requires establishing a mathematical relationship between high-dimensional vector spaces. **Lec_08.pptx** outlines the two fundamental paradigms used to achieve this cross-modal relationship:

```
PARADIGM 1: SPACE ALIGNMENT (Joint Embedding)
Both representation spaces are projected into a BRAND NEW, shared coordinate system.

  Image Space (d_img) ───[ W_img ]───► Aligned Shared Space (d_shared) ◄───[ W_txt ]─── Text Space (d_txt)


PARADIGM 2: SPACE MAPPING (Cross-Modal Projection)
One representation space is projected directly into the pre-existing coordinate space of the other.

  Image Space (d_img) ─────────────────[ W_img ]─────────────────► Text Space (d_txt)
```

### Paradigm Comparison Matrix:

|Metric|Space Alignment (Joint Embedding)|Space Mapping (Cross-Modal Projection)|
|---|---|---|
|**Projection Approach**|Symmetric (both spaces move)|Asymmetric (one space moves to the other)|
|**Learned Weights**|Two matrices: $W_{\text{img}}$ and $W_{\text{txt}}$|One matrix: $W$ (usually mapping image to text)|
|**Target Space**|A newly designed latent space $d_{\text{shared}}$|The pre-existing native space of one modality|
|**Key Use Case**|CLIP, SigLIP, global contrastive pre-training|Multimodal LLMs (e.g., mapping visual tokens to LLM vocabulary)|

## 2. Space Alignment (Symmetric Joint Embedding)

### The Objective

Given an image representation $\vec{v}_{\text{img}} \in \mathbb{R}^{d_{\text{img}}}$ and a text representation $\vec{v}_{\text{txt}} \in \mathbb{R}^{d_{\text{txt}}}$, we learn two linear projection matrices:

- $W_{\text{img}} \in \mathbb{R}^{d_{\text{shared}} \times d_{\text{img}}}$
    
- $W_{\text{txt}} \in \mathbb{R}^{d_{\text{shared}} \times d_{\text{txt}}}$
    

These matrices project both representations into a shared space of dimension $d_{\text{shared}}$:

$$\vec{z}_{\text{img}} = W_{\text{img}} \vec{v}_{\text{img}}, \quad \vec{z}_{\text{txt}} = W_{\text{txt}} \vec{v}_{\text{txt}}$$

We train both projection layers (and the underlying encoders) simultaneously. The learning objective is to optimize the parameters such that the similarity between a matching image-text pair $(\vec{z}_{\text{img}}, \vec{z}_{\text{txt}}^{+})$ is maximized, while the similarity to non-matching pairs $(\vec{z}_{\text{img}}, \vec{z}_{\text{txt}}^{-})$ is minimized:

$$\text{similarity}(\vec{z}_{\text{img}}, \vec{z}_{\text{txt}}^{+}) \uparrow \quad \text{and} \quad \text{similarity}(\vec{z}_{\text{img}}, \vec{z}_{\text{txt}}^{-}) \downarrow$$

## 3. Mathematical Walkthrough: Space Alignment Toy Example

Let us reconstruct and mathematically compute the exact joint embedding toy example from **Lec_08.pptx**.

### Step 1: Inputs & Encoders

We begin with raw, low-level binary features for two images and two texts:

- **Raw Inputs:**
    
    - Image 1 (Cat): $\vec{x}_{\text{im1}} = \begin{bmatrix} 1 & 0 & 1 & 0 & 1 \end{bmatrix}^{\top} \in \mathbb{R}^5$
        
    - Image 2 (Dog): $\vec{x}_{\text{im2}} = \begin{bmatrix} 0 & 1 & 0 & 1 & 0 \end{bmatrix}^{\top} \in \mathbb{R}^5$
        
    - Text 1 ("Cat"): $\vec{x}_{\text{t1}} = \begin{bmatrix} 1 & 0 & 1 \end{bmatrix}^{\top} \in \mathbb{R}^3$
        
    - Text 2 ("Dog"): $\vec{x}_{\text{t2}} = \begin{bmatrix} 0 & 1 & 0 \end{bmatrix}^{\top} \in \mathbb{R}^3$
        
- **Embedding Weights (Modality-Specific Encoders):**
    
    $$W_{\text{vision\_emb}} = \begin{bmatrix} 1 & 0 & 1 & 0 & 1 \\ 0 & 1 & 0 & 1 & 0 \\ 1 & 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 1 & 0 \end{bmatrix} \in \mathbb{R}^{4 \times 5}, \quad W_{\text{txt\_emb}} = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 0 \\ 1 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix} \in \mathbb{R}^{4 \times 3}$$

### Step 2: Compute Initial 4D Embeddings

Now, we compute the unaligned embeddings $\vec{v} = W \cdot \vec{x}$:

- **Image 1:**
    
    $$\vec{v}_{\text{im1}} = W_{\text{vision\_emb}} \cdot \vec{x}_{\text{im1}} = \begin{bmatrix} 1 & 0 & 1 & 0 & 1 \\ 0 & 1 & 0 & 1 & 0 \\ 1 & 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 1 & 0 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \\ 1 \\ 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 1(1) + 0(0) + 1(1) + 0(0) + 1(1) \\ 0(1) + 1(0) + 0(1) + 1(0) + 0(1) \\ 1(1) + 1(0) + 0(1) + 0(0) + 0(1) \\ 0(1) + 0(0) + 1(1) + 1(0) + 0(1) \end{bmatrix} = \begin{bmatrix} 3 \\ 0 \\ 1 \\ 1 \end{bmatrix} \in \mathbb{R}^4$$
- **Text 1:**
    
    $$\vec{v}_{\text{txt1}} = W_{\text{txt\_emb}} \cdot \vec{x}_{\text{t1}} = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 0 \\ 1 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 1(1) + 0(0) + 1(1) \\ 0(1) + 1(0) + 0(1) \\ 1(1) + 1(0) + 0(1) \\ 0(1) + 0(0) + 1(1) \end{bmatrix} = \begin{bmatrix} 2 \\ 0 \\ 1 \\ 1 \end{bmatrix} \in \mathbb{R}^4$$
- **Image 2:**
    
    $$\vec{v}_{\text{im2}} = W_{\text{vision\_emb}} \cdot \vec{x}_{\text{im2}} = \begin{bmatrix} 1 & 0 & 1 & 0 & 1 \\ 0 & 1 & 0 & 1 & 0 \\ 1 & 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 1 & 0 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \\ 0 \\ 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 0 \\ 2 \\ 1 \\ 1 \end{bmatrix} \in \mathbb{R}^4$$
- **Text 2:**
    
    $$\vec{v}_{\text{txt2}} = W_{\text{txt\_emb}} \cdot \vec{x}_{\text{t2}} = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 0 \\ 1 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 0 \\ 1 \\ 1 \\ 0 \end{bmatrix} \in \mathbb{R}^4$$

### Step 3: Projection into the 2D Latent Aligned Space

To align the modalities, we apply the learned projection weights $W_{\text{img}} \in \mathbb{R}^{2 \times 4}$ and $W_{\text{txt}} \in \mathbb{R}^{2 \times 4}$:

$$W_{\text{img}} = \begin{bmatrix} 1 & 0 & 1 & 1 \\ 0 & 1 & 0 & 1 \end{bmatrix}, \quad W_{\text{txt}} = \begin{bmatrix} 1 & 0 & 1 & 1 \\ 0 & 1 & 1 & 0 \end{bmatrix}$$

We calculate the 2D projected coordinates $\vec{z} = W \cdot \vec{v}$:

- **Projected Image 1 (**$\vec{z}_{\text{im1}}$**):**
    
    $$\vec{z}_{\text{im1}} = W_{\text{img}} \vec{v}_{\text{im1}} = \begin{bmatrix} 1 & 0 & 1 & 1 \\ 0 & 1 & 0 & 1 \end{bmatrix} \begin{bmatrix} 3 \\ 0 \\ 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 1(3) + 0(0) + 1(1) + 1(1) \\ 0(3) + 1(0) + 0(1) + 1(1) \end{bmatrix} = \begin{bmatrix} 5 \\ 1 \end{bmatrix} \in \mathbb{R}^2$$
    
    > ⚠️ **Note on Slide Typo:** **Lec_08.pptx** lists the projected vector $\vec{z}_{\text{im1}}$ as $\begin{bmatrix} 5 & 2 \end{bmatrix}^{\top}$. This is an arithmetic typo in the lecture slides. The correct matrix multiplication yields $\vec{z}_{\text{im1}} = \begin{bmatrix} 5 & 1 \end{bmatrix}^{\top}$, as proven above.
    
- **Projected Text 1 (**$\vec{z}_{\text{txt1}}$**):**
    
    $$\vec{z}_{\text{txt1}} = W_{\text{txt}} \vec{v}_{\text{txt1}} = \begin{bmatrix} 1 & 0 & 1 & 1 \\ 0 & 1 & 1 & 0 \end{bmatrix} \begin{bmatrix} 2 \\ 0 \\ 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 1(2) + 0(0) + 1(1) + 1(1) \\ 0(2) + 1(0) + 1(1) + 0(1) \end{bmatrix} = \begin{bmatrix} 4 \\ 1 \end{bmatrix} \in \mathbb{R}^2$$
- **Projected Image 2 (**$\vec{z}_{\text{im2}}$**):**
    
    $$\vec{z}_{\text{im2}} = W_{\text{img}} \vec{v}_{\text{im2}} = \begin{bmatrix} 1 & 0 & 1 & 1 \\ 0 & 1 & 0 & 1 \end{bmatrix} \begin{bmatrix} 0 \\ 2 \\ 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 1(0) + 0(2) + 1(1) + 1(1) \\ 0(0) + 1(2) + 0(1) + 1(1) \end{bmatrix} = \begin{bmatrix} 2 \\ 3 \end{bmatrix} \in \mathbb{R}^2$$
- **Projected Text 2 (**$\vec{z}_{\text{txt2}}$**):**
    
    $$\vec{z}_{\text{txt2}} = W_{\text{txt}} \vec{v}_{\text{txt2}} = \begin{bmatrix} 1 & 0 & 1 & 1 \\ 0 & 1 & 1 & 0 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \\ 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 1(0) + 0(1) + 1(1) + 1(0) \\ 0(0) + 1(1) + 1(1) + 0(0) \end{bmatrix} = \begin{bmatrix} 1 \\ 2 \end{bmatrix} \in \mathbb{R}^2$$

## 4. Space Mapping (Asymmetric Cross-Modal Projection)

### The Objective

Instead of inventing an entirely new shared coordinate system, **Space Mapping** learns a single asymmetric transformation matrix $W_{\text{img}} \in \mathbb{R}^{d_{\text{txt}} \times d_{\text{img}}}$ designed to project visual representations $\vec{v}_{\text{img}}$ directly into the native text coordinate space $\vec{v}_{\text{txt}}$:

$$\vec{v}'_{\text{img}} = W_{\text{img}} \vec{v}_{\text{img}}$$

This is crucial for architectures like Multimodal LLMs where visual input must behave exactly like native text tokens in the model's pre-trained vocabulary space. The objective is:

$$\text{similarity}(W_{\text{img}} \vec{v}_{\text{img}}, \vec{v}_{\text{txt}}) \uparrow$$

## 5. Mathematical Walkthrough: Space Mapping Toy Example

Using the same initial visual embeddings $\vec{v}_{\text{im}}$ and native target text embeddings $\vec{v}_{\text{txt}}$ from our first example, we trace the Space Mapping calculations.

### Step 1: Explicit Projection Matrix

We define an asymmetric mapping matrix $W_{\text{img}} \in \mathbb{R}^{4 \times 4}$:

$$W_{\text{img}} = \begin{bmatrix} 0.67 & 0 & 0 & 0 \\ 0 & 0.5 & 0 & -0.5 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 \end{bmatrix}$$

### Step 2: Mapping Visual Representations directly to Text Coordinates

We project our image vectors into text space via $\vec{v}'_{\text{im}} = W_{\text{img}} \vec{v}_{\text{im}}$:

- **Mapped Image 1 (**$\vec{v}'_{\text{im1}}$**):**
    
    $$\vec{v}'_{\text{im1}} = W_{\text{img}} \vec{v}_{\text{im1}} = \begin{bmatrix} 0.67 & 0 & 0 & 0 \\ 0 & 0.5 & 0 & -0.5 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 \end{bmatrix} \begin{bmatrix} 3 \\ 0 \\ 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 0.67(3) \\ 0.5(0) - 0.5(1) \\ 1(1) \\ 0 \end{bmatrix} = \begin{bmatrix} 2.01 \\ -0.5 \\ 1 \\ 0 \end{bmatrix}$$
- **Mapped Image 2 (**$\vec{v}'_{\text{im2}}$**):**
    
    $$\vec{v}'_{\text{im2}} = W_{\text{img}} \vec{v}_{\text{im2}} = \begin{bmatrix} 0.67 & 0 & 0 & 0 \\ 0 & 0.5 & 0 & -0.5 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 \end{bmatrix} \begin{bmatrix} 0 \\ 2 \\ 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 0.67(0) \\ 0.5(2) - 0.5(1) \\ 1(1) \\ 0 \end{bmatrix} = \begin{bmatrix} 0 \\ 0.5 \\ 1 \\ 0 \end{bmatrix}$$

### Step 3: Similarity Evaluation (Inner Dot Products)

To assess how well this projection has mapped our image vectors to our target text space, we evaluate the similarity scores.

_(Note: Although labeled as `cosinSim` in **Lec_08.pptx**, the numerical outputs calculated in the slides are raw inner dot products of the mapped vectors, which we compute below)._

- **Matching Pair 1 (**$\vec{v}'_{\text{im1}} \cdot \vec{v}_{\text{txt1}}$ **- Cat Image & "Cat" Text):**
    
    $$\vec{v}'_{\text{im1}} \cdot \vec{v}_{\text{txt1}} = \begin{bmatrix} 2.01 \\ -0.5 \\ 1 \\ 0 \end{bmatrix} \cdot \begin{bmatrix} 2 \\ 0 \\ 1 \\ 1 \end{bmatrix} = (2.01 \times 2) + (-0.5 \times 0) + (1 \times 1) + (0 \times 1) = 4.02 + 0 + 1 + 0 = 5.02 \quad (\text{High positive match!})$$
- **Mismatching Pair 1 (**$\vec{v}'_{\text{im1}} \cdot \vec{v}_{\text{txt2}}$ **- Cat Image & "Dog" Text):**
    
    $$\vec{v}'_{\text{im1}} \cdot \vec{v}_{\text{txt2}} = \begin{bmatrix} 2.01 \\ -0.5 \\ 1 \\ 0 \end{bmatrix} \cdot \begin{bmatrix} 0 \\ 1 \\ 1 \\ 0 \end{bmatrix} = (2.01 \times 0) + (-0.5 \times 1) + (1 \times 1) + (0 \times 0) = 0 - 0.5 + 1 + 0 = 0.5 \quad (\text{Low mismatch score!})$$
- **Matching Pair 2 (**$\vec{v}'_{\text{im2}} \cdot \vec{v}_{\text{txt2}}$ **- Dog Image & "Dog" Text):**
    
    $$\vec{v}'_{\text{im2}} \cdot \vec{v}_{\text{txt2}} = \begin{bmatrix} 0 \\ 0.5 \\ 1 \\ 0 \end{bmatrix} \cdot \begin{bmatrix} 0 \\ 1 \\ 1 \\ 0 \end{bmatrix} = (0 \times 0) + (0.5 \times 1) + (1 \times 1) + (0 \times 0) = 0 + 0.5 + 1 + 0 = 1.5 \quad (\text{High positive match!})$$
- **Mismatching Pair 2 (**$\vec{v}'_{\text{im2}} \cdot \vec{v}_{\text{txt1}}$ **- Dog Image & "Cat" Text):**
    
    $$\vec{v}'_{\text{im2}} \cdot \vec{v}_{\text{txt1}} = \begin{bmatrix} 0 \\ 0.5 \\ 1 \\ 0 \end{bmatrix} \cdot \begin{bmatrix} 2 \\ 0 \\ 1 \\ 1 \end{bmatrix} = (0 \times 2) + (0.5 \times 0) + (1 \times 1) + (0 \times 1) = 0 + 0 + 1 + 0 = 1.0 \quad (\text{Low mismatch score!})$$

### Core Takeaway

By designing the transformation matrix $W_{\text{img}}$ correctly, the model maps visual inputs such that **matching pairs** yield substantially higher similarity scores ($5.02$ and $1.5$) than **mismatching pairs** ($0.5$ and $1.0$).

During actual training, backpropagation continuously adjusts the individual parameters inside $W_{\text{img}}$ to widen this contrast gap, ensuring robust cross-modal retrieval.