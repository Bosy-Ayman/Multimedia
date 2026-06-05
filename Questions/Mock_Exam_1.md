# DSAI-413 Multimedia Processing — Mock Exam 1

> **Total Marks: 40** | **Duration: 2 Hours** | **Coverage: Lectures 4–11**

---

## Section A: Multiple Choice Questions (8 Marks)

*Choose the single best answer for each question. Each question is worth **1 mark**.*

---

**Q1.** What does calling `faiss.normalize_L2(vectors)` on your database vectors before adding them to an `IndexFlatIP` index effectively achieve?

- (A) It converts inner product search into L2 (Euclidean) distance search
- **(B) It makes inner product search equivalent to cosine similarity search** ✓
- (C) It reduces the dimensionality of the vectors via PCA
- (D) It converts the index from flat to IVF for faster retrieval

---

**Q2.** In ColPali, the **MaxSim** operator computes the relevance score between a query $Q$ and a document $D$ as:

$$S(Q, D) = \sum_{i=1}^{N_q} \max_{j \in [1, N_d]} \; q_i^\top d_j$$

What does this operator do?

- (A) For each document patch, find the best-matching query token, then sum
- **(B) For each query token, find the best-matching document patch, then sum** ✓
- (C) Compute pairwise cosine similarities and return the global maximum
- (D) Average all query-document dot products across all token-patch pairs

---

**Q3.** In Simulated Annealing, a worse solution (with $\Delta E > 0$) is accepted with probability $P = e^{-\Delta E / T}$. What happens to this acceptance probability as $T \to \infty$?

- (A) $P \to 0$ — worse solutions are never accepted
- **(B) $P \to 1$ — almost all moves are accepted (random walk)** ✓
- (C) $P \to 0.5$ — each move is a coin flip
- (D) $P$ oscillates and is undefined

---

**Q4.** In Stable Diffusion, the **VAE decoder** is responsible for:

- (A) Encoding text prompts into CLIP embeddings
- (B) Adding noise to the latent representation during the forward process
- **(C) Mapping the denoised latent representation back to pixel space** ✓
- (D) Computing the cross-attention between image and text features

---

**Q5.** In Neural Radiance Fields (NeRF), the MLP models a 5D function $F(\mathbf{x}, \mathbf{d}) \to (\mathbf{c}, \sigma)$. Which statement is correct about the density $\sigma$?

- (A) $\sigma$ depends on both the 3D position $\mathbf{x}$ and the viewing direction $\mathbf{d}$
- (B) $\sigma$ depends only on the viewing direction $\mathbf{d}$
- **(C) $\sigma$ depends only on the 3D position $\mathbf{x}$ (not on $\mathbf{d}$)** ✓
- (D) $\sigma$ is a fixed constant learned once during training

---

**Q6.** In the forward diffusion process, the cumulative noise schedule is defined as $\bar{\alpha}_t = \prod_{s=0}^{t-1} \alpha_s$. If $\alpha_0 = 0.9$ and $\alpha_1 = 0.8$, what is $\bar{\alpha}_2$?

- (A) $0.85$
- **(B) $0.72$** ✓
- (C) $1.70$
- (D) $0.17$

---

**Q7.** In Classifier-Free Guidance (CFG), the guided noise prediction is:

$$\hat{\epsilon} = \epsilon_\theta(x_t, \varnothing) + s \cdot \left[\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \varnothing)\right]$$

What happens when the guidance scale $s = 1$?

- (A) The model generates completely random images ignoring the prompt
- (B) The text influence is doubled for more stylized outputs
- **(C) The output equals the standard conditional prediction $\epsilon_\theta(x_t, c)$ — no extra guidance** ✓
- (D) The unconditional and conditional predictions cancel out, producing a blank image

---

**Q8.** What is a key difference between **SigLIP** and the original **CLIP** contrastive loss?

- (A) SigLIP uses a triplet loss while CLIP uses InfoNCE
- (B) SigLIP requires larger batch sizes than CLIP to function properly
- **(C) SigLIP uses a pairwise sigmoid loss on each (image, text) pair, removing the need for a softmax over the full batch** ✓
- (D) SigLIP only trains the image encoder while CLIP trains both encoders

---

## Section B: Tracing Questions (20 Marks)

---

### Question 1: NeRF Volumetric Rendering (10 Marks)

A camera ray is cast with origin $\mathbf{o} = (0, 0, 0)$ and direction $\mathbf{d} = (0, 1, 0)$. The ray is sampled at $N = 4$ points with a uniform step size $\delta_t = 0.5$. The NeRF MLP returns the following density and color values (grayscale, single-channel):

| Sample $i$ | Coordinate $\mathbf{r}(t_i)$ | Density $\sigma_i$ | Color $c_i$ |
|:---:|:---:|:---:|:---:|
| 1 | $(0,\; 0.5,\; 0)$ | 1.0 | 0.3 |
| 2 | $(0,\; 1.0,\; 0)$ | 0.5 | 0.6 |
| 3 | $(0,\; 1.5,\; 0)$ | 2.0 | 0.8 |
| 4 | $(0,\; 2.0,\; 0)$ | 0.0 | 1.0 |

The discrete volumetric rendering equation is:

$$C(\mathbf{r}) = \sum_{i=1}^{N} T_i \cdot \alpha_i \cdot c_i$$

where:

$$\alpha_i = 1 - e^{-\sigma_i \cdot \delta_t}, \qquad T_i = \prod_{j=1}^{i-1}(1 - \alpha_j)$$

**Answer the following:**

**(a)** [2 marks] Compute the opacity $\alpha_i$ for all 4 samples.

**(b)** [3 marks] Compute the transmittance $T_i$ for all 4 samples.

**(c)** [3 marks] Compute each sample's weighted color contribution $C_i = T_i \cdot \alpha_i \cdot c_i$.

**(d)** [2 marks] Compute the final composited pixel color $C(\mathbf{r})$. Explain why sample 4 makes zero contribution despite having $c_4 = 1.0$.

---

### Question 2: Forward Diffusion Process (10 Marks)

Given a $2 \times 2$ image tensor and the following diffusion parameters:

$$x_0 = \begin{bmatrix} 2.0 & 1.0 \\ 0.0 & 3.0 \end{bmatrix}$$

**Noise schedule:** $\alpha_0 = 0.85, \quad \alpha_1 = 0.75$

**Sampled noise tensors:**

$$\epsilon_0 = \begin{bmatrix} 0.3 & -0.5 \\ 1.0 & 0.2 \end{bmatrix}, \qquad \epsilon_1 = \begin{bmatrix} -0.4 & 0.6 \\ 0.1 & -0.8 \end{bmatrix}$$

**Iterative forward step formula:**

$$x_{t+1} = \sqrt{\alpha_t} \cdot x_t + \sqrt{1 - \alpha_t} \cdot \epsilon_t$$

**Answer the following:**

**(a)** [3 marks] Compute $\sqrt{\alpha_0}$ and $\sqrt{1 - \alpha_0}$. Then compute $x_1$ element-by-element.

**(b)** [3 marks] Compute $\sqrt{\alpha_1}$ and $\sqrt{1 - \alpha_1}$. Then compute $x_2$ element-by-element.

**(c)** [2 marks] Compute the cumulative product $\bar{\alpha}_2 = \alpha_0 \cdot \alpha_1$. Also compute $\sqrt{\bar{\alpha}_2}$ and $\sqrt{1 - \bar{\alpha}_2}$.

**(d)** [2 marks] Explain what happens to $x_T$ as $T \to \infty$. Why does $\sqrt{\bar{\alpha}_T} \to 0$, and what does this imply about the signal-to-noise ratio?

---

## Section C: Coding Questions (12 Marks)

---

### Question 3: Temperature-Scaled Softmax (6 Marks)

Given logits $\mathbf{z} = [3.0, \; 1.0, \; 0.5]$, the temperature-scaled softmax is defined as:

$$P(y_i) = \frac{e^{z_i / T}}{\sum_{j} e^{z_j / T}}$$

**Write a Python program that:**

1. **(2 marks)** Implements a function `scaled_softmax(logits, T)` that takes a list of logits and temperature $T$, and returns the probability distribution as a list.

2. **(2 marks)** Computes and prints the probability distribution for $T = 0.5$, $T = 1.0$, and $T = 2.0$.

3. **(2 marks)** Searches for the temperature $T$ (from $0.10$ to $5.00$ in steps of $0.01$) at which the **maximum probability first drops below 0.50**. Print the value of $T$ and the corresponding distribution.

---

### Question 4: ColPali MaxSim Score (6 Marks)

You are given a query with 3 token embeddings and a document page with 4 patch embeddings (all 2D for simplicity):

**Query tokens:**
$$Q = \begin{bmatrix} 0.9 & 0.1 \\ 0.2 & 0.8 \\ 0.5 & 0.5 \end{bmatrix}$$

**Document patches:**
$$D = \begin{bmatrix} 0.8 & 0.3 \\ 0.1 & 0.9 \\ 0.6 & 0.4 \\ 0.3 & 0.7 \end{bmatrix}$$

The MaxSim relevance score is:

$$S(Q, D) = \sum_{i=1}^{N_q} \max_{j \in [1, N_d]} \; q_i^\top d_j$$

**Write a Python program that:**

1. **(3 marks)** Computes the MaxSim score $S(Q, D)$ using the formula above (using dot products, not cosine similarity).

2. **(3 marks)** For each query token, prints the dot product with every document patch, the maximum similarity value, and the index of the best-matching patch.

---

---

# MODEL ANSWERS / SOLUTIONS

---

## Section A: MCQ Answers

| Q | Answer | Key Reasoning |
|---|--------|---------------|
| 1 | **(B)** | After L2-normalizing, $\|v\| = 1$, so $\text{IP}(u,v) = \cos(u,v)$ |
| 2 | **(B)** | MaxSim iterates over query tokens, finding the best patch for each |
| 3 | **(B)** | $\lim_{T\to\infty} e^{-\Delta E/T} = e^0 = 1$; all moves accepted |
| 4 | **(C)** | The decoder maps from latent $z$ back to pixel-space image |
| 5 | **(C)** | Density is view-independent (same geometry from any angle); color depends on $\mathbf{d}$ |
| 6 | **(B)** | $\bar{\alpha}_2 = 0.9 \times 0.8 = 0.72$ |
| 7 | **(C)** | $s=1 \Rightarrow \hat{\epsilon} = \epsilon_\varnothing + 1 \cdot (\epsilon_c - \epsilon_\varnothing) = \epsilon_c$ |
| 8 | **(C)** | SigLIP applies binary sigmoid loss per pair; no softmax denominator over batch |

---

## Section B: Tracing Solutions

---

### Solution 1: NeRF Volumetric Rendering

**Reference values:** $e^{-0.50} = 0.6065, \quad e^{-0.25} = 0.7788, \quad e^{-1.00} = 0.3679, \quad e^{0} = 1.0000$

#### (a) Opacities — $\alpha_i = 1 - e^{-\sigma_i \cdot \delta_t}$ (2 marks)

$$\alpha_1 = 1 - e^{-1.0 \times 0.5} = 1 - e^{-0.50} = 1 - 0.6065 = \boxed{0.3935}$$

$$\alpha_2 = 1 - e^{-0.5 \times 0.5} = 1 - e^{-0.25} = 1 - 0.7788 = \boxed{0.2212}$$

$$\alpha_3 = 1 - e^{-2.0 \times 0.5} = 1 - e^{-1.00} = 1 - 0.3679 = \boxed{0.6321}$$

$$\alpha_4 = 1 - e^{-0.0 \times 0.5} = 1 - e^{0} = 1 - 1 = \boxed{0.0000}$$

#### (b) Transmittances — $T_i = \prod_{j=1}^{i-1}(1 - \alpha_j)$ (3 marks)

$$T_1 = 1.0000 \quad \text{(no preceding samples)}$$

$$T_2 = (1 - \alpha_1) = 1 - 0.3935 = \boxed{0.6065}$$

$$T_3 = (1 - \alpha_1)(1 - \alpha_2) = 0.6065 \times 0.7788 = \boxed{0.4724}$$

$$T_4 = (1 - \alpha_1)(1 - \alpha_2)(1 - \alpha_3) = 0.4724 \times (1 - 0.6321) = 0.4724 \times 0.3679 = \boxed{0.1738}$$

#### (c) Weighted contributions — $C_i = T_i \cdot \alpha_i \cdot c_i$ (3 marks)

$$C_1 = 1.0000 \times 0.3935 \times 0.3 = \boxed{0.1181}$$

$$C_2 = 0.6065 \times 0.2212 \times 0.6 = \boxed{0.0805}$$

$$C_3 = 0.4724 \times 0.6321 \times 0.8 = \boxed{0.2389}$$

$$C_4 = 0.1738 \times 0.0000 \times 1.0 = \boxed{0.0000}$$

#### (d) Final pixel color (2 marks)

$$C(\mathbf{r}) = C_1 + C_2 + C_3 + C_4 = 0.1181 + 0.0805 + 0.2389 + 0.0000 = \boxed{0.4375}$$

**Interpretation:** Sample 4 contributes **nothing** because $\sigma_4 = 0$, which means $\alpha_4 = 1 - e^0 = 0$. A density of zero means that point in space is **completely transparent** (empty space). No matter how bright its color ($c_4 = 1.0$), there is no "material" there to emit or absorb light, so it adds nothing to the final pixel.

**Summary Table:**

| $i$ | $\sigma_i$ | $c_i$ | $\alpha_i$ | $T_i$ | $C_i$ |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 1.0 | 0.3 | 0.3935 | 1.0000 | 0.1181 |
| 2 | 0.5 | 0.6 | 0.2212 | 0.6065 | 0.0805 |
| 3 | 2.0 | 0.8 | 0.6321 | 0.4724 | 0.2389 |
| 4 | 0.0 | 1.0 | 0.0000 | 0.1738 | 0.0000 |
| | | | | **Total** | **0.4375** |

---

### Solution 2: Forward Diffusion Process

#### (a) Compute $x_1$ (3 marks)

$$\sqrt{\alpha_0} = \sqrt{0.85} = 0.9220$$

$$\sqrt{1 - \alpha_0} = \sqrt{0.15} = 0.3873$$

Applying $x_1 = \sqrt{\alpha_0} \cdot x_0 + \sqrt{1 - \alpha_0} \cdot \epsilon_0$ element-by-element:

$$x_1[0,0] = 0.9220 \times 2.0 + 0.3873 \times 0.3 = 1.8440 + 0.1162 = \boxed{1.9602}$$

$$x_1[0,1] = 0.9220 \times 1.0 + 0.3873 \times (-0.5) = 0.9220 - 0.1937 = \boxed{0.7284}$$

$$x_1[1,0] = 0.9220 \times 0.0 + 0.3873 \times 1.0 = 0.0000 + 0.3873 = \boxed{0.3873}$$

$$x_1[1,1] = 0.9220 \times 3.0 + 0.3873 \times 0.2 = 2.7660 + 0.0775 = \boxed{2.8435}$$

$$x_1 = \begin{bmatrix} 1.9602 & 0.7284 \\ 0.3873 & 2.8435 \end{bmatrix}$$

#### (b) Compute $x_2$ (3 marks)

$$\sqrt{\alpha_1} = \sqrt{0.75} = 0.8660$$

$$\sqrt{1 - \alpha_1} = \sqrt{0.25} = 0.5000$$

Applying $x_2 = \sqrt{\alpha_1} \cdot x_1 + \sqrt{1 - \alpha_1} \cdot \epsilon_1$ element-by-element:

$$x_2[0,0] = 0.8660 \times 1.9602 + 0.5000 \times (-0.4) = 1.6975 - 0.2000 = \boxed{1.4975}$$

$$x_2[0,1] = 0.8660 \times 0.7284 + 0.5000 \times 0.6 = 0.6308 + 0.3000 = \boxed{0.9308}$$

$$x_2[1,0] = 0.8660 \times 0.3873 + 0.5000 \times 0.1 = 0.3354 + 0.0500 = \boxed{0.3854}$$

$$x_2[1,1] = 0.8660 \times 2.8435 + 0.5000 \times (-0.8) = 2.4625 - 0.4000 = \boxed{2.0625}$$

$$x_2 = \begin{bmatrix} 1.4975 & 0.9308 \\ 0.3854 & 2.0625 \end{bmatrix}$$

#### (c) Cumulative noise schedule (2 marks)

$$\bar{\alpha}_2 = \alpha_0 \times \alpha_1 = 0.85 \times 0.75 = \boxed{0.6375}$$

$$\sqrt{\bar{\alpha}_2} = \sqrt{0.6375} = \boxed{0.7984}$$

$$\sqrt{1 - \bar{\alpha}_2} = \sqrt{0.3625} = \boxed{0.6021}$$

> **Note:** Using the closed-form formula $x_2 = \sqrt{\bar{\alpha}_2}\, x_0 + \sqrt{1 - \bar{\alpha}_2}\, \bar{\epsilon}$ would produce $x_2$ in a single step (with an appropriately combined noise $\bar{\epsilon}$), which is computationally more efficient than iterating.

#### (d) Behavior as $T \to \infty$ (2 marks)

As $T \to \infty$:

- $\bar{\alpha}_T = \prod_{s=0}^{T-1} \alpha_s$. Since each $\alpha_s < 1$, this product of values less than 1 **shrinks toward zero** as more terms are multiplied.
- Therefore $\sqrt{\bar{\alpha}_T} \to 0$, which means the **signal coefficient** vanishes.
- Simultaneously $\sqrt{1 - \bar{\alpha}_T} \to 1$, which means the **noise coefficient** approaches 1.
- The result is that $x_T \approx \bar{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$ — the original image information is **completely destroyed**, and $x_T$ becomes **pure Gaussian noise**.
- This is by design: diffusion models learn to reverse this process, recovering $x_0$ from noise.

---

## Section C: Coding Solutions

---

### Solution 3: Temperature-Scaled Softmax (6 marks)

```python
import math

# Part 1: scaled_softmax function (2 marks)
def scaled_softmax(logits, T):
    """
    Computes temperature-scaled softmax probabilities.
    
    Args:
        logits: list of raw logit values
        T: temperature parameter (T > 0)
    Returns:
        list of probabilities summing to 1.0
    """
    scaled = [z / T for z in logits]
    # Subtract max for numerical stability
    max_val = max(scaled)
    exps = [math.exp(s - max_val) for s in scaled]
    total = sum(exps)
    return [e / total for e in exps]


# Part 2: Compute for T = 0.5, 1.0, 2.0 (2 marks)
logits = [3.0, 1.0, 0.5]

for T in [0.5, 1.0, 2.0]:
    probs = scaled_softmax(logits, T)
    print(f"T = {T:.1f}: P = [{probs[0]:.4f}, {probs[1]:.4f}, {probs[2]:.4f}]")
```

**Expected Output for Part 2:**

```
T = 0.5: P = [0.9820, 0.0148, 0.0033]
T = 1.0: P = [0.8214, 0.1112, 0.0674]
T = 2.0: P = [0.6044, 0.2224, 0.1732]
```

> At low $T$, the distribution is **sharp** (peaked at the highest logit).
> At high $T$, the distribution **flattens** toward uniform.

```python
# Part 3: Search for T where max probability < 0.50 (2 marks)
logits = [3.0, 1.0, 0.5]

for T_int in range(10, 501):  # T from 0.10 to 5.00, step 0.01
    T = T_int / 100.0
    probs = scaled_softmax(logits, T)
    max_prob = max(probs)
    if max_prob < 0.50:
        print(f"First T where max prob < 0.50: T = {T:.2f}")
        print(f"  Distribution: [{probs[0]:.4f}, {probs[1]:.4f}, {probs[2]:.4f}]")
        print(f"  Max probability: {max_prob:.4f}")
        break
```

**Expected Output for Part 3:**

```
First T where max prob < 0.50: T = 3.24
  Distribution: [0.4997, 0.2753, 0.2250]
  Max probability: 0.4997
```

**Verification at $T = 3.23$:**

$$\text{scaled logits} = [3/3.23,\; 1/3.23,\; 0.5/3.23] = [0.9288,\; 0.3096,\; 0.1548]$$

$$e^{0.9288} = 2.5321, \quad e^{0.3096} = 1.3628, \quad e^{0.1548} = 1.1675$$

$$\text{sum} = 5.0624, \quad P_{\max} = \frac{2.5321}{5.0624} = 0.5003 \;\; (\geq 0.50)$$

At $T = 3.24$ the max probability first drops just below $0.50$, confirming **$T = 3.24$**.

---

### Solution 4: ColPali MaxSim Score (6 marks)

```python
# Part 1 & 2: MaxSim computation with detailed output (6 marks)

def compute_maxsim(Q, D):
    """
    Compute the MaxSim score between query tokens Q and document patches D.
    
    S(Q, D) = sum over each query token of max dot-product to any patch.
    
    Args:
        Q: list of query token vectors (N_q x dim)
        D: list of document patch vectors (N_d x dim)
    Returns:
        total MaxSim score (float)
    """
    total_score = 0.0

    for i, q in enumerate(Q):
        # Compute dot product of query token i with all document patches
        dot_products = []
        for j, d in enumerate(D):
            dp = sum(q[k] * d[k] for k in range(len(q)))
            dot_products.append(dp)

        # Find the maximum similarity and the best-matching patch index
        max_sim = max(dot_products)
        best_patch = dot_products.index(max_sim)

        # Print detailed information for each query token
        print(f"Query token {i} {q}:")
        for j, dp in enumerate(dot_products):
            marker = " <-- MAX" if j == best_patch else ""
            print(f"  dot(q{i}, d{j}) = {dp:.4f}{marker}")
        print(f"  Best match: patch {best_patch}, similarity = {max_sim:.4f}")
        print()

        total_score += max_sim

    return total_score


# Define query and document embeddings
Q = [[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]]
D = [[0.8, 0.3], [0.1, 0.9], [0.6, 0.4], [0.3, 0.7]]

# Compute and display MaxSim score
score = compute_maxsim(Q, D)
print(f"MaxSim Score S(Q, D) = {score:.4f}")
```

**Expected Output:**

```
Query token 0 [0.9, 0.1]:
  dot(q0, d0) = 0.7500 <-- MAX
  dot(q0, d1) = 0.1800
  dot(q0, d2) = 0.5800
  dot(q0, d3) = 0.3400
  Best match: patch 0, similarity = 0.7500

Query token 1 [0.2, 0.8]:
  dot(q1, d0) = 0.4000
  dot(q1, d1) = 0.7400 <-- MAX
  dot(q1, d2) = 0.4400
  dot(q1, d3) = 0.6200
  Best match: patch 1, similarity = 0.7400

Query token 2 [0.5, 0.5]:
  dot(q2, d0) = 0.5500 <-- MAX
  dot(q2, d1) = 0.5000
  dot(q2, d2) = 0.5000
  dot(q2, d3) = 0.5000
  Best match: patch 0, similarity = 0.5500

MaxSim Score S(Q, D) = 2.0400
```

**Hand-Calculated Verification:**

| | $d_0 = [0.8, 0.3]$ | $d_1 = [0.1, 0.9]$ | $d_2 = [0.6, 0.4]$ | $d_3 = [0.3, 0.7]$ | **Max** |
|:---:|:---:|:---:|:---:|:---:|:---:|
| $q_0 = [0.9, 0.1]$ | $0.72 + 0.03 = 0.75$ | $0.09 + 0.09 = 0.18$ | $0.54 + 0.04 = 0.58$ | $0.27 + 0.07 = 0.34$ | **0.75** (patch 0) |
| $q_1 = [0.2, 0.8]$ | $0.16 + 0.24 = 0.40$ | $0.02 + 0.72 = 0.74$ | $0.12 + 0.32 = 0.44$ | $0.06 + 0.56 = 0.62$ | **0.74** (patch 1) |
| $q_2 = [0.5, 0.5]$ | $0.40 + 0.15 = 0.55$ | $0.05 + 0.45 = 0.50$ | $0.30 + 0.20 = 0.50$ | $0.15 + 0.35 = 0.50$ | **0.55** (patch 0) |

$$S(Q, D) = 0.75 + 0.74 + 0.55 = \boxed{2.04}$$

---

*End of Mock Exam 1 — Good luck!*
