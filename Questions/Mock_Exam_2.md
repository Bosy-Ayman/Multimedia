# DSAI-413 — Multimedia Processing — Mock Exam 2

**Final Exam — Spring 2026**
**Total Marks: 40 | Duration: 2 Hours**

> **Instructions:**
> - Answer ALL questions.
> - Show all working for tracing and coding questions.
> - Calculators are permitted; no Internet-connected devices.

---

## Section A: Multiple Choice Questions (8 Marks)

*Choose the single best answer. Each question is worth **1 mark**.*

---

**Q1.** In FAISS, what is the key difference between `IndexFlatL2` and `IndexFlatIP`?

- (A) `IndexFlatL2` uses cosine similarity; `IndexFlatIP` uses Euclidean distance.
- (B) `IndexFlatL2` uses Euclidean (L2) distance; `IndexFlatIP` uses Inner Product similarity.
- (C) `IndexFlatL2` requires normalized vectors; `IndexFlatIP` does not.
- (D) Both use the same metric but differ in memory layout.

---

**Q2.** In ColPali, the **MaxSim** operator computes the relevance score between a query $Q$ and a document page $P$. Which of the following correctly describes MaxSim?

- (A) For each document patch, find the max similarity to any query token, then average.
- (B) For each query token $q_i$, compute $\max_{p_j \in P} \text{sim}(q_i, p_j)$, then sum over all query tokens.
- (C) Compute a single global similarity between the mean query embedding and the mean patch embedding.
- (D) For each query token, find the minimum similarity to any patch, then sum.

---

**Q3.** In Simulated Annealing with a geometric cooling schedule $T_{k+1} = \alpha \cdot T_k$, what does the parameter $\alpha$ control?

- (A) The probability of accepting a worse solution at any temperature.
- (B) The initial temperature $T_0$.
- (C) The rate at which the temperature decays, where $\alpha \in (0, 1)$.
- (D) The number of iterations before the algorithm terminates.

---

**Q4.** What is the input dimensionality of a Neural Radiance Field (NeRF)?

- (A) 3D — only spatial coordinates $(x, y, z)$.
- (B) 4D — spatial coordinates $(x, y, z)$ plus a time step $t$.
- (C) 5D — spatial coordinates $(x, y, z)$ plus viewing direction $(\theta, \phi)$.
- (D) 6D — spatial coordinates $(x, y, z)$ plus RGB color $(r, g, b)$.

---

**Q5.** In Stable Diffusion, what does the VAE Encoder output?

- (A) A full-resolution denoised image of shape $H \times W \times 3$.
- (B) A latent representation $z$ of shape $\frac{H}{8} \times \frac{W}{8} \times 4$.
- (C) A noise prediction $\epsilon_\theta$ of the same shape as the input image.
- (D) A text embedding vector of dimension 768.

---

**Q6.** What is the key difference between **Space Alignment** and **Space Mapping** in multimodal learning?

- (A) Alignment trains only one encoder; Mapping trains both.
- (B) Alignment projects both modalities into a **new shared space**; Mapping projects one modality **into the other's existing space**.
- (C) Alignment requires paired data; Mapping does not.
- (D) There is no difference; the terms are interchangeable.

---

**Q7.** In the forward diffusion process defined by $x_{t+1} = \sqrt{\alpha_t}\, x_t + \sqrt{1 - \alpha_t}\, \epsilon$, where $\epsilon \sim \mathcal{N}(0, I)$, what does $\alpha_t$ control?

- (A) The learning rate of the denoising network.
- (B) The ratio of original signal preserved versus noise added at step $t$.
- (C) The variance of the prior distribution.
- (D) The number of diffusion steps.

---

**Q8.** What is **Classifier-Free Guidance (CFG)** in diffusion models?

- (A) Training a separate classifier network to guide the diffusion process.
- (B) Using two forward passes — one **conditional** ($\epsilon_\theta(x_t, c)$) and one **unconditional** ($\epsilon_\theta(x_t, \varnothing)$) — then computing $\hat{\epsilon} = \epsilon_\theta(x_t, \varnothing) + s \cdot [\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \varnothing)]$ with guidance scale $s$.
- (C) Removing the text encoder entirely and relying on image-only generation.
- (D) Using CLIP scores to rank generated images after sampling.

---

## Section B: Tracing Questions (20 Marks)

### Question 1 — Space Alignment & Space Mapping (10 Marks)

You are building a multimodal retrieval system. You have two images and two text descriptions. Each modality has its own encoder, and both are projected into a shared alignment space.

**Raw Inputs:**

$$
x_{\text{im1}} = \begin{bmatrix} 1 \\ 1 \\ 0 \\ 1 \end{bmatrix} \in \mathbb{R}^4 \quad (\text{Bird image})
\qquad
x_{\text{im2}} = \begin{bmatrix} 0 \\ 1 \\ 1 \\ 0 \end{bmatrix} \in \mathbb{R}^4 \quad (\text{Fish image})
$$

$$
x_{t1} = \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix} \in \mathbb{R}^3 \quad (\text{"Bird"})
\qquad
x_{t2} = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix} \in \mathbb{R}^3 \quad (\text{"Fish"})
$$

**Encoder Weights:**

$$
W_{\text{vision}} = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 1 & 0 & 0 \end{bmatrix} \in \mathbb{R}^{3 \times 4}
\qquad
W_{\text{text}} = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 0 \\ 1 & 1 & 0 \end{bmatrix} \in \mathbb{R}^{3 \times 3}
$$

**Alignment Projection Weights (to shared 2D space):**

$$
W_{\text{img\_proj}} = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \end{bmatrix} \in \mathbb{R}^{2 \times 3}
\qquad
W_{\text{txt\_proj}} = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 0 \end{bmatrix} \in \mathbb{R}^{2 \times 3}
$$

---

**(a)** **(3 marks)** Compute the 3D encoder embeddings: $v_{\text{im1}}$, $v_{\text{im2}}$, $v_{t1}$, $v_{t2}$.

**(b)** **(3 marks)** Compute the 2D aligned projections: $z_{\text{im1}}$, $z_{\text{im2}}$, $z_{t1}$, $z_{t2}$.

**(c)** **(2 marks)** Compute the dot-product similarity for:
- The **matching** pair: $z_{\text{im1}} \cdot z_{t1}$ (Bird image ↔ "Bird" text)
- The **non-matching** pair: $z_{\text{im1}} \cdot z_{t2}$ (Bird image ↔ "Fish" text)

**(d)** **(2 marks)** Does the alignment correctly separate the matching pair from the non-matching pair for the Bird image? What about for the Fish image ($z_{\text{im2}} \cdot z_{t2}$ vs. $z_{\text{im2}} \cdot z_{t1}$)? Explain.

---

### Question 2 — RAG Retrieval, Cosine Similarity & Hallucination (10 Marks)

A Retrieval-Augmented Generation (RAG) system has the following pre-computed document embeddings:

| Document | Content | Embedding Vector |
|----------|---------|-----------------|
| $D_1$ | "The Nile is the longest river in Africa." | $[0.85,\ 0.70,\ 0.10]^T$ |
| $D_2$ | "Cairo is the capital of Egypt." | $[0.80,\ 0.75,\ 0.15]^T$ |
| $D_3$ | "Egypt hosted the Africa Cup of Nations in 2019." | $[0.35,\ 0.20,\ 0.90]^T$ |

A user submits the query:

> **"Which city hosted AFCON 2019?"**

The query is encoded as: $Q = [0.50,\ 0.40,\ 0.70]^T$

---

**(a)** **(2 marks)** Compute the L2 norms: $\|Q\|$, $\|D_1\|$, $\|D_2\|$, $\|D_3\|$.

**(b)** **(4 marks)** Compute the cosine similarity between the query and each document:
$$\text{sim}(Q, D_i) = \frac{Q \cdot D_i}{\|Q\| \cdot \|D_i\|}$$

**(c)** **(2 marks)** Identify the **Top-2** retrieved documents. Write the augmented prompt that would be sent to the LLM.

**(d)** **(2 marks)** The true answer to the query is "Cairo." However, explain why the LLM might arrive at "Cairo" for the **wrong reason**, leading to a hallucination risk.

---

## Section C: Coding Questions (12 Marks)

### Question 3 — NeRF: Finding a Missing Color via Volumetric Rendering (6 Marks)

A NeRF model samples 3 points along a camera ray. The volume densities and delta distances are known, but the color at the second sample is missing.

**Given:**

| Sample $i$ | $\sigma_i$ | $c_i$ | $\delta_i$ |
|:-----------:|:----------:|:-----:|:-----------:|
| 1 | 1.0 | 0.3 | 1.0 |
| 2 | 0.5 | **???** | 1.0 |
| 3 | 2.0 | 0.9 | 1.0 |

**Ground-truth pixel color:** $C_{\text{gt}} = 0.45$

**NeRF Volumetric Rendering Equations:**

$$\alpha_i = 1 - \exp(-\sigma_i \cdot \delta_i)$$

$$T_i = \prod_{j=1}^{i-1}(1 - \alpha_j) \qquad (T_1 = 1)$$

$$C(\mathbf{r}) = \sum_{i=1}^{N} T_i \cdot \alpha_i \cdot c_i$$

**Task:** Write a Python program that:

1. Searches for the missing $c_2$ over `np.linspace(0.0, 1.0, 1000)`.
2. For each candidate $c_2$, computes the full volumetric rendering $C(\mathbf{r})$.
3. Finds the $c_2$ that minimizes $|C(\mathbf{r}) - C_{\text{gt}}|$.
4. Prints the best $c_2$ and the corresponding rendered color.

---

### Question 4 — FAISS: Image Search with Inner Product Similarity (6 Marks)

Write a complete Python function `faiss_image_search(embeddings, query, k=3)` that:

1. Takes a list of image embedding vectors (NumPy arrays) and a single query vector.
2. **Normalizes** all vectors using `faiss.normalize_L2` (for cosine similarity via inner product).
3. Creates a FAISS `IndexFlatIP` index.
4. Adds the embedding vectors to the index.
5. Searches for the **top-k** most similar vectors (default $k=3$).
6. Returns the similarity scores and indices.

---

---

# MODEL ANSWERS

---

## Section A — MCQ Answers

| Q | Answer | Explanation |
|---|--------|-------------|
| 1 | **(B)** | `IndexFlatL2` computes $L_2$ (Euclidean) distance; `IndexFlatIP` computes Inner Product. After L2-normalizing vectors, Inner Product equals cosine similarity. |
| 2 | **(B)** | MaxSim: for each query token $q_i$, find $\max_{p_j} \text{sim}(q_i, p_j)$, then sum: $\text{Score} = \sum_i \max_j \text{sim}(q_i, p_j)$. |
| 3 | **(C)** | $\alpha \in (0,1)$ controls how fast the temperature decays. Smaller $\alpha$ → faster cooling → less exploration. |
| 4 | **(C)** | NeRF input is 5D: 3D position $(x,y,z)$ + 2D viewing direction $(\theta, \phi)$. |
| 5 | **(B)** | The VAE encoder compresses the image from pixel space to a latent of shape $\frac{H}{8} \times \frac{W}{8} \times 4$. |
| 6 | **(B)** | Alignment: both modalities projected to a **new** shared space. Mapping: one modality projected **into** the other's existing space. |
| 7 | **(B)** | $\alpha_t$ controls signal-vs-noise ratio. When $\alpha_t \to 1$, most signal is kept; when $\alpha_t \to 0$, mostly noise. |
| 8 | **(B)** | CFG uses two passes (conditional + unconditional) and extrapolates: $\hat{\epsilon} = \epsilon_\varnothing + s(\epsilon_c - \epsilon_\varnothing)$. Scale $s > 1$ amplifies conditioning. |

---

## Section B — Tracing Answers

### Question 1 — Space Alignment & Space Mapping

#### Part (a): Compute 3D Encoder Embeddings (3 marks)

**$v_{\text{im1}} = W_{\text{vision}} \cdot x_{\text{im1}}$:**

$$
v_{\text{im1}} = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 1 & 0 & 0 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \\ 0 \\ 1 \end{bmatrix}
$$

$$
\text{Row 1: } (1)(1) + (0)(1) + (1)(0) + (0)(1) = 1
$$
$$
\text{Row 2: } (0)(1) + (1)(1) + (0)(0) + (1)(1) = 2
$$
$$
\text{Row 3: } (1)(1) + (1)(1) + (0)(0) + (0)(1) = 2
$$

$$
\boxed{v_{\text{im1}} = \begin{bmatrix} 1 \\ 2 \\ 2 \end{bmatrix}}
$$

---

**$v_{\text{im2}} = W_{\text{vision}} \cdot x_{\text{im2}}$:**

$$
v_{\text{im2}} = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 1 & 0 & 0 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \\ 1 \\ 0 \end{bmatrix}
$$

$$
\text{Row 1: } (1)(0) + (0)(1) + (1)(1) + (0)(0) = 1
$$
$$
\text{Row 2: } (0)(0) + (1)(1) + (0)(1) + (1)(0) = 1
$$
$$
\text{Row 3: } (1)(0) + (1)(1) + (0)(1) + (0)(0) = 1
$$

$$
\boxed{v_{\text{im2}} = \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix}}
$$

---

**$v_{t1} = W_{\text{text}} \cdot x_{t1}$:**

$$
v_{t1} = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 0 \\ 1 & 1 & 0 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}
$$

$$
\text{Row 1: } (1)(1) + (0)(0) + (1)(1) = 2
$$
$$
\text{Row 2: } (0)(1) + (1)(0) + (0)(1) = 0
$$
$$
\text{Row 3: } (1)(1) + (1)(0) + (0)(1) = 1
$$

$$
\boxed{v_{t1} = \begin{bmatrix} 2 \\ 0 \\ 1 \end{bmatrix}}
$$

---

**$v_{t2} = W_{\text{text}} \cdot x_{t2}$:**

$$
v_{t2} = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 0 \\ 1 & 1 & 0 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}
$$

$$
\text{Row 1: } (1)(0) + (0)(1) + (1)(0) = 0
$$
$$
\text{Row 2: } (0)(0) + (1)(1) + (0)(0) = 1
$$
$$
\text{Row 3: } (1)(0) + (1)(1) + (0)(0) = 1
$$

$$
\boxed{v_{t2} = \begin{bmatrix} 0 \\ 1 \\ 1 \end{bmatrix}}
$$

---

#### Part (b): Compute 2D Aligned Projections (3 marks)

**$z_{\text{im1}} = W_{\text{img\_proj}} \cdot v_{\text{im1}}$:**

$$
z_{\text{im1}} = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \end{bmatrix} \begin{bmatrix} 1 \\ 2 \\ 2 \end{bmatrix}
$$

$$
\text{Row 1: } (1)(1) + (0)(2) + (1)(2) = 3
$$
$$
\text{Row 2: } (0)(1) + (1)(2) + (1)(2) = 4
$$

$$
\boxed{z_{\text{im1}} = \begin{bmatrix} 3 \\ 4 \end{bmatrix}}
$$

---

**$z_{\text{im2}} = W_{\text{img\_proj}} \cdot v_{\text{im2}}$:**

$$
z_{\text{im2}} = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix}
$$

$$
\text{Row 1: } (1)(1) + (0)(1) + (1)(1) = 2
$$
$$
\text{Row 2: } (0)(1) + (1)(1) + (1)(1) = 2
$$

$$
\boxed{z_{\text{im2}} = \begin{bmatrix} 2 \\ 2 \end{bmatrix}}
$$

---

**$z_{t1} = W_{\text{txt\_proj}} \cdot v_{t1}$:**

$$
z_{t1} = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 0 \end{bmatrix} \begin{bmatrix} 2 \\ 0 \\ 1 \end{bmatrix}
$$

$$
\text{Row 1: } (1)(2) + (0)(0) + (1)(1) = 3
$$
$$
\text{Row 2: } (0)(2) + (1)(0) + (0)(1) = 0
$$

$$
\boxed{z_{t1} = \begin{bmatrix} 3 \\ 0 \end{bmatrix}}
$$

---

**$z_{t2} = W_{\text{txt\_proj}} \cdot v_{t2}$:**

$$
z_{t2} = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 0 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \\ 1 \end{bmatrix}
$$

$$
\text{Row 1: } (1)(0) + (0)(1) + (1)(1) = 1
$$
$$
\text{Row 2: } (0)(0) + (1)(1) + (0)(1) = 1
$$

$$
\boxed{z_{t2} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}}
$$

---

#### Part (c): Dot-Product Similarities (2 marks)

**Matching pair — Bird image ↔ "Bird" text:**

$$
z_{\text{im1}} \cdot z_{t1} = (3)(3) + (4)(0) = 9 + 0 = \boxed{9}
$$

**Non-matching pair — Bird image ↔ "Fish" text:**

$$
z_{\text{im1}} \cdot z_{t2} = (3)(1) + (4)(1) = 3 + 4 = \boxed{7}
$$

---

#### Part (d): Analysis (2 marks)

**For the Bird image:**

$$
\text{sim}(\text{Bird img}, \text{"Bird"}) = 9 \quad > \quad \text{sim}(\text{Bird img}, \text{"Fish"}) = 7 \quad \checkmark
$$

✅ The alignment **correctly** ranks the matching pair higher for the Bird image.

**For the Fish image:**

$$
z_{\text{im2}} \cdot z_{t2} = (2)(1) + (2)(1) = 4 \quad (\text{match})
$$
$$
z_{\text{im2}} \cdot z_{t1} = (2)(3) + (2)(0) = 6 \quad (\text{mismatch})
$$

$$
\text{sim}(\text{Fish img}, \text{"Fish"}) = 4 \quad < \quad \text{sim}(\text{Fish img}, \text{"Bird"}) = 6 \quad \boldsymbol{\times}
$$

❌ The alignment **fails** for the Fish image — the Fish image is more similar to "Bird" than to "Fish" in the projected space. This indicates the projection weights are **not well-trained** and the model needs further training (e.g., with contrastive loss) to learn better alignment projections.

---

### Question 2 — RAG Retrieval, Cosine Similarity & Hallucination

#### Part (a): L2 Norms (2 marks)

$$
\|Q\| = \sqrt{0.50^2 + 0.40^2 + 0.70^2} = \sqrt{0.25 + 0.16 + 0.49} = \sqrt{0.90} \approx \boxed{0.9487}
$$

$$
\|D_1\| = \sqrt{0.85^2 + 0.70^2 + 0.10^2} = \sqrt{0.7225 + 0.49 + 0.01} = \sqrt{1.2225} \approx \boxed{1.1057}
$$

$$
\|D_2\| = \sqrt{0.80^2 + 0.75^2 + 0.15^2} = \sqrt{0.64 + 0.5625 + 0.0225} = \sqrt{1.225} \approx \boxed{1.1068}
$$

$$
\|D_3\| = \sqrt{0.35^2 + 0.20^2 + 0.90^2} = \sqrt{0.1225 + 0.04 + 0.81} = \sqrt{0.9725} \approx \boxed{0.9862}
$$

---

#### Part (b): Cosine Similarities (4 marks)

**$\text{sim}(Q, D_1)$:**

$$
Q \cdot D_1 = (0.50)(0.85) + (0.40)(0.70) + (0.70)(0.10) = 0.425 + 0.280 + 0.070 = 0.775
$$

$$
\text{sim}(Q, D_1) = \frac{0.775}{0.9487 \times 1.1057} = \frac{0.775}{1.0490} \approx \boxed{0.7388}
$$

---

**$\text{sim}(Q, D_2)$:**

$$
Q \cdot D_2 = (0.50)(0.80) + (0.40)(0.75) + (0.70)(0.15) = 0.400 + 0.300 + 0.105 = 0.805
$$

$$
\text{sim}(Q, D_2) = \frac{0.805}{0.9487 \times 1.1068} = \frac{0.805}{1.0500} \approx \boxed{0.7667}
$$

---

**$\text{sim}(Q, D_3)$:**

$$
Q \cdot D_3 = (0.50)(0.35) + (0.40)(0.20) + (0.70)(0.90) = 0.175 + 0.080 + 0.630 = 0.885
$$

$$
\text{sim}(Q, D_3) = \frac{0.885}{0.9487 \times 0.9862} = \frac{0.885}{0.9356} \approx \boxed{0.9459}
$$

---

#### Part (c): Top-2 Retrieved Documents & Augmented Prompt (2 marks)

**Ranking:**

| Rank | Document | Cosine Similarity |
|------|----------|-------------------|
| 1 | $D_3$ — "Egypt hosted the Africa Cup of Nations in 2019." | 0.9459 |
| 2 | $D_2$ — "Cairo is the capital of Egypt." | 0.7667 |
| 3 | $D_1$ — "The Nile is the longest river in Africa." | 0.7388 |

**Top-2:** $D_3$ and $D_2$.

**Augmented Prompt:**

```
Context:
[1] Egypt hosted the Africa Cup of Nations in 2019.
[2] Cairo is the capital of Egypt.

Question: Which city hosted AFCON 2019?
Answer:
```

---

#### Part (d): Hallucination Risk (2 marks)

The correct answer is **Cairo**, but the RAG system creates a hallucination risk:

- **$D_3$** (highest similarity) mentions that *Egypt* hosted AFCON 2019 but does **not** mention the specific city.
- **$D_2$** (second highest) states that *Cairo is the capital of Egypt* but says **nothing** about AFCON.

The LLM is likely to **compose** information across both documents: it reads that Egypt hosted AFCON 2019 from $D_3$, sees that Cairo is the capital from $D_2$, and **infers** "Cairo hosted AFCON 2019." While the final answer happens to be correct, the reasoning is **grounded on irrelevant context** (capital ≠ host city). This is a form of **coincidental hallucination** — the model arrives at the right answer for the wrong reason. If the host city were different from the capital (e.g., Alexandria), the LLM would have confidently given the wrong answer. This illustrates why RAG does not eliminate hallucinations — it can provide misleading context that the LLM over-relies on.

---

## Section C — Coding Answers

### Question 3 — NeRF: Finding the Missing Color (6 marks)

#### Analytical Solution (for verification):

$$
\alpha_1 = 1 - e^{-1.0 \times 1.0} = 1 - e^{-1} \approx 1 - 0.3679 = 0.6321
$$
$$
\alpha_2 = 1 - e^{-0.5 \times 1.0} = 1 - e^{-0.5} \approx 1 - 0.6065 = 0.3935
$$
$$
\alpha_3 = 1 - e^{-2.0 \times 1.0} = 1 - e^{-2} \approx 1 - 0.1353 = 0.8647
$$

$$
T_1 = 1, \quad T_2 = (1 - \alpha_1) = 0.3679, \quad T_3 = (1 - \alpha_1)(1 - \alpha_2) = 0.3679 \times 0.6065 = 0.2231
$$

$$
C(\mathbf{r}) = T_1 \alpha_1 c_1 + T_2 \alpha_2 c_2 + T_3 \alpha_3 c_3
$$
$$
= (1)(0.6321)(0.3) + (0.3679)(0.3935)(c_2) + (0.2231)(0.8647)(0.9)
$$
$$
= 0.18963 + 0.14478\, c_2 + 0.17355
$$
$$
= 0.36318 + 0.14478\, c_2
$$

Setting $C(\mathbf{r}) = 0.45$:

$$
c_2 = \frac{0.45 - 0.36318}{0.14478} = \frac{0.08682}{0.14478} \approx \boxed{0.5997}
$$

#### Python Code:

```python
import numpy as np

# ---- Given data ----
sigmas   = np.array([1.0, 0.5, 2.0])
colors   = np.array([0.3, None, 0.9])  # c_2 is unknown
delta_t  = np.array([1.0, 1.0, 1.0])
C_gt     = 0.45

# ---- Precompute alpha values ----
alphas = 1 - np.exp(-sigmas * delta_t)

# ---- Search for the best c_2 ----
candidates = np.linspace(0.0, 1.0, 1000)
best_c2    = None
best_error = float('inf')
best_C     = None

for c2 in candidates:
    # Set the color array with current candidate
    c = np.array([0.3, c2, 0.9])

    # Compute transmittance T_i = product of (1 - alpha_j) for j < i
    T = np.ones(3)
    for i in range(1, 3):
        T[i] = T[i - 1] * (1 - alphas[i - 1])

    # Compute rendered color: C(r) = sum(T_i * alpha_i * c_i)
    C_rendered = np.sum(T * alphas * c)

    # Track the best candidate
    error = abs(C_rendered - C_gt)
    if error < best_error:
        best_error = error
        best_c2    = c2
        best_C     = C_rendered

print(f"Best c_2:         {best_c2:.4f}")
print(f"Rendered color:   {best_C:.4f}")
print(f"Target color:     {C_gt:.4f}")
print(f"Absolute error:   {best_error:.6f}")
```

**Expected Output:**
```
Best c_2:         0.5996
Rendered color:   0.4500
Target color:     0.4500
Absolute error:   0.000013
```

---

### Question 4 — FAISS: Image Search with Inner Product Similarity (6 marks)

```python
import numpy as np
import faiss

def faiss_image_search(embeddings, query, k=3):
    """
    Search for the top-k most similar images using FAISS with cosine similarity.

    Args:
        embeddings: list of numpy arrays, each of shape (d,) — the image embeddings.
        query:      numpy array of shape (d,) — the query embedding.
        k:          int — number of nearest neighbors to retrieve.

    Returns:
        scores:  numpy array of shape (1, k) — similarity scores (cosine).
        indices: numpy array of shape (1, k) — indices of the top-k matches.
    """
    # Step 1: Stack embeddings into a matrix of shape (n, d)
    embedding_matrix = np.array(embeddings).astype('float32')
    query_vector     = np.array([query]).astype('float32')  # shape (1, d)

    # Step 2: Normalize all vectors (in-place) so that IP == cosine similarity
    faiss.normalize_L2(embedding_matrix)
    faiss.normalize_L2(query_vector)

    # Step 3: Create a FAISS Inner Product index
    d = embedding_matrix.shape[1]  # dimensionality
    index = faiss.IndexFlatIP(d)

    # Step 4: Add the image embeddings to the index
    index.add(embedding_matrix)

    # Step 5: Search for top-k nearest neighbors
    scores, indices = index.search(query_vector, k)

    # Step 6: Return results
    return scores, indices


# ---- Example Usage ----
if __name__ == "__main__":
    # Create 5 dummy 128-dimensional image embeddings
    np.random.seed(42)
    embeddings = [np.random.randn(128).astype('float32') for _ in range(5)]
    query      = np.random.randn(128).astype('float32')

    scores, indices = faiss_image_search(embeddings, query, k=3)

    print("Top-3 Indices:", indices)
    print("Top-3 Scores: ", scores)
```

**Key Points (Grading Rubric):**

| Criterion | Marks |
|-----------|-------|
| Correct stacking & `float32` casting | 1 |
| `faiss.normalize_L2()` on both embeddings and query | 1 |
| `faiss.IndexFlatIP(d)` with correct dimension | 1 |
| `index.add(embedding_matrix)` | 1 |
| `index.search(query_vector, k)` | 1 |
| Returning scores and indices correctly | 1 |

---

*End of Mock Exam 2*
