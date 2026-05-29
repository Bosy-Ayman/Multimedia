# Masterclass Study Guide: Temperature Scaling, Simulated Annealing, and Hallucination Mechanics

This study guide bridges the conceptual, architectural, and mathematical gaps found in **Lec_09.pptx** regarding optimization temperatures, LLM sampling kinetics, and retrieval-augmented hallucinations.

## 1. Simulated Annealing: Thermodynamic-Inspired Optimization

### 1.1 The Metallurgy Analogy

In metallurgy, **annealing** is a process where a metal is heated to a high temperature (increasing atomic energy and structural mobility) and then cooled slowly. This slow cooling allows atoms to settle into their lowest energy states, resulting in a highly stable, defect-free crystalline lattice.

If cooled too quickly (quenched), the atoms freeze in highly unstable, high-energy local defects.

### 1.2 Mathematical Formulation

In computer science, **Simulated Annealing** is a probabilistic metaheuristic used to find global minima in complex, non-convex optimization landscapes. It mimics metallurgy to avoid getting trapped in **local minima** (where simple greedy hill-climbing algorithms fail).

```
        Energy / Objective f(x)
          ▲
          │    Starting Configuration
          │       \●
          │        \  Perturb (Hill Climbing)
          │   ──┐   ▼      ▲
          │     └───●      │  Perturb (Hill Climbing)
          │      Local     └───●
          │     Minimum         \
          │                      ▼
          │                       ● Global Minimum
          │
          └────────────────────────────────────────► State Space / Variable X
```

At each iteration, the algorithm slightly perturbs the current state $x_{\text{old}}$ to a neighboring state $x_{\text{new}}$. We calculate the change in "energy" (loss/objective function) as:

$$\Delta E = f(x_{\text{new}}) - f(x_{\text{old}})$$

1. **If** $\Delta E < 0$ **(Improved Solution):** The algorithm always accepts the new state.
    
2. **If** $\Delta E \ge 0$ **(Worse Solution):** The algorithm does _not_ immediately reject it. Instead, it accepts the worse state probabilistically to escape local traps. The probability $P$ of accepting a worse solution is dictated by the **Metropolis-Hastings criterion**:
    

$$P = e^{-\frac{\Delta E}{T}}$$

Where:

- $\Delta E$**:** How much worse the proposed solution is.
    
- $T \in (0, \infty)$**:** The system's current "temperature".
    

### 1.3 Temperature Dynamics and Probability Limits

We can mathematically analyze how the system behaves under different temperature states:

- **High Temperature (**$T \to \infty$**):**
    
    $$\lim_{T \to \infty} P = \lim_{T \to \infty} e^{-\frac{\Delta E}{T}} = e^0 = 1.0$$
    
    At very high temperatures, the system is highly chaotic and accepts almost _every_ proposal—even highly detrimental ones. This allows broad, random **exploration** of the parameter landscape.
    
- **Low Temperature (**$T \to 0$**):**
    
    $$\lim_{T \to 0^+} P = \lim_{T \to 0^+} e^{-\frac{\Delta E}{T}} = e^{-\infty} = 0.0$$
    
    As the system cools, the probability of accepting worse steps drops to zero. The algorithm transitions from global exploration to greedy **exploitation** (pure gradient descent / hill climbing).
    

### 1.4 The Cooling Schedule

The system cools gradually according to a predefined **Cooling Schedule**. The most famous geometric schedule is defined as:

$$T_{\text{new}} = \alpha T_{\text{old}}$$

Where $\alpha \in (0, 1)$ is the decay constant. In practice, $\alpha$ is tuned between $0.80$ and $0.99$. If $\alpha$ is too low, the system cools too fast, freezing into a local minimum. If $\alpha$ is too high, optimization is computationally excessive.

## 2. LLM Temperature: Probability Scaling in Generative Spaces

Language models generate text autoregressively by predicting the next token $y_t$ given a history of context tokens. The model outputs a vector of raw, unnormalized scores called **logits** $\mathbf{z} = [z_1, z_2, \dots, z_V]^\top \in \mathbb{R}^V$, where $V$ is the vocabulary size.

### 2.1 The Temperature-Scaled Softmax Equation

To convert raw logits into a valid probability distribution $P$, we apply a scaled version of the Boltzmann/Softmax distribution. By introducing the temperature parameter $T$, we adjust the sampling entropy:

$$P_i = \frac{e^{z_i / T}}{\sum_{j=1}^{V} e^{z_j / T}}$$

Where:

- $z_i$ is the logit score of the $i$-th vocabulary token.
    
- $T > 0$ is the generative temperature control parameter.
    

```
       UNSCALED LOGITS [z_i]       TEMPERATURE SCALING (1/T)       NORMALIZED PROBABILITY [P_i]
          
          [  10.0 (high) ] ──────►  Low T (T=0.1)  ───────►  [ 0.999 (Nearly Deterministic) ]
          [   5.0 (mid)  ]          High contrast            [ 0.001                        ]
          [   1.0 (low)  ]                                   [ 0.000                        ]
          
          [  10.0 (high) ] ──────►  High T (T=2.0) ───────►  [ 0.450 (High Entropy)         ]
          [   5.0 (mid)  ]          Low contrast             [ 0.350                        ]
          [   1.0 (low)  ]                                   [ 0.200                        ]
```

### 2.2 Mathematical Limits of LLM Temperature

- **Low Temperature (**$T \to 0^+$ **- Greedy / Deterministic Mode):** As $T$ approaches $0$, the difference between the maximum logit and all other logits scales exponentially. Let $z_{\max}$ be the highest logit. The ratio for any other logit $z_i < z_{\max}$ is:
    
    $$\frac{e^{z_i / T}}{e^{z_{\max} / T}} = e^{\frac{z_i - z_{\max}}{T}}$$
    
    Because $(z_i - z_{\max}) < 0$, as $T \to 0^+$, this term approaches $e^{-\infty} = 0$. Thus:
    
    $$\lim_{T \to 0^+} P_i = \begin{cases} 1.0 & \text{if } z_i = z_{\max} \\ 0.0 & \text{if } z_i \neq z_{\max} \end{cases}$$
    
    The probability mass collapses entirely onto the single most likely token. Generation becomes highly repetitive, predictable, and logical, but lacks creativity and can become stuck in repetitive loops.
    
- **Default Temperature (**$T = 1.0$**):** The probability distribution is strictly determined by the native unscaled logits output by the transformer:
    
    $$P_i = \frac{e^{z_i}}{\sum e^{z_j}}$$
- **High Temperature (**$T \to \infty$ **- Uniform / Chaotic Mode):** As the temperature $T$ scales to infinity, the exponents approach zero:
    
    $$\lim_{T \to \infty} \frac{z_i}{T} = 0 \implies e^0 = 1$$
    
    Substituting this back into the softmax function:
    
    $$\lim_{T \to \infty} P_i = \frac{1}{\sum_{j=1}^{V} 1} = \frac{1}{V}$$
    
    The probability distribution becomes completely uniform. Every token in the vocabulary has an equal chance of being selected, regardless of context. The model generates random, nonsensical token streams, leading to structural **hallucinations** and broken grammar.
    

## 3. LLM Hallucinations: Core Paradigms

In Large Language Models, **hallucination** is defined as the generation of text that is structurally fluent and rhetorically convincing, but factually false or ungrounded in any source documents.

### 3.1 Why LLMs Hallucinate (Foundational Constraints)

1. **Next-Token Prediction Objective:** The training loss (cross-entropy) forces the model to maximize statistical likelihood of words, _not factual accuracy_. The model is optimized to sound plausible, not to verify truth.
    
2. **Lack of Real-World Grounding:** LLMs do not have an internal, active validation engine or real-time physical senses. Factual knowledge is represented as static associative weights in a massive neural network.
    
3. **Pattern Overgeneralization:** The model learns abstract syntactic structures and associative rules. For example, if it has seen "Paris is the capital of France" and "Berlin is the capital of Germany," it can overgeneralize the pattern "X is the capital of Y" to fill in gaps probabilistically when prompted with unknown configurations.
    

## 4. Tracing Context Injection & Retrieval Failures (Vector Math Walkthrough)

To prevent hallucinations, engineers use **Retrieval-Augmented Generation (RAG)**. RAG retrieves relevant documents from a vector database and inserts them into the prompt context.

However, mathematical errors in embedding spaces and retrieval indices can actually _induce_ hallucinations. Let us trace this mathematically using the data from **Lec_09.pptx**.

### 4.1 Defining the Vector Space Database

We have a document store consisting of three document sentences, represented as normalized 3D vectors:

- $D_1$ **("Paris is the capital of France.")**
    
    $$\vec{v}_{D1} = \begin{bmatrix} 0.92 & 0.81 & 0.05 \end{bmatrix}^{\top}$$
- $D_2$ **("Berlin is the capital of Germany.")**
    
    $$\vec{v}_{D2} = \begin{bmatrix} 0.89 & 0.77 & 0.06 \end{bmatrix}^{\top}$$
- $D_3$ **("France won the 2018 FIFA World Cup.")**
    
    $$\vec{v}_{D3} = \begin{bmatrix} 0.40 & 0.15 & 0.95 \end{bmatrix}^{\top}$$

### 4.2 Case Study 1: Successful Retrieval

A user submits the query: **"What is the capital of France?"**

#### Step 1: Query Vectorization

The query is mapped to embedding vector $Q_1$:

$$\vec{v}_{Q1} = \begin{bmatrix} 0.91 & 0.80 & 0.02 \end{bmatrix}^{\top}$$

#### Step 2: Compute Query Magnitudes

To calculate Cosine Similarity:

$$\text{sim}(Q, D) = \frac{Q \cdot D}{\|Q\| \|D\|}$$

Let's compute the $L_2$ norm (magnitude) of the query and document vectors:

- $\|Q_1\| = \sqrt{0.91^2 + 0.80^2 + 0.02^2} = \sqrt{0.8281 + 0.6400 + 0.0004} = \sqrt{1.4685} \approx \mathbf{1.2118}$
    
- $\|D_1\| = \sqrt{0.92^2 + 0.81^2 + 0.05^2} = \sqrt{0.8464 + 0.6561 + 0.0025} = \sqrt{1.5050} \approx \mathbf{1.2268}$
    
- $\|D_2\| = \sqrt{0.89^2 + 0.77^2 + 0.06^2} = \sqrt{0.7921 + 0.5929 + 0.0036} = \sqrt{1.3886} \approx \mathbf{1.1784}$
    
- $\|D_3\| = \sqrt{0.40^2 + 0.15^2 + 0.95^2} = \sqrt{0.1600 + 0.0225 + 0.9025} = \sqrt{1.0850} \approx \mathbf{1.0416}$
    

#### Step 3: Compute Similarity Metrics

We calculate the cosine similarity between the query and each document:

1. **Similarity to** $D_1$**:**
    
    $$Q_1 \cdot D_1 = (0.91 \times 0.92) + (0.80 \times 0.81) + (0.02 \times 0.05) = 0.8372 + 0.6480 + 0.0010 = 1.4862$$$$\text{sim}(Q_1, D_1) = \frac{1.4862}{1.2118 \times 1.2268} = \frac{1.4862}{1.4866} = \mathbf{0.9997} \approx \mathbf{0.999}$$
2. **Similarity to** $D_2$**:**
    
    $$Q_1 \cdot D_2 = (0.91 \times 0.89) + (0.80 \times 0.77) + (0.02 \times 0.06) = 0.8099 + 0.6160 + 0.0012 = 1.4271$$$$\text{sim}(Q_1, D_2) = \frac{1.4271}{1.2118 \times 1.1784} = \frac{1.4271}{1.4280} = \mathbf{0.9993} \approx \mathbf{0.995}$$
3. **Similarity to** $D_3$**:**
    
    $$Q_1 \cdot D_3 = (0.91 \times 0.40) + (0.80 \times 0.15) + (0.02 \times 0.95) = 0.3640 + 0.1200 + 0.0190 = 0.5030$$$$\text{sim}(Q_1, D_3) = \frac{0.5030}{1.2118 \times 1.0416} = \frac{0.5030}{1.2622} = \mathbf{0.3985} \approx \mathbf{0.31}$$

#### Step 4: Context Injection & LLM Response

Since the Top-2 returned documents are $D_1$ ($\text{sim} = 0.999$) and $D_2$ ($\text{sim} = 0.995$), they are injected into the LLM system prompt:

```
CONTEXT:
- Paris is the capital of France.
- Berlin is the capital of Germany.

QUESTION:
What is the capital of France?

RESPONSE:
Paris
```

The query succeeds because the required factual knowledge was fully present in the retrieved vectors.

### 4.3 Case Study 2: Failed Retrieval (Retrieval-Induced Hallucination)

Now, a user submits a more complex query: **"What city hosted the 2018 World Cup final?"**

#### Step 1: Query Vectorization

The embedding model represents this query as $Q_2$:

$$\vec{v}_{Q2} = \begin{bmatrix} 0.58 & 0.44 & 0.61 \end{bmatrix}^{\top}$$

Magnitude:

$$\|Q_2\| = \sqrt{0.58^2 + 0.44^2 + 0.61^2} = \sqrt{0.3364 + 0.1936 + 0.3721} = \sqrt{0.9021} \approx \mathbf{0.9498}$$

#### Step 2: Compute Similarities

We calculate similarities to identify the closest matches in our database:

1. **Similarity to** $D_1$ **("Paris is the capital of France."):**
    
    $$Q_2 \cdot D_1 = (0.58 \times 0.92) + (0.44 \times 0.81) + (0.61 \times 0.05) = 0.5336 + 0.3564 + 0.0305 = 0.9205$$$$\text{sim}(Q_2, D_1) = \frac{0.9205}{0.9498 \times 1.2268} = \frac{0.9205}{1.1652} \approx \mathbf{0.789} \quad (\approx \mathbf{0.78})$$
2. **Similarity to** $D_2$ **("Berlin is the capital of Germany."):**
    
    $$Q_2 \cdot D_2 = (0.58 \times 0.89) + (0.44 \times 0.77) + (0.61 \times 0.06) = 0.5162 + 0.3388 + 0.0366 = 0.8916$$$$\text{sim}(Q_2, D_2) = \frac{0.8916}{0.9498 \times 1.1784} = \frac{0.8916}{1.1192} \approx \mathbf{0.796} \quad (\approx \mathbf{0.75})$$
3. **Similarity to** $D_3$ **("France won the 2018 FIFA World Cup."):**
    
    $$Q_2 \cdot D_3 = (0.58 \times 0.40) + (0.44 \times 0.15) + (0.61 \times 0.95) = 0.2320 + 0.0660 + 0.5795 = 0.8775$$$$\text{sim}(Q_2, D_3) = \frac{0.8775}{0.9498 \times 1.0416} = \frac{0.8775}{0.9893} \approx \mathbf{0.887} \quad (\approx \mathbf{0.69})$$

#### Step 3: Top Matches Returned

The database ranks the similarity scores and returns the top matches:

- **Rank 1:** $D_3$ (Similarity $\approx 0.69$) — _Correctly identified a relationship to the World Cup._
    
- **Rank 2:** $D_1$ (Similarity $\approx 0.78$) — _Falsely identified because the query mentions "France" and the embedding vectors have heavy structural overlap in spatial directions._
    

The database misses the true answer because **the fact was never stored in the database**:

$$D_{\text{true}} = \text{"The final was hosted in Moscow."} \notin \text{Database}$$

#### Step 4: Injecting Incomplete Context

Because $D_1$ and $D_3$ are the closest vector matches, they are injected into the prompt:

```
CONTEXT:
- Paris is the capital of France.
- France won the 2018 FIFA World Cup.

QUESTION:
What city hosted the 2018 World Cup final?
```

#### Step 5: Probabilistic Completion Error

The LLM reads this injected context. It must satisfy two opposing priorities:

1. **Context Constraint:** Ground the response _only_ using the provided facts.
    
2. **Next-Token Likelihood:** Complete the sentence with high semantic likelihood.
    

The context provides the words **"Paris"**, **"France"**, and **"World Cup"**. The LLM uses these words to construct a statistically likely, fluent completion:

$$\text{"Paris hosted the 2018 World Cup final."} \quad \text{(Factual Hallucination!)}$$

_(Note: The actual match was held at Luzhniki Stadium in Moscow)._

### 4.4 Taxonomy of Retrieval Failure Modes

The above trace demonstrates the four main failure modes of modern RAG systems:

```
  1. EMBEDDING DISTORTION        2. NEAREST-NEIGHBOR LIMITS     3. INCOMPLETE DATABASE         4. PROBABILISTIC FILLER
  High-dimensional words are     Vectors match based on syntax  The true fact is missing       The LLM bridges gaps by
  compressed, warping context    instead of exact semantic      from the vector database       guessing highly plausible
  relationships.                 meanings.                      entirely.                      syntactic relationships.
```