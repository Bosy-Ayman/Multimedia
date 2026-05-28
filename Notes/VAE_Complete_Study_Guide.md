# Variational Autoencoder (VAE) Complete Study Guide

**Course Material:** Lectures 06-12 | **Topics:** Probability Foundations → VAE Theory → Implementation → GANs

---

## Table of Contents

1. [Probability Foundations](#probability-foundations)
2. [Joint Probability & VAE Connection](#joint-probability--vae-connection)
3. [VAE Theory & ELBO](#vae-theory--elbo)
4. [VAE Implementation](#vae-implementation)
5. [Generative Adversarial Networks](#generative-adversarial-networks)
6. [Study Roadmap & Connections](#study-roadmap--connections)

---

## Probability Foundations

### Core Concepts (Lecture 06)

#### 1. Conditional Probability
**What it measures:** Probability of event A given event B has already occurred.

**Formula:**
$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$

**Key Property:** NON-SYMMETRIC
$$P(A \mid B) \neq P(B \mid A)$$

**Real-World Example (Slide 374-375):**
Framework adoption by region:

| Region | Keras | TensorFlow | PyTorch |
|--------|-------|------------|---------|
| Egyptian | 500/1000 | 300/1000 | 200/1000 |
| Indian | 300/2000 | 100/2000 | 1600/2000 |

When you know someone is Egyptian, the probability they use Keras: $P(\text{Keras} \mid \text{Egyptian}) = \frac{500}{1000} = 0.5$

When you know someone uses Keras, the probability they're Egyptian: $P(\text{Egyptian} \mid \text{Keras}) = \frac{500}{1230} \neq 0.5$

**Why This Matters for VAE:** The encoder learns $q(z|x)$ (latent given data) which is conditional probability.

---

#### 2. Joint Probability
**What it measures:** Probability of two events happening together.

**Formula:**
$$P(X, Y) = P(Y, X)$$

**Key Property:** SYMMETRIC

**Matrix Representation (Slide 376-377):**
Normalized against total population of 5000:

| Region | Keras | TensorFlow | PyTorch |
|--------|-------|------------|---------|
| Egyptian | 500/5000 | 300/5000 | 200/5000 |
| Indian | 300/5000 | 100/5000 | 1600/5000 |

Example: $P(\text{Egyptian}, \text{Keras}) = \frac{500}{5000} = 0.1$

**Relationship to Conditional:**
$$P(A, B) = P(A \mid B) \cdot P(B)$$

**Why This Matters for VAE:** VAEs model the joint $p(x, z)$ where:
- $x$ = observed data
- $z$ = latent variables

---

#### 3. Marginal Probability
**What it measures:** Probability of a single event, isolated from other conditions.

**Formulas:**
$$P(A) = \sum_b P(A, b)$$

$$P(A) = \sum_b P(A \mid b)P(b)$$

**Example:**
$P(\text{Keras user}) = P(\text{Keras}, \text{Egyptian}) + P(\text{Keras}, \text{Indian}) + P(\text{Keras}, \text{German}) + P(\text{Keras}, \text{Chinese})$

**Why This Matters for VAE:** The VAE objective aims to maximize $\log p(x)$, the marginal likelihood of data.

---

#### 4. Bayes' Rule
**Formula:**
$$P(A \mid B) = \frac{P(B \mid A)P(A)}{P(B)}$$

**Why This Matters for VAE:** The true posterior $p(z|x)$ in VAE is intractable, so we use Bayes' rule to understand why:
$$p(z \mid x) = \frac{p(x \mid z)p(z)}{p(x)}$$
We can't compute $p(x)$ easily, so we approximate with $q(z|x)$.

---

### Expected Value (Lecture 07)

**Definition:** Long-run average or weighted average of a variable.

**Formula:**
$$E[X] = \sum x \cdot P(x)$$

**Example 1: Light Bulb Quality (Slide 385-390)**
- 95% work ($X=0$), 5% defective ($X=1$)
- $E[X] = (0.05)(1) + (0.95)(0) = 0.05$
- Interpretation: In the long run, 5% of bulbs are defective

**Example 2: Investment Returns (Slide 391-403)**
- Strong market: +$300,000 with 30% probability
- Weak market: -$100,000 with 70% probability
- $E[\text{Return}] = (300,000)(0.3) + (-100,000)(0.7) = \$20,000$
- Interpretation: Expected profit per $1M invested is $20,000

**Why This Matters for VAE:** The ELBO uses expectation:
$$\text{ELBO} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))$$

---

---

## Joint Probability & VAE Connection

### The Central Insight

**Joint Probability Factorization:**
$$p(x, z) = p(x \mid z) \cdot p(z)$$

This simple equation is the **foundation of VAE theory**.

### How They Connect

| Probability Concept | VAE Component | Role |
|-------------------|---------------|------|
| Joint $p(x,z)$ | What VAE models | Target distribution |
| Conditional $p(x\|z)$ | Decoder network | Generates data from latent code |
| Conditional $p(z\|x)$ | Unknown (intractable) | True encoder (can't compute directly) |
| Conditional $q(z\|x)$ | Encoder network | Approximates the true posterior |
| Prior $p(z)$ | Standard normal | Regularizes latent space |

### Decomposition Example

For a VAE on images:
- $x$ = 28×28 MNIST digit
- $z$ = 20-dimensional latent vector
- $p(x, z)$ = joint probability of image and its latent code

$$p(x, z) = \underbrace{p(x \mid z)}_{\text{Decoder}} \cdot \underbrace{p(z)}_{\text{Prior}}$$

The encoder $q(z|x)$ learns to reverse this:
$$q(z \mid x) \approx p(z \mid x) = \frac{p(x, z)}{p(x)}$$

---

---

## VAE Theory & ELBO

### Kullback-Leibler (KL) Divergence (Lecture 07)

**Purpose:** Measures how different one probability distribution is from another.

**Formula:**
$$D_{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}$$

**Key Property:** NON-SYMMETRIC
$$D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$$

**Example (Slide 412-428):**

| x | P(x) | Q(x) |
|---|------|------|
| 1 | 0.5 | 0.4 |
| 2 | 0.3 | 0.4 |
| 3 | 0.2 | 0.2 |

$$D_{KL}(P \| Q) = 0.5 \log\frac{0.5}{0.4} + 0.3 \log\frac{0.3}{0.4} + 0.2 \log\frac{0.2}{0.2}$$
$$= 0.0891 + (-0.0288) + 0 = 0.0603$$

**Why This Matters for VAE:** 
- $D_{KL}(q(z|x) \| p(z|x))$ measures how wrong our encoder approximation is
- We minimize this to make our encoder more accurate

---

### ELBO Derivation (Lecture 08, Slide 429)

**Goal:** Derive a tractable objective for learning VAE parameters.

**Step 1: Start with KL divergence**
$$\mathbb{D}_{KL}(q_\theta(z \mid x) \| p_\theta(z \mid x)) = \int dz \, q_\theta(z \mid x) \log_e \frac{q_\theta(z \mid x)}{p(z \mid x)}$$

**Step 2: Multiply by identity ratio**
$$= \int dz \, q_\theta(z \mid x) \log_e \left[ \frac{q_\theta(z \mid x)}{p(z \mid x)} \cdot \frac{p(x)}{p(x)} \right]$$

**Step 3: Expand the logarithm**
$$= \int dz \, q_\theta(z \mid x) \log_e \frac{q_\theta(z \mid x)}{p(z \mid x)p(x)} + \int dz \, q_\theta(z \mid x) \log_e p(x)$$

**Step 4: Pull out constant**
$$= \int dz \, q_\theta(z \mid x) \log_e \frac{q_\theta(z \mid x)}{p(z \mid x)p(x)} + \log_e p(x) \underbrace{\int dz \, q_\theta(z \mid x)}_{=1}$$

**Step 5: Use joint probability identity** $p(z, x) = p(z \mid x)p(x)$
$$= \int dz \, q_\theta(z \mid x) \log_e \frac{q_\theta(z \mid x)}{p(z, x)} + \log_e p(x)$$

**Step 6: Rearrange to isolate log-likelihood**
$$\log_e p(x) = \int dz \, q_\theta(z \mid x) \log_e \frac{p(z, x)}{q_\theta(z \mid x)} + \mathbb{D}_{KL}(q_\theta(z \mid x) \| p_\theta(z \mid x))$$

**Step 7: Rewrite in expectation form**
$$\log_e p(x) = \mathbb{E}_{q_\theta(z|x)}\left[\log_e \frac{p(z, x)}{q_\theta(z \mid x)}\right] + \mathbb{D}_{KL}(q_\theta(z \mid x) \| p_\theta(z \mid x))$$

---

### ELBO Definition

The **Evidence Lower Bound (ELBO)** is:
$$\text{ELBO} = \mathbb{E}_{q_\theta(z|x)}\left[\log_e \frac{p(z, x)}{q_\theta(z \mid x)}\right]$$

**Interpretation:**
- ELBO is a **lower bound** on $\log p(x)$ (data likelihood)
- The KL divergence gap represents how bad our encoder approximation is
- When $q(z|x) = p(z|x)$, the KL term = 0, and ELBO = $\log p(x)$ (perfect)

---

### ELBO Decomposition (Lecture 08, Slide 432-433)

**Expand using joint factorization:**
$$\text{ELBO} = \mathbb{E}_{q_\theta(z|x)}\left[\log_e \frac{p(z, x)}{q_\theta(z \mid x)}\right]$$

$$= \mathbb{E}_{q_\theta(z|x)}\left[\log_e \frac{p_\theta(x \mid z) p(z)}{q_\theta(z \mid x)}\right]$$

**Split into two terms:**
$$= \mathbb{E}_{q_\theta(z|x)}\left[\log_e p_\theta(x \mid z)\right] + \mathbb{E}_{q_\theta(z|x)}\left[\log_e \frac{p(z)}{q_\theta(z \mid x)}\right]$$

---

### VAE Loss Function (Final Form)

**Two components:**

1. **Reconstruction Loss** (negative of first term)
   $$\mathcal{L}_{\text{recon}} = -\mathbb{E}_{q_\theta(z|x)}[\log p_\theta(x \mid z)]$$
   - Measures how well decoder reconstructs data from latent code
   - For MNIST with binary data: Binary Cross-Entropy
   - For continuous data: Mean Squared Error

2. **KL Regularization Loss**
   $$\mathcal{L}_{KL} = \mathbb{D}_{KL}(q_\theta(z \mid x) \| p(z))$$
   - Measures how different encoder is from prior
   - Pushes latent space to match $p(z) = \mathcal{N}(0, I)$

**Total VAE Loss:**
$$\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{recon}} + \beta \cdot \mathcal{L}_{KL}$$

Where $\beta$ is a weighting hyperparameter (usually 1).

---

### Closed-Form Gaussian Solution (Slide 432-433)

When both $q(z|x)$ and $p(z)$ are Gaussian:
- $p(z) = \mathcal{N}(0, I)$ (standard normal, fixed)
- $q(z|x) = \mathcal{N}(\mu(x), \sigma^2(x))$ (learned by encoder)

The KL divergence has a closed-form solution:
$$\mathbb{D}_{KL}(q(z \mid x) \| p(z)) = -\frac{1}{2} \sum_{j=1}^{J} \left[1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2\right]$$

**Why This Matters:** 
- No need to sample; we can compute the gradient analytically
- Makes VAE training stable and efficient

---

---

## VAE Implementation

### The Reparameterization Trick (Lecture 09, Slide 435-482)

**Problem:** Sampling from $q(z|x)$ blocks gradient backpropagation.

```
x → Encoder → μ(x), σ(x) → [Sample z ~ N(μ, σ²)] → Decoder → x̂
                              ↑ BLOCKED: Can't backprop through sampling!
```

**Solution:** Use the **reparameterization trick**

Instead of sampling directly from $\mathcal{N}(\mu, \sigma^2)$:
1. Sample standard normal: $\epsilon \sim \mathcal{N}(0, 1)$
2. Transform: $z = \mu + \sigma \cdot \epsilon$

```
x → Encoder → μ(x), σ(x) ─→─┐
                             ├─ z = μ + σ·ε ─→ Decoder → x̂
                   Sample ε  │
                    (fixed)──┘
```

Now gradients flow through deterministic transformation!

**Mathematical Proof:**
If $\epsilon \sim \mathcal{N}(0, 1)$, then $z = \mu + \sigma \epsilon \sim \mathcal{N}(\mu, \sigma^2)$ ✓

**Numerical Example (Slide 435-482):**
Sample from $\mathcal{N}(\mu=10, \sigma=2)$:

| $\epsilon$ | $z = 10 + 2\epsilon$ |
|-----------|-------------------|
| -1.2 | 7.6 |
| 0.3 | 10.6 |
| 1.5 | 13.0 |
| -0.7 | 8.6 |
| 0.0 | 10.0 |

All values stay within reasonable range due to 68-95-99.7 rule.

---

### Keras/TensorFlow Implementation (Lecture 09, Slide 494-526)

**Sampling Layer:**
```python
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding the input."""
    
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        
        # Reparameterization trick: sample ε and return μ + σ·ε
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
```

**Complete VAE Model:**
```python
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        
        # Track losses
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            
            # Reconstruction loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2),
                )
            )
            
            # KL divergence loss
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
```

**Encoder Architecture:**
```python
latent_dim = 20
encoder_inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)

z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])

encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
```

**Decoder Architecture:**
```python
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)

decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
```

**Training:**
```python
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))

# Load MNIST
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, -1).astype("float32") / 255.0
x_test = np.expand_dims(x_test, -1).astype("float32") / 255.0

# Train
vae.fit(x_train, epochs=30, batch_size=128, validation_data=(x_test, x_test))
```

---

### Sampling & Generation (Lecture 09, Slide 527-549)

**Generate Random Images:**
```python
# Sample from prior p(z) = N(0, I)
n_samples = 10
z_samples = np.random.normal(loc=0, scale=1, size=(n_samples, latent_dim))
generated_images = decoder.predict(z_samples)

# Display: [1, 2, 4, 0, 3, 1, 9, 2, 8, 4] - random synthetic digits
```

**Interpolation in Latent Space:**
```python
# Encode a test image
x_test_sample = x_test[11:12]
z_mean, z_log_var, _ = encoder.predict(x_test_sample)

# Create variations by sampling around z_mean
n_variations = 20
epsilon_samples = np.random.normal(size=(n_variations, latent_dim))
z_variations = z_mean + np.exp(0.5 * z_log_var) * epsilon_samples
generated_variations = decoder.predict(z_variations)

# Display: 20 smooth variations of the target digit
```

---

---

## Generative Adversarial Networks

### GAN Architecture (Lecture 10, Slide 550-582)

**Two competing networks:**

| Component | Input | Output | Role |
|-----------|-------|--------|------|
| Generator $G$ | Random noise $z$ | Synthetic data | Create fake samples |
| Discriminator $D$ | Data or fake | Probability [0,1] | Distinguish real vs fake |

**Workflow:**
```
1. Random noise z ─→ Generator G ─→ Fake image
                                       ↓
2. Real image ─→ Discriminator D ─→ P(real) 
   Fake image ─→ Discriminator D ─→ P(real)
                     ↓
3. Feedback: D learns to distinguish, G learns to fool D
```

---

### GAN Loss Functions (Lecture 10, Slide 598-601)

**General Objective:**
$$L = -[\log(D(x)) + \log(1 - D(G(z)))]$$

**Discriminator Loss** (maximize = minimize negative)
$$L_D = -[\log(D(x)) + \log(1 - D(G(z)))]$$

**Generator Loss**
$$L_G = -\log(D(G(z)))$$

**Interpretation:**
- $D(x)$ = discriminator output for real data (want → 1)
- $D(G(z))$ = discriminator output for fake data (want → 0)
- Generator wants to maximize $D(G(z))$ (fool discriminator)
- Discriminator wants to minimize both: real loss + fake loss

---

### GAN Training Loop (Lecture 10, Slide 583-597)

**Step 1: Train Discriminator**
```
- Freeze Generator weights
- Generate fake batch: G(z)
- Get real batch: x
- Compute loss: D wants real → 1, fake → 0
- Update D parameters only
```

**Step 2: Train Generator**
```
- Freeze Discriminator weights
- Generate fake batch: G(z)
- Compute loss: G wants D(G(z)) → 1
- Update G parameters only
```

**Step 3: Repeat**
- Alternate between training D and G until convergence
- Monitor both losses

---

### PyTorch GAN Implementation (Lecture 11, Slide 602-675)

**Generator Network:**
```python
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, kernel_size=7, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)
```

**Discriminator Network:**
```python
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
```

**Training Setup:**
```python
G = Generator().to(device)
D = Discriminator().to(device)
criterion = nn.BCELoss()
opt_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
opt_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
```

**Training Loop:**
```python
def train(dataloader, epochs):
    for epoch in range(epochs):
        for batch_idx, (real_imgs, _) in enumerate(dataloader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            
            # Labels
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # ===== Train Discriminator =====
            z = torch.randn(batch_size, latent_dim, 1, 1).to(device)
            fake_imgs = G(z)
            
            real_output = D(real_imgs)
            fake_output = D(fake_imgs.detach())  # Detach to not update G
            
            real_loss = criterion(real_output, real_labels)
            fake_loss = criterion(fake_output, fake_labels)
            d_loss = real_loss + fake_loss

            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

            # ===== Train Generator =====
            z = torch.randn(batch_size, latent_dim, 1, 1).to(device)
            fake_imgs = G(z)
            output = D(fake_imgs)  # No detach - compute gradient for G
            g_loss = criterion(output, real_labels)  # Want D to think it's real

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

            # Print progress
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch}/{epochs}] Batch [{batch_idx}]'
                      f' D Loss: {d_loss.item():.4f} G Loss: {g_loss.item():.4f}')
```

**Training Progression (Lecture 11):**

| Epoch | Stage | D Loss | G Loss | Quality |
|-------|-------|--------|--------|---------|
| 1 | Start | 0.6650 | 0.3423 | Noise |
| 1 | End | 0.0000 | 10.6752 | Noise |
| 4 | Early | 0.0003 | 10.4591 | Rough shapes |
| 7 | Mid | 0.0921 | 4.3651 | Clear digits |
| 20 | Trained | 0.0906 | 4.1331 | Sharp digits |

---

### Conditional GAN (Lecture 12, Slide 676-734)

**Key Idea:** Control what digit to generate by conditioning on class label.

**Conditional Generator:**
```python
class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10, embedding_dim=10):
        super().__init__()
        # Embedding for class label
        self.label_emb = nn.Embedding(num_classes, embedding_dim)
        
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + embedding_dim, 256, 7, 1, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        # Embed labels
        label_embedding = self.label_emb(labels)
        # Reshape to match spatial dimensions
        label_embedding = label_embedding.unsqueeze(2).unsqueeze(3)
        # Concatenate noise and label embedding
        x = torch.cat([z, label_embedding], dim=1)
        return self.model(x)
```

**Conditional Discriminator:**
```python
class ConditionalDiscriminator(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Embedding for class label (matches image spatial size)
        self.label_emb = nn.Embedding(num_classes, 28 * 28)
        
        self.model = nn.Sequential(
            nn.Conv2d(2, 64, 4, 2, 1),  # 2 channels: image + label
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # Embed labels
        label_embedding = self.label_emb(labels)
        # Reshape to spatial dimensions
        label_embedding = label_embedding.view(labels.size(0), 1, 28, 28)
        # Concatenate image and label
        x = torch.cat([img, label_embedding], dim=1)
        return self.model(x)
```

**Generation Example:**
```python
# Generate digit 7
G.eval()
with torch.no_grad():
    label = torch.tensor([7]).to(device)  # Class 7
    z = torch.randn(1, latent_dim, 1, 1).to(device)
    generated_img = G(z, label).cpu()
    # Output: Single digit "7"
```

**Training Results (Lecture 12):**
- Epoch 1: Faint digit shapes organized by class
- Epoch 2: Clear, distinct digits organized by row (Label 0, Label 1, ..., Label 9)

---

---

## Study Roadmap & Connections

### Learning Path

```
┌─────────────────────────────────────────────────────┐
│  FOUNDATIONS (Lectures 06-07)                       │
│  • Conditional, Joint, Marginal Probability        │
│  • Expected Value                                   │
│  • Bayes' Rule & KL Divergence                      │
└──────────────┬──────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────┐
│  VAE THEORY (Lectures 08)                           │
│  • Joint Probability Factorization: p(x,z)=p(x|z)p(z)│
│  • ELBO Derivation (9 steps!)                       │
│  • Loss Decomposition:                              │
│    - Reconstruction: E[log p(x|z)]                  │
│    - KL Regularization: D_KL(q||p)                 │
└──────────────┬──────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────┐
│  VAE IMPLEMENTATION (Lectures 09)                   │
│  • Reparameterization Trick: z = μ + σ·ε            │
│  • Keras/TensorFlow Code                           │
│  • Sampling and Generation                          │
└──────────────┬──────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────┐
│  GANs (Lectures 10-12)                              │
│  • Generator vs Discriminator                       │
│  • Loss Functions & Training Loop                   │
│  • PyTorch Implementation                           │
│  • Conditional GANs (CG AN)                         │
└─────────────────────────────────────────────────────┘
```

---

### Key Conceptual Connections

**1. Joint Probability → ELBO Derivation**
- $p(x,z) = p(x|z) \cdot p(z)$ (factorization)
- Appears in Step 6 of ELBO derivation
- Leads to two components in final loss

**2. KL Divergence → VAE Regularization**
- Measures distance between $q(z|x)$ and $p(z|x)$
- Final VAE loss includes $D_{KL}(q(z|x) \| p(z))$
- Closed-form solution for Gaussians

**3. Conditional Probability → VAE Encoder**
- Encoder learns $q(z|x)$ (latent given data)
- Approximates true posterior $p(z|x)$ from Bayes' rule
- Non-symmetric: different from $q(x|z)$ (decoder)

**4. Reparameterization Trick → Gradient Flow**
- Enables backpropagation through sampling
- Fundamental to VAE training stability
- Makes ELBO gradients computable

**5. Reconstruction vs Regularization Trade-off**
- Reconstruction loss: Fidelity to training data
- KL loss: Constraints on latent space
- $\beta$ parameter controls trade-off

**6. GAN vs VAE**
- **VAE:** Explicit probability model (ELBO)
- **GAN:** Implicit model (adversarial training)
- Both generate new samples
- VAE has inference (encode), GAN does not

---

### Common Misconceptions

| Misconception | Reality |
|---|---|
| VAE directly samples from $p(z\|x)$ | VAE uses $q(z\|x)$ encoder as approximation |
| KL loss forces latent space to be standard normal | KL loss encourages it; reconstruction loss prevents collapse |
| Joint probability is used only in ELBO | Joint appears throughout VAE theory |
| GAN and VAE produce same quality | Different trade-offs: VAE preserves structure, GAN sharper |
| Reparameterization trick changes distribution | It preserves distribution while enabling gradients |

---

### Questions to Test Understanding

**Probability:**
1. Why is $P(A\|B) \neq P(B\|A)$?
2. How does joint probability relate to conditional and marginal?
3. What does KL divergence measure?

**VAE Theory:**
4. Why can't we compute $p(z\|x)$ directly?
5. What would happen if you removed the KL term from loss?
6. Why does ELBO have a gap from $\log p(x)$?

**Implementation:**
7. How does reparameterization trick enable backprop?
8. What's the intuition behind $z = \mu + \sigma \cdot \epsilon$?
9. How does interpolation in latent space show VAE learned structure?

**Advanced:**
10. How would you modify VAE for discrete latent variables?
11. What's the connection between VAE and PCA?
12. Why does Conditional GAN embed labels separately?

---

### Key Formulas Summary

| Concept | Formula | Purpose |
|---------|---------|---------|
| Conditional Prob | $P(A\|B) = P(A,B)/P(B)$ | Prob of event given another |
| Joint Prob | $p(x,z) = p(x\|z)p(z)$ | Core VAE factorization |
| Expected Value | $E[X] = \sum x \cdot P(x)$ | Long-run average |
| KL Divergence | $D_{KL}(P\|\|Q) = \sum P(x)\log(P(x)/Q(x))$ | Distribution distance |
| ELBO | $\mathbb{E}_{q}[\log p(x,z)/q(z\|x)]$ | Lower bound on log-likelihood |
| Reconstruction | $-\mathbb{E}_q[\log p(x\|z)]$ | Data fidelity |
| Regularization | $-D_{KL}(q(z\|x)\|\|p(z))$ | Latent space constraint |
| Reparameterization | $z = \mu + \sigma \cdot \epsilon$ | Enable gradient through sample |
| GAN Discriminator | $L_D = -[\log D(x) + \log(1-D(G(z)))]$ | Real vs fake classification |
| GAN Generator | $L_G = -\log(D(G(z)))$ | Fool discriminator |

---

## Additional Resources & Notes

### Implementation Tips
- Use `tf.exp(0.5 * z_log_var)` instead of `tf.sqrt(...)` for numerical stability
- Binary Cross-Entropy assumes data in [0,1]
- Start with $\beta=1$; adjust if reconstruction/regularization imbalance
- Monitor both losses during training

### Debugging Checklist
- [ ] Encoder outputs reasonable mean and variance
- [ ] KL loss decreases during training
- [ ] Reconstruction loss decreases during training
- [ ] Generated samples show diversity (not mode collapse)
- [ ] Latent space interpolation is smooth
- [ ] Loss values don't explode or vanish

### Extensions to Explore
- **β-VAE:** Increase $\beta$ for more disentangled representations
- **VQ-VAE:** Discrete latents instead of continuous
- **VAE-GAN:** Combine VAE reconstruction with GAN discrimination
- **Conditional VAE:** Add class labels like in CGAN

---

**Last Updated:** Study Guide for Lectures 06-12 (Full Course)  
**Recommended Study Time:** 8-12 hours for complete understanding
