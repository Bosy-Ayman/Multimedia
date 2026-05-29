# Masterclass Study Guide: Neural Radiance Fields (NeRF) and Volumetric Rendering

This study guide bridges the conceptual and mathematical gaps found in **Lec_08.pptx** regarding Neural Radiance Fields (NeRF) and volumetric ray marching.

## 1. The Core Paradigm Shift: Traditional Graphics vs. NeRF

To understand NeRF, we must contrast it with traditional 3D pipelines:

```
TRADITIONAL 3D GRAPHICS (Forward Rendering)
[Explicit 3D Geometry] ───(Rasterization / Ray Tracing)───► [2D Image Projection]
(Meshes, Textures, Materials, Lights)

NEURAL RADIANCE FIELDS (Inverse Rendering / View Synthesis)
[Many Registered 2D Images] ───(Neural Field Optimization)───► [Implicit 3D Scene Representation]
                                                                        │
                                                               (Volumetric Ray Marching)
                                                                        │
                                                                        ▼
                                                             [Novel view synthesis]
```

- **Traditional 3D Graphics:** Starts with explicit geometry (like triangle meshes with coordinates $V \in \mathbb{R}^{3}$) and texture coordinates. It simulates light interacting with these surfaces to produce 2D pixel projections.
    
- **NeRF:** Starts with a collection of 2D images of a static scene whose camera poses (origins and directions) are pre-calculated. It trains an implicit neural model to represent the scene as a continuous volumetric field. You can then render photorealistic novel views from any arbitrary camera angle.
    

## 2. Defining the 5D Neural Radiance Field Function

NeRF models a continuous static scene as a 5D vector-valued function $F_{\Theta}$. The neural network (typically a Multi-Layer Perceptron optimized with weights $\Theta$) learns the mapping:

$$F_{\Theta}: (\mathbf{x}, \mathbf{d}) \longrightarrow (\vec{c}, \sigma)$$

### The Inputs

1. **Spatial Position (**$\mathbf{x}$**):** A 3D coordinate vector representing a point in space:
    
    $$\mathbf{x} = (x, y, z) \in \mathbb{R}^3$$
2. **Viewing Direction (**$\mathbf{d}$**):** Represented in spherical coordinates by elevation and azimuth angles, or as a 3D unit direction vector:
    
    $$\mathbf{d} = (\theta, \phi) \equiv (d_x, d_y, d_z) \in \mathbb{S}^2$$

### The Outputs

1. **Emitted Color (**$\vec{c}$**):** The view-dependent RGB color at that spatial coordinate:
    
    $$\vec{c} = (R, G, B) \in [0, 1]^3$$
2. **Volume Density (**$\sigma$**):** The differential probability of the ray hitting an infinitesimal particle at spatial coordinate $\mathbf{x}$. It acts as physical "opacity" or "thickness" of space:
    
    $$\sigma(\mathbf{x}) \in [0, \infty)$$

### Crucial Architecture Constraint: Isotropic Density vs. Anisotropic Color

To ensure that the reconstructed 3D geometry remains static and physically consistent across multiple views, NeRF enforces a strict architectural bottleneck:

- **Density** $\sigma$ must only depend on the spatial position $\mathbf{x}$. The network cannot look at the viewing direction $\mathbf{d}$ to calculate $\sigma$. This ensures solid shapes do not warp or disappear when looked at from different angles:
    
    $$\sigma = \text{MLP}_{\text{geometry}}(\mathbf{x})$$
- **Color** $\vec{c}$ depends on _both_ position $\mathbf{x}$ and viewing direction $\mathbf{d}$. This allows the model to represent view-dependent effects like specular highlights on glossy surfaces, reflections, and sheen:
    
    $$\vec{c} = \text{MLP}_{\text{color}}(\mathbf{x}, \mathbf{d})$$

## 3. Volumetric Rendering Equations (Theory)

To render a pixel's color, NeRF casts a camera ray from the camera center through that pixel into the 3D volume, evaluates the network at multiple points along the ray, and composites the outputs.

### 3.1 Defining the Camera Ray

A ray starting at camera origin $\vec{O}$ and traveling along unit direction $\vec{d}$ is parameterized by the distance variable $t$:

$$\vec{r}(t) = \vec{O} + t\vec{d}, \quad t \in [t_n, t_f]$$

Where $t_n$ is the **Near Plane** (closest rendering distance) and $t_f$ is the **Far Plane** (farthest rendering distance).

### 3.2 Continuous Volumetric Rendering Formulation

The expected color $C(\vec{r})$ of a camera ray $\vec{r}(t)$ between near plane $t_n$ and far plane $t_f$ is defined by the continuous integral:

$$C(\vec{r}) = \int_{t_n}^{t_f} T(t) \cdot \sigma(\vec{r}(t)) \cdot \vec{c}(\vec{r}(t), \vec{d}) \, dt$$

Where:

- $\sigma(\vec{r}(t))$**:** The volume density at point $t$. Higher density means more light is emitted/scattered at this point.
    
- $\vec{c}(\vec{r}(t), \vec{d})$**:** The emitted color from point $t$ back toward the camera direction $\vec{d}$.
    
- $T(t)$ **(Accumulated Transmittance):** The probability that the ray travels from the near plane $t_n$ to the point $t$ without hitting any intervening particles. It is defined as:
    

$$T(t) = \exp \left( -\int_{t_n}^{t} \sigma(\vec{r}(s)) \, ds \right)$$

#### Physical Intuition of Transmittance ($T(t)$):

At the near plane ($t = t_n$), no volume has been traversed, so $T(t_n) = e^0 = 1$. This means the ray is completely clear. As the ray marches deeper into the scene ($t > t_n$), it encounters particles of density $\sigma(s)$. These particles block and scatter the light, causing $T(t)$ to decay toward $0$. If the ray hits a solid wall, the transmittance behind that wall falls to zero, meaning any points behind the wall do not contribute to the final pixel color.

### 3.3 Discrete Volumetric Rendering Formulation

Because we cannot evaluate a continuous neural network at infinitely many points along the ray, we must approximate the integrals numerically. We partition the range $[t_n, t_f]$ into $N$ evenly spaced (or stratified) intervals:

$$[t_i, t_{i+1}], \quad \Delta t_i = t_{i+1} - t_i$$

Using numerical quadrature, we approximate the continuous equations into discrete versions:

$$C(\vec{r}) \approx \sum_{i=1}^{N} T_i \cdot \alpha_i \cdot \vec{c}_i$$

Where:

1. **Interval Opacity (**$\alpha_i$**):** The probability that light is blocked/reflected within the finite interval $i$:
    
    $$\alpha_i = 1 - e^{-\sigma_i \Delta t_i}$$
    
    _(Note: If density_ $\sigma_i = 0$_, then opacity_ $\alpha_i = 0$_. If density_ $\sigma_i \to \infty$_, then opacity_ $\alpha_i \to 1$_)._
    
2. **Discrete Accumulated Transmittance (**$T_i$**):** The probability that light survives all previous intervals ($1$ to $i-1$) to reach the current interval $i$:
    
    $$T_i = \exp \left( -\sum_{j=1}^{i-1} \sigma_j \Delta t_j \right) = \prod_{j=1}^{i-1} (1 - \alpha_j)$$

## 4. Mathematical Toy Example (Trace Walkthrough)

Let us mathematically reconstruct and calculate the numerical toy example presented in **Lec_08.pptx**.

### Step 1: Initialize Camera Ray Parameters

We define a camera ray with:

- **Camera Origin (**$\vec{O}$**):** $(0, 0, 0)$
    
- **Ray Direction (**$\vec{d}$**):** $(1, 0, 0)$ (marching straight along the positive x-axis)
    
- **Step Size (**$\Delta t_i$**):** $\Delta t = 1$
    
- **Total Samples (**$N$**):** $3$
    

The ray equation is:

$$\vec{r}(t) = \vec{O} + t\vec{d} = (0, 0, 0) + t(1, 0, 0) = (t, 0, 0)$$

### Step 2: Compute Sample Point Coordinates

Using our parameter steps $t_1 = 1, t_2 = 2, t_3 = 3$:

- **Sample 1:** $\vec{r}(1) = (1, 0, 0)$
    
- **Sample 2:** $\vec{r}(2) = (2, 0, 0)$
    
- **Sample 3:** $\vec{r}(3) = (3, 0, 0)$
    

### Step 3: Neural Network Predictions

Assume our NeRF model predicts the following density ($\sigma_i$) and color intensity ($\vec{c}_i$) at these points:

|Sample ($i$)|Coordinate $\vec{r}(t_i)$|Predicted Density ($\sigma_i$)|Predicted Color ($\vec{c}_i$)|
|---|---|---|---|
|$1$|$(1, 0, 0)$|$0.5$|$0.2$|
|$2$|$(2, 0, 0)$|$1.0$|$0.7$|
|$3$|$(3, 0, 0)$|$2.0$|$1.0$|

### Step 4: Calculate Interval Opacities ($\alpha_i$)

Using the formula $\alpha_i = 1 - e^{-\sigma_i \Delta t}$ with $\Delta t = 1$:

- **Sample 1:**
    
    $$\alpha_1 = 1 - e^{-0.5(1)} = 1 - 0.60653 = 0.39347 \quad (\approx \mathbf{0.3935})$$
- **Sample 2:**
    
    $$\alpha_2 = 1 - e^{-1.0(1)} = 1 - 0.36788 = 0.63212 \quad (\approx \mathbf{0.6321})$$
- **Sample 3:**
    
    $$\alpha_3 = 1 - e^{-2.0(1)} = 1 - 0.13534 = 0.86466 \quad (\approx \mathbf{0.8647})$$

### Step 5: Calculate Accumulated Transmittance ($T_i$)

Using the product formula $T_i = \prod_{j=1}^{i-1} (1 - \alpha_j)$:

- **Sample 1 (**$T_1$**):** By definition, nothing exists before the first sample point:
    
    $$T_1 = 1.0$$
- **Sample 2 (**$T_2$**):**
    
    $$T_2 = T_1 \cdot (1 - \alpha_1) = 1.0 \times (1 - 0.39347) = 0.60653 \quad (\approx \mathbf{0.6065})$$
- **Sample 3 (**$T_3$**):**
    
    $$T_3 = T_2 \cdot (1 - \alpha_2) = 0.60653 \times (1 - 0.63212) = 0.60653 \times 0.36788 = 0.22313 \quad (\approx \mathbf{0.2231})$$

### Step 6: Compute Composited Color Contributions ($T_i \alpha_i \vec{c}_i$)

We compute the final color contribution of each sample along the ray path:

- **Sample 1 Contribution:**
    
    $$C_1 = T_1 \cdot \alpha_1 \cdot \vec{c}_1 = 1.0 \times 0.3935 \times 0.2 = \mathbf{0.0787}$$
- **Sample 2 Contribution:**
    
    $$C_2 = T_2 \cdot \alpha_2 \cdot \vec{c}_2 = 0.6065 \times 0.6321 \times 0.7 = 0.38337 \times 0.7 \approx \mathbf{0.2683}$$
- **Sample 3 Contribution:**
    
    $$C_3 = T_3 \cdot \alpha_3 \cdot \vec{c}_3 = 0.2231 \times 0.8647 \times 1.0 = 0.19291 \times 1.0 \approx \mathbf{0.1929}$$

### Step 7: Final Composited Ray Pixel Color ($C(\vec{r})$)

Summing the individual contributions of all samples yields the final predicted color rendered for this pixel:

$$C(\vec{r}) = C_1 + C_2 + C_3$$$$C(\vec{r}) = 0.0787 + 0.2683 + 0.1929 = \mathbf{0.5399}$$

**Result Interpretation:** The rendered grayscale pixel value is $0.5399$ (on a scale from $0$ to $1$). Even though the far background is bright ($c_3 = 1.0$) and dense ($\sigma_3 = 2.0$), its contribution to the camera is dampened because the medium before it has absorbed a portion of the light ($T_3 = 0.2231$).

## 5. How NeRF Optimization Works (Loss Function)

To optimize the MLP parameters ($\Theta$), we compute the squared error loss over a mini-batch of rays ($R$) between our rendered volumetric colors ($C(\vec{r})$) and the ground-truth pixel colors ($C_{\text{GT}}(\vec{r})$) extracted from our known input images:

$$\mathcal{L} = \sum_{\vec{r} \in R} \left\| \hat{C}(\vec{r}) - C_{\text{GT}}(\vec{r}) \right\|_2^2$$

Through backpropagation, gradient updates adjust the MLP's internal weights, forcing it to output accurate density and view-dependent color representations for every single point in the 3D scene coordinate envelope.