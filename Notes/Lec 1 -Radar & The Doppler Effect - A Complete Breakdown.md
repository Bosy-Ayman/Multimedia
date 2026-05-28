## Part 1: The Core Concept (The "Why")

Imagine you are standing on a sidewalk and an ambulance drives by with its sirens blaring. As it comes **toward** you, the pitch sounds high. As it moves **away**, the pitch suddenly drops and sounds deeper.

This is the **Doppler Effect**. It applies to sound waves, but it also applies to the electromagnetic waves used by Radar systems.

1. **Transmitted Signal:** The radar shoots out a continuous wave of energy at a specific, known frequency.
    
2. **Received Signal:** That wave bounces off a target (like a car or a bicycle) and returns to the radar.
    

If the target is moving, it "squishes" or "stretches" the wave as it bounces back.

- By measuring the **difference in frequency** between the wave we sent and the wave we got back (the Doppler Shift, or **Δf**), we can calculate exactly how fast the target is moving and in what direction.
    

## Part 2: The Math of Velocity

The lecture gives us a simple, powerful formula to find the target's velocity:

**ΔV = (Δf * λ) / 2**

- **ΔV** = The velocity of the target (How fast is it moving relative to the radar?)
    
- **Δf** = The Doppler frequency shift (The difference between transmitted and received frequency, measured in Hertz).
    
- **λ (Lambda)** = The wavelength of the radar (measured in meters).
    
- **Divided by 2**: Because the wave has to travel _to_ the target and _back_ (a two-way trip).
    

### Velocity Examples (assuming a radar wavelength λ = 0.03 m):

- **Negative Shift (-200 Hz):** (-200 * 0.03) / 2 = **-3.0 m/s** (Moving away)
    
- **Positive Shift (+100 Hz):** (100 * 0.03) / 2 = **+1.5 m/s** (Moving toward)
    
- **Zero Shift (0 Hz):** (0 * 0.03) / 2 = **0 m/s** (Stationary)
    

## Part 3: The Distance Problem & "Chirp" Signals

Lecture 1 let us find velocity using a continuous wave. But there's a problem: a plain continuous wave can't tell us **distance**. Because every single wave looks identical, we can't tell exactly _which_ wave bounced back, so we can't measure how long the trip took.

**The Solution: The Chirp Signal** Think of a chirp like a slide whistle. Instead of blowing one constant note, you start at a low pitch and smoothly slide up to a high pitch.

In radar, a chirp is a signal whose frequency changes continuously over time in a linear pattern.

**The Formula:** `f(t) = f0 + S * t`

- **f0** = The starting frequency (Carrier Frequency)
    
- **S** = The Chirp Slope (How fast the pitch is rising)
    
- **t** = time
    

_Your Touch (The Dominant Frequency):_ Because the frequency is always changing, if we want to know the "dominant" (average) frequency over a specific window of time, we just calculate the frequency at the start of the window, the frequency at the end of the window, and average them!

> _Example:_ Starts at `1.5`, ends at `2.5`. The dominant frequency is `(1.5 + 2.5) / 2 = 2`.

## Part 4: How Chirps Measure Distance (The "Beat")

Here is the genius of Chirp radar (often called FMCW radar):

1. **Transmit (Tx):** The radar starts singing a note that goes higher and higher.
    
2. **Receive (Rx):** The wave bounces off a car and comes back. Because it took time to travel, the "echo" we hear is slightly delayed.
    
3. **The Mixer:** Because of the delay, the pitch of the echo (Rx) is _lower_ than the pitch the rada r is currently singing (Tx). The radar mixes these two signals together (`Tx × Rx`).
    
4. **The Beat Signal:** Mixing them produces a brand new signal called the **Beat Signal**. The frequency of this beat (`f_beat`) is literally just the difference in pitch between the Tx and Rx signals at that exact moment.
    

**The Golden Rule:** The further away the target is, the longer the echo takes to return. The longer the echo takes, the _bigger_ the pitch difference will be between Tx and Rx. **Therefore, a higher beat frequency means a further distance!**

## Part 5: The Math of Distance

Now we can calculate exactly how far away the target is using this formula:

**R = (f_beat * c) / (2 * S)**

- **R (Range)** = Distance to the target (in meters).
    
- **f_beat** = The Beat Frequency we just created (in Hertz).
    
- **c** = The Speed of Light (Radar waves travel at the speed of light: **3 × 10⁸ m/s**).
    
- **S** = The Chirp Slope (How steep the frequency slide is).
    
- **Divided by 2**: Again, because the wave had to make a round trip (out and back).
    

### The Examples (Watch the Unit Conversions!)

_This is where most students get tripped up on exams. You MUST convert everything into standard Hertz (Hz) and Seconds (s) before doing the math._

**Example 1:**

- **Beat Frequency:** 200 kHz -> _(Kilo means 1,000)_ -> **2 × 10⁵ Hz**
    
- **Chirp Slope (S):** 20 MHz / μs.
    
    - _Wait!_ We have Mega (10⁶) divided by Micro (10⁻⁶).
        
    - 20 × 10⁶ / 10⁻⁶ = 20 × 10¹² = **2 × 10¹³ Hz/s**
        
- **The Math:** R = (2×10⁵ * 3×10⁸) / (2 * 2×10¹³)
    
- R = (6 × 10¹³) / (4 × 10¹³) = **1.5 meters**
    

**Example 2:**

- **Beat Frequency:** 2 MHz -> _(Mega means 1,000,000)_ -> **2 × 10⁶ Hz**
    
- **Chirp Slope (S):** 20 MHz / μs -> **2 × 10¹³ Hz/s** _(same as above)_
    
- **The Math:** R = (2×10⁶ * 3×10⁸) / (2 * 2×10¹³)
    
- R = (6 × 10¹⁴) / (4 × 10¹³) = **15 meters**
    

_(Notice how in Example 2, the Beat Frequency was 10 times higher than Example 1, which perfectly resulted in a distance that was exactly 10 times further away!)_

## Part 6: Demystifying the Python Code

_(Previously Part 3 & 4 from Lecture 1 regarding reading standard Doppler spectrographs)_

The second half of your first lecture takes real radar data of a bicycle and plots it using Python's `matplotlib`. Let's translate the code into plain English.

```python
# 1. Loading the Data
filename = "moving_target_dataset.npy"
signatures = np.load(filename, allow_pickle=True)
```

- **What it does:** Loads a raw dataset containing radar recordings of different objects.
    

```python
# 2. Processing the Signal Math
arr = signature['signature']
arr = 20 * np.log10(np.abs(arr)).transpose()
```

- **What it does:** `20 * np.log10` converts the raw, linear math into **Decibels (dB)**. This compresses the data so human eyes can see both the strong and weak signals on a screen without the strong ones blinding out everything else.
    

```python
# 3. Setting the Visual Boundaries
prf = signature['radar_parameters']['prf']
extent=[0, arr.shape[1], -int(prf/2), int(prf / 2)]
```

- **What it does:** This sets up the X and Y axes for the graph (`extent = [x_min, x_max, y_min, y_max]`).
    
- **X-Axis (Time):** Runs from `0` to `arr.shape[1]` (the total number of time samples).
    
- **Y-Axis (Frequency/Velocity):** Runs from `-PRF/2` to `+PRF/2`. PRF (Pulse Repetition Frequency) is the radar's "framerate".
    

```python
# 4. Coloring the Graph
plt.imshow(arr, cmap='jet', aspect='auto', vmax=np.max(arr) - 20, vmin=np.max(arr) - 70)
```

- **What it does:** Draws the actual image. `vmax` and `vmin` act as a contrast filter to ignore the background static (blue/green), and only highlight the strong, solid radar reflections of the actual target (red).