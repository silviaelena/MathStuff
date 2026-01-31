# Summary: FedLGAN - BiLSTM + Attention + GAN for Time-Series Anomaly Detection and Repair

This file summarizes the key concepts, mathematics, and examples from the FedLGAN paper on anomaly detection and repair in hydrological telemetry data. The method combines **Generative Adversarial Networks (GANs)**, **Long Short-Term Memory (LSTM) networks with attention**, and **federated learning (FL)** to detect anomalies and reconstruct missing or corrupted data while preserving privacy.

---

## 1️⃣ Tensors and Inputs

| Symbol | Meaning | Shape / Type | Example |
|--------|---------|--------------|---------|
| X | Input sequence | L × f tensor | 24 × 2 (24 hours, 2 features: water level + rainfall) |
| I | Tensor representation of sequence | L × f | Same as X |
| h_t | LSTM hidden state at time t | Vector of size h | [0.1, -0.05, 0.2] for h=3 |
| C_t | LSTM cell state at time t | Vector of size h | Long-term memory of sequence |
| H | Concatenated BiLSTM hidden states | L × 2h | 24 × 6 for h=3 |
| r | Attention-weighted context vector | Vector of size 2h | Summary vector of important sequence info |
| α | Attention weights | Vector of size L | [0.05, 0.05, 0.2,...] |
| θ_d | Discriminator parameters | All weights/biases | Matrices for LSTM, attention, FC layers |
| θ_g | Generator parameters | All weights/biases | Matrices for LSTM, FC layers |
| Î | Reconstructed sequence | L × f | G(I) output sequence |
| Ĩ | Interpolated sequence for gradient penalty | L × f | ε I + (1-ε)G(I) |

> Tensors represent the data at each stage: inputs, hidden memory, reconstructed sequences, and attention-weighted summaries.

---

## 2️⃣ Hidden States

- **Conceptually**: The LSTM hidden state \(h_t\) is the network's memory at time t, summarizing all previous inputs. It allows the network to remember long-term patterns and dependencies in time-series data.
- **Mathematically**:

\(
h_t = o_t \odot \tanh(C_t), \quad C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
\)

- **Example:**
  - Input I_1 = [3.2, 0.5], h_0 = [0,0,0] → h_1 = [0.1, -0.05, 0.2]
  - Next input I_2 = [3.3, 0.0] → h_2 = updated memory incorporating I_1 + I_2

- **BiLSTM:** Concatenates forward and backward states:

\(
h_t = [h_t^\text{forward} \oplus h_t^\text{backward}] \in \mathbb{R}^{2h}
\)

> Concatenation preserves information from past and future, allowing more context for anomaly detection.

---

## 3️⃣ Attention Mechanism

1. Compute intermediate matrix:
\(
M = \tanh(H)  \in \mathbb{R}^{L \times 2h}
\)

2. Compute raw scores:
\(
s = w^T M  \in \mathbb{R}^{L} \)

3. Apply softmax to get attention weights:
\(
\alpha = \text{softmax}(s)  \in \mathbb{R}^{L}
\)

4. Compute context vector (weighted sum of hidden states):
\(
r = H \alpha^T  \in \mathbb{R}^{2h}
\)

> Attention focuses on the most important time steps, e.g., spikes in river level or rainfall, improving detection and reconstruction of anomalies.

---

## 4️⃣ GAN Loss Functions

### 4.1 Discriminator Update

\(
\nabla_{\theta_d} \frac{1}{M} \sum_{m=1}^M \Big[ \log D(I_k^{(m)}) + \log(1-D(G(I_k^{(m)}))) \Big]
\)

- D assigns low scores to real sequences and high scores to generated sequences.
- Gradient penalty stabilizes training:
\(
\lambda (||\nabla_{Ĩ_k} D(Ĩ_k^{(m)})||_2 - 1)^2
\)

### 4.2 Generator Update

\(
\nabla_{\theta_g} \frac{1}{M} \sum_{m=1}^M \Big[ \log(1 - D(G(I_k^{(m)}))) + || I_k^{(m)} - \hat{I}_k^{(m)} ||^2 \Big]
\)

- G tries to **fool D** (adversarial loss) and **reconstruct normal sequences** (reconstruction loss), which repairs anomalous data.

### Example shapes:
| Symbol | Shape | Meaning |
|--------|-------|---------|
| D(I) | scalar / vector L | probability sequence is real |
| G(I) | L × f | reconstructed sequence |
| α | L | attention weights |
| r | 2h | context vector |

---

## 5️⃣ Federated Learning & Iterations

- M = number of **local training iterations**
- **Within each iteration**:
  1. Freeze G → update D (possibly multiple times)
  2. Freeze D → update G (once)
- After M iterations:
  - G updated M times total
  - D updated ≥ M times (if multiple D updates per iteration)

> Federated learning allows each client to train locally without sharing raw data, preserving privacy while improving the global model.

---

## 6️⃣ Summary of Algorithm Flow (Example)

1. Initialize hidden states h_0 = 0, C_0 = 0
2. Forward BiLSTM → get H
3. Apply attention → get context vector r
4. **Discriminator update:** compute loss, gradient penalty, update θ_d (D learns to detect anomalies)
5. **Generator update:** compute adversarial + reconstruction loss, update θ_g (G learns to repair data)
6. Repeat M local iterations → send updates to global server → federated averaging

> Example with 24-hour hydrology sequence, 2 features, hidden size h=3:
> - h_0 = [0,0,0], C_0 = [0,0,0]
> - BiLSTM concatenated h_t = 6
> - α vector of size 24, context vector r = 6
> - Output Î = 24×2 reconstructed sequence

---

**Key Intuition:**
- **Hidden state** = memory of sequence
- **BiLSTM + attention** = captures past & future context and highlights important timesteps
- **GAN** = adversarial learning detects anomalies and reconstructs missing/corrupted data
- **Federated learning** = enables distributed training without sharing raw data, preserving privacy
- Together, this framework detects anomalies and **repairs sequences** effectively in time-series data

