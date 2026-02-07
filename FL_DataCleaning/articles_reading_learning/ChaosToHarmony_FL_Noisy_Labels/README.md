# Analysis: Labeling Chaos to Learning Harmony: Federated Learning with Noisy Labels

**Authors:** Vasileios Tsouvalas, Aaqib Saeed, Tanir Özcelebi, Nirvana Meratnia  
**Institution:** Eindhoven University of Technology, The Netherlands  
**Published:** ACM Transactions on Intelligent Systems and Technology, Volume 15, Issue 2 (April 2024)  
**DOI:** https://doi.org/10.1145/3626242  
**Accepted:** 15 August 2023 | Published Online: 09 October 2023  
**Citations:** 18 | Downloads: 2,451 (as of Feb 2, 2026)

---

## Summary Table

| Column | Answer | Details |
|--------|--------|---------|
| **ML or FL** | **FL (Federated Learning)** | Distributed machine learning paradigm where labeling effort is entrusted to clients. Uses FedAvg as base algorithm. Addresses label noise across FL initialization, on-device training, and server aggregation stages. |
| **Handles fairness?** | **No** | Does not address fairness, equity, or bias concerns. While NA-FedAvg re-weights clients based on data quality (potentially disadvantaging clients with noisier labels), this is for model accuracy, not fairness. No discussion of: equal performance across clients, demographic parity, group fairness, or bias mitigation. The accommodation of different device capabilities is about computational efficiency, not fairness. |
| **Outliers** | **Indirectly (via embeddings)** | Uses kNN on embeddings to detect outliers in feature space (noisy labels appear as outliers in neighborhood). Not explicitly an outlier detection method, but noisy samples are treated as outliers in label space. |
| **Missing values** | **No** | Not addressed. Focus is exclusively on label noise/corruption, not missing data or features. |
| **Label errors** | **✓ Yes (Primary Focus)** | Core contribution: handles label noise (mislabeled data) across different FL stages. Addresses heterogeneous noise distributions across clients (per-client noise profiles). Handles synthetic noise (controlled experiments) and real-world human annotation errors (CIFAR-10N/100N). |
| **Repair data?** | **✓ Yes - LABELS ONLY (not data values)** | **Repairs/corrects mislabeled data through three approaches:** (1) **NNC (Nearest Neighbor Correction):** Directly corrects wrong labels using kNN majority vote on embeddings (changes y to y_vote). (2) **AKD (Adaptive Knowledge Distillation):** Implicitly corrects by using clean supervision signal (embeddings/logits) to prevent memorization of wrong labels. (3) **NA-FedAvg (Noise-Aware FedAvg):** Mitigates (not corrects) by downweighting clients with many label errors during aggregation. **Note:** Does NOT repair missing values, outliers, or corrupted feature data - only label errors. |
| **Topics** | **Federated learning, Noisy labels, Label correction, Deep learning, Knowledge distillation, Non-IID data, Label noise estimation, k-Nearest Neighbors, Self-supervised learning, Energy-based scoring, Noise matrices, Confidence Learning** | Addresses practical FL scenarios where users provide labels through interaction (keyboard suggestions, voice commands) or weak labeling, resulting in noisy annotations. |
| **Datasets** | **Vision:** CIFAR-10, Fashion-MNIST, PathMNIST, EuroSAT; **Audio:** Speech Commands (v2); **Real-world noisy:** CIFAR-10N/100N (human-annotated) | All publicly available datasets with standard train/test splits. CIFAR-10N/100N contain real human annotation errors (~40% noise) from Amazon Mechanical Turk. Vision: object detection, clothing classification, pathology, landmark classification. Audio: keyword spotting (12 classes). |
| **Code** | **✓ Yes - Available on GitHub** | Repository: https://github.com/FederatedML/FedLN. Implements all three FedLN approaches (NNC, AKD, NA-FedAvg), noise injection schemes, and all baseline methods. Uses Flower framework for FL simulation. |
| **Implemented?** | **✓ Yes - Extensively tested** | Comprehensive experiments: 5 datasets, multiple noise profiles (nl: 0-100%, ns: 0-100%), 30 clients, 200 federated rounds, 3 trials per setting. Compares against 7 baselines (FedAvg, Label Smoothing, Bi-Tempered loss, Confidence Learning, FedCorr, centralized methods). Reports 22% average improvement over FedAvg at 60% noise level, 9% improvement on real-world CIFAR-10N/100N. |

---

## Key Findings

### 1. **Problem Definition**

#### Challenge: Label Noise in Federated Learning
- **Centralized ML:** Can verify label quality through crowdsourcing, expert annotation, validation on aggregated data
- **Federated setting:** 
  - Data annotation performed by users (e.g., keyboard suggestions, voice commands)
  - No way to verify label quality (data never leaves device)
  - Weak labeling or programmatic labeling functions introduce noise
  - **Per-client noise profiles:** Noise varies by client characteristics, user expertise, annotation system
  - Scarce data per client makes noise more impactful

#### Why Existing Methods Fail:
- **Centralized approaches (CL, GAIN, etc.):** Assume IID data and sufficient samples per class
- **FL approaches (FedCorr, etc.):** 
  - Require excessive on-device computation (GMM training, pseudo-labeling)
  - Need additional clean data on server
  - Communicate client-sensitive information (features, centroids)

#### Label Noise Definitions:
1. **Noise Level (nl):** Amount of label noise, nl = 1 - diag(Q_{y|y*})
   - nl=0: Clean dataset (all labels correct)
   - nl=1: Completely noisy dataset
   
2. **Noise Sparsity (ns):** Shape of noise distribution
   - ns=0: Uniform noise (any class confused with any other)
   - ns=1: Class flipping (e.g., cat ↔ dog only)
   - High ns: Related classes confused (cat→tiger, not cat→airplane)

3. **Noise Matrix Q_{y|y*}:** C×C matrix where Q_{ij} = P(y=i | y*=j)
   - Each column: probability distribution for true label y*=j
   - In FL: Each client m has unique Q^m (heterogeneous noise profiles)

---

### 2. **Methodology: FedLN Framework**

FedLN proposes **three distinct approaches** operating at different FL stages, suitable for different device computational capabilities:

#### **Approach 1: NNC (Nearest Neighbor Correction)** - High compute
**Stage:** FL initialization (one-time)  
**Target devices:** High-end smartphones, tablets with sufficient compute/storage

**Process:**
1. **Extract embeddings:** Use self-supervised pre-trained model (CLIP ViT-B/32 for vision, TRILLsson for audio)
   - Forward pass only, no training
   - Embeddings are noise-tolerant (trained without labels)
2. **Detect noisy labels:** For each sample, find k=100 nearest neighbors in embedding space
3. **Correct labels:** Majority vote among k neighbors
   - If predicted label ≠ current label → update label
   - Equation: y_vote = argmax_{c∈C} Σ_{j∈[k]} 1(y_j = c)
4. **Train normally:** After correction, standard FedAvg on corrected datasets

**Noise estimation:** nl^m = (# samples where y_vote ≠ y_current) / N_m

**Advantages:**
- Most accurate label correction (Table 2: 93% AUC for nl=40%)
- Not affected by client model quality
- One-time computational cost

**Limitations:**
- Requires pre-trained model storage temporarily
- Fails on class-flipping with high noise (ns=100%, nl=70%): neighborhoods become mixed
- Embedding quality critical

---

#### **Approach 2: AKD (Adaptive Knowledge Distillation)** - Medium compute
**Stage:** On-device model training (every round)  
**Target devices:** Mid-range devices with moderate compute

**Process:**
1. **Estimate noise:** Same as NNC (embeddings) or NA-FedAvg (model confidence)
2. **Adaptive loss:** For clients with nl > ε, add distillation loss:
   ```
   L = L_CE(y, p_θ(y|x)) + β · u_ε(nl) · L_KD(e, p_θ(e|x))
   ```
   - u_ε: Heaviside function (activates if nl > ε)
   - β=10: distillation weight
   - L_KD: Either MAE (embeddings) or KL divergence (logits)

3. **Two variants:**
   - **AKD (Embeddings):** Distill to match pre-trained model embeddings
   - **AKD (Logits):** Distill to match server global model outputs (temperature T=2)

**Advantages:**
- Handles class-flipping scenarios (where NNC fails)
- Provides "clean" supervision signal uncorrelated with noisy labels
- Outperforms NNC on clean data (Table 3: higher accuracy when nl=0%)

**Limitations:**
- Computational overhead every round (forward pass through teacher)
- Requires longer training (R > 200) to surpass NNC

---

#### **Approach 3: NA-FedAvg (Noise-Aware Federated Averaging)** - Low compute
**Stage:** Server-side aggregation (after round R_w=30)  
**Target devices:** Wearables, edge devices, low-end smartphones

**Process:**
1. **Warm-up phase:** Train normally for R_w=30 rounds
2. **Noise estimation round (R_w):**
   - Each client computes energy scores for all samples:
     ```
     E(x) = -T · log Σ_c exp(f_c(x) / T)
     ```
   - Compute scores using local model θ_m and global model θ_G
   - Send scoring sets (S_θ_G, S_θ_m) to server (no raw data!)
   
3. **Server-side noise estimation:**
   - Compute threshold τ_ν = P_ν(S_θ_G) (ν=75th percentile)
   - Estimate per-client noise: nl^m = (# samples below τ_ν) / N_m

4. **Noise-aware aggregation:** Modify FedAvg weights:
   ```
   γ_m = (1 - nl^m) · (N_m / N)
   ```
   - Clients with cleaner labels have higher weight

**Advantages:**
- **Minimal on-device overhead:** Only logsumexp operation
- **No additional data:** No pre-trained models needed
- **Lightweight communication:** Only scores sent once
- Suitable for resource-constrained devices

**Limitations:**
- Requires some "clean" clients (F < 100%)
- Performance degrades if all clients equally noisy (Table 5)
- Less accurate noise estimation than embeddings (Table 2)

---

### 3. **Label Noise Detection Mechanisms**

#### **Embedding-Based Discovery:**
- **AUC scores (Table 2):**
  - nl=40%, ns=40%: **93.43%** (CIFAR-10), **83.07%** (Speech Commands)
  - nl=70%, ns=70%: **62.29%** (CIFAR-10), **60.11%** (Speech Commands)
- **Robustness:** Unaffected by client models or noise during training
- **Failure mode:** High sparsity + high noise → neighborhoods become mixed

#### **Model Confidence-Based (Energy Scores):**
- **Why Energy > Softmax:** Less susceptible to overconfident predictions on out-of-distribution samples
- **AUC scores (Table 2):**
  - nl=40%, ns=40%: 65.07% (CIFAR-10), 69.59% (Speech Commands)
  - nl=70%, ns=70%: 54.40% (CIFAR-10), 59.33% (Speech Commands)
- **Optimal timing:** R_w=30 (Figure 5 shows AUC plateau after 30 rounds)
- **Key insight:** Clean vs noisy clients show clear separation in energy scores (Figure 6)

#### **Comparison with Baselines:**
- **Confidence Learning (CL):** Prunes data based on probabilistic thresholds
  - Requires 20 local epochs initialization
  - Lower AUC than energy scores
- **FedCorr:** GMM-based on loss scores
  - More computationally expensive
  - Similar AUC to energy scores but higher overhead

---

### 4. **Experimental Setup**

#### **Federated Parameters (Table 1):**
- **Clients (M):** 30
- **Rounds (R):** 200 (500 for CIFAR-100N)
- **Local steps (E):** 1
- **Participation rate (q):** 80%
- **Data distribution variance (σ):** 25% (non-IID, imbalanced)
- **Noisy clients percentage (F):** 0-100%

#### **Noise Injection:**
- Synthetic noise: Construct Q_{y|y*} based on nl and ns
- Per-client unique noise matrices (heterogeneous profiles)
- Random partitioning → naturally imbalanced clients

#### **Model Architectures:**
- **Vision:** ResNet-20 (compact for on-device learning)
- **Audio:** 4-block CNN (temporal + frequency convolutions, inspired by mobile architectures)
- **Alternative tested:** ConvMixer-128/4 (~0.1M parameters)

#### **Optimization:**
- **CIFAR-10, Fashion-MNIST:** SGD, lr=0.1, momentum=0.9
- **PathMNIST, EuroSAT, Speech Commands:** Adam, lr=0.001
- **Augmentation:** Random flip, crop, Cutout (vision); log-Mel spectrograms (audio)

#### **Pre-trained Models:**
- **Vision:** CLIP ViT-B/32 (OpenAI)
- **Audio:** TRILLsson v3 (EfficientNetv2-B3)
- Downloaded directly on-device, used for single forward pass

#### **Evaluation:**
- 3 independent trials per setting (different seeds)
- Average accuracy on clean test sets
- Comparison against 7 baselines

---

### 5. **Results**

#### **Main Results (Table 3):**

**CIFAR-10 @ nl=70%, ns=40%:**
- FedAvg: 56.53%
- Label Smoothing: 57.61%
- Confidence Learning: 60.28%
- FedCorr: 68.17%
- **NNC (FedLN):** **72.63%** (+16.1% vs FedAvg)
- **AKD (FedLN):** **68.06%** (+11.5% vs FedAvg)
- **NA-FedAvg (FedLN):** **66.04%** (+9.5% vs FedAvg)

**Fashion-MNIST @ nl=70%, ns=40%:**
- FedAvg: 56.24%
- FedCorr: 79.28%
- **NNC:** **84.91%** (+28.7% vs FedAvg)
- **AKD:** **78.63%** (+22.4% vs FedAvg)
- **NA-FedAvg:** **78.68%** (+22.4% vs FedAvg)

**Speech Commands @ nl=70%, ns=40%:**
- FedAvg: 71.28%
- FedCorr: N/A (MixUp not applicable to audio)
- **NNC:** **96.11%** (+24.8% vs FedAvg)
- **AKD:** **76.63%** (+5.4% vs FedAvg)
- **NA-FedAvg:** **81.91%** (+10.6% vs FedAvg)

#### **Average Improvement @ nl=60%:**
- **22% improvement** across all datasets compared to FedAvg

#### **Key Observations:**

1. **NNC dominates most scenarios:**
   - Best for moderate noise (nl=40%, 70% with ns < 70%)
   - Exception: Class-flipping (ns=100%) at high noise → embeddings fail

2. **AKD excels in specific cases:**
   - Class-flipping scenarios (Table 3: ns=100% columns)
   - Clean data (nl=0%): Outperforms all methods on vision datasets
   - Long training horizons (CIFAR-100N with R=500)

3. **NA-FedAvg competitive with minimal overhead:**
   - Performance close to FedCorr without computational burden
   - Especially effective for high sparsity noise (ns=70-100%)

4. **FedLN > Centralized methods in FL:**
   - Label Smoothing: Inconsistent (helps low sparsity, hurts high)
   - Bi-Tempered loss: Only helps high noise levels
   - Confidence Learning: +4% average but unstable

5. **FedCorr comparison:**
   - FedLN comparable or better performance
   - FedLN more efficient: FedCorr requires 250 rounds (5×150 preprocessing + 95 fine-tune + 100 standard)
   - FedLN requires 200 rounds total

---

#### **Architecture Robustness (Table 4):**

**ConvMixer-128/4 on CIFAR-10 @ nl=70%, ns=40%:**
- FedAvg: 63.94%
- **NNC:** 76.59% (+12.7%)
- **NA-FedAvg:** 68.97% (+5.0%)

**Conclusion:** FedLN efficacy independent of architecture choice

---

#### **Varying Noisy Clients (Table 5):**

**CIFAR-10 @ nl=40%, ns=40%:**

| % Noisy Clients (F) | NNC Accuracy | NA-FedAvg Accuracy |
|---------------------|--------------|-------------------|
| 25% | 74.84% | 74.84% |
| 50% | 74.97% | 73.45% |
| 75% | 74.96% | 70.74% |
| 100% | 73.41% | 62.62% (-12%) |

**Key Insights:**
- **NNC robust:** Minimal impact from increasing F (embeddings unaffected)
- **NA-FedAvg sensitive:** Requires some clean clients (F < 100%)
- With 20% clean clients: NA-FedAvg reaches ~70% accuracy

---

#### **Label Noise on "Clean" Clients (Table 6):**

**CIFAR-10 @ nl=40%, injecting X% of nl to "clean" clients:**

| Noise on Clean | NNC | NA-FedAvg |
|----------------|-----|-----------|
| 0% | 73.68% | 69.52% |
| 10% | 74.57% | 68.12% (-1.4%) |
| 25% | 72.94% | 66.17% (-3.4%) |
| 100% | 74.01% | 62.62% (-6.9%) |

**Conclusions:**
- **NNC unaffected:** Embeddings remain noise-tolerant
- **NA-FedAvg degrades:** Relies on clean client models for noise detection
- NNC preferable when strong assumptions about clean clients are violated

---

#### **Real-World Human Annotation (Figure 8):**

**CIFAR-10N (nl ≈ 40% real noise):**
- FedAvg: ~70%
- Confidence Learning: ~72%
- FedCorr: ~74%
- **NNC:** **~76%** (+6%)
- **AKD:** **~75%** (+5%)
- **NA-FedAvg:** **~74%** (+4%)

**CIFAR-100N (100 classes, R=500):**
- FedAvg: ~44%
- **AKD (best):** **~52%** (+8%)
- NNC: ~49%
- NA-FedAvg: ~47%

**Key Findings:**
- Real-world noise patterns handled effectively
- AKD benefits from longer training (R=500)
- Performance within 2% of synthetic noise experiments
- Validates practical applicability for everyday users (e.g., Gboard keyboard suggestions)

---

### 6. **Technical Contributions**

1. **First comprehensive FL label noise framework** with solutions at three stages:
   - Initialization (NNC)
   - Training (AKD)
   - Aggregation (NA-FedAvg)

2. **Heterogeneous noise handling:** Per-client noise profiles (distinct Q^m matrices)

3. **Compute-adaptive solutions:** Low/medium/high-end devices accommodated

4. **No server-side clean data:** Unlike FedNoisy, FedLabelFair (require validation set)

5. **No client-sensitive data sharing:** Unlike FedDR-Filter (sends features), RFL (sends centroids)

6. **Single-round noise estimation:** Efficient compared to multi-stage approaches (FedCorr)

7. **Energy-based scoring for FL:** First use of energy scores for label noise detection in federated setting

8. **Embeddings for label correction:** Novel use of self-supervised pre-trained models for noise detection without training

---

### 7. **Limitations & Future Work**

#### **Limitations:**
1. **NNC failure on class-flipping:** High sparsity + high noise → mixed neighborhoods
2. **NA-FedAvg requires clean clients:** Performance degrades when F=100%
3. **Pre-trained model availability:** NNC/AKD assume accessible pre-trained models for domain
4. **Communication overhead:** NA-FedAvg requires all clients in round R_w (can be asynchronous)
5. **Honest-but-curious server:** Not Byzantine-robust
6. **Moderate scale:** 30 clients, specific datasets

#### **Future Directions (from paper):**
1. **Computational efficiency analysis:** Real hardware measurements (not just FLOPs)
2. **Detection of real noisy samples:** Identify which specific samples are mislabeled (for manual review)
3. **Hybrid approaches:** Combine NNC + AKD + NA-FedAvg in single scheme
4. **Adaptive R_w selection:** Automatically determine noise estimation round
5. **Communication cost optimization:** Compression, quantization of scores

#### **Additional Extensions:**
1. **Byzantine robustness:** Malicious clients providing false noise estimates
2. **Vertical FL:** Handle label noise when features distributed across clients
3. **Class-imbalanced noise:** Address scenarios where noise concentrated in specific classes
4. **Active learning integration:** Query users for labels on high-uncertainty samples
5. **Continual learning:** Handle label noise in non-stationary data distributions
6. **Differential privacy:** Formal privacy guarantees beyond secure aggregation
7. **Multi-modal fusion:** Combine multiple pre-trained models (vision + text)

---

## Comparison with Baseline Methods

### Centralized Methods:

| Method | Type | CIFAR-10 @ nl=40%, ns=40% | Notes |
|--------|------|--------------------------|-------|
| **Supervised (FedAvg)** | Baseline | 64.02% | Standard federated training |
| **Label Smoothing** | Regularization | 66.58% (+2.6%) | α=0.2, softens labels |
| **Bi-Tempered Loss** | Robust Loss | 65.66% (+1.6%) | Heavy-tailed softmax |
| **Confidence Learning** | Data Pruning | 67.65% (+3.6%) | Prunes noisy samples |

### FL-Specific Methods:

| Method | Type | CIFAR-10 @ nl=40%, ns=40% | Overhead |
|--------|------|--------------------------|----------|
| **FedCorr** | Multi-stage | 71.14% (+7.1%) | 250 rounds, GMM training |
| **NNC (FedLN)** | Label Correction | **73.64% (+9.6%)** | Pre-trained model (one-time) |
| **AKD (FedLN)** | Knowledge Distillation | **69.18% (+5.2%)** | Distillation every round |
| **NA-FedAvg (FedLN)** | Noise-Aware Aggregation | **70.73% (+6.7%)** | Minimal (one-time scoring) |

### Detection Performance (AUC):

| Method | CIFAR-10 @ nl=40%, ns=40% | Speech Commands @ nl=40%, ns=40% |
|--------|--------------------------|----------------------------------|
| Confidence Learning | 58.13% | 61.85% |
| FedCorr | 64.51% | N/A |
| Softmax Score | 55.27% | 57.98% |
| **Energy Score** | **65.07%** | **69.59%** |
| **Embeddings (NNC)** | **93.43%** | **83.07%** |

---

## Algorithm Pseudocode (Simplified)

### Algorithm 1: FedLN Framework

```python
# SERVER SIDE
def FedLN_Server(M, R, R_w, approach):
    θ_G ← initialize_global_model()
    
    for r in range(1, R+1):
        # Select clients
        M' ← random_sample(M, q=0.8)
        
        if r == 1 and approach in ['NNC', 'AKD_emb']:
            # Initialization: Noise estimation via embeddings
            for m in M':
                n_l^m ← estimate_noise_embeddings(m)
        
        if r == R_w and approach == 'NA-FedAvg':
            # Noise estimation via model confidence
            for m in M':
                (S_θG^m, S_θm^m) ← compute_scores(m, θ_G)
                n_l^m ← estimate_noise_scores(S_θG^m, S_θm^m)
        
        # Client updates
        for m in M':
            θ_m ← ClientUpdate(θ_G, m, approach, n_l^m)
        
        # Aggregation
        if approach == 'NA-FedAvg' and r > R_w:
            # Noise-aware weighted averaging
            θ_G ← Σ (1 - n_l^m) × (N_m / N) × θ_m
        else:
            # Standard FedAvg
            θ_G ← Σ (N_m / N) × θ_m
    
    return θ_G

# CLIENT SIDE
def ClientUpdate(θ_G, m, approach, n_l):
    θ_m ← θ_G
    
    if approach == 'NNC':
        # Label correction (done once at initialization)
        D_m ← correct_labels_kNN(D_m, embeddings)
    
    for epoch in range(E):
        for batch in D_m:
            if approach == 'AKD' and n_l > ε:
                # Adaptive Knowledge Distillation
                loss ← L_CE + β × L_KD
            else:
                loss ← L_CE
            
            θ_m ← θ_m - η × ∇loss
    
    return θ_m

# NOISE ESTIMATION
def estimate_noise_embeddings(m):
    embeddings ← extract_embeddings(D_m, pretrained_model)
    mismatches ← 0
    
    for i, (x_i, y_i) in enumerate(D_m):
        # Find k nearest neighbors
        neighbors ← kNN(embeddings[i], embeddings, k=100)
        y_vote ← majority_vote(neighbors)
        
        if y_vote != y_i:
            mismatches += 1
    
    return mismatches / len(D_m)

def estimate_noise_scores(S_θG, S_θm):
    # Compute threshold from global model scores
    τ_ν ← percentile(S_θG, 75)
    
    # Count samples below threshold
    noisy_count ← sum(s < τ_ν for s in S_θm)
    
    return noisy_count / len(S_θm)
```

---

## Data Characteristics

### Datasets Used:

#### Vision:
1. **CIFAR-10:**
   - Classes: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
   - Samples: 50,000 train, 10,000 test
   - Task: Object detection

2. **Fashion-MNIST:**
   - Classes: 10 (clothing items)
   - Samples: 60,000 train, 10,000 test
   - Task: Clothing classification

3. **PathMNIST:**
   - Classes: 9 (pathology types)
   - Samples: Medical imaging
   - Task: Pathology reporting

4. **EuroSAT:**
   - Classes: 10 (land types)
   - Samples: Satellite imagery
   - Task: Landmark classification

5. **CIFAR-10N / CIFAR-100N:**
   - Real human annotations from Amazon Mechanical Turk
   - Noise level: ~40% (naturally occurring)
   - Used for validation on real-world noise

#### Audio:
1. **Speech Commands (v2):**
   - Classes: 12 (target keywords)
   - Format: 1-second audio clips
   - Features: 64 log-Mel spectrograms (25ms window, 10ms hop)
   - Task: Keyword spotting

### Data Distribution:
- **Non-IID:** σ=25% variance across clients
- **Imbalanced:** Random partitioning with controlled variance
- **Heterogeneous noise:** Each client has unique noise matrix Q^m

### Noise Injection Process:
1. Construct noise matrix Q_{y|y*} per client based on (nl, ns)
2. For each sample with true label y*:
   - Flip to label y with probability Q_{y|y*}
3. Result: Synthetic controllable noise for experiments

---

## Code Repository Information

**GitHub:** https://github.com/FederatedML/FedLN

### Expected Structure:
- **FL Framework:** Flower (v1.0+)
- **Main training scripts:**
  - `fedln_ncc.py` - Nearest Neighbor Correction approach
  - `fedln_akd.py` - Adaptive Knowledge Distillation approach
  - `fedln_nafedavg.py` - Noise-Aware FedAvg approach
  
- **Noise utilities:**
  - `noise_injection.py` - Generate Q matrices, inject label noise
  - `noise_estimation.py` - Embedding-based and score-based detection
  
- **Models:**
  - `resnet.py` - ResNet-20 for vision
  - `audio_cnn.py` - 4-block CNN for audio
  - `convmixer.py` - ConvMixer-128/4 alternative
  
- **Baselines:**
  - `label_smoothing.py` - LS regularization
  - `bitempered_loss.py` - Bi-tempered loss
  - `confidence_learning.py` - CL data pruning
  - `fedcorr.py` - Multi-stage FedCorr implementation
  
- **Pre-trained models:**
  - CLIP ViT-B/32 (vision)
  - TRILLsson v3 (audio)
  
- **Experiments:**
  - `run_synthetic_noise.py` - Controlled noise experiments
  - `run_real_noise.py` - CIFAR-10N/100N experiments
  - `run_ablations.py` - Architecture, noise profile variations

---

## Relation to Other Papers

### Label Noise Learning (Centralized):

1. **Confident Learning (CL)** - Northcutt et al., 2019
   - Prunes noisy samples via probabilistic thresholds
   - **FedLN advantage:** Better detection (energy > softmax), no data pruning

2. **GAIN** - Yoon et al., 2018
   - Generative adversarial imputation for missing data
   - **Difference:** GAIN handles missing values, FedLN handles label noise

3. **Bi-Tempered Loss** - Amid et al., 2019
   - Robust loss with heavy-tailed softmax
   - **FedLN advantage:** Consistent across noise profiles (bi-temp inconsistent)

4. **Label Smoothing** - Müller et al., 2020
   - Regularization via soft labels
   - **FedLN advantage:** Active correction vs passive regularization

5. **Co-Learning** - Song et al., 2020
   - Cooperative supervised + self-supervised learning
   - **Potential integration:** Combine with AKD distillation

### Label Noise in FL:

1. **FedCorr** - Xu et al., 2022
   - Multi-stage: GMM detection → clean training → pseudo-labeling → standard FL
   - **FedLN advantages:**
     - Simpler (single-stage)
     - More efficient (200 vs 250 rounds)
     - Lower on-device compute (no GMM)
   - **FedCorr advantages:**
     - Pseudo-labeling may help extreme noise

2. **CLC** - Zeng et al., 2022
   - Consensus-based label correction via client cooperation
   - **FedLN advantage:** No additional client communication (CLC exchanges corrections)

3. **FedDR-Filter** - Duan et al., 2022
   - Communicates data features to server for filtering
   - **FedLN advantage:** No client-sensitive data exchange (privacy)

4. **RFL** - Yang et al., 2022
   - Communicates class centroids for decision boundaries
   - **FedLN advantage:** No feature sharing, works with heterogeneous noise

5. **FedNoisy** - Fang & Ye, 2022
   - Assumes server has clean validation set
   - **FedLN advantage:** No server-side clean data required

### Self-Supervised Pre-training:

1. **CLIP** - Radford et al., 2021
   - Vision-language pre-training
   - **FedLN use:** Embedding extraction for NNC/AKD

2. **TRILLsson** - Shor & Venugopalan, 2022
   - Distilled audio representations
   - **FedLN use:** Audio embeddings for Speech Commands

### Novel Aspects:

1. **First** to provide compute-adaptive solutions (3 approaches for different devices)
2. **First** to use energy scores for label noise detection in FL
3. **First** comprehensive framework handling noise at initialization, training, and aggregation
4. **First** to validate on real human annotation errors (CIFAR-10N/100N) in FL
5. Shows embeddings robust to label noise (not previously demonstrated in FL context)

---

## Practical Impact

### Who Benefits:

1. **Mobile device users:**
   - Keyboard query suggestions (Gboard)
   - Voice assistants (Siri, Alexa)
   - Photo organization (automatic tagging)

2. **Wearable users:**
   - Health monitoring (activity recognition)
   - Sleep tracking
   - Fall detection

3. **IoT deployments:**
   - Smart home devices
   - Industrial sensors
   - Autonomous vehicles

4. **Privacy-sensitive domains:**
   - Healthcare (patient data labeling)
   - Finance (transaction classification)
   - Legal (document categorization)

### Real-World Scenarios:

1. **Weak supervision:** Programmatic labeling functions (e.g., heuristics) introduce noise
2. **User annotation:** Non-expert users label data through interaction
3. **Automatic labeling:** Pre-trained models generate noisy pseudo-labels
4. **Crowdsourcing:** Distributed annotation with varying expertise levels
5. **Long-tail classes:** Rare categories more susceptible to mislabeling

### Deployment Considerations:

| Device Type | Recommended Approach | Rationale |
|-------------|---------------------|-----------|
| Wearables, Edge | NA-FedAvg | Minimal compute, battery-efficient |
| Smartphones (mid) | AKD (Logits) | No pre-trained model storage needed |
| Smartphones (high) | NNC or AKD (Emb) | Best accuracy, sufficient resources |
| Tablets, Laptops | NNC | Maximum accuracy, no constraints |

---

## Classification

- **Domain:** Federated Learning, Robust Machine Learning
- **ML Task:** Multi-class classification with noisy labels
- **Learning Paradigm:** Federated learning (horizontal)
- **Model Types:** ResNet-20, Custom CNN, ConvMixer
- **Data Modality:** Vision (images), Audio (spectrograms)
- **Privacy Method:** Secure aggregation, no raw data sharing
- **Robustness:** Label noise, non-IID data, heterogeneous clients
- **Evaluation:** Classification accuracy on clean test sets
- **Simulation:** Flower framework with 30 clients, 80% participation

---

## Citations & Impact

- **Published:** ACM Transactions on Intelligent Systems and Technology (TIST)
- **Journal Rank:** Q1, ACM Premier Journal
- **Year:** 2024 (accepted Aug 2023, published Feb 2024)
- **Citations:** 18 (as of Feb 2, 2026)
- **Downloads:** 2,451
- **Open Access:** Supported by Eindhoven University of Technology
- **Institution:** Eindhoven University of Technology (TU/e), Netherlands
- **Funding:** ECSEL Joint Undertaking (Distributed Artificial Intelligent Systems project)
- **Code:** Open-sourced on GitHub with permissive license

---

## Personal Notes

### Strengths:

1. **Comprehensive framework:** Three distinct approaches for different device capabilities (excellent practical consideration)

2. **Thorough evaluation:**
   - 5 datasets (vision + audio)
   - Multiple noise profiles (nl × ns combinations)
   - Real-world validation (CIFAR-10N/100N)
   - 3 trials per setting for statistical robustness

3. **Strong baselines:** Compares against both centralized (LS, Bi-Temp, CL) and FL-specific (FedCorr) methods

4. **Practical insights:**
   - Energy scores vs softmax (Figure 5)
   - Optimal timing for noise estimation (R_w=30)
   - Clear separation in score distributions (Figure 6)
   - Failure mode analysis (high sparsity class-flipping)

5. **Reproducibility:** Open-source code, detailed hyperparameters, standard datasets

6. **Privacy-preserving:** No client data or features shared (unlike FedDR-Filter, RFL)

7. **Novel findings:**
   - Embeddings robust to label noise in FL
   - Energy scores effective for noise detection
   - NA-FedAvg competitive with minimal overhead

### Weaknesses:

1. **Limited scale:** 30 clients (practical FL deployments: 100s-1000s)

2. **Pre-trained model assumption:** NNC/AKD require domain-specific models
   - Not all domains have CLIP/TRILLsson equivalents
   - Storage/download overhead for resource-constrained devices

3. **NA-FedAvg requires clean clients:** Fails when F=100% (Table 5)
   - Unclear how to detect this scenario in practice
   - No fallback mechanism proposed

4. **Synchronous aggregation in NA-FedAvg:** All clients must participate in R_w
   - Mentioned as "can be asynchronous" but not demonstrated

5. **Byzantine attacks not addressed:** Malicious clients could:
   - Provide false noise estimates
   - Poison embeddings/scores
   - Target specific classes

6. **Communication cost not analyzed:**
   - Scoring set sizes for NA-FedAvg
   - Embedding sizes if transmitted (though not needed in current design)

7. **Hyperparameter sensitivity:**
   - k=100 for kNN (not ablated)
   - R_w=30 (tested only on one dataset, Figure 5)
   - ν=75 percentile (not justified)
   - β=10, T=2 for AKD (mentioned as "working well" without ablation)

8. **Class imbalance not explored:** All experiments assume balanced classes
   - Real FL often has long-tail distributions
   - Noise + imbalance interactions unknown

### Potential Extensions:

1. **Hybrid FedLN:**
   - Combine all three approaches in adaptive framework
   - Start with NA-FedAvg, escalate to AKD/NNC if noise detected
   - Device capability-aware selection

2. **Byzantine robustness:**
   - Robust aggregation (Krum, Median, Trimmed Mean)
   - Detect malicious noise estimates
   - Sybil attack resistance

3. **Adaptive hyperparameters:**
   - Learn R_w, k, ν, β from validation performance
   - Per-client adaptive thresholds

4. **Class-imbalanced scenarios:**
   - Per-class noise estimation
   - Long-tail robustness
   - Few-shot learning integration

5. **Vertical FL extension:**
   - Label noise when labels held by single party
   - Split learning with noisy annotations

6. **Semi-supervised FL:**
   - Combine with unlabeled data
   - Pseudo-labeling with noise-aware confidence

7. **Continual learning:**
   - Non-stationary noise distributions
   - Concept drift + label noise

8. **Differential privacy:**
   - DP-FedAvg with noise estimation
   - Privacy-utility tradeoff analysis

9. **Communication efficiency:**
   - Quantize scoring sets
   - Compress embeddings if transmitted
   - Gradient compression (TopK, sparsification)

10. **Hardware efficiency:**
    - On-device benchmarks (latency, energy, memory)
    - Model quantization (INT8)
    - Pruning for resource-constrained devices

11. **Domain adaptation:**
    - Transfer learning for embeddings
    - Few-shot adaptation of pre-trained models
    - Domain-specific noise patterns

12. **Active learning:**
    - Query labels for high-uncertainty samples
    - User feedback on corrections
    - Prioritize clients for annotation

### Questions for Authors:

1. How to detect F=100% scenario in practice? (NA-FedAvg failure mode)
2. Sensitivity to R_w across different datasets/models?
3. Asynchronous NA-FedAvg implementation details?
4. Hyperparameter selection guidelines (k, ν, β, T)?
5. Computational cost breakdown (pre-trained inference, kNN search, energy scores)?
6. Byzantine robustness evaluation?
7. Real deployment experience (if any)?

### Overall Assessment:

**Excellent contribution** to federated learning under label noise. Key strengths:
- **Practical focus:** Device capability considerations rare in FL literature
- **Comprehensive:** Multiple approaches, extensive evaluation
- **Reproducible:** Open-source code, standard datasets
- **Novel insights:** Embeddings robustness, energy scores, minimal-overhead NA-FedAvg

**Primary limitations:**
- Scale (30 clients)
- Pre-trained model dependency (NNC/AKD)
- NA-FedAvg requires clean clients

**Recommended for:**
- Researchers: Strong baseline for FL label noise
- Practitioners: Deployment-ready approaches (choose based on device capability)
- Educators: Clear presentation of FL challenges and solutions

**Impact:** High potential for real-world FL deployments (Gboard, mobile sensing, IoT). Addresses critical practical challenge (user-generated noisy labels) with pragmatic solutions.


