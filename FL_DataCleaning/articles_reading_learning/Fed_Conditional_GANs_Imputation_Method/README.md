# Analysis: Federated Conditional Generative Adversarial Nets Imputation Method for Air Quality Missing Data

**Authors:** Xu Zhou, Xiaofeng Liu, Gongjin Lan, Jian Wu  
**Published:** Knowledge-Based Systems 228 (2021) 107261  
**DOI:** https://doi.org/10.1016/j.knosys.2021.107261

---

## Summary Table

| Column | Answer | Details |
|--------|--------|---------|
| **ML or FL** | **FL (Federated Learning)** | Uses horizontal federated learning with client-server architecture. First paper to propose conditional GAN imputation under federated learning framework. |
| **Handles fairness?** | **No** | Paper does not address fairness, equity, or bias concerns. Focus is on privacy-preserving collaboration and imputation accuracy. |
| **Outliers** | **No** | Paper does not address outlier detection or handling. Mentions outliers occur in spatial air quality data but does not develop methods to handle them. |
| **Missing values** | **✓ Yes (Primary Focus)** | Core contribution: imputes missing air quality data using conditional GAN. Handles MAR (Missing At Random) assumption. Addresses varying missing rates (5%-40%), different gap lengths, and missing patterns. |
| **Label errors** | **No** | Not addressed. This is an unsupervised imputation problem, not a classification task with labels. |
| **Repair data?** | **✓ Yes** | Generates/imputes missing values to complete the dataset. Uses conditional GAN with Wasserstein distance and "Hint mask" trick to repair incomplete air quality measurements. |
| **Topics** | **Air quality monitoring, Missing data imputation, Generative Adversarial Networks (GANs), Conditional GANs, Federated Learning, Privacy-preserving ML, Non-IID data, Wasserstein distance, Horizontal FL** | Addresses privacy constraints where organizations cannot share raw monitoring data due to security, competition, and regulations. |
| **Datasets** | **Real-world air quality data from Changzhou, China (2016)** | 6 AQM stations (A-Central, B-ZhongLou, C-XinBei, D-WuJin, E-AnJia, F-JinKai). Uses 3 stations (A, B, E) in experiments. Hourly measurements of 6 pollutants: NOx, O3, SO2, CO, PM10, PM2.5. Total ~8554 rows. Missing rates: 0.349%-0.658%. |
| **Code** | **✓ Yes - Available on GitHub** | Repository: `zxecho/FGAN_for_air_quality_data_imputation` (mentioned in paper line 51). Contains implementation of FCGAI algorithm, FedAvg, GAIN baseline, and conditional GAN models. |
| **Implemented?** | **✓ Yes - Fully implemented and tested** | Extensive experiments on real data with 3 runs per experiment, 5 test sets. Compares FCGAI vs local GAN, GAIN, EM, MICE, linear/spline interpolation, nearest neighbor. RMSE used as evaluation metric. |

---

## Key Findings

### 1. **Problem Definition**
- **Challenge:** Air quality monitoring networks collect data with missing values, but data cannot be shared between organizations due to:
  - Privacy concerns
  - Security regulations
  - Industrial competition
  - Government restrictions
- **Traditional GAN limitation:** Requires large centralized datasets with IID distribution
- **Solution:** Federated learning enables collaborative training without data exchange

### 2. **Methodology: FCGAI (Federated Conditional Generative Adversarial Imputation)**

#### Architecture Components:
1. **Generator (G):**
   - Input: Observed data (X_obs), mask (M), noise (Z)
   - Output: Imputed complete data
   - Loss: Combines adversarial loss + MSE loss (α=10)
   - 3-layer FCN with ReLU activation, Sigmoid output

2. **Discriminator (D):**
   - Input: Imputed data + Hint mask
   - Output: Probability of distinguishing real vs imputed components
   - 3-layer FCN with LeakyReLU(0.2)
   - Uses "Hint mask" trick: provides partial mask information (p_hint=0.9) to prevent overfitting

3. **Improvements over vanilla conditional GAN:**
   - **Wasserstein distance with gradient penalty (λ=10):** Improves training stability, prevents mode collapse
   - **Hint mechanism:** Discriminator receives sampled mask (not full mask) to force generator to learn better distribution

#### Federated Learning Framework:
- **Type:** Horizontal federated learning (same features, different samples)
- **Architecture:** Client-server (C/S)
- **Algorithm:** FedAvg (Federated Averaging)
- **Process:**
  1. N participants train local conditional GAN models
  2. After Loc_b=10 local epochs, send encrypted model parameters to server
  3. Server aggregates parameters: w^(t+1) = Σ(w̄_n × w_n)
  4. Server returns aggregated parameters to clients
  5. Clients update local models and continue training
  6. Repeat for E=300 epochs

### 3. **Experimental Setup**

#### Hyperparameters:
- Epochs: 2,000 (local GAN) vs 300 (federated, with 10 local updates each)
- Learning rate: 1×10^-4 (Adam optimizer)
- Hidden layer dim: 64
- Minibatch size: 64
- Noise input dim: 128
- α (MSE weight): 10
- λ (Wasserstein GP): 10
- p_hint: 0.9

#### Experimental Scenarios:
1. **Single location (Station A only):**
   - Simulates 3 participants with different sampling frequencies (2, 3, 5)
   - Different missing rates: (5%, 10%, 15%), (5%, 15%, 30%), (5%, 20%, 30%)
   - Tests non-IID conditions

2. **Multiple locations (Stations A, B, E):**
   - Simulates 3 participants from different geographical locations
   - Same missing rate configurations
   - Tests geographical distribution effects

### 4. **Results**

#### Performance (RMSE):
- **FCGAI outperforms:**
  - Classic methods: Linear interpolation, spline, nearest neighbor, EM, MICE
  - MissForest, Matrix Completion
  - GAIN (reproduced baseline)
  - **Local GAN models** (key finding!)

#### Single Location Results (Station A):
| Missing Rates | Participant | Local GAN RMSE | Fed GAN RMSE | Improvement |
|---------------|-------------|----------------|--------------|-------------|
| 5%, 10%, 15% | P1 | 0.0659(±0.0066) | 0.0650(±0.0067) | Better |
| | P2 | 0.0628(±0.0069) | 0.0607(±0.0069) | **Significant** |
| | P3 | 0.0607(±0.0031) | 0.0582(±0.0023) | **Significant** |

#### Multi-Location Results:
| Stations | Participant | Local GAN RMSE | Fed GAN RMSE | Improvement |
|----------|-------------|----------------|--------------|-------------|
| A:5%, B:10%, E:15% | P1 | 0.0618 | 0.0615 | Slight |
| | P2 | 0.0576 | 0.0565 | Better |
| | P3 | 0.0586 | 0.0570 | **Significant** |
| A:5%, B:20%, E:30% | P3 | 0.0655 | 0.0605 | **Large (7.6%)** |

#### Key Observations:
1. **Federated model more stable:** Lower variance in RMSE across runs
2. **Training stability:** Federated training curves much smoother than local training (less oscillation)
3. **Helps poor data holders:** Participants with high missing rates (P3 typically) benefit most from federated approach
4. **Geographical distribution:** Multi-location data improves overall performance compared to single location

### 5. **Technical Contributions**

1. **First federated GAN for missing data imputation** in air quality monitoring
2. **Conditional GAN adaptation:** Generator conditioned on observed data, discriminator predicts mask
3. **Wasserstein distance + Hint mask:** Improves training stability and convergence
4. **Privacy-preserving:** No raw data exchange, only encrypted model parameters
5. **Handles non-IID data:** Different sampling frequencies, missing patterns, data quality levels

### 6. **Limitations & Future Work**

- **Time continuity not fully exploited:** Future work to incorporate RNN/LSTM for temporal patterns
- **Assumes MAR:** Does not handle MNAR (Missing Not At Random)
- **Honest-but-curious server:** Security model assumes server won't collude
- **Communication overhead:** Requires multiple rounds of parameter exchange

---

## Comparison with Baseline Methods

| Method | Type | RMSE @ 10% missing | Notes |
|--------|------|-------------------|-------|
| Linear Interpolation | Classic | ~0.08-0.09 | Simple baseline |
| Spline | Classic | ~0.08 | Cubic polynomial fitting |
| Nearest Neighbor | Classic | ~0.08 | Uses gap endpoints |
| EM | Statistical | ~0.065 | Iterative expectation-maximization |
| MICE | Multiple Imputation | ~0.065 | Chained equations |
| MissForest | ML | ~0.07 | Random forest-based |
| Matrix Completion | ML | ~0.07 | Low-rank approximation |
| GAIN (reproduced) | Deep Learning | ~0.065 | GAN-based imputation |
| **Local Conditional GAN** | Deep Learning | ~0.063 | Authors' non-federated version |
| **FCGAI (Proposed)** | Federated DL | **~0.058-0.061** | **Best performance** |

---

## Data Characteristics

### Stations Used:
- **Station A (Central):** Total missing rate 0.658%, row-wise 0.319%, column-wise 0.465%
- **Station B (ZhongLou):** Total missing rate 0.457%, row-wise 0.387%, column-wise 0.321%
- **Station E (AnJia):** Total missing rate 0.349%, row-wise 0.125%, column-wise 0.222%

### Missing Patterns:
- Missing completely at random (MCAR)
- Missing at random (MAR) - paper's assumption
- Different gap lengths (continuous missing segments)
- Different sampling frequencies across participants

### Pollutants Measured:
- PM2.5, PM10, SO2, CO, O3, NOx
- Hourly measurements
- Year: 2016
- Location: Changzhou, China

---

## Code Repository Information

**GitHub:** zxecho/FGAN_for_air_quality_data_imputation

Expected files based on paper:
- `Fed_gain_main.py` - Main federated training script
- `FedAvg.py` - Federated averaging algorithm implementation
- `GAIN_model.py` - Baseline GAIN implementation
- `CGAI_model.py` - Conditional GAN imputation model
- Algorithm 1: Conditional GAN imputation (lines 265-348)
- Algorithm 2: Federated Averaging (lines 313-375)
- Algorithm 3: FCGAI (lines 380-413)

---

## Relation to Other Papers

### Related Work Referenced:
1. **GAIN** (Yoon et al., 2018): Generative Adversarial Imputation Nets - baseline method
2. **WGAN** (Arjovsky et al., 2017): Wasserstein GAN for improved training
3. **Conditional GAN** (Mirza & Osindero, 2014): Extended GAN with conditional information
4. **FedAvg** (McMahan et al., 2017): Federated averaging algorithm
5. **E2GAN** (Luo et al., 2019): End-to-end GAN for time series imputation

### Novel Aspects:
- **First** to combine conditional GAN + federated learning for missing data
- **First** federated approach for air quality imputation
- Shows federated training improves GAN stability (unexpected finding)

---

## Practical Impact

### Who Benefits:
1. **Poor data holders:** Organizations with limited/low-quality data can improve models
2. **Privacy-sensitive organizations:** Railway stations, hospitals, secret units with proprietary data
3. **Multi-institutional collaborations:** Government + companies + research institutions

### Use Cases:
- Air quality monitoring networks
- Environmental sensor networks
- IoT device networks with privacy constraints
- Healthcare data (mentioned as related application)
- Distributed edge computing scenarios

---

## Classification

- **Domain:** Environmental monitoring, Air quality
- **ML Task:** Missing data imputation (unsupervised generative modeling)
- **Learning Paradigm:** Federated learning (horizontal)
- **Model Type:** Generative Adversarial Network (conditional)
- **Data Type:** Multivariate time series
- **Privacy Method:** Secure aggregation, encrypted parameter exchange
- **Evaluation:** Root Mean Square Error (RMSE)

---

## Citations & Impact

- **Published:** Knowledge-Based Systems (Q1 journal, IF ~8.0)
- **Year:** 2021
- **Research Groups:** Hohai University (China), Southern University of Science and Technology, VU University Amsterdam
- **Funding:** National Key R&D Program, Jiangsu grants, Guangdong Forestry, Changzhou International Cooperation

---

## Personal Notes

### Strengths:
1. Clear problem motivation (real-world privacy constraints)
2. Solid experimental design with multiple scenarios
3. Ablation-like comparisons (single vs multi-location, different missing rates)
4. Unexpected finding about federated training stability
5. Open-sourced code repository

### Weaknesses:
1. Small scale (3 participants, 1 city, 6 stations)
2. Assumes honest-but-curious server (not Byzantine-robust)
3. Communication costs not analyzed
4. No temporal modeling (RNN/LSTM)
5. Only handles MAR, not MNAR
6. Does not address outliers or anomalies

### Potential Extensions:
1. Add LSTM/GRU for temporal dependencies
2. Scale to more participants and cities
3. Robust aggregation against Byzantine participants
4. Handle MNAR mechanisms
5. Integrate outlier detection
6. Differential privacy guarantees
7. Communication efficiency (compression, quantization)


