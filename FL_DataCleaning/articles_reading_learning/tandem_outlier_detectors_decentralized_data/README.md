# Paper Analysis: Tandem Outlier Detectors for Decentralized Data

## Authors
- Marco Heyden (marco.heyden@kit.edu) - Karlsruhe Institute of Technology
- Jürgen Wilwer (juergen.wilwer@kit.edu) - Karlsruhe Institute of Technology
- Edouard Fouché (edouard.fouche@kit.edu) - Karlsruhe Institute of Technology
- Vadim Arzamasov (vadim.arzamasov@kit.edu) - Karlsruhe Institute of Technology
- Steffen Thoma (thoma@fzi.de) - FZI Research Center for Information Technology
- Sven Matthiesen (sven.matthiesen@kit.edu) - Karlsruhe Institute of Technology
- Thomas Gwosch (thomas.gwosch@kit.edu) - Karlsruhe Institute of Technology

## Publication Details
- **Conference**: SSDBM 2022 (34th International Conference on Scientific and Statistical Database Management)
- **Date**: July 6-8, 2022, Copenhagen, Denmark
- **DOI**: https://doi.org/10.1145/3538712.3538748
- **Citations**: 1 (as of January 31, 2026)
- **Downloads**: 88
- **Type**: Short Paper

---

## Summary Table

| ML or FL | Handles fairness? | Outliers | Missing values | Label errors | Repair data? | Topics | Datasets | Code | Implemented? |
|----------|-------------------|----------|----------------|--------------|--------------|--------|----------|------|--------------|
| **FL** (Federated Learning) | **No** | **Yes** (Local, Global, and Partition outliers) | **No** | **No** | **No** | Outlier detection, Federated Learning, Edge Computing, Autoencoders, Unsupervised Learning, Privacy-preserving | Synthetic: 10-component Gaussian Mixture Model (30 clients, 1000 obs each, 10 dimensions); Real-world: Power Tool Study (15 participants, cordless screwdriver, 1665 features from battery current/voltage, IMU data) | **Yes** - GitHub: https://github.com/heymarco/TandemOutlierDetection | **Yes** (by authors) |

---

## Detailed Analysis

### 1. ML or FL Approach
**Federated Learning (FL)**
- Uses Federated Averaging approach with Autoencoders (AEs)
- Each device trains locally (one epoch per round)
- 20 communication rounds with batch size B = 32
- No raw data is exchanged between devices
- Privacy-preserving approach

### 2. Handles Fairness?
**No**
- The paper does not address fairness issues
- Focus is on distinguishing outlier types, not on fairness across devices or populations
- No discussion of bias, equity, or fairness metrics

### 3. Outliers
**Yes - Primary Focus**

The paper's main contribution is detecting **three types of outliers**:

#### a) **Global Outliers**
- Definition: Observations that deviate from data on ALL devices
- Example: Mistakes in power tool operation (e.g., drilling in unintended material)
- Detection: Identified by both federated (F) and local (Li) detectors
- Condition: `os_F > ε_F AND os_L > ε_L`

#### b) **Local Outliers**
- Definition: Observations that deviate from their partition but NOT from other devices' data
- Example: Using a drill meant for wood on metal (anomalous on that device, but normal elsewhere)
- Detection: Identified by local detector only
- Condition: `os_L > ε_L AND os_F ≤ ε_F`

#### c) **Partition Outliers**
- Definition: An entire partition (device's dataset) is outlying compared to other devices
- Example: Unskilled worker making systematic mistakes (all their data deviates)
- Detection: Uses decentralized Mann-Whitney-U test on aggregated outlier scores
- Method: Compare outlier score distributions across devices

#### Detection Method: **"Tandem" Approach**
- Combines two outlier detectors per device:
  - **F (Federated)**: Trained on global data via FL, identifies global outliers
  - **Li (Local)**: Trained only on local partition, identifies local + global outliers
- Key insight: Combining both allows discrimination between outlier types

#### Technical Implementation:
- Uses Autoencoders (AEs) for both F and Li
- Architecture: One hidden layer of size η · d (η = 0.7 synthetic, η = 0.4 real data)
- Activation: ReLu (hidden), Sigmoid/Linear (output)
- Thresholds: ε_F and ε_L set to 96th percentiles of outlier scores

#### Performance (Synthetic Data):
- Outperforms pure federated and pure local approaches
- Higher precision and recall for mixed outlier scenarios
- F1 scores improve when combining both detectors

### 4. Missing Values
**No**
- Not addressed in the paper
- No discussion of handling missing data
- Assumes complete sensor readings

### 5. Label Errors
**No**
- The approach is **unsupervised**
- No labels are used (by design)
- Focus is on scenarios where "outlier labels are often hard to obtain"

### 6. Repair Data?
**No**
- Paper focuses on **detection**, not repair
- Identifies and classifies outliers but does not propose correction methods
- Goal is to find mistakes/anomalies, not fix them

### 7. Topics Covered

#### Primary Topics:
1. **Outlier/Anomaly Detection** - Main focus
2. **Federated Learning** - Core methodology
3. **Edge Computing** - Application context
4. **Unsupervised Learning** - No labels required
5. **Privacy-Preserving Machine Learning** - No raw data exchange

#### Technical Components:
- **Autoencoders (AEs)** - Neural network for outlier detection
- **Federated Averaging** - Training protocol
- **Mann-Whitney-U Test** - Statistical test for partition outliers
- **Decentralized Data Analysis** - Processing at the edge

#### Application Domains:
- **Smart Power Tools** - Main case study
- **Predictive Maintenance**
- **Energy Consumption Analysis**
- **IoT/Smart Devices**

### 8. Datasets

#### A) **Synthetic Data**
- **Type**: Gaussian Mixture Model (GMM)
- **Configuration**:
  - 10-component GMM representing global distribution X_G
  - 10 dimensions
  - 30 clients (devices)
  - 1,000 observations per client (|di| = 1000)
  - Each client samples from 5 randomly chosen components (representing Xi)
  - Total global dataset D = 30,000 observations
- **Outliers**:
  - 4% outliers per partition
  - Variable ratio between local and global outliers (6 experiments)
  - Local outliers: sampled from patterns not in local data
  - Global outliers: random observations
- **Purpose**: Ablation study and comparison with baselines

#### B) **Real-World Data: Power Tool Study**
- **Type**: User study with sensor data
- **Participants**: 15 individuals
- **Device**: Cordless screwdriver PDC 18/4 Quaddrive by Festool
- **Sensors**:
  - Data logger measuring battery current and voltage
  - Inertial Measurement Units (IMUs) for accelerations and angular velocities
- **Features**: 1,665 features extracted using TSFEL (Time Series Feature Extraction Library)
- **Preprocessing**: Removed inactive phases of power tool
- **Partition Sizes**: |di| ∈ [240, 718] observations per participant
- **Use Case**: Detecting mistakes in power tool operation
  - Normal screwing: characteristic current pattern with spike when screw enters wood
  - Global anomaly: operator slipped off screw head
  - Local anomaly: minor deviations from usual patterns
  - Partition outlier: non-professional usage patterns (e.g., participant c4)
- **Availability**: **Open source** (GitHub link provided)
- **Video Analysis**: Used to confirm identified anomalies

### 9. Code Availability
**Yes - Open Source**
- **Repository**: https://github.com/heymarco/TandemOutlierDetection
- **Contents**: 
  - Code for all experiments
  - Open source power tool dataset
  - Reproducibility ensured
- **Language**: Likely Python (uses neural networks, standard ML libraries)
- **Dependencies** (inferred):
  - Autoencoders implementation
  - Federated Learning framework
  - TSFEL library for feature extraction
  - Statistical testing libraries (Mann-Whitney-U)

### 10. Implemented?
**Yes - Fully Implemented by Authors**
- Complete implementation available on GitHub
- Tested on both synthetic and real-world data
- Real user study conducted with 15 participants
- Results published with reproducible experiments
- 10 repetitions with different random seeds for robustness

---

## Key Contributions

1. **Novel Outlier Taxonomy**: First to distinguish local, global, and partition outliers in decentralized data
2. **Tandem Detection Framework**: Combines local and federated outlier detection
3. **Privacy-Preserving**: Only shares aggregated outlier scores, not raw data
4. **Real-World Validation**: User study with power tools demonstrates practical applicability
5. **Open Science**: Dataset and code publicly available

---

## Methodology Details

### Notation
- Network N = {c1, c2, ..., c|N|} with |N| devices (clients)
- Each client ci has partition di = {xi1, xi2, ..., xi|di|}
- Global dataset D = d1 ∪ d2 ∪ ... ∪ d|N|
- Observations xij ∈ Rd (d-dimensional)
- Xi: underlying distribution for partition i
- XG: global distribution

### Outlier Score Distributions
- F returns os_F (outlier scores from federated detector)
- Li returns os_L (outlier scores from local detector)
- Scores sampled from unknown distributions OS_F and OS_L

### Partition Outlier Detection Process
1. **Score Aggregation**: 
   - Each client divides sorted federated scores into bins of size b
   - Computes mean for each bin → os*i
   - Reduces data transmission b-fold
   - Masks information about individual point outliers
   
2. **Server Evaluation**:
   - Computes p-value using Mann-Whitney-U statistic
   - Compares os*i against (os*1 ∪ os*2 ∪ ... ∪ os*|N|) \ os*i
   
3. **Client Evaluation**:
   - Checks if pi < α (significance level, typically 0.05)
   - Trade-off: larger b = better compression, smaller b = better statistical power
   - Recommendation: b = 10 for small partitions, b > 100 for large partitions

---

## Experimental Results

### Synthetic Data (Local & Global Outliers)
- **Setup**: 4% total outliers, varying ratio local:global
- **Metrics**: Precision, Recall, F1 score
- **Finding**: Tandem outperforms both pure local and pure federated approaches
- **Surprise**: Tandem even beats federated-only when data has only global outliers

### Synthetic Data (Partition Outliers)
- **Test**: Shift data of c0 by various standard deviations
- **Results**: 
  - High p-values when shift = 0.0 (correct inlier identification)
  - p decreases as shift increases (correctly detects outliers)
  - More sensitive with larger |di| and smaller b

### Power Tool Data
- **Local Outlier**: Minor deviations in usage pattern (would have been false alarm without global context)
- **Global Outlier**: Operator slipped off screw head (absent characteristic pattern)
- **Partition Outlier**: Participant c4 identified as significantly different (non-professional usage)

---

## Limitations and Future Work

### Stated by Authors:
- **Future Extensions**:
  - Data stream setting (currently batch)
  - Applications in healthcare and energy domains

### Implicit Limitations:
- Short paper format (4 pages) - limited depth
- Small real-world study (15 participants)
- Single application domain demonstrated (power tools)
- Requires setting multiple hyperparameters (ε_F, ε_L, α, b)
- Does not handle missing data
- Does not repair detected outliers

---

## Related Work Comparison

### Key Differences from Existing Approaches:
1. **vs. Centralized Approaches**: Don't transfer raw data (privacy-preserving)
2. **vs. Local-Only Approaches**: Leverage information from peer devices
3. **vs. Pure FL Outlier Detection**: Can distinguish between outlier types
4. **vs. Wireless Sensor Networks**: Handles multivariate data, mobile devices, privacy

### Unique Contribution:
"To the best of our knowledge, our method is the first to achieve this [distinguishing local, global, and partition outliers]."

---

## CCS Concepts
- Computing methodologies → Anomaly detection
- Distributed artificial intelligence

## Keywords
- outlier detection
- federated learning
- edge computing

