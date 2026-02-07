# Paper Analysis: Outlier Detection by Privacy-Preserving Ensemble Decision Tree Using Homomorphic Encryption

**Authors:** Kengo Itokazu, Lihua Wang, Seiichi Ozawa  
**Venue:** 2021 International Joint Conference on Neural Networks (IJCNN)  
**Year:** 2021

---

## Summary Table

| Category | Answer | Details |
|----------|--------|---------|
| **ML or FL** | **FL (Federated Learning)** | The paper extends Isolation Forest to a federated learning setting where multiple organizations collaboratively train a model without sharing raw data |
| **Handles fairness?** | **No** | Fairness is not discussed in this paper |
| **Outliers** | **Yes** | Core focus of the paper - detecting outliers/anomalies in horizontally partitioned data across multiple organizations |
| **Missing values** | **No** | Missing data imputation is not addressed |
| **Label errors** | **No** | Label noise or label correction is not addressed |
| **Repair data?** | **Yes - Detects outliers for removal** | The paper focuses on detecting and eliminating outliers to improve system performance. This is data-level repair (removing anomalous data points), not label repair |
| **Topics** | Privacy-preserving machine learning, Federated learning, Outlier detection, Decision tree ensemble, Isolation Forest, Homomorphic encryption, Anomaly detection |
| **Datasets** | Credit Card (fraud detection), Forest Cover, Shuttle, Annthyroid, Http (attack detection), Smtp (attack detection) | See Table I in paper for full details |
| **Code** | **No** | No code availability mentioned in the paper |
| **Implemented?** | **Yes** | Experimental results demonstrate the pp-iForest achieves stable performance across 1-8 organizations |

---

## Key Contributions

1. **Privacy-Preserving Isolation Forest (pp-iForest)**: Extends the classical Isolation Forest algorithm to a federated learning scenario using additive homomorphic encryption

2. **Security Model**: Multiple organizations can collaboratively detect outliers without:
   - Sharing raw data with each other
   - Revealing their data to the central server
   - Exposing complete model information to the server

3. **Performance**: Demonstrates that pp-iForest achieves comparable AUC performance to centralized Isolation Forest even when data is distributed across multiple organizations

---

## Technical Approach

### Core Algorithm: Isolation Forest
- **Principle**: Anomalies are easier to isolate than normal data (require fewer random partitions)
- **Method**: Ensemble of decision trees trained on random subsamples
- **Output**: Anomaly score from 0 to 1 (1 = likely outlier, 0 = likely normal)

### Privacy Mechanism: Additive Homomorphic Encryption
- Uses Microsoft SEAL library for encryption
- Allows addition operations on encrypted data: `Enc(m1) ⊕ Enc(m2) = Enc(m1 + m2)`
- Organizations encrypt partial model information before sending to server
- Server performs aggregations on encrypted data without decryption

### Federated Learning Protocol
1. Organizations {O1, ..., ON} and central server S
2. Each organization trains local decision trees on their data
3. Server randomly selects an organization to perform split
4. Selected organization encrypts split information (feature, threshold)
5. Server aggregates encrypted node statistics across organizations
6. All organizations update shared model simultaneously

---

## Experimental Results

### Datasets Evaluated
1. **Credit Card** (284,807 samples, 28 features): Fraud detection (0.17% anomalies)
2. **Forest Cover** (286,048 samples, 10 features): Class 4 vs Class 2 (0.9% anomalies)
3. **Shuttle** (49,097 samples, 9 features): Multiple anomaly classes (7% anomalies)
4. **Annthyroid** (7,200 samples, 6 features): Thyroid condition detection (7.4% anomalies)
5. **Http** (567,497 samples, 3 features): Network attack detection (0.4% anomalies)
6. **Smtp** (95,156 samples, 3 features): Email attack detection (0.03% anomalies)

### Key Findings

1. **Baseline Comparison** (Table II):
   - pp-iForest and iForest achieve comparable performance
   - Both significantly outperform LOF (Local Outlier Factor) and OCSVM (One-Class SVM)
   - Example (Credit Card): iForest=0.95, pp-iForest=0.95, LOF=0.78, OCSVM=0.52

2. **Scalability** (Number of Organizations):
   - Tested with N = 1, 2, 4, 8 organizations
   - Performance remains stable as organizations increase
   - Minimal AUC degradation even with 8 organizations
   - Example (Http): N=1: 1.00, N=2: 0.99, N=4: 0.99, N=8: 0.97

3. **Robustness to Data Distribution Bias** (Table III):
   - Tested with anomaly bias (50%, 60%, 70% concentration)
   - Tested with sample size bias (50%, 60%, 70% concentration)
   - Performance remains stable under both types of bias
   - Demonstrates practical applicability where organizations have different data characteristics

### Configuration
- Number of trees (NT): 100
- Training samples per tree (ψ): 256
- Maximum tree depth: log2(256) ≈ 8

---

## Important Distinctions

### Data Repair vs. Label Repair
- **This paper addresses: Data Repair** (outlier detection and elimination)
- **Not addressed: Label Repair** (correcting mislabeled training examples)

The paper detects anomalous data points that should be removed from the dataset to improve downstream system performance, not correcting incorrect labels on otherwise valid data points.

### Security Model
- **Threat Model**: Semi-honest adversaries (honest-but-curious)
- **Assumptions**:
  - No collusion among organizations
  - No collusion between organizations and server
  - Secure channels between each organization and server
- **Privacy Guarantees**:
  - Raw data never leaves organizations
  - Server cannot decrypt model parameters
  - Organizations cannot infer other organizations' data distributions

---

## Limitations and Future Work

### Limitations (implicit)
1. No code release for reproducibility
2. Semi-honest security model (not malicious adversaries)
3. Assumes no collusion between parties
4. Communication overhead not analyzed
5. Does not handle missing data or label noise

### Application Scenarios
- **Financial**: Fraud detection across multiple banks
- **Healthcare**: Anomaly detection in medical records across hospitals
- **Cybersecurity**: Network intrusion detection across organizations
- **Manufacturing**: Quality control across multiple factories

---

## Related Work References

1. **Isolation Forest**: Liu et al., 2008 - Original unsupervised outlier detection method
2. **Federated Learning**: Konečný et al., 2016 - Distributed optimization for on-device intelligence
3. **Privacy-Preserving LOF**: Li et al., 2015 - Privacy-preserving outlier detection
4. **Microsoft SEAL**: Microsoft Research, 2018 - Homomorphic encryption library
5. **LOF**: Breunig et al., 2000 - Local Outlier Factor algorithm
6. **OCSVM**: Khan & Madden, 2010 - One-Class SVM survey

---

## Key Takeaways

1. ✅ **Federated Learning**: Successfully extends Isolation Forest to federated setting
2. ✅ **Privacy Preservation**: Uses homomorphic encryption to protect data
3. ✅ **Performance**: Achieves comparable performance to centralized approach
4. ✅ **Scalability**: Works well with multiple organizations (1-8 tested)
5. ✅ **Robustness**: Stable under data distribution biases
6. ❌ **No fairness considerations**: Does not address fairness across organizations
7. ❌ **No missing data handling**: Assumes complete data
8. ❌ **No label noise handling**: Focuses on outlier detection, not label correction
9. ⚠️ **No code**: Implementation not publicly available

---

## Comparison with Other FL Data Cleaning Papers

### Unique Aspects of This Paper
- **Focus**: Outlier detection (not imputation or label correction)
- **Method**: Decision tree ensemble with homomorphic encryption
- **Setting**: Horizontally partitioned data (different individuals across organizations)
- **Privacy**: Cryptographic approach (not differential privacy)

### Complementary to Other Approaches
- Could be combined with missing data imputation methods (FedIMPUTE, Cafe)
- Could be applied before label noise correction (FedCorr, ChaosToHarmony)
- Addresses a different data quality issue than other FL data cleaning papers

