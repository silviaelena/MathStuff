# Cafe: Improved Federated Data Imputation by Leveraging Missing Data Heterogeneity - Paper Analysis

## Paper Information
**Title:** Cafe: Improved Federated Data Imputation by Leveraging Missing Data Heterogeneity

**Authors:** Sitao Min, Hafiz Asif, Xinyue Wang, Jaideep Vaidya (Rutgers University & Hofstra University)

**Publication:** IEEE (based on formatting)

## Summary Table

| Column | Value |
|--------|-------|
| **ML or FL** | FL (Federated Learning) |
| **Handles fairness?** | No - Not explicitly addressed |
| **Outliers** | No - Not the focus |
| **Missing values** | **Yes - Primary focus** |
| **Label errors** | No - Not addressed |
| **Repair data?** | **Yes - Imputes missing values** |
| **Topics** | Missing data imputation, Missing data heterogeneity, MNAR mechanisms, Personalized federated learning, Complementarity-adjusted averaging, ICE (Iterative Chained Equations), Data quality in FL |
| **Datasets** | Codon, Codrna, MIMICIII, Heart, Genetic, HHNP (Heritage Health Provider Networks), eICU (Philips eICU Collaborative Database) |
| **Code** | **Yes** - Available at https://github.com/sitaomin1994/federatedimputation |
| **Implemented?** | **Yes** - Full implementation with PyTorch 11.2, Sklearn, Ridge/Logistic Regression |

---

## Detailed Analysis

### 1. Core Problem & Contribution

**Problem Addressed:**
- How to handle missing values in federated learning settings when missing data mechanisms are:
  - Complex (MNAR - Missing Not At Random)
  - Heterogeneous across clients (different organizations have different reasons for missingness)
  - Current federated imputation methods suffer significant quality loss in these scenarios

**Key Innovation:**
The paper introduces **Cafe (Complementarity Adjusted Federated) averaging**, a personalized FL approach that:
- Recognizes that heterogeneity in missing mechanisms creates complementarity between clients
- Leverages this complementarity to improve imputation quality
- Computes personalized weights calibrated for heterogeneity levels
- Develops personalized imputation models for each client

### 2. Missing Data Mechanisms

The paper focuses on three types of missing mechanisms:

1. **MCAR (Missing Completely At Random):** Missing values independent of data distribution
2. **MAR (Missing At Random):** Missing values depend on observable data
3. **MNAR (Missing Not At Random):** Missing values depend on unobservable data - **Primary focus**

**Real-world example from the paper:**
- Hospital 1: Records only aberrant glucose values (normal values missing) due to documentation practices
- Hospital 2: Has abnormal glucose values missing because patients with abnormal levels don't disclose due to low trust
- This creates **heterogeneous missing mechanisms** that can be complementary

### 3. Key Concepts

#### a) **Complementarity**
The paper introduces the concept of "missing mechanism complementarity":
- When one client's observable data distribution can inform another client's missing data distribution
- Four levels defined:
  1. **Perfect complementarity:** M(L, 0.3) and M(R, 0.7) - opposite sides, complementary rates
  2. **Imperfect complementarity:** Different directions, non-complementary rates
  3. **One-sided complementarity:** Same direction, different rates
  4. **No complementarity:** Identical mechanisms

#### b) **Complementarity Score**
Computed using cosine similarity between missing mechanism models:

```
Complementarity-Score = 1 - (⟨ξₖᶠ, ξₗᶠ⟩) / (‖ξₖᶠ‖ · ‖ξₗᶠ‖)
```

Where:
- ξₖᶠ = missing mechanism model for feature f at client k
- Ranges from 0 (identical, no complementarity) to 1 (significant complementarity)

### 4. The Cafe Algorithm

**Two main algorithms:**

#### Algorithm 1: Cafe ICE (Main algorithm)
1. **Setup Phase:**
   - Identify incomplete features
   - Compute missing indicators
   - Perform naive federated imputation (using global mean)

2. **Iterative Imputation Phase (for each round t, each feature f):**
   - **Clients fit local models:**
     - Imputation model θₖᶠ (using Linear Regression on f-observable samples)
     - Mechanism model ξₖᶠ (using Logistic Regression to predict missingness)
   - **Server aggregation:**
     - Receives all models from clients
     - Computes Cafe averaging (Algorithm 2)
     - Returns personalized models to each client
   - **Clients update imputations:**
     - Use personalized model to impute missing values
     - Prepare for next iteration

#### Algorithm 2: Cafe Averaging
For each client k, computes personalized aggregated model:

```
θ̃ₖᶠ = γ · θₖᶠ + (1 - γ) · Σ(wₖₗ · θₗᶠ)  [for ℓ ≠ k]
```

Where:
- **wₖₗ** = normalized weight combining complementarity and sample size
- **aₖₗ** = α · cₖₗ + (1 - α) · sₗ
  - cₖₗ = complementarity score between k and ℓ
  - sₗ = relative sample size of client ℓ
- **Hyperparameters:**
  - α ∈ [0, 1]: balances complementarity vs. sample size
  - β ∈ [2, 4]: controls weight normalization
  - γ ∈ [0.05, 0.3]: balances own model vs. aggregated model

**Optimal parameters found:** α = 0.95, γ = 0.05, β = 4 (works across all datasets)

### 5. Datasets Used

#### Synthetic Missing Data Scenarios
Applied to 5 primary datasets:
1. **Codon** (13,008 samples, 26 features, 10 classes) - Codon usage frequencies
2. **Codrna** (20,000 samples, 9 features, 2 classes) - Non-coding RNA detection
3. **MIMICIII** (20,000 samples, 51 features, 2 classes) - ICU patient records
4. **Heart** (20,000 samples, 25 features, 2 classes) - Heart disease prediction
5. **Genetic** (20,000 samples, 33 features, 2 classes) - Genetic variant conflicts

#### Real-world Federated Datasets
6. **HHNP** (87,956 samples, 32 features) - Heritage Health Provider Networks, 6 providers selected
7. **eICU** (12,889 samples, 24 features) - 10 hospitals selected from 208 U.S. hospitals

**Missing Data Simulation:**
- **Quantile-based:** MNAR-Left (erase values in [0, ρ]) vs. MNAR-Right (erase values in [1-ρ, 1])
- **Logit-based:** Complex MNAR using logistic regression with random weights
- Missing rates: ρ ∈ {0.3, 0.4, 0.5, 0.6, 0.7}

### 6. Evaluation Scenarios

**Complementarity Scenarios (S1-S4):**
- **Ideal:** Half clients M(L, 0.5), half M(R, 0.5)
- **S1 - Perfect complementarity:** Each feature has perfectly complementary mechanisms across clients
- **S2 - Imperfect complementarity:** Features have imperfectly complementary mechanisms
- **S3 - One-sided complementarity:** Same direction, different rates
- **S4 - No complementarity:** All clients identical mechanisms

**Complex Scenarios:**
- **Complex #1:** Random quantile-based mechanisms per feature/client (independence)
- **Complex #2:** Logit-based mechanisms (correlated with multiple features)

**Sample Size Distributions:**
- **Uni:** Equal sample sizes across clients
- **Dir:** Dirichlet distribution (moderate heterogeneity, α = 0.1)
- **HS (Hub-Spoke):** One hub with 50% data, others with 5% each

### 7. Evaluation Metrics

#### Imputation Quality
- **RMSE** (Root Mean Square Error) - Lower is better
- **Sliced-WS** (Sliced Wasserstein Distance) - Lower is better
- **t-SNE plots** - Visual comparison of imputed vs. ground truth

#### Downstream Task Performance
- **AUROC** (Area Under ROC Curve)
- **F1-Score**
- **Accuracy** (mentioned but not fully reported)
- **AUPRC** (Area Under Precision-Recall Curve) (mentioned but not fully reported)

Used federated learning with FedAvg protocol and 2-layer neural networks for downstream prediction.

### 8. Key Results

#### Complementarity Score Verification
- Heatmaps show Cafe effectively captures complementarity levels
- Perfect complementarity (S1): scores near 1.0 between complementary clients, near 0 within same mechanism
- No complementarity (S4): uniformly low scores

#### Performance Across Complementarity Levels
- **High complementarity (Ideal, S1, S2):** Cafe significantly outperforms baselines
- **Lower complementarity (S3, S4):** Cafe matches baseline performance
- Demonstrates robustness and adaptability

#### Complex Scenarios
- **RMSE:** Cafe achieves significantly lower errors than all baselines (FedICE, FedMIWAE, FedGAIN, Local)
- **t-SNE:** Cafe's imputed data closely aligns with ground truth
- **Convergence:** All methods converge in 5-10 iterations, but Cafe achieves much lower final error
- **Federated Prediction:** Cafe outperforms baselines, especially on Codon (multi-class task)
- **Lower variation:** Cafe shows more stable performance across multiple runs

#### Real-world Federated Data (HHNP & eICU)
**HHNP Dataset:**
- RMSE: 29.5% reduction (0.183 → 0.129) vs. FedICE
- F1-Score: +3.9% improvement (72.1% → 76.0%)
- AUPRC: +5.7% improvement (78.8% → 84.5%)

**eICU Dataset:**
- Similar improvements observed
- Validates effectiveness in real-world federated healthcare settings

#### Sample Size Robustness
- Minimal variation across different sample size distributions (Uni, Dir, HS)
- Actually improves performance when sample sizes differ (shows complementarity value)

#### Scalability
- Performance stable with 2-10 complementarity clients
- Even a few complementary clients markedly enhance imputation
- Runtime comparable to baselines (additional mechanism model fitting is minimal overhead)

**Runtime (Table III) - seconds across all scenarios:**
| Method | Codrna | Codon | MIMICIII | Heart | Genetic |
|--------|--------|-------|----------|-------|---------|
| Local | 120.79 | 160.88 | 654.9 | 271.64 | 330.55 |
| FedICE | 126.09 | 160.15 | 688.75 | 281.33 | 341.17 |
| **Cafe ICE** | **126.48** | **163.78** | **744.06** | **293.19** | **370.99** |

Very comparable to FedICE (~2-8% overhead)

### 9. Baselines Compared

1. **Local ICE:** Each client imputes independently (no collaboration)
2. **FedICE:** Federated ICE with sample-size weighted averaging
3. **FedMIWAE:** Federated version of MIWAE (VAE-based imputation)
4. **FedGAIN:** Federated version of GAIN (GAN-based imputation)

Cafe outperforms all in heterogeneous scenarios and matches them in homogeneous scenarios.

### 10. Implementation Details

**Technology Stack:**
- **Imputation framework:** Modified Sklearn's Iterative Imputer
- **Imputation models:** Ridge Regression (Linear Regression with L2 regularization)
- **Mechanism models:** Logistic Regression
- **Iterations:** 20 rounds
- **Downstream FL:** PyTorch 11.2, FedAvg algorithm
- **Neural networks:** 2-layer with 32-64 hidden nodes, Adam optimizer, lr=0.001
- **Experiments:** 5 independent runs with different randomness, average reported

**Hyperparameter Grid Search:**
- α: 0.5 to 1.0
- β: 2 to 4
- γ: 0.05 to 0.3
- Best universal set: α = 0.95, γ = 0.05, β = 4

### 11. Theoretical Considerations

**Privacy:**
- Only model parameters (θₖᶠ and ξₖᶠ) transferred to server
- Significantly limits information disclosure
- Raw data never shared
- Explicit privacy guarantees left for future work

**Convergence:**
- Existing FL theory applies to special cases (extreme α, γ values)
- General case more complex due to:
  - Lack of strong convexity
  - Non-smooth loss functions
  - Lipschitz continuity may not apply
- Full theoretical analysis left for future work

### 12. Key Insights

1. **Heterogeneity can be beneficial:** Unlike traditional FL where heterogeneity is a challenge, in missing data imputation, heterogeneous missing mechanisms can create complementarity that improves imputation

2. **MNAR is addressable:** First imputation method that naturally handles MNAR missingness in federated settings

3. **Personalization is crucial:** Single global model fails in heterogeneous scenarios; personalized models are necessary

4. **Mechanism modeling works:** Using simple logistic regression to model missing mechanisms provides effective proxy for heterogeneity measurement

5. **Practical applicability:** Works on real-world federated healthcare datasets with naturally partitioned data

### 13. Limitations & Future Work

**Current Limitations:**
1. No explicit privacy guarantees (differential privacy)
2. Limited theoretical convergence analysis for general case
3. Focus on single imputation (not multiple imputation like MICE)
4. Only numeric features tested (mentions categorical extensibility)

**Future Directions:**
1. **Multiple imputation:** Extend to Bayesian models for posterior distribution sampling
2. **Privacy guarantees:** Add differential privacy mechanisms
3. **Theoretical analysis:** Develop convergence guarantees for general case
4. **Hyperparameter optimization:** Use RL or Bayesian optimization instead of grid search
5. **Domain knowledge integration:** Allow clients to set personalized γ values based on local knowledge
6. **Categorical features:** Full implementation and testing with categorical data

### 14. Related Areas

**Missing Data Imputation Methods:**
- Generative: EM, MCMC, GANs
- Hot-deck: KNN, Graph Neural Networks
- Discriminative: MICE/ICE (focus of this paper)
- Recent: LLMs, Optimal Transport

**Federated Learning Challenges:**
- Data heterogeneity (non-IID data)
- System heterogeneity (different compute capabilities)
- Model heterogeneity (different model architectures)
- Missing data heterogeneity (**focus of this paper**)

### 15. Clinical Relevance

**Healthcare Missing Data Causes:**
- **Individual level:** Privacy concerns, trust, health condition severity, race, gender
- **Organizational level:** Administrative policies, data collection protocols, documentation practices, clinical practices

**Examples from paper:**
- Zachariasse et al.: Missing triage data in European EDs due to staff shortages and bad protocols
- Wells et al.: Missing clinical data at Cleveland Clinic due to non-digitized paper records

This makes heterogeneous missing mechanisms the norm, not the exception in federated healthcare.

---

## Comparison with Other Data Quality Papers

### Cafe vs. FedCorr

| Aspect | Cafe (This Paper) | FedCorr (Different Paper) |
|--------|-------------------|---------------------------|
| **Focus** | Missing data imputation | Label noise correction |
| **Data Quality Issue** | Missing values (MNAR) | Label errors |
| **Repair Method** | Impute missing values using complementarity | Correct noisy labels using LID scores |
| **Heterogeneity** | Missing mechanism heterogeneity | Label noise heterogeneity |
| **Key Technique** | Complementarity-adjusted averaging | Multi-stage correction with LID |
| **Personalization** | Yes (personalized imputation models) | Yes (identifies noisy clients) |
| **Handles** | Complex MNAR mechanisms | Heterogeneous label noise |

Both papers address data quality in federated learning but tackle different problems. Cafe handles missing data, while FedCorr handles label errors.

---

## Key Takeaways

✅ **What Cafe Does:**
- Handles missing values in federated learning
- Works with complex MNAR mechanisms
- Leverages heterogeneity as a benefit (complementarity)
- Creates personalized imputation models
- Outperforms state-of-the-art in heterogeneous scenarios

❌ **What Cafe Does NOT Do:**
- Does not handle outliers
- Does not correct label errors
- Does not explicitly address fairness
- Does not provide differential privacy guarantees (yet)

🎯 **Best Use Cases:**
- Federated healthcare networks with heterogeneous missing data
- Situations where different organizations have different reasons for missing data
- MNAR missing mechanisms
- When complementarity exists between clients' observable and missing data distributions

