# Analysis: FedIMPUTE - Privacy-preserving missing value imputation for multi-site heterogeneous electronic health records

**Authors:** Siqi Li, Mengying Yan, Ruizhi Yuan, Molei Liu, Nan Liu, Chuan Hong  
**Affiliations:** Duke-NUS Medical School, Duke University, Columbia University, Peking University  
**Published:** Journal of Biomedical Informatics (2025)  
**DOI:** https://doi.org/10.1016/j.jbi.2025.104780

---

## Summary Table

| Column | Answer | Details |
|--------|--------|---------|
| **ML or FL** | **FL (Federated Learning)** | Communication-efficient federated learning framework for collaborative missing value imputation across multiple healthcare institutions. Uses statistical FL approaches (DAC algorithm) rather than engineering-based approaches. |
| **Handles fairness?** | **No** | Paper does not explicitly address fairness, equity, or bias concerns. However, it helps sites with high missing data rates and small sample sizes by leveraging information from other sites. |
| **Outliers** | **No** | Paper does not address outlier detection or handling. Focus is on missing data imputation only. |
| **Missing values** | **✓ Yes (Primary Focus)** | Core contribution: privacy-preserving missing value imputation (MVI) for Electronic Health Records (EHRs). Handles three types of missingness: MCAR (Missing Completely At Random), MAR (Missing At Random), and MNAR (Missing Not At Random). Tested with missing rates up to 80%. |
| **Label errors** | **No** | Not addressed. This paper focuses on missing predictor values, not label/outcome errors. |
| **Repair data?** | **✓ Yes** | Imputes missing values to complete datasets. Uses regression-based imputation methods (MICE and single regression imputation) federated across multiple sites. Each site obtains a complete imputed dataset for downstream prediction tasks. |
| **Topics** | **Federated Learning, Missing Value Imputation, Electronic Health Records (EHR), Privacy-Preserving ML, Clinical Decision Support, Multi-site Collaboration, Heterogeneous Data, MICE Imputation, Regression Imputation, DAC Algorithm (Divide-and-Conquer), Communication-Efficient FL, Emergency Department Prediction, Risk Prediction** | Addresses privacy constraints in healthcare where hospitals cannot share patient-level data due to GDPR, HIPAA, and other regulations. |
| **Datasets** | **Simulations + Duke University Health System (DUHS) Real-world EHR data (2019)** | **Simulation:** 3 sites with sample sizes n1=1000, n2=2000, n3=3000; 20+1 predictors; missing rates of ~80%, ~40%, 0%; 500 replicates per setting. **Real Data:** Emergency Department (ED) visits from 3 Duke hospitals (DRAH: N=15,298 with ~75% missing; DRH: N=23,820 with ~45% missing; DUH: N=40,669 with 0% missing). Predictors include demographics (age, sex), vital signs (pulse, SBP, DBP, SpO2, temperature, respiration, acuity level), and comorbidities (tumors, diabetes, renal disease). Outcome: inpatient admission (binary). Variables with missingness: Systolic Blood Pressure (SBP) and Diastolic Blood Pressure (DBP). |
| **Code** | **✓ Yes - Available on GitHub** | Repository: https://github.com/flemergency/FedIMPUTE. Contains R code for reproducing simulated data. Real-world datasets are private and cannot be shared due to data privacy regulations. |
| **Implemented?** | **✓ Yes - Fully implemented and tested** | Three FedIMPUTE variants implemented: **FedIMPUTE-Basic** (Algorithm 1 - simple parameter weighting), **FedIMPUTE-Pooled** (Algorithm 2 - pooled parameters), **FedIMPUTE-DAC** (Algorithm 3 - divide-and-conquer FL). Compared against baselines: complete case analysis, local mean imputation, local MICE imputation, local single regression imputation. Evaluation metrics: AUROC, AUPRC, TPR@FPR=0.05 for downstream prediction; MSE for imputation accuracy. |

---

## Key Findings

### 1. **Problem Definition**

#### Challenges:
- **Missing data is pervasive in clinical research:** Leads to biased results, reduced predictive power, and unreliable clinical decisions
- **Cross-institutional collaboration barriers:** 
  - Privacy regulations (HIPAA, GDPR, PDPA) prevent data pooling
  - Traditional FL focuses on modeling, not data preprocessing
  - Most FL studies assume sites independently handle missingness using basic methods (mean/median imputation or complete case analysis)
- **Sites with high missingness suffer:** Small hospitals or regional centers often have higher missing data rates and smaller sample sizes, leading to poor local imputation quality

#### Current Gap:
- FL has been widely applied to healthcare modeling (COVID-19 prediction, phenotyping, treatment optimization) but **not** to missing value imputation
- Only one prior work (Zhou et al. 2021) applied FL-based MVI, but for air quality data, not clinical/biomedical data
- Existing FL studies largely ignore the data quality issues that arise from heterogeneous missing data patterns

### 2. **Methodology: FedIMPUTE**

FedIMPUTE is a **communication-efficient federated learning framework** that enables multiple sites to collaboratively perform missing value imputation without sharing patient-level data.

#### Three Variants:

##### **FedIMPUTE-Basic (Algorithm 1)**
- **Approach:** Simple federated mean/median/mode imputation
- **Process:**
  1. Each site computes local mean/median/mode for variables with missing values
  2. Federate by computing weighted average: ω̂ᶠᵢ = (1/nₜ) Σ(nₖ × ω̂ᵢᵏ)
  3. Each site uses federated parameter to impute local missing values
- **Communication:** Single round (very efficient)
- **Use case:** Simple baseline for comparison

##### **FedIMPUTE-Pooled (Algorithm 2)**
- **Approach:** Federation of regression imputation models via direct parameter pooling
- **Process:**
  1. Each site fits local imputation model (single regression or MICE) for each variable with missingness
  2. Federate parameters via weighted averaging: ω̂ᶠᵢ = (1/nₜ) Σ(nₖ × ω̂ᵢᵏ)
  3. Broadcast federated parameters to all sites
  4. Each site uses federated model to impute local data
- **Communication:** One round per missing variable
- **Limitation:** May underperform when large sample size sites dominate the weighted average

##### **FedIMPUTE-DAC (Algorithm 3) - Main Contribution**
- **Approach:** Advanced federation using the Divide-and-Conquer (DAC) algorithm
- **DAC Background:** Originally proposed by Hong et al. (2022) for fitting sparse logistic regression to distributed datasets. Achieves similar statistical efficiency as full-sample estimator with minimal communication.
- **Process:**
  1. **Step 1:** Each site fits local imputation model (MICE or single regression) and broadcasts score vector Û and Hessian matrix Â
  2. **Step 2:** Compute federated parameters:
     - Aggregate scores: Ûᴰᴬᶜⁱ = (1/K) Σ Ûᵏⁱ
     - Aggregate Hessian: Âᴰᴬᶜⁱ = (1/K) Σ Âᵏⁱ
     - Obtain initial estimator β̃ from a selected site (preferably large sample size, low missing rate)
     - Compute federated estimator: β̂ᴰᴬᶜⁱ = β̃ + (Âᴰᴬᶜⁱ)⁻¹ × Ûᴰᴬᶜⁱ(β̃)
  3. **Step 3:** Broadcast federated parameters to all sites; each site imputes using federated model
- **Communication:** 2-3 rounds (still very efficient compared to engineering-based FL)
- **Advantages:**
  - Statistically efficient
  - Robust to heterogeneity (covariate distribution shifts, model shifts)
  - No central server required (peer-to-peer summary statistics sharing)

#### Missing Data Mechanisms Handled:

1. **MCAR (Missing Completely At Random):**
   - f(Iⱼ | X₋ⱼ, φ) = f(Iⱼ | φ)
   - Missingness does not depend on any data values

2. **MAR (Missing At Random):**
   - f(Iⱼ | X₋ⱼ, φ) = f(Iⱼ | X̂ₒᵦₛ₋ⱼ, φ)
   - Missingness depends only on observed values, not missing values

3. **MNAR (Missing Not At Random):**
   - f(Iⱼ | X₋ⱼ, φ) = f(Iⱼ | Xⱼ, X₋ⱼ, φ)
   - Missingness depends on the missing values themselves

#### Imputation Methods Used:

1. **Mean/Median Imputation:** Replace missing values with mean (continuous) or median (categorical)

2. **Single Regression Imputation:** Regress variable Xⱼ on all remaining predictors Xᵢ (j ≠ i)

3. **MICE (Multivariate Imputation by Chained Equations):**
   - Gibbs sampler iterations:
     - For X¹: draw imputations X^(t+1)¹ from P(X¹ | X^(t+1)², X^t³, ..., X^tᴾ)
     - For X²: draw imputations X^(t+1)² from P(X² | X^(t+1)¹, X^(t+1)³, ..., X^tᴾ)
     - ...
     - For Xᴾ: draw imputations X^(t+1)ᴾ from P(Xᴾ | X^(t+1)¹, X^(t+1)², ..., X^(t+1)ᴾ⁻¹)
   - Assumes MAR
   - Most commonly used MVI method in practice

### 3. **Experimental Setup**

#### Simulation Studies:

**Setup:**
- **3 sites:** S₁ (high missingness, small sample), S₂ (moderate missingness, medium sample), S₃ (no missingness, large sample)
- **Sample sizes:** n₁=1000, n₂=2000, n₃=3000
- **Predictors:** p = 20 + 1 (intercept); first two predictors (X¹, X²) have missing values
- **Missing rates:** S₁ ≈ 80%, S₂ ≈ 40%, S₃ = 0%
- **Outcome:** Binary outcome Y generated via logistic regression; prevalence ≈ 0.5; 6 non-zero effect sizes
- **Replicates:** 500 simulations per setting

**Four Simulation Trends:**
1. **Change in S₁ sample size:** n₁ = 100, 200, ..., 1000
2. **Change in missingness control variables:** nC = 5, 6, ..., 19 for MAR; 20 for MNAR
3. **Change in covariate distribution (P(X)):** 
   - Introduce covariate shift via mean/variance shifts
   - X^S₁ ~ N(μ, σ), X^S₂ ~ N(α₁μ, α₂σ), X^S₃ ~ N((1-α₁)μ, (1-α₂)σ)
   - α = 0.1, 0.2, ..., 0.8
4. **Change in model distribution (P(Y|X)):**
   - Introduce model shift via effect size shifts
   - β^S₁, δβ^S₁, (1-δ)β^S₁ for S₁, S₂, S₃
   - δ = 0.1, 0.2, ..., 0.9

**Baseline Methods Compared:**
1. Original data (no missingness) - ideal scenario
2. Complete case (remove all missing observations) - naive approach
3. Local mean imputation
4. Local MICE imputation
5. Local single regression imputation
6. FedIMPUTE-Basic
7. FedIMPUTE-Pooled (two variants: sample-size weighted, average pooling)
8. **FedIMPUTE-DAC** (main method)

**Evaluation Metrics (for downstream prediction task):**
- **AUROC:** Area Under Receiver Operating Characteristic curve
- **AUPRC:** Area Under Precision-Recall Curve
- **TPR @ FPR=0.05:** True Positive Rate when controlling False Positive Rate at 5%

#### Real-World Application:

**Data Source:** Duke University Health System (DUHS) via Duke Clinical Research Datamart (CRDM)

**Cohort:**
- **Time period:** All ED visits in 2019
- **Sites:**
  - Site 1: Duke Raleigh Hospital (DRAH), N=15,298, ~75% missing
  - Site 2: Duke Regional Hospital (DRH), N=23,820, ~45% missing
  - Site 3: Duke University Hospital (DUH), N=40,669, 0% missing
- **Data preprocessing:** Down-sampled outcome=0 cases by 1/3 to balance prevalence; included only samples without initial missingness to evaluate imputation accurately

**Outcome:** Inpatient admission (binary)

**Predictors (3 categories):**
1. **Demographics:** Age (years), Sex (Male/Female)
2. **Vital Signs:** Pulse (beats/min), Systolic Blood Pressure (SBP, mm Hg), Diastolic Blood Pressure (DBP, mm Hg), Oxygen Saturation (SpO2, %), Temperature (°F), Respiration (times/min), Acuity Level (1-5)
3. **Comorbidities (ICD-9-CM codes):** Local tumor, Metastatic tumor, Diabetes with complications, Diabetes without complications, Renal disease

**Variables with artificially induced missingness:** SBP, DBP

**Missingness mechanisms tested:**
- **MCAR:** Fixed probability (0.75 for Site 1, 0.45 for Site 2)
- **MAR:** Control variables = sex and temperature, effect sizes = 0.1 and 0.2
- **MNAR:** Missing variables (SBP, DBP) themselves as control variables, effect size = 0.1

**Evaluation Metrics:**
- **MSE:** Mean Squared Error between imputed and original values
- **AUROC:** For downstream prediction task

### 4. **Results**

#### Simulation Results:

**Key Findings (Figures 2 and 3):**

1. **FedIMPUTE-DAC consistently outperforms all baseline methods:**
   - Higher AUROC and TPR across all simulation settings
   - Performance closest to the ideal scenario (original data with no missingness)

2. **Robustness to heterogeneity:**
   - **Model shifts (trend D):** FedIMPUTE-DAC remains stable even when effect sizes differ up to 1.9× between sites
   - **Covariate shifts (trend C):** Performance stable across varying covariate distributions
   - Other methods (local imputation, FedIMPUTE-Basic) show noticeable performance drops under heterogeneity

3. **Sample size effects (trend A):**
   - FedIMPUTE-DAC maintains high performance even when S₁ has very small sample size (n₁=100)
   - Local imputation methods degrade significantly with smaller sample sizes

4. **Missingness control variables (trend B):**
   - FedIMPUTE-DAC stable across varying numbers of control variables (5-20)
   - Effective under MCAR, MAR, and MNAR mechanisms

5. **FedIMPUTE-Pooled sometimes underperforms local imputation:**
   - Reason: Large sample size sites dominate the weighted average
   - Simplistic federation (direct parameter pooling) can be less effective than localized solutions

6. **Extreme heterogeneity scenarios:**
   - When model heterogeneity is extremely high (sites have very different outcome models), all methods degrade
   - Aligns with real-world FL practice: FL only practical when sites share sufficient model similarity

**Visual Evidence:**
- **Figure 2 (AUROC trends):** FedIMPUTE-DAC consistently achieves higher AUROC across all four simulation trends
- **Figure 3 (TPR trends):** FedIMPUTE-DAC superior and stable across various conditions; baseline methods suffer significant drops under extreme conditions

#### Real-World Results (Duke ED Data):

**Table 1 - Cohort Description:**
| | Site 1 (DRAH) | Site 2 (DRH) | Site 3 (DUH) |
|---|---|---|---|
| **N** | 15,298 | 23,820 | 40,669 |
| **Outcome (admission)** | 6,093 (39.8%) | 11,128 (46.7%) | 22,440 (55.2%) |
| **Missing rate** | ~75% | ~45% | 0% |
| **Age, mean (sd)** | 51.6 (22.3) | 48.8 (22.7) | 44.8 (24.5) |
| **Sex (Female)** | 8,827 (57.7%) | 14,432 (60.6%) | 22,694 (55.8%) |
| **SBP, mean (sd)** | 133 (23.2) | 133 (23.3) | 132 (22.5) |
| **DBP, mean (sd)** | 78.9 (16.4) | 80.4 (13.8) | 79.8 (15.3) |

**Performance Results (Site 1 - highest challenge):**

1. **MSE (Mean Squared Error) - Figure 4:**
   - **FedIMPUTE-DAC MICE:** Achieves **lowest MSE** in all three missing mechanisms (MCAR, MAR, MNAR) for both SBP and DBP
   - **FedIMPUTE-DAC Single:** Also achieves **lowest MSE** in all settings
   - Superior imputation accuracy compared to all baseline methods

2. **AUROC for downstream prediction:**
   - All methods achieve comparable AUROC of **0.89** for Site 1
   - Reason: Even with data shifts or additional noise, AUROC (ranking metric) remains robust
   - **Interpretation:** FedIMPUTE's main advantage is **accurate data recovery** (low MSE), not necessarily dramatic improvement in prediction metrics like AUROC

3. **Statistical significance of SBP/DBP:**
   - Both SBP and DBP have p-values < 0.05 in full model
   - However, excluding them minimally changes AUC/Brier scores due to presence of other significant predictors
   - **Key takeaway:** FedIMPUTE excels at recovering original data values, which is crucial for interpretability and data quality

### 5. **Advantages of FedIMPUTE**

#### 1. **Communication Efficiency:**
- **1-3 rounds of communication** vs. traditional engineering-based FL (often 100+ rounds)
- Only shares **non-patient-level summary statistics** (score vectors, Hessian matrices)
- No need for central server (peer-to-peer sharing)
- Easy to implement without complex FL infrastructure

#### 2. **Statistical Approach vs. Engineering Approach:**
- **Engineering-based FL** (e.g., FedAvg):
  - Requires central server
  - 100+ communication rounds
  - Challenging for hospitals with secure computing systems that prohibit external server connections
  - Requires additional safeguards against privacy attacks
  - Labor-intensive to develop
  
- **FedIMPUTE (statistical approach):**
  - No central server
  - 1-3 communication rounds
  - Simple summary statistics exchange
  - Easier to implement in real-world healthcare settings with strict privacy regulations

#### 3. **Transparency and Interpretability:**
- Uses **regression-based imputation** (white-box models)
- More interpretable than black-box models (e.g., neural networks, GANs)
- Minimizes impact on downstream explanation tools (LIME, SHAP)
- Reduces noise in feature importance analysis

#### 4. **Robustness to Heterogeneity:**
- Handles **covariate distribution shifts** (different patient populations)
- Handles **model shifts** (different predictor-outcome relationships)
- Effective under all missing mechanisms (MCAR, MAR, MNAR)

#### 5. **Benefits Small/Underserved Sites:**
- Sites with high missing rates and small sample sizes gain most benefit
- Leverages information from larger, more complete sites
- Improves data quality without compromising privacy

### 6. **Limitations and Future Work**

#### Current Limitations:

1. **Same predictors required:**
   - All sites must use identical predictor sets
   - Real-world healthcare: sites may collect different variables due to different clinical practices

2. **Real-world validation limited:**
   - Duke EHR experiment is still a simulation (artificially induced missingness)
   - Sites involved do not have actual privacy constraints between them
   - Need true multi-institutional study with real privacy barriers

3. **Limited to structured data:**
   - Current methods focus on tabular EHR data
   - Does not handle unstructured data (clinical notes, images)

4. **Extreme heterogeneity:**
   - When sites have vastly different outcome models, all FL methods (including FedIMPUTE) degrade
   - FL only practical when sites share sufficient model similarity

#### Future Directions:

1. **Leverage synthetic data strategies:**
   - Enable flexibility to accommodate variations in predictor sets across sites
   - Allow collaboration even when sites collect different variables

2. **Real-world multi-institutional studies:**
   - Collaborate with partners who have strict data-sharing privacy constraints
   - Evaluate FedIMPUTE in true real-world settings with actual privacy barriers

3. **Extend to unstructured data:**
   - Adapt methods for clinical notes, medical images, waveform data

4. **Heterogeneity-aware methods:**
   - Develop techniques to detect and handle extreme heterogeneity
   - Automatic site clustering: identify subgroups of similar sites for collaboration

5. **Fairness and equity:**
   - Investigate whether FedIMPUTE reduces health disparities
   - Ensure imputation quality is equitable across demographic groups

### 7. **Comparison with Related Work**

#### Prior FL-based MVI:

**Zhou et al. (2021) - FCGAI (Federated Conditional GAN for Air Quality Imputation):**
- **Domain:** Air quality monitoring (environmental science)
- **Method:** Conditional GAN with Wasserstein distance, FedAvg aggregation
- **Architecture:** Client-server with central server
- **Communication:** 300 epochs with aggregation every 10 local epochs
- **Limitation:** Engineering-based approach, requires central server

**FedIMPUTE (2025):**
- **Domain:** Healthcare (Electronic Health Records)
- **Method:** Statistical regression (MICE, single regression) with DAC algorithm
- **Architecture:** Peer-to-peer (no central server)
- **Communication:** 1-3 rounds
- **Advantage:** Communication-efficient, easy to implement, interpretable

#### Traditional MVI Methods (Non-FL):

1. **Mean/Median Imputation:** Simple but ignores relationships between variables
2. **MICE:** Gold standard for centralized data, assumes MAR
3. **Single Regression:** Regresses missing variable on others
4. **Complete Case Analysis:** Removes all incomplete observations (wastes data, introduces bias)

**FedIMPUTE vs. Local MVI:**
- Local methods suffer when site has small sample size and high missing rate
- FedIMPUTE leverages external information while preserving privacy

### 8. **Implications for Practice**

#### For Healthcare Institutions:

1. **Enable multi-site collaboration without data sharing:**
   - Hospitals can collaborate on data quality improvement
   - Comply with HIPAA, GDPR, PDPA regulations
   - No need for complex data use agreements

2. **Improve data quality for underserved sites:**
   - Small regional hospitals benefit from large academic medical centers' data
   - Reduces health disparities in data quality

3. **Easy implementation:**
   - No need for expensive FL infrastructure
   - Simple summary statistics exchange via secure email or file transfer
   - No central server = lower security risks

#### For Researchers:

1. **Better downstream modeling:**
   - Higher quality imputed data leads to better predictive models
   - Reduced bias in multi-site clinical studies

2. **Transparency and interpretability:**
   - Regression-based imputation is explainable
   - Easier to gain institutional review board (IRB) approval

3. **Foundation for FL preprocessing:**
   - Demonstrates FL can be applied to data preprocessing, not just modeling
   - Opens door for other FL-based preprocessing tasks (outlier detection, data cleaning, feature engineering)

#### For Policy Makers:

1. **Privacy-preserving collaboration model:**
   - Demonstrates feasible approach for multi-institutional research under strict privacy laws
   - Can inform policy on data sharing and collaborative research

2. **Equity considerations:**
   - FedIMPUTE helps smaller, underserved institutions improve data quality
   - Can reduce disparities in research capacity

---

## Technical Details

### Missing Data Notation:

- **Y:** Outcome variable
- **X:** (P+1) × 1 vector of predictors (first element is 1 for intercept)
- **K:** Number of sites (S₁, S₂, ..., Sₖ)
- **𝐏ₖ:** Set of predictor indices with missing values at site k
- **𝐑₀:** Union of all missing features across all sites: 𝐑₀ = ⋃ₖ₌₁ᴷ 𝐏ₖ
- **d:** Cardinality of 𝐑₀ (number of unique variables with missingness)
- **Iⱼ:** Missing-data indicator vector for variable Xⱼ
- **X̂ₒᵦₛ, X̂ₘᵢₛ:** Observed and missing components of X

### DAC Algorithm Details:

**Divide-and-Conquer (DAC) for Sparse Logistic Regression:**

Given distributed datasets at K sites, goal is to fit logistic regression without pooling data.

**Key idea:** Use linearization to approximate the global objective function.

**Steps:**
1. Each site k computes:
   - Score vector: Ûₖ(β) = ∂ℓₖ(β)/∂β (gradient of local log-likelihood)
   - Hessian matrix: Âₖ(β) = -∂²ℓₖ(β)/∂β∂β' (second derivative)

2. Central aggregation (or peer-to-peer):
   - Û_DAC = (1/K) Σₖ Ûₖ(β₀)
   - Â_DAC = (1/K) Σₖ Âₖ(β₀)

3. Update estimator:
   - β̂_DAC = β₀ + Â_DAC⁻¹ × Û_DAC(β₀)
   - β₀ is initial estimator from a selected site (chosen for large sample size, low missing rate)

**Properties:**
- Achieves similar statistical efficiency as full-sample estimator
- Minimal communication: only score vectors and Hessian matrices
- Robust to heterogeneity when sites share sufficient similarity

**Application to MVI:**
- For each variable Xⱼ with missingness, treat imputation as regression problem
- Regress Xⱼ on X₋ⱼ (all other variables)
- Apply DAC to federate regression parameters
- Use federated model to impute missing values

### Ordering of Missing Variables:

**Recommendation:** Form {𝐑₀} such that missing rate rᵢ satisfies rᵢ ≥ rⱼ for all i > j

**Rationale:**
- Impute variables with lower missing rates first
- Use imputed variables to help impute variables with higher missing rates
- Similar to MICE's chained equations approach

---

## Reproducibility

### Code Availability:
- **GitHub:** https://github.com/flemergency/FedIMPUTE
- **Language:** R
- **Contents:** Code to reproduce simulated data

### Data Availability:
- **Simulated data:** Can be reproduced using provided R code
- **Real-world data:** Private, cannot be shared due to Duke University Health System data privacy regulations

### Hyperparameters (not explicitly stated, inferred from methods):
- **Missing rates:** S₁ ≈ 80%, S₂ ≈ 40%, S₃ = 0%
- **Sample sizes:** n₁=1000, n₂=2000, n₃=3000 (default)
- **Predictors:** p = 20 + 1
- **Outcome prevalence:** ≈ 0.5
- **Non-zero effects:** s = 6
- **Simulation replicates:** 500

---

## Key Contributions

1. **First FL-based MVI method for structured clinical data (EHRs)**
2. **Communication-efficient approach** (1-3 rounds vs. 100+ for traditional FL)
3. **Three algorithms:** FedIMPUTE-Basic, FedIMPUTE-Pooled, FedIMPUTE-DAC
4. **Handles heterogeneity:** Robust to covariate and model shifts
5. **Handles all missing mechanisms:** MCAR, MAR, MNAR
6. **No central server required:** Peer-to-peer summary statistics sharing
7. **Interpretable:** Uses transparent regression-based methods
8. **Benefits underserved sites:** Improves data quality for sites with high missingness and small sample sizes
9. **Proof-of-concept on real EHR data:** Duke ED dataset with 79,787 patients

---

## Citation

```bibtex
@article{li2025fedimpute,
  title={FedIMPUTE: Privacy-preserving missing value imputation for multi-site heterogeneous electronic health records},
  author={Li, Siqi and Yan, Mengying and Yuan, Ruizhi and Liu, Molei and Liu, Nan and Hong, Chuan},
  journal={Journal of Biomedical Informatics},
  year={2025},
  volume={},
  pages={104780},
  doi={10.1016/j.jbi.2025.104780},
  publisher={Elsevier}
}
```

---

## Related Papers

1. **Zhou et al. (2021):** Federated conditional generative adversarial nets imputation method for air quality missing data
2. **Hong et al. (2022):** A divide-and-conquer method for sparse risk prediction and evaluation (DAC algorithm)
3. **Van Buuren & Oudshoorn (1999):** Flexible multivariate imputation by MICE
4. **Little & Rubin (2019):** Statistical analysis with missing data (missing data theory)

---

## Summary

FedIMPUTE is a **groundbreaking communication-efficient federated learning framework** for privacy-preserving missing value imputation in healthcare. It is the **first method to apply FL specifically to structured clinical data preprocessing**, addressing a critical gap in multi-site clinical research. By leveraging the DAC algorithm, FedIMPUTE achieves **superior imputation accuracy** with minimal communication (1-3 rounds), making it practical for real-world healthcare settings with strict privacy regulations. The method is **transparent, interpretable, and robust to heterogeneity**, and it particularly benefits smaller sites with high missing rates and limited sample sizes. Validated through extensive simulations and proof-of-concept on Duke University Health System's emergency department data (79,787 patients), FedIMPUTE demonstrates **state-of-the-art performance** and opens new avenues for FL-based data preprocessing in biomedical research.

