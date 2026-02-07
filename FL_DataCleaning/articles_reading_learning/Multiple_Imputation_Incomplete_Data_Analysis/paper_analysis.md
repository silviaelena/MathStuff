# Paper Analysis: Multiple Imputation for Analysis of Incomplete Data in Distributed Health Data Networks

## Paper Information
- **Title**: Multiple imputation for analysis of incomplete data in distributed health data networks
- **Authors**: Changgee Chang, Yi Deng, Xiaoqian Jiang, Qi Long
- **Affiliations**: University of Pennsylvania, Emory University, University of Texas Health Science Center at Houston
- **Published**: Nature Communications (2020)
- **DOI**: https://doi.org/10.1038/s41467-020-19270-2
- **Code**: https://github.com/changgee/MIDist

---

## Classification Table

| Element | Answer |
|---------|--------|
| **ML or FL** | **FL (Federated Learning / Distributed Learning)** - Operates on horizontally partitioned data across multiple sites without sharing subject-level data. This is essentially federated learning applied to distributed health data networks (DHDNs). |
| **Handles fairness?** | **No** - Paper does not explicitly address fairness, equity, or bias concerns. Focus is on privacy protection and statistical validity. |
| **Outliers** | **No** - Paper does not address outlier detection or handling. |
| **Missing values** | **✓ Yes (Primary Focus)** - Core contribution: communication-efficient distributed multiple imputation (MI) methods for incomplete data in distributed health data networks. Handles three missing data mechanisms: MCAR (Missing Completely At Random), MAR (Missing At Random), and MNAR (Missing Not At Random). Tested with missing rates up to 50%. |
| **Label errors** | **No** - Not addressed. This paper focuses on missing predictor/covariate values, not label/outcome errors. |
| **Repair data?** | **✓ Yes** - Imputes missing values to complete datasets using multiple imputation. Four distributed MI methods developed (iMI, avgmMI, cslMI, siMI) plus MICE variants for general missing patterns. Each site obtains multiple imputed datasets that account for imputation uncertainty. |
| **Topics** | **Federated Learning, Distributed Health Data Networks (DHDNs), Multiple Imputation, Missing Data, Privacy-Preserving ML, Electronic Health Records (EHR), Communication-Efficient Algorithms, Horizontally Partitioned Data, MAR/MCAR/MNAR, MICE (Multiple Imputation by Chained Equations), Statistical Inference, Rubin's Rule, Bayesian Imputation, AVGM Algorithm, CSL Algorithm** |
| **Datasets** | **Simulations + Georgia Coverdell Acute Stroke Registry (GCASR) Real-world EHR data (2005-2013)** - **Simulation:** 3 scenarios with univariate and general missing patterns; 5-10 sites; sample sizes 250-1000; missing rates ~20-50%. **Real Data:** Acute stroke dataset from 75 hospitals in Georgia with 68,287 patients, 203 variables. Analysis performed on 66-75 hospitals depending on missing data constraints. |
| **Code** | **✓ Yes - Available on GitHub** - Repository: https://github.com/changgee/MIDist. Contains R code with custom implementations and uses existing R package 'mice' for MICE implementation. |
| **Implemented?** | **✓ Yes - Fully implemented and tested** - Four distributed MI methods: **iMI** (independent MI at each site), **avgmMI** (average mixture approach), **cslMI** (communication-efficient surrogate likelihood), **siMI** (sufficient information MI). Plus four distributed MICE variants: **iMICE**, **avgmMICE**, **cslMICE**, **siMICE**. Evaluation metrics: Bias, SD, rMSE of regression coefficients; communication costs; statistical significance consistency. |

---

## Summary

This paper represents the **first systematic investigation of multiple imputation methods for horizontally partitioned incomplete data in distributed health data networks (DHDNs)**. Unlike traditional MI methods that require pooling data into a central repository, these distributed approaches enable hospitals and healthcare systems to collaboratively handle missing data while preserving patient privacy by never sharing subject-level information.

---

## Key Findings

### 1. **Problem Definition**

#### Challenges in DHDNs:

1. **Privacy and Regulatory Barriers:**
   - Government regulations (e.g., VA policies, HIPAA)
   - Institutional policies prevent data pooling
   - De-identification alone is inadequate for privacy protection
   - Background information can enable re-identification attacks

2. **Missing Data Pervasive in EHRs:**
   - EHRs collected during healthcare delivery have extensive missingness
   - Improper handling leads to biased results and invalid conclusions
   - Can cause inappropriate healthcare and policy decisions
   - Reduces usable sample size and analysis power

3. **Existing MI Methods Require Centralized Data:**
   - Standard MI methods (e.g., MICE) assume access to pooled data
   - Not feasible in DHDNs where data cannot be combined
   - Current DHDNs (e.g., pSCANNER) lack tools for distributed missing data

#### Missing Data Mechanisms:

1. **MCAR (Missing Completely At Random):** Missingness independent of any data
2. **MAR (Missing At Random):** Missingness depends only on observed data (assumed throughout paper)
3. **MNAR (Missing Not At Random):** Missingness depends on missing values themselves

---

### 2. **Methodology: Four Distributed MI Approaches**

All methods share the goal of imputing missing values without sharing subject-level data between sites.

#### **iMI (Independent MI)**
- **Approach:** Each site performs MI independently using only local data
- **Communication:** 0 rounds (most efficient)
- **Advantages:**
  - Privacy-preserving (no communication)
  - Simple to implement
- **Limitations:**
  - Fails to leverage information from other sites
  - Large variability in imputation
  - Cannot handle variables missing for all observations in a site
  - Unstable when site has small sample size

#### **avgmMI (Average Mixture MI)**
- **Approach:** Inspired by Zhang et al. (2013) AVGM algorithm for distributed complete data
- **Process:**
  1. Each site fits imputation model locally → obtains estimate α̂^(k)
  2. Central site computes weighted average: α̂_avgm = Σ w_k α̂^(k) where w_k ∝ n_c^(k) (complete cases)
  3. Sample imputation parameters from N(α̂_avgm, Cov(α̂_avgm))
  4. Broadcast sampled parameters to all sites for imputation
- **Communication:** 2 one-way rounds
- **Advantages:**
  - Communication-efficient
  - Achieves optimal convergence rate asymptotically
  - Less affected by sample distribution when imputing continuous variables
- **Limitations:**
  - Local estimates volatile when sample size is small
  - Substantially affected by uneven sample distribution for binary variables
  - Cannot handle variables missing for all observations in a site

#### **cslMI (Communication-Efficient Surrogate Likelihood MI)**
- **Approach:** Inspired by Jordan et al. (2019) CSL algorithm
- **Process:**
  1. Central site (largest sample size) computes initial estimate α̃
  2. All sites send gradient ∇L_k(α̃) to central site
  3. Central site computes: α̂_csl = arg min L̃(α) where L̃ uses curvature from central site and pooled derivatives
  4. Covariance: Cov(α̂_csl) = N_c/n^(1) × [∇²L_1(α)]^(-1)
  5. Sample and broadcast parameters for imputation
- **Communication:** 3 one-way rounds (1 more than avgmMI)
- **Advantages:**
  - Achieves optimal convergence rate when central site has majority of samples
  - Communication-efficient
- **Limitations:**
  - Performance sensitive to sample size at central site
  - Worse when samples evenly distributed (central site is small)
  - Can fail to converge when sample sizes are small and evenly distributed
  - Cannot handle variables missing for all observations in a site

#### **siMI (Sufficient Information MI)**
- **Approach:** Uses only aggregated statistics sufficient to reproduce pooled data results
- **Process:**
  1. For linear regression imputation: Each site sends X^(k)ᵀX^(k) and X^(k)ᵀx_{1,c}^(k)
  2. For GLMs: Iterative Newton updates requiring derivatives and curvature per iteration
  3. Compute global imputation model equivalent to pooled data
  4. Sample parameters and broadcast for imputation
- **Communication:** 
  - 2 rounds for linear imputation models
  - Many rounds for nonlinear models (one round per Newton iteration)
- **Advantages:**
  - Produces same results as standard MI on pooled data (gold standard)
  - Can handle variables missing for all observations in a site
  - Unbiased across all settings
- **Limitations:**
  - Not communication-efficient for GLMs (iterative fitting)
  - Transmits entire design matrix structure (higher privacy risk)
  - More communications = higher cost

---

### 3. **Distributed MICE for General Missing Patterns**

When multiple variables have missing values, MICE (Multiple Imputation by Chained Equations) is the standard approach. The paper extends all four methods to MICE:

#### MICE Algorithm (Standard):
1. Initialize missing values (e.g., random sampling from observed)
2. For each variable X_j with missingness (j = 1, ..., q):
   - Impute X_j using MI method, treating imputed values of other variables as observed
3. Repeat until distribution stabilizes
4. Generate M imputed datasets
5. Fit analysis model on each → combine via Rubin's rule

#### Distributed MICE Variants:
- **iMICE:** Each site runs MICE independently (0 communication)
- **avgmMICE:** Uses avgmMI for each variable in MICE chain (communication per variable × iterations)
- **cslMICE:** Uses cslMI for each variable in MICE chain
- **siMICE:** Uses siMI for each variable in MICE chain

#### Communication Costs:
- **iMICE:** 0 (most efficient)
- **avgmMICE, cslMICE, siMICE:** Proportional to M (imputations) × variables × iterations
- For general missing patterns, only iMICE is highly communication-efficient
- Example: Scenario 3 with N=1000 requires ~1290-1935 communications for distributed MICE methods

---

### 4. **Experimental Setup**

#### Simulation Studies:

**Three Scenarios:**

1. **Scenario 1: Continuous variable with univariate missingness**
   - p = 2 predictors (X₁ has missing values, X₂ fully observed)
   - X₂ ~ U(-3, 3); X₁|X₂ ~ N(0.2 - 0.5X₂, 1)
   - Y = θ₀ + θ₁X₁ + θ₂X₂ + ε where all θⱼ = 1
   - X₁ missing with probability [1 + exp(-0.3 + 0.2Y - 0.1X₂)]^(-1) → ~50% missing (MAR)

2. **Scenario 2: Binary variable with univariate missingness**
   - Same as Scenario 1 but X₁ ~ Bernoulli(p) where p = [1 + exp(-0.2 + 0.5X₂)]^(-1)
   - ~50% missing rate

3. **Scenario 3: General missing pattern (3 variables missing)**
   - p = 5 predictors; X₁, X₂, X₃ have missing values; X₄, X₅ fully observed
   - X₄, X₅ ~ N(0,1) i.i.d.
   - (X₁, X₂, X₃)|X₄, X₅ ~ N(μ_X, Σ_X) with correlations
   - Y = θ₀ + ΣθⱼXⱼ + ε where all θⱼ = 1
   - Missing generated via logistic regression on Y and fully observed variables
   - ~20% missing per variable, 50% complete case rate

**Simulation Settings (15 configurations):**

| Type | K (sites) | N (total) | Distribution |
|------|-----------|-----------|--------------|
| U (Uneven) | 5 | 250, 500, 1000 | Site 1: majority; others: 15 each |
| U (Uneven) | 10 | 250, 500, 1000 | Site 1: majority; others: 15 each |
| E (Even) | 5 | 250, 500, 1000 | Equal samples per site |
| E (Even) | 10 | 250, 500, 1000 | Equal samples per site |

**Evaluation Metrics:**
- Bias(θ) = ||Eθ̂ - θ₀||₂
- SD(θ) = √E||θ̂ - Eθ̂||₂²
- rMSE(θ) = √E||θ̂ - θ₀||₂²
- Communication costs (number of one-way transmissions)

**Benchmarks:**
- **CD (Complete Data):** Analysis on data before missing values introduced (ideal)
- **CC (Complete Case):** Analysis using only complete cases (naive, biased under MAR/MNAR)

#### Real Data Analysis: Georgia Coverdell Acute Stroke Registry (GCASR)

**Dataset:**
- **Time period:** 2005-2013
- **Hospitals:** 75 hospitals in Georgia
- **Patients:** 68,287 with clinically diagnosed acute stroke
- **Coverage:** ~80% of acute stroke admissions in Georgia
- **Variables:** 203 total, many with missing values

**Analysis Goal:**
- Fit linear regression model for arrival-to-CT time (quality indicator)
- Assess effect of 14 features on CT timing

**Features:**
- **Patient characteristics:** Age, gender, race (African American, White)
- **Pre-hospital:** EMS prenotiﬁcation, arrival time (day/night), NPO status
- **Clinical history:** NIH stroke score, serum total lipid, history of stroke/TIA/cardiac valve prosthesis, family history of stroke
- **Coverage:** Health insurance by Medicare

**Missing Data Characteristics:**
- Only gender and race fully observed
- Missing rates range from 0.04% to 50.73%
- Some hospitals have variables missing for ALL observations
- **Hospitals removed for iMICE:** 9 hospitals (66 remaining with 67,944 patients)
- Sample size per hospital: 18 to 4,333 (median: 578)
- Complete cases across all hospitals: 13,353

**Imputation Setup:**
- M = 20 imputed datasets per method
- Two versions of cslMICE:
  - **cslMICE(M):** Central site = largest hospital (N=4,333)
  - **cslMICE(m):** Central site = median-sized hospital (N=578)

**Evaluation:**
- Compare parameter estimates and 95% CIs across methods
- Count discrepancies in statistical significance (α=0.05) or sign/direction vs. siMICE
- Assess robustness by removing large hospitals (thresholds T = 500, 300, 100)

---

### 5. **Results**

#### Simulation Results:

**Key Findings (Tables 2-4):**

1. **Complete Case (CC) is biased regardless of sample size:**
   - Expected under MAR mechanism
   - Confirms need for MI methods

2. **siMI is gold standard:**
   - Unbiased across all settings (comparable to CD)
   - Slightly larger SD than CD due to imputation uncertainty
   - Serves as benchmark for other methods

3. **iMI performance depends on sample distribution:**
   - **Even distribution:** Less biased, stable performance
   - **Uneven distribution:** Large bias when most sites have small samples
   - **Binary variables (Scenario 2):** More affected than continuous (Scenario 1)
   - **K=10 vs K=5:** Bias increases substantially with more sites (harder to fit stable local models)

4. **avgmMI performance:**
   - **Continuous variables:** Hardly affected by sample distribution (robust)
   - **Binary variables:** Substantially influenced, especially with K=10
   - Achieves low bias when samples are even or total N is large
   - Communication-efficient (2 rounds)

5. **cslMI performance:**
   - **Best when central site has majority of samples (uneven distribution)**
   - **Worse when samples evenly distributed** (central site too small)
   - **Convergence issues:** Some failures when N=250, K=10, even distribution
   - Comparable to siMI when central site is large
   - Sensitive to central site sample size

6. **Communication costs:**
   - **iMI/iMICE:** 0 communications
   - **avgmMI:** 2 rounds for univariate patterns
   - **cslMI:** 3 rounds for univariate patterns
   - **siMI:** 2 rounds for linear models; 10-11 rounds for GLMs (Scenario 2)
   - **MICE methods:** 1290-1935 rounds for Scenario 3 (not communication-efficient except iMICE)

7. **Sample size effects:**
   - All MI methods improve as N increases
   - Bias and rMSE decrease with larger samples
   - siMI maintains superior performance across all N

**Visual Evidence:**
- Tables show avgmMI and siMI consistently achieve low bias and rMSE
- cslMI competitive when uneven distribution favors central site
- iMI struggles with small, uneven samples

#### Real Data Results (GCASR):

**Main Analysis (66-75 hospitals depending on method):**

**Table 5: Method Comparison**

| Method | # Communications | # Discrepancies vs. siMICE |
|--------|------------------|----------------------------|
| CC | 0 | 8 |
| iMICE | 0 | 3 |
| avgmMICE | 4,730 | 2 |
| cslMICE(M) | 7,095 | 2 |
| cslMICE(m) | 7,095 | 4 |
| siMICE | 25,397 | — (benchmark) |

**Key Observations:**

1. **Complete Case (CC) yields most discrepancies:**
   - 8 features show disagreement in statistical significance or direction
   - Demonstrates need for adequate missing data handling

2. **iMICE shows notable discrepancies:**
   - "NIH stroke score," "Serum total lipid," "NPO" differ from siMICE
   - Cannot use 9 hospitals where at least one variable is completely missing
   - Still offers substantial communication savings (0 vs. 25,397)

3. **avgmMICE and cslMICE(M) perform best:**
   - Only 2 discrepancies each vs. siMICE
   - avgmMICE: Most similar CI locations to siMICE; wider intervals for some features
   - cslMICE(M): Most similar CI lengths to siMICE (leverages large central site's curvature)
   - Both offer 5-6× communication savings vs. siMICE

4. **cslMICE(m) has more discrepancies:**
   - 4 discrepancies when using median-sized central site
   - Confirms sensitivity to central site sample size

5. **Robustness to sample size constraints:**
   - When removing large hospitals (T=500, 300, 100), discrepancies increase for all methods
   - **avgmMICE more robust** than cslMICE(M) as sample sizes decrease
   - cslMICE(M) discrepancies grow faster under moderate-to-small sample sizes

**Figure 1: Forest Plot**
- Shows parameter estimates and 95% CIs for all 14 features + intercept
- Visual comparison across CC, iMICE, avgmMICE, cslMICE(M), cslMICE(m), siMICE
- avgmMICE CI locations most similar to siMICE
- cslMICE(M) CI lengths most similar to siMICE

---

### 6. **Advantages of Distributed MI Methods**

#### 1. **Privacy Preservation:**
- **Subject-level data never shared** between sites
- Only aggregated statistics transmitted (model parameters, summary statistics)
- Enhances patient privacy protection
- Strengthens public trust in sensitive health data analysis
- Complies with institutional policies and regulations (HIPAA, GDPR, VA policies)

#### 2. **Communication Efficiency:**
- **iMI/iMICE:** 0 communications (most efficient)
- **avgmMI/cslMI:** 2-3 rounds for univariate patterns (highly efficient)
- **siMI:** 2 rounds for linear models (efficient); many rounds for GLMs (less efficient)
- Compared to pooling data (one-time massive transfer) or iterative FL (100+ rounds), distributed MI is practical

#### 3. **Statistical Validity:**
- Enables proper statistical inference (hypothesis testing, confidence intervals)
- Accounts for imputation uncertainty via multiple imputations
- Rubin's rule combines results appropriately
- **siMI reproduces exact pooled-data results** (gold standard)

#### 4. **Collaboration Without Pooling:**
- Lowers hurdles for inter-institutional collaboration
- No need for central data repository (costly, difficult to secure)
- Eliminates many security, proprietary, legal concerns
- Applicable to distributed data networks of any size

#### 5. **Flexibility:**
- Handles univariate and general missing patterns
- Supports various imputation models (linear, logistic, etc.)
- Accommodates different sample sizes and distributions across sites
- Can be extended to other missing data methods

---

### 7. **Limitations and Future Work**

#### Current Limitations:

1. **Communication costs for general missing patterns:**
   - MICE-based methods (except iMICE) require many communications
   - Proportional to M × variables × iterations
   - May not be practical for very large networks or high-dimensional data

2. **siMI privacy risk:**
   - Transmits entire design matrix structure (X^(k)ᵀX^(k))
   - Summary statistics may still leak individual-level information
   - Differential privacy step could strengthen privacy but not implemented

3. **iMI/avgmMI/cslMI cannot handle completely missing variables:**
   - When a variable is missing for ALL observations in a site, that site cannot participate in imputing that variable
   - Only siMI enables full participation
   - May exclude sites from analysis

4. **Method selection complexity:**
   - **avgmMI:** Better for even sample distribution, continuous variables
   - **cslMI:** Better when central site has majority of samples
   - **iMI:** Simple but unstable for small sites
   - **siMI:** Gold standard but high communication cost for GLMs
   - Practitioners must choose based on network characteristics

5. **Heterogeneity assumptions:**
   - Methods assume heterogeneity can be adequately adjusted in imputation models
   - If not, iMI (no borrowing) may be preferred
   - No explicit method for detecting when borrowing is inappropriate

6. **No fairness or equity considerations:**
   - Does not address whether imputation quality is equitable across demographic groups
   - No analysis of disparities in imputation performance

7. **Limited to structured data:**
   - Focus on tabular EHR data
   - Does not handle unstructured data (clinical notes, images, waveforms)

#### Future Directions:

1. **More communication-efficient MICE methods:**
   - Critical need for handling general missing patterns
   - Potential approaches: adaptive communication, variable grouping, approximations

2. **Enhanced privacy preservation:**
   - Add differential privacy steps to summary statistics
   - Secure multiparty computation for aggregation
   - Homomorphic encryption for parameter transmission

3. **Robust imputation methods:**
   - Extend to Predictive Mean Matching (PMM)
   - Random Forest imputation
   - Generic imputation methods (more challenging, require careful investigation)

4. **Heterogeneity detection and handling:**
   - Automatic site clustering based on similarity
   - Methods to detect when borrowing information is harmful
   - Site-specific vs. global imputation model selection

5. **Fairness and equity:**
   - Ensure imputation quality equitable across demographic groups
   - Investigate impact on health disparities
   - Fairness-aware imputation methods

6. **Real-world deployment:**
   - Integration with existing DHDNs (pSCANNER, PCORnet, Sentinel)
   - Software tools for practitioners
   - Best practice guidelines

---

### 8. **Comparison with Related Work**

#### Traditional MI Methods (Non-Distributed):

1. **Rubin (1987):** Multiple Imputation framework
2. **Van Buuren & Groothuis-Oudshoorn (2011):** MICE algorithm
3. **Standard MI:** Requires pooled data, not applicable to DHDNs

**Distributed MI (this paper) vs. Traditional MI:**
- Enables MI without pooling data (privacy-preserving)
- siMI reproduces traditional MI results exactly
- Other methods trade some accuracy for communication efficiency

#### Prior Distributed Imputation Work:

**Jagannathan & Wright (2008):**
- Privacy-preserving lazy decision-tree imputation
- **Limitations:** 
  - Single imputation only (underestimates uncertainty, improper inference)
  - Not directly applicable to general missing patterns
  - Designed for only two sources
  - Complex decision tree may overfit
  - Not communication-efficient

**This paper vs. Jagannathan & Wright:**
- Multiple imputation (proper uncertainty quantification)
- General missing patterns via MICE
- Scalable to K > 2 sites
- Communication-efficient
- Various method options (iMI, avgmMI, cslMI, siMI)

#### Distributed Learning Methods (for Complete Data):

**Zhang et al. (2013) - AVGM:**
- Average mixture algorithm for distributed complete data
- Achieves optimal convergence rate
- Inspired avgmMI method

**Jordan et al. (2019) - CSL:**
- Communication-efficient surrogate likelihood
- Uses central site's curvature and pooled derivatives
- Inspired cslMI method

**This paper extends these methods to handle missing data:**
- Adapts AVGM/CSL for imputation model fitting
- Sampling from parameter distributions for MI
- Extension to MICE for general missing patterns

---

### 9. **Technical Innovations**

#### 1. **Adaptation of Distributed Learning Algorithms to MI:**
- First application of AVGM and CSL to missing data problem
- Novel sampling schemes for imputation parameters
- Handling of variance parameters (τ²) in distributed setting

#### 2. **Error Variance Sampling for avgmMI:**
- SSE = Σ_k SSE^(k) (pooled sum of squared errors)
- τ² ~ IG(N_c/2, SSE/2)
- α ~ N(α̂_avgm, τ²/N_c² × Σ n_c^(k)² × (Z^(k)ᵀZ^(k) + λI)^(-1))
- Ensures positive variance parameter

#### 3. **Error Variance Sampling for cslMI:**
- SSE = ||x_{1,c}^(1) - Z^(1)α̂_csl||²/n_c^(1) (asymptotic approximation using central site only)
- No additional communication required
- Trade-off: less accurate in finite samples when central site is small

#### 4. **Extension to MICE:**
- Direct application of each MI method to each variable in MICE chain
- Ordering recommendation: impute variables with lower missing rates first
- Iterative refinement until imputation distributions stabilize

#### 5. **Communication Cost Analysis:**
- Detailed accounting of one-way transmissions
- Comparison across methods and scenarios
- Demonstrates impracticality of MICE extensions except iMICE for high communication costs

---

### 10. **Implications for Practice**

#### For Healthcare Institutions:

1. **Enable multi-site collaboration:**
   - Hospitals can collaborate on research without sharing patient data
   - Comply with HIPAA, GDPR, VA policies
   - Lower barriers to inter-institutional research

2. **Improve analysis validity:**
   - Proper handling of missing data reduces bias
   - Better clinical decisions and policy recommendations
   - Increased statistical power by leveraging multi-site data

3. **Practical implementation:**
   - Methods implementable with existing R packages (mice)
   - Code publicly available (https://github.com/changgee/MIDist)
   - Can be integrated into existing DHDNs

#### For Researchers:

1. **Principled missing data handling in DHDNs:**
   - No longer need to resort to complete case analysis or simple mean imputation
   - Proper statistical inference with confidence intervals, p-values
   - Accounts for imputation uncertainty via Rubin's rule

2. **Method selection guidance:**
   - **Small sites, unstable local models:** Use avgmMI, cslMI, or siMI (borrow information)
   - **Evenly distributed samples, continuous variables:** Use avgmMI
   - **Uneven distribution, large central site:** Use cslMI
   - **Communication constraints, stable local models:** Use iMI
   - **Need exact pooled-data results, linear models:** Use siMI
   - **Need exact results, GLMs, communication OK:** Use siMI (accept higher cost)

3. **Research quality improvement:**
   - Reduces bias in multi-site clinical studies
   - Increases generalizability by leveraging diverse populations
   - Strengthens public trust through privacy preservation

#### For Policy Makers:

1. **Privacy-preserving collaboration model:**
   - Demonstrates feasible approach for multi-institutional research under strict regulations
   - Can inform policies on data sharing and collaborative research
   - Balances scientific rigor with privacy protection

2. **Support for distributed research infrastructure:**
   - DHDNs (pSCANNER, PCORnet, Sentinel) need missing data tools
   - This work provides methodological foundation
   - Can guide funding priorities for health data networks

---

### 11. **Key Equations and Algorithms**

#### Multiple Imputation Framework:

**Analysis Model:**
```
y = θ₀ + θ₁x₁ + ... + θₚxₚ + ε
```

**Imputation Model (Linear, Continuous X₁):**
```
X₁ = α₀ + α₁Y + Σⱼ₌₂ᵖ αⱼXⱼ + ζ,  ζ ~ N(0, τ²)

Priors:
π(τ²) ∝ IG(1/2, 1/2)
α|τ² ~ N(0, τ²λ⁻¹I)

Posterior:
τ²|Z_c ~ IG((N_c + 1)/2, (SSE + 1)/2)
α|τ², Z_c ~ N((Z_c^T Z_c + λI)⁻¹ Z_c^T x_{1,c}, τ²(Z_c^T Z_c + λI)⁻¹)

where SSE = x_{1,c}^T x_{1,c} - x_{1,c}^T Z_c(Z_c^T Z_c + λI)⁻¹ Z_c^T x_{1,c}
```

**Imputation Model (Logistic, Binary X₁):**
```
X₁ ~ Bernoulli(expit(z^T α))

Prior: α ~ N(0, λ⁻¹I)

Approximate posterior:
Cov(α̂) ≈ (Z_c^T W_c Z_c + λI)⁻¹
where W_c is diagonal with w_ii = expit(z_i^T α̂)(1 - expit(z_i^T α̂))

Sample: α ~ N(α̂, Cov(α̂))
```

#### avgmMI Algorithm:

```
For k = 1 to K:
  Fit local imputation model → α̂^(k), Cov(α̂^(k))
  Send α̂^(k), Cov(α̂^(k)) to central site

Central site:
  α̂_avgm = (1/N_c) Σ n_c^(k) α̂^(k)
  Cov(α̂_avgm) = (1/N_c²) Σ (n_c^(k))² Cov(α̂^(k))

For m = 1 to M:
  Sample α_m ~ N(α̂_avgm, Cov(α̂_avgm))
  Send α_m to all sites
  Each site imputes missing values using α_m
  Each site fits analysis model → θ̂_m, Var(θ̂_m)

Combine via Rubin's rule:
  θ̂ = (1/M) Σ θ̂_m
  Var(θ̂) = (1/M) Σ Var(θ̂_m) + (1 + 1/M) × (1/(M-1)) Σ (θ̂_m - θ̂)(θ̂_m - θ̂)^T
```

#### cslMI Algorithm:

```
Central site (site 1):
  Find α̃ = arg min_α L_1(α) (initial estimate)
  Send α̃ to all sites

For k = 1 to K:
  Compute ∇L_k(α)|_{α=α̃}
  Send ∇L_k(α̃) to central site

Central site:
  α̂_csl = arg min_α L̃(α)
  where L̃(α) = L_1(α) - ⟨∇L_1(α̃) - ∇L(α̃), α⟩
  Cov(α̂_csl) = (N_c/n^(1)) × [∇²L_1(α)]^{-1}|_{α=α̂_csl}

For m = 1 to M:
  Sample α_m ~ N(α̂_csl, Cov(α̂_csl))
  Send α_m to all sites
  [Imputation and analysis as in avgmMI]
```

#### siMI Algorithm (Linear Model):

```
For k = 1 to K:
  Compute X^(k)ᵀ X^(k) and X^(k)ᵀ x_{1,c}^(k)
  Send to central site

Central site:
  X^T X = Σ X^(k)ᵀ X^(k)
  X^T x_{1,c} = Σ X^(k)ᵀ x_{1,c}^(k)
  Compute α̂_si = (X^T X + λI)⁻¹ X^T x_{1,c}
  Compute SSE, τ² posterior, α posterior as in standard MI

For m = 1 to M:
  Sample (τ², α_m) from joint posterior
  Send α_m to all sites
  [Imputation and analysis as in avgmMI]
```

---

### 12. **Reproducibility**

#### Code Availability:
- **GitHub:** https://github.com/changgee/MIDist
- **Language:** R
- **Dependencies:** R package 'mice'
- **Contents:** 
  - Custom implementations of iMI, avgmMI, cslMI, siMI
  - Extensions to iMICE, avgmMICE, cslMICE, siMICE
  - Simulation code for all three scenarios
  - Analysis code for GCASR real data

#### Data Availability:
- **Simulated data:** Can be reproduced using provided code
- **GCASR data:** Restrictions apply; request access from Georgia Coverdell Acute Stroke Registry

#### Simulation Parameters:
- **Sites:** K = 5 or 10
- **Sample sizes:** N = 250, 500, or 1000
- **Sample distributions:** Uneven (U) or Even (E)
- **Missing rates:** ~20-50%
- **Imputations:** M = 20
- **Monte Carlo replicates:** 1000
- **Regularization:** λ chosen to be small (minimize bias from regularization)

#### Real Data Parameters:
- **Imputations:** M = 20
- **Central site for cslMICE(M):** Hospital with largest sample (N=4,333)
- **Central site for cslMICE(m):** Hospital with median sample (N=578)

---

### 13. **Key Contributions**

1. **First distributed MI methods for horizontally partitioned incomplete data in DHDNs**
2. **Four communication-efficient approaches:** iMI, avgmMI, cslMI, siMI (plus MICE variants)
3. **Novel adaptation of AVGM and CSL algorithms to missing data problem**
4. **Handles univariate and general missing patterns** via MICE extension
5. **Privacy-preserving:** Subject-level data never shared
6. **Statistical validity:** Proper inference via multiple imputation and Rubin's rule
7. **Extensive simulation studies** across 15 configurations
8. **Real-world validation** on GCASR acute stroke dataset (68,287 patients, 75 hospitals)
9. **Publicly available code** (R implementation)
10. **Guidance for method selection** based on network characteristics

---

### 14. **Citation**

```bibtex
@article{chang2020multiple,
  title={Multiple imputation for analysis of incomplete data in distributed health data networks},
  author={Chang, Changgee and Deng, Yi and Jiang, Xiaoqian and Long, Qi},
  journal={Nature Communications},
  volume={11},
  number={1},
  pages={5467},
  year={2020},
  publisher={Nature Publishing Group},
  doi={10.1038/s41467-020-19270-2}
}
```

---

### 15. **Related Papers**

1. **Rubin, D.B. (1987):** Multiple Imputation for Nonresponse in Surveys (MI framework)
2. **Van Buuren & Groothuis-Oudshoorn (2011):** MICE R package
3. **Zhang et al. (2013):** Communication-efficient algorithms for statistical optimization (AVGM)
4. **Jordan et al. (2019):** Communication-efficient distributed statistical inference (CSL)
5. **Jagannathan & Wright (2008):** Privacy-preserving imputation (prior distributed imputation work)
6. **Little & Rubin (2014):** Statistical Analysis With Missing Data (missing data theory)

---

### 16. **Comparison with FedIMPUTE (2025)**

This 2020 paper and FedIMPUTE (2025) both address distributed missing value imputation in healthcare but differ in approach:

| Aspect | This Paper (2020) | FedIMPUTE (2025) |
|--------|-------------------|-------------------|
| **Method** | Statistical approach (AVGM, CSL, SI) | Statistical approach (DAC algorithm) |
| **Communication** | 0-3 rounds (MI); 100s-1000s (MICE) | 1-3 rounds (always efficient) |
| **Imputation** | Bayesian MI (linear, logistic regression) | Regression-based (MICE, single regression) |
| **Heterogeneity** | Assumed addressable via covariates | Explicitly tested (covariate & model shifts) |
| **Evaluation** | Bias, SD, rMSE of regression coefficients | MSE of imputed values + downstream AUROC |
| **Dataset** | GCASR (68K patients, 75 hospitals, 203 vars) | Duke EHR (80K patients, 3 hospitals, 20 vars) |
| **Missing rates** | 20-50% | Up to 80% |
| **Missing mechanisms** | MCAR, MAR, MNAR (assumed, not tested) | MCAR, MAR, MNAR (explicitly tested) |

**FedIMPUTE builds upon this foundational work** with more focus on robustness to heterogeneity and practical deployment in EHR settings.

---

## Summary

This **groundbreaking paper** is the **first systematic investigation of multiple imputation methods for distributed incomplete data in health data networks**. By developing four communication-efficient privacy-preserving approaches (iMI, avgmMI, cslMI, siMI) and their MICE extensions, the work enables hospitals and healthcare systems to collaboratively handle missing data while complying with strict privacy regulations. The methods achieve **statistical validity comparable to pooled-data analysis** while **never sharing subject-level information**, making them practical for real-world DHDNs. Validated through extensive simulations and application to a 68,287-patient acute stroke registry across 75 hospitals, the paper provides both **methodological innovation** and **practical implementation guidance** (publicly available R code). It lays the foundation for **privacy-preserving distributed data preprocessing** in biomedical research and has inspired subsequent work like FedIMPUTE (2025).

