# Multiple Imputation for Analysis of Incomplete Data in Distributed Health Data Networks

## 📄 Paper Information

**Title:** Multiple imputation for analysis of incomplete data in distributed health data networks

**Authors:** Changgee Chang, Yi Deng, Xiaoqian Jiang, Qi Long

**Journal:** Nature Communications (2020)

**DOI:** https://doi.org/10.1038/s41467-020-19270-2

**Code:** https://github.com/changgee/MIDist

---

## 📁 Files in This Directory

| File | Description |
|------|-------------|
| `MultipleImputationForIncompleteDataAnalysis.docx` | Original paper (Word format) |
| `MultipleImputationForIncompleteDataAnalysis.pdf` | Original paper (PDF format) |
| `multiple_imputation_paper_text.txt` | **Extracted text** from the DOCX file for easy reference |
| `paper_analysis.md` | **Comprehensive analysis** with detailed methodology, results, and technical innovations |
| `paper_summary_table.md` | **Quick reference table** with classification, method comparison, and usage guidance |
| `README.md` | This file - navigation guide |

---

## 🎯 Quick Classification

| Category | Answer |
|----------|--------|
| **ML or FL** | **FL (Federated Learning)** - Distributed learning on horizontally partitioned data |
| **Handles fairness?** | No |
| **Outliers** | No |
| **Missing values** | ✅ **Yes (Primary Focus)** |
| **Label errors** | No |
| **Repair data?** | ✅ **Yes** - Imputes missing values using multiple imputation |

---

## 🔑 Key Contributions

This is the **first paper to develop multiple imputation methods for distributed incomplete data** in health data networks.

### Four Distributed MI Methods:

1. **iMI (Independent MI):** Each site imputes independently (0 communication)
2. **avgmMI (Average Mixture MI):** Weighted average of site estimates (2 communications)
3. **cslMI (Surrogate Likelihood MI):** Uses central site curvature (3 communications)
4. **siMI (Sufficient Information MI):** Gold standard, reproduces pooled-data results (2+ communications)

### MICE Extensions for General Missing Patterns:
- iMICE, avgmMICE, cslMICE, siMICE

---

## 📊 Datasets

### Simulations:
- 3 scenarios (univariate & general missing patterns)
- 5-10 sites
- Sample sizes: 250-1000
- Missing rates: 20-50%

### Real Data:
- **Georgia Coverdell Acute Stroke Registry (GCASR)**
- 68,287 patients from 75 hospitals (2005-2013)
- 203 variables with extensive missingness
- Analyzed 66-75 hospitals depending on method

---

## 🏆 Main Results

### Real Data (GCASR) - Discrepancies vs. Gold Standard (siMICE):

| Method | Communications | Discrepancies |
|--------|----------------|---------------|
| Complete Case | 0 | 8 ❌ |
| iMICE | 0 | 3 |
| **avgmMICE** | 4,730 | **2** ✅ |
| **cslMICE(M)** | 7,095 | **2** ✅ |
| cslMICE(m) | 7,095 | 4 |
| siMICE | 25,397 | — (benchmark) |

**avgmMICE and cslMICE(M) achieve near-gold-standard accuracy with 5-6× fewer communications!**

---

## 💡 When to Use Each Method

### Use **iMI/iMICE** when:
- ❌ No communication possible
- ✅ Sites have stable local models
- ✅ Variables not completely missing in any site

### Use **avgmMI/avgmMICE** when:
- ✅ Samples evenly distributed across sites
- ✅ Imputing continuous variables
- ✅ Need balance of accuracy & efficiency

### Use **cslMI/cslMICE** when:
- ✅ One site has majority of samples
- ✅ Central site has large sample size
- ✅ Need good accuracy with moderate communication

### Use **siMI/siMICE** when:
- ✅ Need exact pooled-data results
- ✅ Variables completely missing in some sites
- ✅ Highest accuracy required
- ⚠️ Can accept higher communication cost

---

## 🔬 Technical Innovation

### Novel Adaptations:
- First application of **AVGM** and **CSL** algorithms to missing data
- Bayesian imputation with proper uncertainty quantification
- Extension to **MICE** for general missing patterns
- Communication cost analysis

### Privacy Preservation:
- ✅ Subject-level data **never shared**
- ✅ Only aggregated statistics transmitted
- ✅ Complies with HIPAA, GDPR, VA policies

### Statistical Validity:
- ✅ Proper inference via **Rubin's rule**
- ✅ Accounts for imputation uncertainty
- ✅ **siMI reproduces exact pooled-data results**

---

## 📖 How to Read This Analysis

1. **Start here (README.md)** for overview
2. **Read `paper_summary_table.md`** for quick reference
3. **Deep dive into `paper_analysis.md`** for comprehensive understanding
4. **Reference `multiple_imputation_paper_text.txt`** for original paper text

---

## 🔗 Related Work

### Foundational Papers:
- **Rubin (1987):** Multiple Imputation framework
- **Van Buuren & Groothuis-Oudshoorn (2011):** MICE algorithm
- **Zhang et al. (2013):** AVGM for distributed learning
- **Jordan et al. (2019):** CSL for communication-efficient inference

### Follow-up Work:
- **FedIMPUTE (2025):** Builds upon this work with DAC algorithm, tests robustness to heterogeneity

---

## 💻 Implementation

### Code Repository:
- **GitHub:** https://github.com/changgee/MIDist
- **Language:** R
- **Dependencies:** R package 'mice'

### Reproducibility:
- ✅ Simulation code provided
- ✅ All methods fully implemented
- ⚠️ Real GCASR data has access restrictions

---

## 🎓 Citation

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

## ✅ Summary

This groundbreaking paper enables **privacy-preserving collaborative missing data handling** in distributed health data networks. By developing **four communication-efficient multiple imputation methods** that never share subject-level data, it addresses a critical gap in multi-institutional healthcare research. The methods achieve **statistical validity comparable to pooled-data analysis** while complying with strict privacy regulations, making them practical for real-world deployment in systems like pSCANNER, PCORnet, and Sentinel.

**Key achievement:** avgmMICE and cslMICE achieve near-gold-standard accuracy with **5-6× fewer communications** than the exact method (siMICE).

