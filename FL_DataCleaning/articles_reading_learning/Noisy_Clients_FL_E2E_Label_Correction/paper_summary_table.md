# Paper Summary Table: Tackling Noisy Clients in Federated Learning with End-to-end Label Correction

| Element | Answer |
|---------|--------|
| **ML or FL** | FL (Federated Learning) |
| **Handles fairness?** | No |
| **Outliers** | No |
| **Missing values** | No |
| **Label errors** | Yes - Primary focus of the paper. Addresses heterogeneous label noise across clients with varying noise rates |
| **Repair data?** | Yes - End-to-end label correction via backpropagation. Learns possible ground-truth labels for noisy clients' datasets |
| **Topics** | Federated Learning, Noisy Label Learning, Label Noise Correction, Noisy Client Detection, Data Quality, Privacy-Preserving Learning, Non-IID Data, Robust Training, Two-Stage Framework |
| **Datasets** | CIFAR-10, CIFAR-100, CIFAR-10-N (human annotation, 40.2% noise), CIFAR-100-N (human annotation, 40.2% noise), Clothing1M (1M images, 38.46% systematic noise) |
| **Code** | Yes - https://github.com/Sprinter1999/FedELC |
| **Implemented?** | Yes - 16 baseline methods implemented and compared including FedAvg, FedProx, FedExP, Krum, TrimmedMean, Median, Co-teaching, Co-teaching+, Joint Optim, SELFIE, Symmetric CE, DivideMix, Robust FL, FedLSR, FedRN, FedNoRo |

## Key Method
Two-stage framework:
1. **Stage 1**: Detect noisy clients using Gaussian Mixture Model (GMM) on class-wise loss
2. **Stage 2**: Correct labels of noisy clients via end-to-end learning with triplet supervision (classification loss + compatibility loss + entropy loss)

## Key Results
- Best performance on CIFAR-10: 76.81% Precision, 76.72% Recall
- Best performance on Clothing1M: 71.64% accuracy
- Successfully improves data quality by correcting local labels

