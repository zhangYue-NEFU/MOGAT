MOGAT: A Multi-Omics Graph-Attention Transformer for Pan-Cancer Survival Prediction
MOGAT is a novel deep learning framework designed for pan-cancer survival prediction and biomarker discovery. By integrating Stacked Denoising Autoencoders (SDAE), Graph Attention Networks (GAT), and Transformer architectures, MOGAT effectively captures both the molecular heterogeneity and common biological principles across different cancer types within a single unified model.

Paper: MOGAT: A Multi-Omics Graph-Attention Transformer for Pan-Cancer Survival Prediction and Biomarker Discovery

🚀 Key Features
Multi-Omics Integration: Seamlessly integrates mRNA expression, miRNA expression, and Proteomics (RPPA) data.

Graph Attention Mechanism: Utilizes GAT (Graph Attention Network) with Global Attention Pooling to extract topological features from static Protein-Protein Interaction (PPI) networks.

Transformer Fusion: Employs a Transformer Encoder to capture long-range dependencies and interactions between different omics modalities.

Pan-Cancer Modeling: Innovative use of Cancer Embeddings allows a single model to robustly predict survival outcomes across 18+ cancer types (e.g., LUAD, STAD, BRCA).

Rigorous Validation (Zero-Leakage): Implements a strict "Zero-Leakage" data processing protocol where imputation, scaling, and feature selection are fitted solely on the training set to ensure reproducibility and validity.

🛠️ System Architecture
The MOGAT framework consists of three main modules:

Feature Encoders:

mRNA & miRNA: Processed via AutoPower2AE (Stacked Denoising Autoencoder) for dimensionality reduction and robust feature extraction.

Protein: Mapped onto a PPI network (STRING database) and processed via DynamicGNN (Multi-head GAT + Attention Pooling).

Multi-Modal Fusion:

Features are fused using ProjectedTransformerFusion, incorporating learnable Cancer Embeddings to distinguish specific cancer patterns.

Survival Prediction:

The fused representation is passed to a prediction head optimized via Cox Proportional Hazards Loss.
<img width="2500" height="1406" alt="幻灯片1" src="https://github.com/user-attachments/assets/90348bd7-2dc2-415f-960d-f2ad8ec4d737" />
<img width="2500" height="1406" alt="幻灯片1" src="https://github.com/user-attachments/assets/047e2da9-b875-4ed5-a4a6-65a3a6a0bac1" />

