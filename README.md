# Vision-Language-Joint-Predictive-Embedding-JEPA-Inspired-
A JEPA-inspired framework that predicts semantic meaning directly in latent space instead of generating tokens autoregressively, enabling faster and more stable multimodal representation learning.

A research-oriented implementation of joint image–text representation learning using contrastive learning and JEPA-style predictive objectives on the Flickr8k dataset.
This project explores how visual embeddings can be aligned with semantic text embeddings using deep neural networks, and investigates the effect of masked predictive learning inspired by the JEPA (Joint Embedding Predictive Architecture) paradigm.

# Project Objectives

Learn a shared embedding space between images and captions.
Predict semantic text embeddings directly from images.
Evaluate retrieval quality using cosine similarity and Recall@K.
Analyze embedding geometry using t-SNE visualization.
Compare:

Approach 1 — Direct Contrastive Alignment (No Masking)

Approach 2 — JEPA-Inspired Masked Predictive Learning

# Model Architecture

🔹 Image Encoder
ResNet-18 (ImageNet pretrained)
Final projection to 512-dimensional embedding
🔹 Text Encoder
BERT-base-uncased
CLS token embedding projected to 512 dimensions
🔹 Predictor Network
Multi-layer MLP with BatchNorm and ReLU
Predicts target text embedding from context embeddings

# Dataset
Flickr8k Dataset
~40,000 image–caption pairs
Subsampled to 15k–25k pairs for faster training
Each image has multiple captions
Preprocessing:
Images resized to 224×224 and normalized
Captions tokenized using BERT tokenizer (max length = 32)

# Training Approaches

# Approach 1 — Direct Contrastive Alignment (No Masking)

Objective:
Predict caption embedding directly from image embedding using InfoNCE contrastive loss.

![Non-Masking Architecture](working%20of%20nonmasking.png)

Loss:
InfoNCE contrastive loss on cosine similarity

Results:

| Metric                 | Value  |
| ---------------------- | ------ |
| Mean Cosine Similarity | ~0.79  |
| Recall@1               | ~12.7% |
| Recall@5               | ~46.9% |
| Recall@10              | ~63.6% |

Observations:

Strong alignment between predicted and true embeddings.
t-SNE visualization shows overlapping manifolds.
Retrieval quality is stable and consistent.
Fast convergence.

# Approach 2 — JEPA-Inspired Masked Predictive Learning

Objective:
Predict the full semantic embedding of a caption using:

i)Image embedding

ii)Masked caption embedding (partial text)

![Masking Architecture](working%20of%20masking.png)

This mimics JEPA behavior:
Partially observed context
Predict latent semantic target
No explicit label supervision

Key Differences: 

Random token masking in captions
Contextual predictive learning
Harder optimization problem

Results:

| Metric    | Value  |
| --------- | ------ |
| Recall@1  | ~3.0%  |
| Recall@5  | ~8.9%  |
| Recall@10 | ~13.4% |

Observations:

Embeddings form structured clusters but remain globally misaligned.
t-SNE shows separated manifolds between predicted and true embeddings.
Masking increases representation difficulty.
Highlights alignment challenges in predictive embedding models.

# Evaluation Metrics

i) Cosine Similarity Distribution

ii) Recall@K Retrieval Accuracy

iii) t-SNE Visualization
iv) Confidence Interval Estimation
v) Embedding Alignment Curves

# Visualizations

Included plots:

Cosine similarity histogram ,
Retrieval performance curves , 
t-SNE embedding projections 

These provide insight into representation geometry and alignment behavior.

# Key Learnings

Direct contrastive learning provides strong alignment quickly.
JEPA-style masked prediction learns structure but struggles with manifold alignment without long training and stabilization.
Visualization is critical for diagnosing embedding collapse and drift.
Predictive objectives require stronger regularization and larger datasets.

# Technologies Used

Python

PyTorch

Torchvision

HuggingFace Transformers

NumPy

Scikit-learn

Matplotlib

Kaggle GPU

# Author
# Abhijnan Das
