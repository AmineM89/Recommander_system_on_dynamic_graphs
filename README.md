# Recommender System Prediction with Dynamic Graph Representation Learning

## Overview
This project explores link prediction in dynamic graphs for recommender systems. Unlike static graphs, dynamic graphs evolve over time, making them more suited for modeling real-world networks. Our work aims to improve the **flexibility** and **robustness** of link prediction models by addressing two key challenges:

1. **Cold Start Problem** – Predicting links for nodes that were not seen during training.
2. **Sensitivity to Noise** – Analyzing how random edge additions impact model performance.

We base our experiments on the **Euler model**, a state-of-the-art approach combining Graph Neural Networks (GNNs) and Recurrent Neural Networks (RNNs) for topological and temporal encoding, respectively.

## Features
- **Link Prediction & Link Detection** on dynamic graphs.
- **Node2Vec Enhancements** for unseen nodes (cold start).
- **Impact Analysis of Noisy Edges** on model performance.
- **Evaluation on Multiple Datasets** (Enron10, DBLP, FB).

## Model Architecture
The Euler model consists of:
- **GCN Encoder** for topological embeddings.
- **RNN Encoder** for temporal embeddings.
- **Decoder** using dot product similarity.
- **Loss Function** optimizing link prediction accuracy.

## Experimental Setup
- **Datasets**: Enron10, DBLP, FB.
- **Metrics**: Average Precision (AP) & Area Under the Curve (AUC).
- **Baseline Comparison** with original Euler model results.
- **Node2Vec Integration**: Replacing or concatenating embeddings for unseen nodes.
- **Noise Injection**: Adding 1%, 5%, and 10% random edges to test robustness.

## Key Findings
- **Cold Start Improvements**: Concatenating Node2Vec embeddings improved performance for DBLP and FB datasets.
- **Noise Sensitivity**: Increased noise led to degraded performance in most cases, except for DBLP, where it reduced overfitting.
- **Performance vs. Dataset Characteristics**: The proportion of new nodes significantly impacted results.

## Future Work
- Extending experiments to other dynamic graph models.
- Investigating additional augmentation techniques for cold start.
- Exploring adversarial training to counteract noisy edges.

## References
See the [full paper](https://github.com/user-attachments/files/18648354/recommander.systems.pdf) for detailed methodology and experimental results.
