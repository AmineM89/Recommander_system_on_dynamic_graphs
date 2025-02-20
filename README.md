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

## Implementation Details
### Modifications to Euler Model
Several modifications were made to the original Euler implementation to enhance its performance and flexibility:
- **Custom Node2Vec Integration**: Added an option to either replace or concatenate embeddings for unseen nodes.
- **Optimized RNN Architectures**: Replaced vanilla RNN with GRU/LSTM options for better temporal encoding.
- **Hyperparameter Tuning**: Added configuration support for batch sizes, learning rates, and embedding dimensions.

### Node Embeddings Strategy
For handling new (unseen) nodes during link prediction, we experimented with two different strategies:
1. **Replacement Strategy**: The embeddings of new nodes are entirely replaced by Node2Vec embeddings. This allows the model to leverage pre-trained structural representations of the graph.
2. **Concatenation Strategy**: Instead of replacing, we concatenate the Node2Vec embeddings with the GCN-generated embeddings. This approach enriches the node representations by combining topological (Node2Vec) and learned graph structure features (GCN).

Experimental results indicate that the concatenation strategy generally performs better, especially for datasets with a high proportion of new nodes (e.g., DBLP and FB). However, in some cases, replacing the embeddings yielded comparable results, particularly in link detection tasks.

## Node2Vec Training
The following code snippet demonstrates the training of Node2Vec embeddings for dynamic graphs:

```python
if embed_type is not 'default':
    try:
        embeds = torch.load(f"n2v_embeds_{dataset}.pth")
    except:
        start_idx = 0
        embeds = []
        for i in range(len(data.eis)):
            ei = data.tr(start_idx + i)
            num_nodes = data.x.size(0)

            all_nodes = torch.arange(num_nodes)
            connected_nodes = ei.view(1,-1).unique()
            remaining_nodes = torch.tensor(list(set(all_nodes.numpy()) - set(connected_nodes.numpy())))
            dummy_edges = remaining_nodes.repeat(2,1)
            all_edges = torch.cat((ei,dummy_edges),dim=1)

            model_n2v = Node2Vec(all_edges,
                                embedding_dim=32,
                                walk_length=50,
                                context_size=25,
                                walks_per_node=25,
                                num_negative_samples=1,
                                p=1.0,
                                q=1.0,
                                sparse=True)

            loader = model_n2v.loader(batch_size=128, shuffle=True)
            optimizer = torch.optim.SparseAdam(list(model_n2v.parameters()), lr=0.01)

            def train():
                model_n2v.train()
                total_loss = 0
                for pos_rw, neg_rw in loader:
                    optimizer.zero_grad()
                    loss = model_n2v.loss(pos_rw, neg_rw)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                return total_loss / len(loader)

            for epoch in range(1, 151):
                loss = train()
                model_n2v.eval()
                z = model_n2v()

            model_n2v.eval()
            z = model_n2v()
            embeds.append(z)
        embeds = torch.stack(embeds)
        torch.save(embeds, f"n2v_embeds_{dataset}.pth")
```

This script trains Node2Vec embeddings for different graph timestamps and saves them for later use in the model.

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

## Full Report
For detailed methodology and experimental results, see the full report

<a href="https://github.com/user-attachments/files/18669799/recommander.systems.1.pdf">
    <img src= "https://github.com/user-attachments/assets/9b65e15a-a895-468b-80b5-c6f50df0ad42" alt="Full report" width="800"/>
</a>

## References
For more details on the original Euler model, please refer to the original paper:
King, I. J., & Huang, H. H. (2023). Euler: Detecting network lateral movement via scalable temporal link prediction. ACM Transactions on Privacy and Security.
