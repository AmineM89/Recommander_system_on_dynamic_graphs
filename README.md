# 🔗 Recommender System Prediction with Dynamic Graph Representation Learning

> **Link prediction in evolving graphs using GNNs and RNNs – enhanced with Node2Vec and robust to noisy data.**

---

## 🧠 Overview
This project tackles link prediction in **dynamic graphs**, a crucial task in recommender systems and fraud detection. Unlike static graphs, dynamic graphs evolve over time, capturing real-world complexity more effectively.

We address two key challenges:

1. ❄️ **Cold Start Problem** – Predicting links for unseen nodes.
2. 🧪 **Noise Robustness** – Measuring how random edge additions affect performance.

Our approach is based on the **Euler model**: a state-of-the-art hybrid architecture combining Graph Neural Networks (GNNs) and Recurrent Neural Networks (RNNs) for topological and temporal learning.

---

## 🚀 Features
- ✅ Link prediction on **temporal** and **topological** patterns.
- 🔍 **Cold Start node handling** via Node2Vec (replacement or concatenation).
- 🔁 Noise injection (1–10%) to test robustness.
- 📊 Evaluation on 3 datasets: `Enron10`, `DBLP`, and `Facebook`.

---

## 🧱 Model Architecture

| Component | Role |
|----------|------|
| `GCN` Encoder | Learns graph structure embeddings |
| `RNN` Encoder | Captures temporal dynamics |
| `Dot Product Decoder` | Predicts edge existence |
| `Binary Cross Entropy` | Optimizes link prediction |

---

## ⚙️ Implementation Highlights

### 🧩 Euler Model Enhancements
- 🔀 **Node2Vec** for new nodes (2 strategies: replace vs. concat)
- 🧠 Swappable RNN cells (`RNN`, `GRU`, `LSTM`)
- ⚙️ Full config support: learning rate, batch size, embedding dims...

### 🧠 Node Embedding Strategies
| Strategy | Description | Best for |
|---------|-------------|----------|
| **Replacement** | Use Node2Vec only | Simpler graphs |
| **Concatenation** | Combine Node2Vec + GCN | High-cold-start scenarios |

🔎 *Concatenation yields better results on `DBLP` and `Facebook` datasets.*

---

## 🧪 Node2Vec Training Example

```python
model_n2v = Node2Vec(
    edges, embedding_dim=32, walk_length=50,
    context_size=25, walks_per_node=25, 
    num_negative_samples=1, sparse=True
)

optimizer = torch.optim.SparseAdam(model_n2v.parameters(), lr=0.01)

for epoch in range(150):
    model_n2v.train()
    for pos_rw, neg_rw in model_n2v.loader(batch_size=128):
        optimizer.zero_grad()
        loss = model_n2v.loss(pos_rw, neg_rw)
        loss.backward()
        optimizer.step()
```

Embeddings are saved per graph snapshot and reused during training.

---

## 🧪 Experimental Setup

| Parameter | Value |
|----------|-------|
| 📁 Datasets | `Enron10`, `DBLP`, `Facebook` |
| 📏 Metrics | `AUC`, `Average Precision (AP)` |
| ⚖️ Baseline | Original Euler model |
| 💥 Noise Levels | 0%, 1%, 5%, 10% |

---

## 📈 Key Results

- 🎯 **Cold Start**: Node2Vec concatenation significantly improves performance for unseen nodes.
- ⚠️ **Noise Sensitivity**: Most models degrade with noise, but DBLP shows resilience (less overfitting).
- 📐 **Architecture Flexibility**: Swappable RNNs and embedding fusion yield consistent gains.

---

## 🔮 Future Work
- 📚 Extend to other dynamic GNNs (e.g., TGN, DySAT).
- 🧬 Integrate graph data augmentation techniques.
- 🛡️ Explore adversarial training for robustness.

---

## 📄 Full Report & Visuals

👉 **[📘 Read the Full Report (PDF)](https://github.com/user-attachments/files/18669799/recommander.systems.1.pdf)**

![Preview](https://github.com/user-attachments/assets/9b65e15a-a895-468b-80b5-c6f50df0ad42)

---

## 📚 References

> King, I. J., & Huang, H. H. (2023).  
> *Euler: Detecting network lateral movement via scalable temporal link prediction*.  
> _ACM Transactions on Privacy and Security_.

---

## 🧑‍💻 Author

**Amine Mohabeddine** – [GitHub](https://github.com/AmineM89) | [LinkedIn](https://linkedin.com/in/AmineM89)
