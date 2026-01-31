"""
Graph Convolutional Networks (GCN) - Complete Example
======================================================

This script demonstrates:
1. USE CASE: Social network node classification
2. MATHEMATICS: Complete mathematical derivation
3. CODE: From-scratch implementation using PyTorch

Author: Silvia Nistor
Date: 2026-01-31
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

print("="*80)
print("GRAPH CONVOLUTIONAL NETWORKS (GCN) - COMPLETE EXAMPLE")
print("="*80)

# ============================================================================
# PART 1: USE CASE - Social Network Node Classification
# ============================================================================

print("\n" + "="*80)
print("PART 1: USE CASE")
print("="*80)

print("""
PROBLEM: Social Network Community Detection
-------------------------------------------
Given:
  - Nodes: People in a social network
  - Edges: Friendships between people
  - Features: Each person has attributes (age, interests, activity level)
  - Labels: Some people are labeled with their community/group

Goal: Predict the community membership of unlabeled users using:
  1. Their own features
  2. The network structure (who they're connected to)

Why GCN?
  Traditional neural networks ignore graph structure. GCNs aggregate information
  from neighboring nodes, leveraging the principle: "you are similar to your friends."
""")

# ============================================================================
# PART 2: MATHEMATICAL FOUNDATION
# ============================================================================

print("\n" + "="*80)
print("PART 2: MATHEMATICAL FOUNDATION")
print("="*80)

print("""
Graph Representation
--------------------
A graph G = (V, E) consists of:
  - V: set of N nodes
  - E: set of edges
  - A ∈ ℝ^(N×N): adjacency matrix where A_ij = 1 if nodes i and j are connected
  - X ∈ ℝ^(N×F): node feature matrix (F features per node)

Graph Convolution Operation
----------------------------
Core idea: Aggregate features from neighboring nodes

Naive approach:
  H^(l+1) = σ(A H^(l) W^(l))

Problem: Doesn't include the node's own features, and different degree nodes 
         get different scale features.

Solution: Add self-loops and normalize

1. Add self-loops: Ã = A + I_N
2. Compute degree matrix: D̃_ii = Σ_j Ã_ij
3. Symmetric normalization:

  H^(l+1) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l))

Where:
  - H^(l) ∈ ℝ^(N×d_l): node representations at layer l (with H^(0) = X)
  - W^(l) ∈ ℝ^(d_l × d_(l+1)): trainable weight matrix
  - σ: activation function (e.g., ReLU)
  - D̃^(-1/2) Ã D̃^(-1/2): normalized adjacency matrix

Why This Works
--------------
The normalized adjacency matrix computes a weighted average of neighbor features:
  - Nodes with many neighbors (high degree) don't dominate
  - Each node contributes based on both its degree and its neighbor's degree

Multi-layer GCN
---------------
Stack multiple layers to aggregate information from k-hop neighborhoods:

  H^(1) = σ(D̃^(-1/2) Ã D̃^(-1/2) X W^(0))
  H^(2) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(1) W^(1))
  Z = softmax(D̃^(-1/2) Ã D̃^(-1/2) H^(2) W^(2))
""")

# ============================================================================
# PART 3: DATA PREPARATION
# ============================================================================

print("\n" + "="*80)
print("PART 3: DATA PREPARATION - Zachary's Karate Club")
print("="*80)

print("""
Zachary's Karate Club Dataset
------------------------------
This famous dataset represents friendships in a karate club.
The club split into two groups after a conflict between the instructor (Mr. Hi)
and the club administrator (Officer).
""")

# Load Zachary's Karate Club graph
G = nx.karate_club_graph()
print(f"\nDataset Statistics:")
print(f"  Number of nodes: {G.number_of_nodes()}")
print(f"  Number of edges: {G.number_of_edges()}")
print(f"  Average degree: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")

# Create adjacency matrix
A = nx.adjacency_matrix(G).todense()
A = np.array(A)
N = A.shape[0]

print(f"\nAdjacency matrix shape: {A.shape}")
print(f"Adjacency matrix (first 5×5):")
print(A[:5, :5])

# ============================================================================
# PART 4: NORMALIZE ADJACENCY MATRIX
# ============================================================================

print("\n" + "="*80)
print("PART 4: ADJACENCY MATRIX NORMALIZATION")
print("="*80)

def normalize_adjacency(A):
    """
    Compute the symmetrically normalized adjacency matrix.
    
    Returns: D^(-1/2) * (A + I) * D^(-1/2)
    """
    # Add self-loops
    A_tilde = A + np.eye(A.shape[0])
    
    # Compute degree matrix
    D_tilde = np.array(np.sum(A_tilde, axis=1)).flatten()
    
    # Compute D^(-1/2)
    D_inv_sqrt = np.diag(np.power(D_tilde, -0.5))
    
    # Compute normalized adjacency
    A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt
    
    return A_norm

A_norm = normalize_adjacency(A)
print(f"Normalized adjacency matrix shape: {A_norm.shape}")
print(f"\nNormalized adjacency (first 5×5):")
print(np.round(A_norm[:5, :5], 3))

print("""
Key Properties:
  - Each row sums to approximately 1 (normalized)
  - Diagonal values included (self-loops)
  - Symmetric matrix (undirected graph)
""")

# ============================================================================
# PART 5: PREPARE FEATURES AND LABELS
# ============================================================================

print("\n" + "="*80)
print("PART 5: FEATURES AND LABELS")
print("="*80)

# Create node features (one-hot encoding of node IDs)
# In practice, use real features like age, interests, etc.
X = np.eye(N)

print(f"Feature matrix shape: {X.shape}")
print(f"Features per node: {X.shape[1]}")
print(f"\nNote: Using one-hot encoding for simplicity.")
print(f"      In practice, use meaningful features!")

# Create labels (0 = Mr. Hi's group, 1 = Officer's group)
labels = np.array([0 if G.nodes[node]['club'] == 'Mr. Hi' else 1 
                   for node in G.nodes()])

print(f"\nLabels shape: {labels.shape}")
print(f"Class distribution:")
print(f"  Mr. Hi's group (0): {np.sum(labels == 0)} members")
print(f"  Officer's group (1): {np.sum(labels == 1)} members")

# Convert to PyTorch tensors
A_norm_tensor = torch.FloatTensor(A_norm)
X_tensor = torch.FloatTensor(X)
labels_tensor = torch.LongTensor(labels)

# ============================================================================
# PART 6: GCN IMPLEMENTATION
# ============================================================================

print("\n" + "="*80)
print("PART 6: GCN IMPLEMENTATION")
print("="*80)

class GraphConvolutionLayer(nn.Module):
    """
    Graph Convolution Layer
    
    Implements: H^(l+1) = σ(A_norm * H^(l) * W^(l))
    """
    def __init__(self, in_features, out_features):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights using Xavier uniform initialization"""
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        """
        Forward pass
        
        Args:
            x: Node features [N, in_features]
            adj: Normalized adjacency matrix [N, N]
        
        Returns:
            output: Updated node features [N, out_features]
        """
        # Step 1: Linear transformation H^(l) * W^(l)
        support = torch.mm(x, self.weight)
        
        # Step 2: Graph convolution A_norm * (H^(l) * W^(l))
        output = torch.mm(adj, support)
        
        # Step 3: Add bias
        output = output + self.bias
        
        return output
    
    def __repr__(self):
        return f'{self.__class__.__name__} ({self.in_features} -> {self.out_features})'


class GCN(nn.Module):
    """
    Two-layer Graph Convolutional Network for node classification
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(GCN, self).__init__()
        
        self.gc1 = GraphConvolutionLayer(input_dim, hidden_dim)
        self.gc2 = GraphConvolutionLayer(hidden_dim, output_dim)
        self.dropout = dropout
    
    def forward(self, x, adj):
        """
        Forward pass through the network
        
        Args:
            x: Node features [N, input_dim]
            adj: Normalized adjacency matrix [N, N]
        
        Returns:
            output: Log probabilities for each class [N, output_dim]
        """
        # First layer: GC + ReLU + Dropout
        x = self.gc1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        
        # Second layer: GC + Log Softmax
        x = self.gc2(x, adj)
        output = F.log_softmax(x, dim=1)
        
        return output


# Create model
input_dim = X.shape[1]
hidden_dim = 16
output_dim = 2

model = GCN(input_dim, hidden_dim, output_dim, dropout=0.5)
print(f"\nModel Architecture:")
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")

# ============================================================================
# PART 7: TRAINING
# ============================================================================

print("\n" + "="*80)
print("PART 7: TRAINING")
print("="*80)

# Training setup
optimizer = Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.NLLLoss()

# For this small dataset, use all nodes for training
train_mask = torch.ones(N, dtype=torch.bool)

def train_epoch():
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    output = model(X_tensor, A_norm_tensor)
    
    # Compute loss
    loss = criterion(output[train_mask], labels_tensor[train_mask])
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()

def evaluate():
    model.eval()
    with torch.no_grad():
        output = model(X_tensor, A_norm_tensor)
        predictions = output.argmax(dim=1)
        accuracy = (predictions == labels_tensor).float().mean()
    return accuracy.item(), predictions

# Training loop
epochs = 200
losses = []
accuracies = []

print(f"\nTraining for {epochs} epochs...\n")
for epoch in range(epochs):
    loss = train_epoch()
    accuracy, _ = evaluate()
    
    losses.append(loss)
    accuracies.append(accuracy)
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {loss:.4f} | Accuracy: {accuracy:.4f}")

print("\n✓ Training complete!")

# ============================================================================
# PART 8: EVALUATION
# ============================================================================

print("\n" + "="*80)
print("PART 8: EVALUATION")
print("="*80)

final_accuracy, predictions = evaluate()
predictions_np = predictions.numpy()

print(f"\nFinal Results:")
print(f"  Accuracy: {final_accuracy:.2%}")
print(f"  Correct predictions: {np.sum(predictions_np == labels)}/{N}")
print(f"  Misclassified nodes: {np.sum(predictions_np != labels)}")

# Show misclassified nodes
misclassified = np.where(predictions_np != labels)[0]
if len(misclassified) > 0:
    print(f"\nMisclassified node IDs: {list(misclassified)}")
else:
    print("\n✓ Perfect classification!")

# ============================================================================
# PART 9: ANALYZE EMBEDDINGS
# ============================================================================

print("\n" + "="*80)
print("PART 9: NODE EMBEDDINGS")
print("="*80)

# Extract embeddings from the hidden layer
model.eval()
with torch.no_grad():
    x = model.gc1(X_tensor, A_norm_tensor)
    embeddings = F.relu(x)
    embeddings_np = embeddings.numpy()

print(f"\nEmbedding shape: {embeddings_np.shape}")
print(f"Each node is represented by a {embeddings_np.shape[1]}-dimensional vector")

# Compute separation between groups in embedding space
group0_embeddings = embeddings_np[labels == 0]
group1_embeddings = embeddings_np[labels == 1]

group0_mean = group0_embeddings.mean(axis=0)
group1_mean = group1_embeddings.mean(axis=0)
separation = np.linalg.norm(group0_mean - group1_mean)

print(f"\nEmbedding Statistics:")
print(f"  Distance between group centroids: {separation:.3f}")
print(f"  Within-group variance (Group 0): {group0_embeddings.var():.3f}")
print(f"  Within-group variance (Group 1): {group1_embeddings.var():.3f}")

# ============================================================================
# PART 10: VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("PART 10: VISUALIZATIONS")
print("="*80)

# Create visualizations
fig = plt.figure(figsize=(20, 12))

# 1. Network with ground truth
ax1 = plt.subplot(2, 3, 1)
pos = nx.spring_layout(G, seed=42)
colors_true = ['red' if label == 0 else 'blue' for label in labels]
nx.draw_networkx(G, pos, node_color=colors_true, node_size=500,
                with_labels=True, font_color='white', font_weight='bold', ax=ax1)
ax1.set_title("Ground Truth\n(Red = Mr. Hi, Blue = Officer)", fontsize=14, fontweight='bold')
ax1.axis('off')

# 2. Network with predictions
ax2 = plt.subplot(2, 3, 2)
colors_pred = ['red' if pred == 0 else 'blue' for pred in predictions_np]
edgecolors = ['yellow' if predictions_np[i] != labels[i] else 'black' for i in range(N)]
linewidths = [4 if predictions_np[i] != labels[i] else 1 for i in range(N)]
nx.draw_networkx(G, pos, node_color=colors_pred, node_size=500,
                with_labels=True, font_color='white', font_weight='bold',
                edgecolors=edgecolors, linewidths=linewidths, ax=ax2)
ax2.set_title(f"GCN Predictions\n(Accuracy: {final_accuracy:.2%})", 
             fontsize=14, fontweight='bold')
ax2.axis('off')

# 3. Adjacency matrix heatmap
ax3 = plt.subplot(2, 3, 3)
im = ax3.imshow(A, cmap='YlOrRd', interpolation='nearest')
ax3.set_title("Adjacency Matrix", fontsize=14, fontweight='bold')
ax3.set_xlabel("Node ID")
ax3.set_ylabel("Node ID")
plt.colorbar(im, ax=ax3)

# 4. Training loss
ax4 = plt.subplot(2, 3, 4)
ax4.plot(losses, linewidth=2, color='orange')
ax4.set_xlabel('Epoch', fontsize=12)
ax4.set_ylabel('Loss', fontsize=12)
ax4.set_title('Training Loss', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)

# 5. Training accuracy
ax5 = plt.subplot(2, 3, 5)
ax5.plot(accuracies, linewidth=2, color='green')
ax5.set_xlabel('Epoch', fontsize=12)
ax5.set_ylabel('Accuracy', fontsize=12)
ax5.set_title('Training Accuracy', fontsize=14, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.set_ylim([0, 1.05])

# 6. Node embeddings (2D projection using PCA)
ax6 = plt.subplot(2, 3, 6)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings_np)
scatter = ax6.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                     c=labels, cmap='coolwarm', s=200, alpha=0.7, edgecolors='black')
ax6.set_xlabel('First Principal Component', fontsize=12)
ax6.set_ylabel('Second Principal Component', fontsize=12)
ax6.set_title(f'Learned Node Embeddings (2D)\nExplained Var: {pca.explained_variance_ratio_.sum():.1%}',
             fontsize=14, fontweight='bold')
ax6.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax6, label='Class')

plt.tight_layout()
plt.savefig('/Users/snistor/Documents/Maths/MathStuff/gcn_results.png', dpi=150, bbox_inches='tight')
print("\n✓ Visualization saved as 'gcn_results.png'")
plt.show()

# ============================================================================
# PART 11: KEY INSIGHTS
# ============================================================================

print("\n" + "="*80)
print("PART 11: KEY INSIGHTS")
print("="*80)

print("""
What We Learned
---------------

1. Graph Structure Matters
   The GCN achieves high accuracy by leveraging both node features AND the
   network structure. Traditional MLPs would ignore the friendship connections.

2. Information Propagation
   Each layer aggregates information from neighbors:
     - Layer 1: Direct neighbors (1-hop)
     - Layer 2: Neighbors of neighbors (2-hop)
   
   For k layers, the network considers k-hop neighborhoods.

3. Learned Embeddings
   The hidden layer creates meaningful node embeddings where similar nodes
   (same class) cluster together in the embedding space.

4. Semi-supervised Learning
   In practice, you only need labels for a small subset of nodes.
   The graph structure helps propagate information to unlabeled nodes.

Mathematical Intuition
----------------------
The normalized adjacency matrix D̃^(-1/2) Ã D̃^(-1/2) acts as a diffusion operator:
  - It smooths features across the graph
  - Connected nodes influence each other
  - The normalization ensures stability (values don't explode or vanish)

Computational Complexity
------------------------
For a graph with N nodes, E edges, and F features:
  - Forward pass: O(E·F·H) where H is hidden dimension
  - Memory: O(N·F) for features, O(E) for adjacency
  
GCNs are efficient because they leverage the sparse structure of real graphs!

Applications
------------
1. Social Networks: Community detection, influence prediction, recommendation
2. Molecular Chemistry: Predicting molecular properties (atoms=nodes, bonds=edges)
3. Traffic Networks: Traffic flow prediction, route optimization
4. Citation Networks: Paper classification, citation recommendation
5. Knowledge Graphs: Link prediction, entity classification
6. Computer Vision: Scene graph understanding, point cloud processing

Advanced GCN Variants
---------------------
- GraphSAGE: Samples neighbors for scalability
- GAT (Graph Attention Networks): Learn attention weights for neighbors
- GIN (Graph Isomorphism Network): More expressive than standard GCN
- DGCRN: Dynamic graphs for temporal data

Tips for Your Own GCN
----------------------
1. Feature Engineering: Good node features improve performance significantly
2. Depth: 2-3 layers usually sufficient (too deep causes over-smoothing)
3. Normalization: Symmetric normalization works well in practice
4. Regularization: Dropout and weight decay prevent overfitting
5. Scalability: For large graphs, use mini-batch training with neighbor sampling

""")

print("="*80)
print("COMPLETE GCN TUTORIAL FINISHED!")
print("="*80)

print("""
References
----------
1. Kipf & Welling (2017): "Semi-Supervised Classification with Graph Convolutional Networks"
2. Hamilton et al. (2017): "Inductive Representation Learning on Large Graphs" (GraphSAGE)
3. Veličković et al. (2018): "Graph Attention Networks" (GAT)
4. Wu et al. (2021): "A Comprehensive Survey on Graph Neural Networks"

Created: 2026-01-31
""")

