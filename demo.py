"""
GraphECG Demo

Demonstrates graph construction and inference with variable lead configurations.
"""

import numpy as np
import torch
from torch_geometric.data import Batch
from graphecg import ECGGraphBuilder, GraphECG

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Initialize
    builder = ECGGraphBuilder()
    model = GraphECG().to(device)
    model.eval()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Simulate ECG data: (12 leads, 2500 samples)
    ecg = np.random.randn(12, 2500).astype(np.float32)
    tabular = torch.randn(1, 7).to(device)

    # Example 1: Full 12-lead
    print("\n12-lead input:")
    graph = builder.build_from_array(ecg, bidirectional=True)
    graph = graph.to(device)
    with torch.no_grad():
        out = model(graph, tabular)
    print(f"  Edges: {graph.edge_index.shape[1]}")
    print(f"  Output: {torch.sigmoid(out['logits']).item():.4f}")

    # Example 2: 6-lead (limb leads only)
    print("\n6-lead input (limb leads):")
    graph = builder.build_from_array(ecg, lead_indices=[0, 1, 2, 3, 4, 5], bidirectional=True)
    graph = graph.to(device)
    with torch.no_grad():
        out = model(graph, tabular)
    print(f"  Edges: {graph.edge_index.shape[1]}")
    print(f"  Output: {torch.sigmoid(out['logits']).item():.4f}")

    # Example 3: Single lead
    print("\nSingle lead (Lead II):")
    graph = builder.build_from_array(ecg, lead_indices=[1], bidirectional=True)
    graph = graph.to(device)
    with torch.no_grad():
        out = model(graph, tabular)
    print(f"  Edges: {graph.edge_index.shape[1]}")
    print(f"  Output: {torch.sigmoid(out['logits']).item():.4f}")

    # Example 4: Batched inference
    print("\nBatched inference (4 samples with varying leads):")
    graphs = []
    for i in range(4):
        ecg_i = np.random.randn(12, 2500).astype(np.float32)
        n_leads = np.random.randint(3, 13)
        indices = sorted(np.random.choice(12, n_leads, replace=False).tolist())
        graphs.append(builder.build_from_array(ecg_i, lead_indices=indices, bidirectional=True))
    batch = Batch.from_data_list(graphs).to(device)
    tabular_batch = torch.randn(4, 7).to(device)
    with torch.no_grad():
        out = model(batch, tabular_batch)
    print(f"  Total edges: {batch.edge_index.shape[1]}")
    print(f"  Outputs: {torch.sigmoid(out['logits']).squeeze().tolist()}")

if __name__ == '__main__':
    main()
