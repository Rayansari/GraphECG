# GraphECG

Electrode-aware graph neural network for flexible-lead ECG analysis.

## Overview

GraphECG represents ECGs as directed graphs where electrodes are nodes and leads are edges carrying waveform embeddings. This formulation supports variable lead configurations through subgraph construction without architectural modification.

## Installation

```bash
pip install torch torch-geometric numpy
```

## Usage

```python
import numpy as np
import torch
from graphecg import ECGGraphBuilder, GraphECG

# Initialize
builder = ECGGraphBuilder()
model = GraphECG()

# Build graph from 12-lead ECG array (12, 2500)
ecg = np.random.randn(12, 2500).astype(np.float32)
graph = builder.build_from_array(ecg, bidirectional=True)

# Or from specific leads
graph = builder.build_from_array(ecg, lead_indices=[0, 1, 2], bidirectional=True)

# Or from named leads
graph = builder.build_graph({'I': ecg[0], 'II': ecg[1]}, bidirectional=True)

# Or from an arbitrary electrode pair (non-standard / wearable vectors)
graph = builder.build_custom([('V2', 'V3', ecg[8] - ecg[7])], bidirectional=True)

# Inference
tabular = torch.randn(1, 7)  # Clinical features
output = model(graph, tabular)
prob = torch.sigmoid(output['logits'])
```

## Training on your own data / task

`GraphECG` is a standard `nn.Module` — bring your own training loop. It supports
multiclass tasks and works with or without tabular features:

```python
# 3-class task, no tabular features:
model = GraphECG(num_classes=3, tabular_dim=0)
logits = model(graph)['logits']          # (B, 3); pass tabular=None or omit it

# binary task with 7 tabular features (the default / pretrained EchoNext setup):
model = GraphECG(num_classes=1, tabular_dim=7)
```

Build a `torch_geometric` batch from per-sample graphs with
`torch_geometric.data.Batch.from_data_list([...])`, then optimize with your loss
of choice. Reduced-lead inputs need no special handling: pass only the leads you
have to `build_from_array`, and the model runs on the resulting subgraph.

## Graph Representation

- **13 electrode nodes**: WCT, RA, LA, LL, V1-V6, and 3 virtual midpoints for augmented leads
- **Directed edges**: Each lead is an edge from source to target electrode
- **Bidirectional**: Reverse edges carry negated signals (physics: V_AB = -V_BA)
- **Flexible**: Any lead subset creates a corresponding subgraph

## Citation

```bibtex
@article{graphecg2025,
  title={Electrode-Aware Message Passing for Flexible-Lead ECG Analysis},
  author={Ansari, Rayan and Musa, Mohammed and Bandyopadhyay, Sabyasachi and Rogers, Albert J.},
  year={2025}
}
```

## License

MIT
