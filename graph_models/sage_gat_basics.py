"""Session 15: inspect untrained GraphSAGE and two-head GAT layer shapes."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch_geometric.nn import SAGEConv, GATConv
from graph_models.gcn_basics import make_toy_graph


def main():
    torch.manual_seed(42)
    graph = make_toy_graph()
    sage = SAGEConv(in_channels=2, out_channels=8)
    gat = GATConv(in_channels=2, out_channels=8, heads=2, concat=True)
    with torch.no_grad():
        sage_output = sage(graph.x, graph.edge_index)
        gat_output = gat(graph.x, graph.edge_index)
    print("Input shape:", tuple(graph.x.shape))
    print("GraphSAGE output shape:", tuple(sage_output.shape))
    print("GAT output shape:", tuple(gat_output.shape))
    print("GAT concatenates two heads of eight features: 2 * 8 = 16.")
    print("These are randomly initialized layer outputs, without training.")


if __name__ == "__main__":
    main()
