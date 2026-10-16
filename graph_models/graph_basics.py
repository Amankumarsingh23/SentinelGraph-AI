"""Session 12: explore a small, entirely synthetic account-device network."""

import networkx as nx


def build_toy_graph():
    graph = nx.Graph()
    graph.add_nodes_from(["A", "B", "C", "E"], node_type="account")
    graph.add_nodes_from(["D1", "D2"], node_type="device")
    graph.add_edges_from(
        [("A", "D1"), ("B", "D1"), ("C", "D1"), ("E", "D2")],
        relation="uses_device",
    )
    return graph


def main():
    graph = build_toy_graph()
    print("Synthetic account-device graph")
    print("Number of nodes:", graph.number_of_nodes())
    print("Number of edges:", graph.number_of_edges())
    print("\nDevice D1 degree:", graph.degree("D1"))
    print("Device D2 degree:", graph.degree("D2"))

    # Sort printed nodes/components so output is reproducible across runs.
    components = sorted(sorted(nodes) for nodes in nx.connected_components(graph))
    print("\nNumber of connected components:", len(components))
    for component in components:
        print(component)
    print("\nShortest path from A to C:", " -> ".join(nx.shortest_path(graph, "A", "C")))
    print(f"D1 degree centrality: {nx.degree_centrality(graph)['D1']:.3g}")
    print("\nA shared device is a relationship, not proof of fraud.")


if __name__ == "__main__":
    main()
