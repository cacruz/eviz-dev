from dataclasses import dataclass
from glob import glob
import json
from typing import Any, Optional, List
import networkx as nx
from eviz.lib.data.reader import DataReader


@dataclass
class GraphDataReader(DataReader):
    """ Class definitions for reading graph data files using NetworkX.
    Supports various graph formats including GraphML, GML, GEXF, Pajek,
    Edge Lists, Adjacency Lists, and JSON.
    """
    file_path: str = None
    format: str = "auto"  # Can be "auto", "graphml", "gml", "gexf", "pajek", "edgelist", "adjlist", "json"

    def __post_init__(self):
        super().__post_init__()
        self.findex = 0
        self._readers = {
            "graphml": nx.read_graphml,
            "gml": nx.read_gml,
            "gexf": nx.read_gexf,
            "pajek": nx.read_pajek,
            "edgelist": nx.read_edgelist,
            "adjlist": nx.read_adjlist,
            "json": self._read_json
        }

    def _read_json(self, file_path: str) -> nx.Graph:
        """Read a JSON file and convert it to a NetworkX graph

        Args:
            file_path: Path to the JSON file

        Returns:
            A NetworkX graph object
        """
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Create a new graph
        G = nx.Graph()

        # Try to determine the JSON graph format
        if isinstance(data, dict) and 'nodes' in data:
            # This looks like a node-link format
            self.logger.info("Detected node-link JSON format")

            # Add nodes with their attributes
            for node_data in data['nodes']:
                if 'id' in node_data:
                    node_id = node_data.pop('id')
                    G.add_node(node_id, **node_data)
                else:
                    # If no id is provided, assume it's in the order they appear
                    G.add_node(len(G.nodes), **node_data)

            # Add edges with their attributes
            if 'links' in data:
                for edge_data in data['links']:
                    source = edge_data.pop('source', None)
                    target = edge_data.pop('target', None)
                    if source is not None and target is not None:
                        G.add_edge(source, target, **edge_data)
            elif 'edges' in data:
                for edge_data in data['edges']:
                    source = edge_data.pop('source', None)
                    target = edge_data.pop('target', None)
                    if source is not None and target is not None:
                        G.add_edge(source, target, **edge_data)

        elif isinstance(data, dict) and all(isinstance(k, str) for k in data.keys()):
            # This might be an adjacency format where keys are nodes
            self.logger.info("Detected adjacency-like JSON format")

            # Add nodes and their attributes
            for node, attrs in data.items():
                if isinstance(attrs, dict):
                    # Check if the dict contains both node attributes and adjacency info
                    neighbors = attrs.pop('neighbors', {})
                    G.add_node(node, **attrs)

                    # Add edges from this node to its neighbors
                    for neighbor, edge_attrs in neighbors.items():
                        if isinstance(edge_attrs, dict):
                            G.add_edge(node, neighbor, **edge_attrs)
                        else:
                            G.add_edge(node, neighbor, weight=edge_attrs)
                else:
                    # Simpler format where value is just a list of neighbors
                    G.add_node(node)
                    for neighbor in attrs:
                        G.add_edge(node, neighbor)

        else:
            # Try as a plain adjacency list/matrix
            self.logger.warning(
                "JSON format not recognized, attempting to interpret as adjacency data")
            try:
                for i, node_data in enumerate(data):
                    G.add_node(i)
                    if isinstance(node_data, list):
                        for j, weight in enumerate(node_data):
                            if weight:
                                G.add_edge(i, j, weight=weight)
            except:
                self.logger.error(f"Could not interpret JSON file {file_path} as a graph")
                raise ValueError(f"Could not interpret JSON file {file_path} as a graph")

        # Check if we successfully created a graph
        if len(G.nodes) == 0:
            self.logger.error(f"No nodes found in JSON file {file_path}")
            raise ValueError(f"No nodes found in JSON file {file_path}")

        return G

    def _detect_format(self, file_path: str) -> str:
        """Detect the graph format based on file extension"""
        lower_path = file_path.lower()

        if lower_path.endswith(".json"):
            return "json"
        elif lower_path.endswith(".graphml"):
            return "graphml"
        elif lower_path.endswith(".gml"):
            return "gml"
        elif lower_path.endswith(".gexf"):
            return "gexf"
        elif lower_path.endswith(".net"):
            return "pajek"
        elif lower_path.endswith(".edgelist") or lower_path.endswith(".edges"):
            return "edgelist"
        elif lower_path.endswith(".adjlist") or lower_path.endswith(".adj"):
            return "adjlist"
        else:
            self.logger.warning(
                f"Unable to detect format for {file_path}, defaulting to GraphML")
            return "graphml"

    def read_data(self, file_path: str, format: Optional[str] = None) -> Any:
        """ Reads graph data files and returns NetworkX graph objects

        Args:
            file_path: Path to the graph file(s), can include wildcards
            format: Optional format override (graphml, gml, gexf, pajek, edgelist, adjlist, json)
                    If not specified, will use the class format or auto-detect from extension

        Returns:
            A NetworkX graph object or a list of graph objects if multiple files
        """
        self.logger.info(f"Reading graph data from {file_path}")
        self.file_path = file_path
        files = glob(self.file_path)

        # Use parameter format if provided, otherwise use class format
        file_format = format if format else self.format

        # Create a list to hold all graphs if we're reading multiple files
        all_graphs = []

        try:
            if not files:
                self.logger.warning(f"No files found matching pattern: {self.file_path}")
                return None

            for f in files:
                # Auto-detect format if needed
                if file_format == "auto":
                    detected_format = self._detect_format(f)
                else:
                    detected_format = file_format

                if detected_format not in self._readers:
                    self.logger.error(f"Unsupported graph format: {detected_format}")
                    continue

                # Read the graph using the appropriate reader
                reader_func = self._readers[detected_format]
                graph = reader_func(f)

                self.logger.info(
                    f"Read graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")

                # Process the graph data
                processed_graph = self._process_graph(graph)
                all_graphs.append(processed_graph)

        except Exception as e:
            self.logger.error(f"An error occurred while reading the graph data: {str(e)}")
            return None

        self.findex += 1

        # Save all graphs to the datasets list
        if len(all_graphs) == 1:
            result = all_graphs[0]
        else:
            result = all_graphs

        self.datasets.append(result)
        return result

    def _process_graph(self, graph: nx.Graph) -> nx.Graph:
        """Process the graph before returning it.
        Override this method in subclasses to implement custom processing.

        Args:
            graph: The NetworkX graph to process

        Returns:
            The processed NetworkX graph
        """
        # This is a placeholder for any graph-specific processing
        # Similar to _process_data in the CSVDataReader
        return graph

    def get_node_attributes(self, graph: nx.Graph) -> dict:
        """Get all node attributes from the graph

        Args:
            graph: A NetworkX graph object

        Returns:
            Dictionary of node attributes
        """
        return {attr: nx.get_node_attributes(graph, attr) for attr in
                set().union(*(d.keys() for _, d in graph.nodes(data=True))) if attr}

    def get_edge_attributes(self, graph: nx.Graph) -> dict:
        """Get all edge attributes from the graph

        Args:
            graph: A NetworkX graph object

        Returns:
            Dictionary of edge attributes
        """
        return {attr: nx.get_edge_attributes(graph, attr) for attr in
                set().union(*(d.keys() for _, _, d in graph.edges(data=True))) if attr}

    def visualize(self, graph, layout="spring", node_attr=None, edge_attr=None,
                  title=None,
                  filename=None, show=True, figsize=(10, 8)):
        """Visualize the graph using matplotlib

        Args:
            graph: A NetworkX graph object
            layout: Layout algorithm ('spring', 'circular', 'kamada_kawai', 'spectral', 'shell')
            node_attr: Node attribute to use for node labels
            edge_attr: Edge attribute to use for edge labels
            title: Title for the visualization
            filename: If specified, save visualization to this file
            show: Whether to display the visualization
            figsize: Figure size as a tuple (width, height)

        Returns:
            matplotlib figure object
        """
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=figsize)

        # Choose layout algorithm
        if layout == "spring":
            pos = nx.spring_layout(graph, seed=42)
        elif layout == "circular":
            pos = nx.circular_layout(graph)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(graph)
        elif layout == "spectral":
            pos = nx.spectral_layout(graph)
        elif layout == "shell":
            pos = nx.shell_layout(graph)
        else:
            self.logger.warning(f"Layout '{layout}' not recognized, using spring layout")
            pos = nx.spring_layout(graph, seed=42)

        # Draw nodes
        nx.draw_networkx_nodes(graph, pos, node_size=700, node_color='skyblue')

        # Draw node labels
        if node_attr and node_attr in self.get_node_attributes(graph):
            labels = nx.get_node_attributes(graph, node_attr)
            nx.draw_networkx_labels(graph, pos, labels=labels)
        else:
            nx.draw_networkx_labels(graph, pos)

        # Draw edges
        # Check if weight attribute exists for edges
        if 'weight' in self.get_edge_attributes(graph):
            edge_weights = [graph[u][v]['weight'] * 2 for u, v in graph.edges()]
            nx.draw_networkx_edges(graph, pos, width=edge_weights)
        else:
            nx.draw_networkx_edges(graph, pos)

        # Draw edge labels if attribute specified
        if edge_attr and edge_attr in self.get_edge_attributes(graph):
            edge_labels = nx.get_edge_attributes(graph, edge_attr)
            nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)

        # Set title
        if title:
            plt.title(title)
        else:
            plt.title(
                f"Graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")

        plt.axis('off')
        plt.tight_layout()

        # Save if filename provided
        if filename:
            plt.savefig(filename, bbox_inches='tight')

        # Show plot if requested
        if show:
            plt.show()

        return fig

    def to_json(self, graph, file_path):
        """Save the graph to a JSON file

        Args:
            graph: A NetworkX graph object
            file_path: Path to save the JSON file
        """
        # Convert the graph to a node-link format
        data = nx.node_link_data(graph)

        # Write to file
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

        self.logger.info(f"Saved graph to {file_path}")