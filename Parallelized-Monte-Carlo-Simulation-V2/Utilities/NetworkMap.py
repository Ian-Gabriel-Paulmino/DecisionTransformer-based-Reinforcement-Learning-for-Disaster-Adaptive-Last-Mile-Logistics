import osmnx as ox
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from typing import Dict, Any, Tuple
import networkx as nx
import random

class NetworkMap:

    """ Custom Network class to handle graph and network processes"""
    def __init__(self,place_query):
        self.place_query = place_query
        # Raw unprocessed network from osmnx
        self.network = None

        # Processed networkx map
        self.G = None

        self.start_node = None
        self.delivery_nodes = None

        # Estimated ideal route (Node 1 -> Node 2) without disasters
        self.route = []

        # Estimated ideal paths from route (Node 1 -> path -> Node 2) without disasters
        self.path_sequence = []

        # Ideal time
        self.ideal_delivery_time = 0

    
    def download_map(self,network_type="drive",which_results=1):
        print(f"Downloading map for {self.place_query}...")

        graph = ox.graph_from_place(self.place_query,network_type=network_type,which_result=which_results)
        graph = ox.add_edge_bearings(graph)

        # Plot the graph for ppt
        # fig, ax = plt.subplots(figsize=(10, 10)) 
        # ox.plot_graph(graph, ax=ax, node_size=0, edge_linewidth=0.5, bgcolor='w')

        # plt.show()

        # Save graph instance to network attribute for processing
        self.network = graph
    

    def network_to_networkx(self):
        """
        Convert a custom Network instance to a NetworkX graph.
        
        Args:
            network: A Network instance
            
        Returns:
            A NetworkX graph with the same nodes, edges, and attributes
        """
        # Create an empty directed graph
        G = nx.DiGraph()

        # Process nodes
        for node_id, node_data in self.network.nodes(data=True):
            # Attribute extraction from nodes (x and y for lat,lon)
            attributes = {
                'x': node_data['x'],
                'y': node_data['y'],
                'osmid': node_id
            }

            # Copy other data to nodes
            for key in ['street_count', 'highway', 'ref']:
                if key in node_data:
                    attributes[key] = node_data[key]

            G.add_node(node_id, **attributes)

        # Process edges
        for source, target, edge_data in self.network.edges(data=True):
            # Collection to hold attributes of the nodes
            attributes = {}

            for key in ['length', 'name', 'highway', 'oneway', 'lanes']:
                if key in edge_data:
                    attributes[key] = edge_data[key]

            # Extract geometry if available
            if 'geometry' in edge_data:
                # Check if geometry already has coordinates or needs to be processed
                geometry = edge_data['geometry']
                if hasattr(geometry, 'coords'):
                    # If it's a geometry object with coords method (like a LineString)
                    attributes['geometry'] = list(geometry.coords)
                elif isinstance(geometry, list):
                    # If it's already a list of coordinates
                    attributes['geometry'] = geometry
                else:
                    # Skip geometry if it's in an unexpected format
                    print(f"Skipping unsupported geometry format: {type(geometry)}")

            G.add_edge(source, target, **attributes)
        

        # Save instance of processes networkx graph to attribute for global use
        self.G = G


    def initialize_edge_disaster_attributes(self):
        """Initialize disaster-related attributes for all edges in the graph"""

        """ 
        Purpose: 
            Determine if edge is problematic
            If it is problematic assign prone_values

            Flood_prone values:
                If near_waters: 0.7
                Random values from 0.3 to 0.9
            
            Landslide_prone values:
                Random values from 0.3 to 0.9
        """
        
        landslide_prone_edges = []
        flood_prone_edges = []

        try:
            # Get edge attributes to identify potential flood and landslide prone areas
            for u, v, data in self.G.edges(data=True):
                # Initialize disaster probability attributes
                
                # Default values
                data['flood_prone'] = 0.0  # Probability of flooding (0.0 to 1.0)
                data['landslide_prone'] = 0.0  # Probability of landslide (0.0 to 1.0)
                data['currently_flooded'] = False
                data['currently_landslide'] = False
               
                
                # Assign flood-prone probabilities based on:
                # 1. Edges near rivers/water (we'll use a heuristic based on names)
                if 'name' in data:
                    name = data['name'].lower() if isinstance(data['name'], str) else ""
                    if any(water_keyword in name for water_keyword in ['river', 'creek', 'stream', 'bridge']):
                        data['flood_prone'] = 0.7

                        if (u,v) not in flood_prone_edges:
                            flood_prone_edges.append((u,v))
                
            
                
                # Randomly assign some edges as flood/landslide prone
                if data['flood_prone'] == 0 and random.random() < 0.20:

                    # Activation value (How likely will this activate)
                    data['flood_prone'] = random.uniform(0.5, 1.0)
                    if (u,v) not in flood_prone_edges:
                            flood_prone_edges.append((u,v))
                    
                if data['landslide_prone'] == 0 and random.random() < 0.20:

                    # Activation value (How likely will this activate)
                    data['landslide_prone'] = random.uniform(0.5, 1.0)
                    if (u,v) not in landslide_prone_edges:
                            landslide_prone_edges.append((u,v))
                
            return landslide_prone_edges,flood_prone_edges

        except Exception as e:
            raise ValueError(f"Error in adding Problematic Nodes {e}")
        

    def select_fixed_points(self,num_delivery_points=70):
        """Select fixed start and delivery points for consistency"""

        # Get all nodes sorted by ID to ensure consistency
        all_nodes = sorted(list(self.G.nodes()))

        # Choose nodes at regular intervals for better distribution
        node_count = len(all_nodes)
        interval = node_count // (num_delivery_points + 1)

        start_node = all_nodes[0]

        delivery_nodes = []

        for i in range(1,num_delivery_points + 1):
            # Jump to each interval
            index = i * interval
            if index < node_count:
                delivery_nodes.append(all_nodes[index])
        

        # Initialize values for start and delivery nodes
        self.start_node = start_node
        self.delivery_nodes = delivery_nodes

        return start_node, delivery_nodes
    

    def find_shortest_path(self, source_id, target_id, weight='length'):
        """
        Find the shortest path between two nodes in the network.
        
        Args:
            G: A NetworkX graph
            source_id: ID of the source node
            target_id: ID of the target node
            weight: Edge attribute to use as weight (default: 'length')
            
        Returns:
            A tuple of (path, path_length) where path is a list of node IDs
            and path_length is the total weight of the path
        """
        # Check if nodes exist
        if source_id not in self.G.nodes or target_id not in self.G.nodes:
            raise ValueError(f"Source or target node not found in the network")
        
        # Find shortest path
        try:
            path = nx.shortest_path(self.G, source=source_id, target=target_id, weight=weight)
            
            # Calculate path length
            path_length = 0
            for i in range(len(path)-1):
                edge_data = self.G.get_edge_data(path[i], path[i+1])
                if weight in edge_data:
                    path_length += edge_data[weight]
                else:
                    # If the specified weight attribute is missing, count as 1
                    path_length += 1
                    
            return path, path_length
        except nx.NetworkXNoPath:
            return None, float('inf')



    def nearest_neighbor_route(self):
        """Find route using nearest neighbor heuristic"""

        unvisited = set(self.delivery_nodes)
        current_node = self.start_node

        # Route starts with start node
        route = [self.start_node]
        path_sequence = [self.start_node]
        total_distance = 0


        while unvisited:
            next_node = None
            shortest_distance = float('inf')
            shortest_path = None

            for node in unvisited:
                path,distance = self.find_shortest_path(current_node,node)
                if distance < shortest_distance and path:
                    shortest_distance = distance
                    shortest_path = path
                    next_node = node

            if next_node:
                # Add next node to route
                route.append(next_node)
                
                # Add path to complete path sequence (not including the first node since its already present)
                path_sequence.extend(shortest_path[1:])
                total_distance += shortest_distance
                unvisited.remove(next_node)
                current_node = next_node
            else:
                print("Could not find path to remaining delivery nodes")
                break
        
        # Initialize routes and path sequence for class use
        self.route = route
        self.path_sequence = path_sequence
        
        return route, total_distance, path_sequence
    

    
    """
    ===================================================================================================================
    🔨  Utility and visualization Functions
    ===================================================================================================================
    """ 
    def calculate_travel_time(self,distance_meters,speed_meters_per_minute=500):
        """Calculate travel time in minutes based on distance and speed"""
        self.ideal_delivery_time = distance_meters / speed_meters_per_minute


    def visualize_delivery_nodes(self):
        """Visualize the delivery route with start, delivery points, and path"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot the base graph
        pos = {node: (data['x'], data['y']) for node, data in self.G.nodes(data=True)}
        nx.draw_networkx_nodes(self.G, pos, node_size=5, node_color='lightgray', ax=ax)
        nx.draw_networkx_edges(self.G, pos, edge_color='lightgray', width=0.5, alpha=0.6, ax=ax)
        
        # Highlight delivery nodes
        nx.draw_networkx_nodes(self.G, pos, nodelist=self.delivery_nodes, node_color='green', node_size=100, ax=ax)
        
        # Highlight start node
        nx.draw_networkx_nodes(self.G, pos, nodelist=[self.start_node], node_color='red', node_size=150, ax=ax)
        
        # Create legend
        legend_elements = [
            Patch(facecolor='red', edgecolor='black', label='Start Node'),
            Patch(facecolor='green', edgecolor='black', label='Delivery Nodes')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.set_axis_off()
        plt.title("Delivery Nodes and Starting Point")
        return fig, ax
    


    
    def visualize_delivery_route(self):
        """Visualize the delivery route with start, delivery points, and path"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot the base graph
        pos = {node: (data['x'], data['y']) for node, data in self.G.nodes(data=True)}
        nx.draw_networkx_nodes(self.G, pos, node_size=5, node_color='lightgray', ax=ax)
        nx.draw_networkx_edges(self.G, pos, edge_color='lightgray', width=0.5, alpha=0.6, ax=ax)
        
        # Plot the path
        path_edges = list(zip(self.path_sequence[:-1], self.path_sequence[1:]))
        nx.draw_networkx_edges(self.G, pos, edgelist=path_edges, edge_color='blue', width=2.0, ax=ax)
        
        # Highlight delivery nodes
        nx.draw_networkx_nodes(self.G, pos, nodelist=self.delivery_nodes, node_color='green', node_size=100, ax=ax)
        
        # Highlight start node
        nx.draw_networkx_nodes(self.G, pos, nodelist=[self.start_node], node_color='red', node_size=150, ax=ax)
        
        # Add labels for the delivery sequence
        labels = {}
        for i, node in enumerate(self.route):
            if i == 0:
                labels[node] = f"Start"
            else:
                labels[node] = f"Destination {i}"
        
        nx.draw_networkx_labels(self.G, pos, labels=labels, font_color='black', font_weight='bold')
        
        ax.set_axis_off()
        plt.title("Delivery Route using Networkx nearest_path function")
        return fig, ax
    




    def visualize_delivery_route_with_disaster(self, landslide_prone, flood_prone):
        """Visualize the delivery route with start, delivery points, and path"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot the base graph
        pos = {node: (data['x'], data['y']) for node, data in self.G.nodes(data=True)}
        nx.draw_networkx_nodes(self.G, pos, node_size=5, node_color='lightgray', ax=ax)
        nx.draw_networkx_edges(self.G, pos, edge_color='lightgray', width=0.5, alpha=0.6, ax=ax)
        
        # Highlight flood-prone edges (light blue)
        nx.draw_networkx_edges(self.G, pos, edgelist=flood_prone, edge_color='aqua', width=1.5, style='dashed', ax=ax)
        
        # Highlight landslide-prone edges (orange)
        nx.draw_networkx_edges(self.G, pos, edgelist=landslide_prone, edge_color='orange', width=1.5, style='dotted', ax=ax)

        # Plot the path
        path_edges = list(zip(self.path_sequence[:-1], self.path_sequence[1:]))
        nx.draw_networkx_edges(self.G, pos, edgelist=path_edges, edge_color='blue', width=2.0, ax=ax)
        
        # Highlight delivery nodes
        nx.draw_networkx_nodes(self.G, pos, nodelist=self.delivery_nodes, node_color='green', node_size=100, ax=ax)
        
        # Highlight start node
        nx.draw_networkx_nodes(self.G, pos, nodelist=[self.start_node], node_color='red', node_size=150, ax=ax)
        
        # Add labels for the delivery sequence
        labels = {}
        for i, node in enumerate(self.route):
            if i == 0:
                labels[node] = f"Start"
            else:
                labels[node] = f"Destination {i}"
        
        nx.draw_networkx_labels(self.G, pos, labels=labels, font_color='black', font_weight='bold')
        
        ax.set_axis_off()
        plt.title("Randomly initialized disaster prone edges")
        return fig, ax