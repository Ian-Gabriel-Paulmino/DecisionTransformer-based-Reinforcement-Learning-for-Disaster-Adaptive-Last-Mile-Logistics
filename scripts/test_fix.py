from DecisionTransformer.data.state_encoder import StateEncoder
from DecisionTransformer.Utilities import NetworkMap

network_map = NetworkMap("La Trinidad, Benguet, Philippines")
network_map.download_map()
network_map.network_to_networkx()

encoder = StateEncoder(network_map.G, node_embedding_dim=64)
print("State dimension:", encoder.state_dim)
sample_state = {
    'current_node': list(network_map.G.nodes())[0],
    'remaining_nodes': [],
    'rain_intensity': 3,
    'position_in_route': 2,
    'total_nodes': 10,
    'disaster_context': {'floods_nearby': 2, 'landslides_nearby': 1}
}
vec = encoder.encode_state(sample_state)
print("Encoded vector length:", len(vec))