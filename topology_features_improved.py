import os

import networkx as nx
import numpy as np
import torch

from AMDGT_original.data_preprocess import k_matrix


def _compute_graph_topology(adj_matrix):
    graph = nx.from_numpy_array(adj_matrix)
    num_nodes = adj_matrix.shape[0]

    degree = np.array([graph.degree(i) for i in range(num_nodes)], dtype=np.float32)
    degree = degree / max(degree.max(), 1.0)

    weighted_degree = np.array([graph.degree(i, weight='weight') for i in range(num_nodes)], dtype=np.float32)
    weighted_degree = weighted_degree / max(weighted_degree.max(), 1.0)

    clustering = nx.clustering(graph, weight='weight')
    clustering = np.array([clustering[i] for i in range(num_nodes)], dtype=np.float32)

    try:
        pagerank = nx.pagerank(graph, weight='weight', max_iter=100)
        pagerank = np.array([pagerank[i] for i in range(num_nodes)], dtype=np.float32)
    except nx.PowerIterationFailedConvergence:
        pagerank = np.ones(num_nodes, dtype=np.float32) / max(num_nodes, 1)
    pagerank = pagerank / max(pagerank.max(), 1.0)

    avg_neighbor_degree = nx.average_neighbor_degree(graph, weight='weight')
    avg_neighbor_degree = np.array([avg_neighbor_degree.get(i, 0.0) for i in range(num_nodes)], dtype=np.float32)
    avg_neighbor_degree = avg_neighbor_degree / max(avg_neighbor_degree.max(), 1.0)

    return np.stack(
        [degree, weighted_degree, clustering, pagerank, avg_neighbor_degree],
        axis=1,
    )


def _association_degree(associations, num_entities, entity_col):
    degree = np.zeros(num_entities, dtype=np.float32)
    for row in associations:
        idx = int(row[entity_col])
        if 0 <= idx < num_entities:
            degree[idx] += 1
    return degree / max(degree.max(), 1.0)


def extract_topology_features(data, args, force_recompute=False):
    cache_dir = os.path.join(args.data_dir, 'topology_cache')
    os.makedirs(cache_dir, exist_ok=True)

    drug_cache = os.path.join(cache_dir, f'drug_topo_k{args.neighbor}_n{args.drug_number}.pt')
    disease_cache = os.path.join(cache_dir, f'disease_topo_k{args.neighbor}_n{args.disease_number}.pt')

    if not force_recompute and os.path.exists(drug_cache) and os.path.exists(disease_cache):
        drug_topo = torch.load(drug_cache, weights_only=False)
        disease_topo = torch.load(disease_cache, weights_only=False)
        if drug_topo.shape[0] == args.drug_number and disease_topo.shape[0] == args.disease_number:
            return drug_topo.float(), disease_topo.float()

    drug_knn = k_matrix(data['drs'], args.neighbor)
    disease_knn = k_matrix(data['dis'], args.neighbor)

    drug_graph_features = _compute_graph_topology(drug_knn)
    disease_graph_features = _compute_graph_topology(disease_knn)

    drug_disease_degree = _association_degree(data['drdi'], args.drug_number, entity_col=0)
    drug_protein_degree = _association_degree(data['drpr'], args.drug_number, entity_col=0)

    disease_drug_degree = _association_degree(data['drdi'], args.disease_number, entity_col=1)
    disease_protein_degree = _association_degree(data['dipr'], args.disease_number, entity_col=0)

    drug_topology = np.concatenate(
        [drug_graph_features, drug_disease_degree[:, None], drug_protein_degree[:, None]],
        axis=1,
    )
    disease_topology = np.concatenate(
        [disease_graph_features, disease_drug_degree[:, None], disease_protein_degree[:, None]],
        axis=1,
    )

    drug_topology = torch.tensor(drug_topology, dtype=torch.float32)
    disease_topology = torch.tensor(disease_topology, dtype=torch.float32)

    torch.save(drug_topology, drug_cache)
    torch.save(disease_topology, disease_cache)

    return drug_topology, disease_topology
