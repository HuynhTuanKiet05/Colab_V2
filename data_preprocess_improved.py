import os
import dgl
import networkx as nx
import numpy as np
import torch

from AMDGT_original.data_preprocess import data_processing as original_data_processing
from AMDGT_original.data_preprocess import dgl_similarity_graph as original_dgl_similarity_graph
from AMDGT_original.data_preprocess import get_adj
from AMDGT_original.data_preprocess import get_data as original_get_data
from AMDGT_original.data_preprocess import k_fold as original_k_fold
from AMDGT_original.data_preprocess import k_matrix

device = torch.device(os.environ.get('AMDGT_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'))

def get_data(args):
    return original_get_data(args)


def data_processing(data, args):
    return original_data_processing(data, args)


def k_fold(data, args):
    # Reuse the original split logic so baseline and improved runs share the
    # exact same folds during 10-fold cross-validation.
    return original_k_fold(data, args)


def dgl_similarity_graph(data, args):
    return original_dgl_similarity_graph(data, args)


def _build_similarity_graph(matrix, feature_key, k):
    graph_matrix = k_matrix(matrix, k)
    if hasattr(nx, 'from_numpy_array'):
        nx_graph = nx.from_numpy_array(graph_matrix)
    else:
        nx_graph = nx.from_numpy_matrix(graph_matrix)
    graph = dgl.from_networkx(nx_graph)
    graph.ndata[feature_key] = torch.tensor(matrix)
    return graph


def dgl_similarity_view_graphs(data, args):
    drug_graphs = {
        'fingerprint': _build_similarity_graph(data['drf'], 'drs', args.neighbor),
        'gip': _build_similarity_graph(data['drg'], 'drs', args.neighbor),
        'consensus': _build_similarity_graph(data['drs'], 'drs', args.neighbor),
    }
    disease_graphs = {
        'phenotype': _build_similarity_graph(data['dip'], 'dis', args.neighbor),
        'gip': _build_similarity_graph(data['dig'], 'dis', args.neighbor),
        'consensus': _build_similarity_graph(data['dis'], 'dis', args.neighbor),
    }
    return drug_graphs, disease_graphs, data


def dgl_heterograph(data, drdi, args):
    def to_edge_tuple(edges):
        if edges.size == 0:
            empty = np.array([], dtype=np.int64)
            return (empty, empty)
        return (edges[:, 0], edges[:, 1])

    drdi_list, drpr_list, dipr_list = [], [], []
    for i in range(drdi.shape[0]):
        drdi_list.append(drdi[i])
    for i in range(data['drpr'].shape[0]):
        drpr_list.append(data['drpr'][i])
    for i in range(data['dipr'].shape[0]):
        dipr_list.append(data['dipr'][i])

    drdi_arr = np.asarray(drdi_list, dtype=int) if len(drdi_list) > 0 else np.empty((0, 2), dtype=int)
    drpr_arr = np.asarray(drpr_list, dtype=int) if len(drpr_list) > 0 else np.empty((0, 2), dtype=int)
    dipr_arr = np.asarray(dipr_list, dtype=int) if len(dipr_list) > 0 else np.empty((0, 2), dtype=int)

    node_dict = {
        'drug': args.drug_number,
        'disease': args.disease_number,
        'protein': args.protein_number
    }

    heterograph_dict = {
        ('drug', 'association', 'disease'): to_edge_tuple(drdi_arr),
        ('disease', 'association_rev', 'drug'): to_edge_tuple(drdi_arr[:, ::-1] if drdi_arr.size > 0 else drdi_arr),
        ('drug', 'association', 'protein'): to_edge_tuple(drpr_arr),
        ('protein', 'association_rev', 'drug'): to_edge_tuple(drpr_arr[:, ::-1] if drpr_arr.size > 0 else drpr_arr),
        ('disease', 'association', 'protein'): to_edge_tuple(dipr_arr),
        ('protein', 'association_rev', 'disease'): to_edge_tuple(dipr_arr[:, ::-1] if dipr_arr.size > 0 else dipr_arr)
    }

    data['feature_dict'] ={
        'drug': torch.tensor(data['drugfeature']),
        'disease': torch.tensor(data['diseasefeature']),
        'protein': torch.tensor(data['proteinfeature'])
    }

    drdipr_graph = dgl.heterograph(heterograph_dict, num_nodes_dict=node_dict)

    edge_stats = {
        'pair_bias': torch.log1p(torch.tensor(
            [drdipr_graph.num_edges(('drug', 'association', 'disease')),
             drdipr_graph.num_edges(('drug', 'association', 'protein')),
             drdipr_graph.num_edges(('disease', 'association', 'protein'))],
            dtype=torch.float32,
        )).mean().view(1, 1)
    }

    return drdipr_graph, data, edge_stats





