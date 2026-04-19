import dgl
import dgl.function as fn
import dgl.nn.functional as dglf
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


class RLGHGTLayer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        out_dim,
        num_heads,
        canonical_etypes,
        node_types,
        metapaths_by_target=None,
        dropout=0.2,
        use_relation_attention=True,
        use_metapath=True,
        use_global=True,
        use_topological=True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.node_types = list(node_types)
        self.canonical_etypes = list(canonical_etypes)
        self.rel_ids = {etype: i for i, etype in enumerate(self.canonical_etypes)}
        self.metapaths_by_target = metapaths_by_target or {}
        self.use_relation_attention = use_relation_attention
        self.use_metapath = use_metapath
        self.use_global = use_global
        self.use_topological = use_topological

        self.q_proj = nn.ModuleDict({ntype: nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, out_dim)) for ntype in self.node_types})
        self.k_proj = nn.ModuleDict({ntype: nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, out_dim)) for ntype in self.node_types})
        self.v_proj = nn.ModuleDict({ntype: nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, out_dim)) for ntype in self.node_types})
        self.rel_k = nn.Embedding(len(self.canonical_etypes), out_dim)
        self.rel_v = nn.Embedding(len(self.canonical_etypes), out_dim)
        self.rel_bias = nn.Embedding(len(self.canonical_etypes), num_heads)
        self.rel_scale = nn.Embedding(len(self.canonical_etypes), 1)

        self.meta_q = nn.ModuleDict({ntype: nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim)) for ntype in self.node_types})
        self.meta_v = nn.ModuleDict({ntype: nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim)) for ntype in self.node_types})

        self.out_proj = nn.ModuleDict({ntype: nn.Sequential(nn.LayerNorm(out_dim), nn.Linear(out_dim, hidden_dim)) for ntype in self.node_types})
        self.topo_proj = nn.ModuleDict({ntype: nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim)) for ntype in self.node_types})
        self.global_proj = nn.ModuleDict({ntype: nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim)) for ntype in self.node_types})
        self.meta_gate = nn.ModuleDict({ntype: nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid()) for ntype in self.node_types})
        self.branch_gate = nn.ModuleDict({ntype: nn.Sequential(nn.Linear(hidden_dim * 4, hidden_dim), nn.Sigmoid()) for ntype in self.node_types})
        self.ffn = nn.ModuleDict({
            ntype: nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
            ) for ntype in self.node_types
        })
        self.norm1 = nn.ModuleDict({ntype: nn.LayerNorm(hidden_dim) for ntype in self.node_types})
        self.norm2 = nn.ModuleDict({ntype: nn.LayerNorm(hidden_dim) for ntype in self.node_types})
        self.dropout = nn.Dropout(dropout)

    def _attn_message(self, rel_graph, h, srctype, dsttype, rel_id):
        with rel_graph.local_scope():
            q = self.q_proj[dsttype](h[dsttype]).view(-1, self.num_heads, self.head_dim)
            k = self.k_proj[srctype](h[srctype]).view(-1, self.num_heads, self.head_dim)
            v = self.v_proj[srctype](h[srctype]).view(-1, self.num_heads, self.head_dim)
            if self.use_relation_attention:
                rk = self.rel_k.weight[rel_id].view(1, self.num_heads, self.head_dim)
                rv = self.rel_v.weight[rel_id].view(1, self.num_heads, self.head_dim)
                rel_bias = self.rel_bias.weight[rel_id].view(1, self.num_heads)
                rel_scale = torch.sigmoid(self.rel_scale.weight[rel_id]).view(1, 1, 1) + 1.0
                k = k + rk
                v = v + rv
            else:
                rel_bias = torch.zeros(1, self.num_heads, device=q.device)
                rel_scale = 1.0

            rel_graph.srcdata['k'] = k
            rel_graph.srcdata['v'] = v
            rel_graph.dstdata['q'] = q
            rel_graph.apply_edges(fn.u_dot_v('k', 'q', 'score'))
            score = rel_graph.edata['score'] / (self.head_dim ** 0.5)
            score = score * rel_scale
            score = score + rel_bias.view(1, self.num_heads, 1)
            attn = dglf.edge_softmax(rel_graph, score)
            rel_graph.edata['a'] = attn
            rel_graph.update_all(fn.u_mul_e('v', 'a', 'm'), fn.sum('m', 'out'))
            out = rel_graph.dstdata['out'].reshape(-1, self.out_dim)
            return out

    def _metapath_message(self, g, h_dict, target_type, metapath_graphs=None):
        meta_out = torch.zeros(g.num_nodes(target_type), self.hidden_dim, device=h_dict[target_type].device)
        if not self.use_metapath:
            return meta_out

        if not metapath_graphs:
            return meta_out

        target_q = self.meta_q[target_type](h_dict[target_type])
        path_msgs = []
        path_scores = []
        for src_type, rg in metapath_graphs:
            with rg.local_scope():
                src_h = self.meta_v[src_type](h_dict[src_type])
                rg.srcdata['h'] = src_h
                rg.dstdata['q'] = target_q
                rg.apply_edges(fn.u_add_v('h', 'q', 'score'))
                score = torch.tanh(rg.edata['score']).mean(dim=-1, keepdim=True)
                attn = dglf.edge_softmax(rg, score)
                rg.edata['a'] = attn
                rg.update_all(fn.u_mul_e('h', 'a', 'm'), fn.sum('m', 'out'))
                out = rg.dstdata['out']
                if out.dim() == 1:
                    out = out.unsqueeze(-1)
                path_msgs.append(out)
                pooled = out.mean(dim=0, keepdim=True)
                path_scores.append(pooled)

        if not path_msgs:
            return meta_out

        if len(path_msgs) == 1:
            return path_msgs[0]

        scores = torch.cat(path_scores, dim=0)
        weights = torch.softmax(scores.mean(dim=1, keepdim=True), dim=0)
        stacked = torch.stack(path_msgs, dim=0)
        return (weights.unsqueeze(-1) * stacked).sum(dim=0)

    def forward(self, g, h_dict, metapath_cache=None):
        local_out = {ntype: torch.zeros(g.num_nodes(ntype), self.out_dim, device=h_dict[ntype].device) for ntype in self.node_types}
        for canonical_etype in self.canonical_etypes:
            srctype, _, dsttype = canonical_etype
            rel_graph = g[canonical_etype]
            rel_id = self.rel_ids[canonical_etype]
            if rel_graph.num_edges() == 0:
                continue
            local_out[dsttype] = local_out[dsttype] + self._attn_message(rel_graph, h_dict, srctype, dsttype, rel_id)

        new_h = {}
        layer_w = torch.softmax(self.layer_alpha, dim=0)
        cur_layer_w = layer_w.mean()
        for ntype in self.node_types:
            local = self.out_proj[ntype](local_out[ntype])

            meta_msg = torch.zeros_like(local)
            if self.use_metapath:
                graphs_for_target = None if metapath_cache is None else metapath_cache.get(ntype)
                meta_msg = self._metapath_message(g, h_dict, ntype, graphs_for_target)
                meta = self.meta_gate[ntype](torch.cat([h_dict[ntype], meta_msg], dim=-1))
            else:
                meta = torch.zeros_like(local)

            if self.use_metapath and self.use_topological:
                topo = self.topo_proj[ntype](meta_msg)
            else:
                topo = torch.zeros_like(local)

            if self.use_global:
                global_ctx = self.global_proj[ntype](h_dict[ntype].mean(dim=0, keepdim=True)).repeat(
                    h_dict[ntype].shape[0], 1
                )
            else:
                global_ctx = torch.zeros_like(local)

            # Branch-specific gating with relation/meta/global focus.
            branch_stack = torch.stack([local, meta, topo, global_ctx], dim=1)
            branch_gate = self.branch_gate[ntype](torch.cat([local, meta, topo, global_ctx], dim=-1))
            branch_mix = (branch_gate.unsqueeze(1) * branch_stack).sum(dim=1)

            fused = h_dict[ntype] + cur_layer_w * self.dropout(branch_mix)
            fused = self.norm1[ntype](fused)
            fused = self.norm2[ntype](fused + self.dropout(self.ffn[ntype](fused)))
            new_h[ntype] = fused
        return new_h


class RLGHGT(nn.Module):
    def __init__(
        self,
        hidden_dim,
        out_dim,
        num_heads,
        num_layers,
        canonical_etypes,
        node_types,
        dropout=0.2,
        use_relation_attention=True,
        use_metapath=True,
        use_global=True,
        use_topological=True,
        ):
        super().__init__()
        metapaths_by_target = {
            'drug': [
                [('drug', 'association', 'protein'), ('protein', 'association_rev', 'drug')],
                [('drug', 'association', 'disease'), ('disease', 'association_rev', 'drug')],
                [('drug', 'association', 'protein'), ('protein', 'association_rev', 'disease'), ('disease', 'association_rev', 'drug')],
            ],
            'disease': [
                [('disease', 'association', 'protein'), ('protein', 'association_rev', 'disease')],
                [('disease', 'association_rev', 'drug'), ('drug', 'association', 'disease')],
                [('disease', 'association', 'protein'), ('protein', 'association_rev', 'drug'), ('drug', 'association', 'disease')],
            ],
            'protein': [
                [('protein', 'association_rev', 'drug'), ('drug', 'association', 'protein')],
                [('protein', 'association_rev', 'disease'), ('disease', 'association', 'protein')],
            ],
        }
        self.metapaths_by_target = metapaths_by_target
        self.layers = nn.ModuleList([
            RLGHGTLayer(
                hidden_dim,
                out_dim,
                num_heads,
                canonical_etypes,
                node_types,
                metapaths_by_target,
                dropout,
                use_relation_attention=use_relation_attention,
                use_metapath=use_metapath,
                use_global=use_global,
                use_topological=use_topological,
            )
            for _ in range(num_layers)
        ])
        self.layer_scores = nn.ModuleDict({ntype: nn.Linear(hidden_dim, 1) for ntype in node_types})
        self.layer_alpha = nn.Parameter(torch.zeros(num_layers))
        self.node_types = list(node_types)
        self._metapath_cache_key: Optional[int] = None
        self._metapath_cache: Dict[str, List[Tuple[str, dgl.DGLGraph]]] = {}

    def _build_metapath_cache(self, g):
        cache: Dict[str, List[Tuple[str, dgl.DGLGraph]]] = {}
        for target_type, metapaths in self.metapaths_by_target.items():
            cache[target_type] = []
            for metapath in metapaths:
                reachable_graph = dgl.metapath_reachable_graph(g, metapath)
                if reachable_graph.num_edges() == 0:
                    continue
                cache[target_type].append((metapath[0][0], reachable_graph))
        return cache

    def _get_metapath_cache(self, g):
        cache_key = id(g)
        if self._metapath_cache_key != cache_key:
            self._metapath_cache = self._build_metapath_cache(g)
            self._metapath_cache_key = cache_key
        return self._metapath_cache

    def forward(self, g, h_dict):
        hs = {ntype: [h_dict[ntype]] for ntype in self.node_types}
        cur = h_dict
        metapath_cache = None
        if self.layers and self.layers[0].use_metapath:
            metapath_cache = self._get_metapath_cache(g)
        for layer in self.layers:
            cur = layer(g, cur, metapath_cache)
            for ntype in self.node_types:
                hs[ntype].append(cur[ntype])

        out = {}
        for ntype in self.node_types:
            stack = torch.stack(hs[ntype], dim=1)
            score = self.layer_scores[ntype](stack).squeeze(-1)
            alpha = torch.softmax(score, dim=1).unsqueeze(-1)
            out[ntype] = (alpha * stack).sum(dim=1)
        return out
