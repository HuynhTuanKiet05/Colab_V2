import dgl
import dgl.nn.pytorch
import torch
import torch.nn as nn
import torch.nn.functional as fn

from AMDGT_original.model import gt_net_drug, gt_net_disease


class TopologyEncoder(nn.Module):
    def __init__(self, topo_feat_dim=7, hidden_dim=128, out_dim=200, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(topo_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, topo_features):
        return self.encoder(topo_features)


class MultiViewFusion(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        hidden = max(dim // 2, 64)
        self.score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, views):
        weights = torch.softmax(self.score(views).squeeze(-1), dim=1)
        fused = (weights.unsqueeze(-1) * views).sum(dim=1)
        return self.out_norm(fused), weights


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, view1, view2):
        z1 = fn.normalize(view1, dim=1)
        z2 = fn.normalize(view2, dim=1)
        similarity = torch.mm(z1, z2.t()) / self.temperature
        labels = torch.arange(z1.shape[0], device=z1.device)
        return fn.cross_entropy(similarity, labels)


class MultiViewContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.loss = ContrastiveLoss(temperature=temperature)

    def forward(self, sim_view, assoc_view, topo_view):
        loss_sa = self.loss(sim_view, assoc_view)
        loss_st = self.loss(sim_view, topo_view)
        loss_at = self.loss(assoc_view, topo_view)
        return (loss_sa + loss_st + loss_at) / 3.0


class FuzzyGate(nn.Module):
    def __init__(self, base_dim, topo_dim, dropout=0.1, gate_mode='vector', gate_bias_init=-2.0):
        super().__init__()
        if gate_mode not in {'scalar', 'vector'}:
            raise ValueError(f'Unsupported gate_mode: {gate_mode}')
        self.gate_mode = gate_mode
        gate_out_dim = 1 if gate_mode == 'scalar' else base_dim

        self.topo_proj = nn.Sequential(
            nn.Linear(topo_dim, base_dim),
            nn.LayerNorm(base_dim),
            nn.ReLU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(base_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, gate_out_dim),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)
        self.topo_norm = nn.LayerNorm(base_dim)
        nn.init.constant_(self.gate[-2].bias, gate_bias_init)

    def forward(self, base_repr, topo_repr):
        topo_proj = self.topo_norm(self.dropout(self.topo_proj(topo_repr)))
        gate = self.gate(torch.cat([base_repr, topo_proj], dim=-1))
        return base_repr + gate * topo_proj


class HybridPairDecoder(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        hidden = max(dim * 2, 256)
        compact = max(dim, 128)
        self.bilinear = nn.Bilinear(dim, dim, 2)
        self.skip = nn.Linear(dim, 2)
        self.decoder = nn.Sequential(
            nn.LayerNorm(dim * 4 + 3),
            nn.Linear(dim * 4 + 3, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, compact),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(compact, 2),
        )

    def forward(self, drug_repr, disease_repr, topology_score=None, pair_bias=None):
        pair_mul = drug_repr * disease_repr
        pair_diff = torch.abs(drug_repr - disease_repr)
        pair_sum = drug_repr + disease_repr
        pair_sqdiff = (drug_repr - disease_repr) ** 2
        pair_dot = (drug_repr * disease_repr).sum(dim=-1, keepdim=True)
        pair_cos = fn.cosine_similarity(drug_repr, disease_repr, dim=-1).unsqueeze(-1)
        if topology_score is None:
            topology_score = torch.zeros_like(pair_dot)
        if pair_bias is not None:
            topology_score = topology_score + pair_bias
        features = torch.cat([pair_mul, pair_diff, pair_sum, pair_sqdiff, pair_dot, pair_cos, topology_score], dim=-1)
        return self.bilinear(drug_repr, disease_repr) + self.skip(pair_mul) + self.decoder(features)


class PairEnsembleGate(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        hidden = max(dim, 256)
        self.gate = nn.Sequential(
            nn.LayerNorm(dim * 3 + 3),
            nn.Linear(dim * 3 + 3, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, pair_mul, pair_diff, pair_sum, pair_dot, pair_cos, topology_score):
        features = torch.cat([pair_mul, pair_diff, pair_sum, pair_dot, pair_cos, topology_score], dim=-1)
        return torch.sigmoid(self.gate(features))


class TMC_AMDGT_RVG(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device

        self.drug_linear = nn.Linear(300, args.hgt_in_dim)
        self.disease_linear = nn.Linear(64, args.hgt_in_dim)
        self.protein_linear = nn.Linear(320, args.hgt_in_dim)

        self.drug_view_encoders = nn.ModuleDict({
            'fingerprint': gt_net_drug.GraphTransformer(
                self.device,
                args.gt_layer,
                args.drug_number,
                args.gt_out_dim,
                args.gt_out_dim,
                args.gt_head,
                args.dropout,
            ),
            'gip': gt_net_drug.GraphTransformer(
                self.device,
                args.gt_layer,
                args.drug_number,
                args.gt_out_dim,
                args.gt_out_dim,
                args.gt_head,
                args.dropout,
            ),
            'consensus': gt_net_drug.GraphTransformer(
                self.device,
                args.gt_layer,
                args.drug_number,
                args.gt_out_dim,
                args.gt_out_dim,
                args.gt_head,
                args.dropout,
            ),
        })
        self.disease_view_encoders = nn.ModuleDict({
            'phenotype': gt_net_disease.GraphTransformer(
                self.device,
                args.gt_layer,
                args.disease_number,
                args.gt_out_dim,
                args.gt_out_dim,
                args.gt_head,
                args.dropout,
            ),
            'gip': gt_net_disease.GraphTransformer(
                self.device,
                args.gt_layer,
                args.disease_number,
                args.gt_out_dim,
                args.gt_out_dim,
                args.gt_head,
                args.dropout,
            ),
            'consensus': gt_net_disease.GraphTransformer(
                self.device,
                args.gt_layer,
                args.disease_number,
                args.gt_out_dim,
                args.gt_out_dim,
                args.gt_head,
                args.dropout,
            ),
        })
        self.drug_view_fusion = MultiViewFusion(args.gt_out_dim, dropout=args.dropout)
        self.disease_view_fusion = MultiViewFusion(args.gt_out_dim, dropout=args.dropout)

        self.hgt_shared = dgl.nn.pytorch.conv.HGTConv(
            args.hgt_in_dim,
            int(args.hgt_in_dim / args.hgt_head),
            args.hgt_head,
            3,
            3,
            args.dropout,
        )
        self.hgt_last = dgl.nn.pytorch.conv.HGTConv(
            args.hgt_in_dim,
            args.hgt_head_dim,
            args.hgt_head,
            3,
            3,
            args.dropout,
        )
        self.hgt_layers = nn.ModuleList()
        for _ in range(args.hgt_layer - 1):
            self.hgt_layers.append(self.hgt_shared)
        self.hgt_layers.append(self.hgt_last)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.gt_out_dim,
            nhead=args.tr_head,
            dropout=args.dropout,
            activation='gelu',
            batch_first=True,
        )
        self.drug_trans = nn.TransformerEncoder(encoder_layer, num_layers=args.tr_layer)
        self.disease_trans = nn.TransformerEncoder(encoder_layer, num_layers=args.tr_layer)

        base_dim = args.gt_out_dim * 2
        self.drug_topology_encoder = TopologyEncoder(
            topo_feat_dim=args.topo_feat_dim,
            hidden_dim=args.topo_hidden,
            out_dim=args.gt_out_dim,
            dropout=args.dropout,
        )
        self.disease_topology_encoder = TopologyEncoder(
            topo_feat_dim=args.topo_feat_dim,
            hidden_dim=args.topo_hidden,
            out_dim=args.gt_out_dim,
            dropout=args.dropout,
        )
        self.drug_gate = FuzzyGate(
            base_dim=base_dim,
            topo_dim=args.gt_out_dim,
            dropout=args.dropout,
            gate_mode=args.gate_mode,
            gate_bias_init=args.gate_bias_init,
        )
        self.disease_gate = FuzzyGate(
            base_dim=base_dim,
            topo_dim=args.gt_out_dim,
            dropout=args.dropout,
            gate_mode=args.gate_mode,
            gate_bias_init=args.gate_bias_init,
        )

        self.contrastive_loss = MultiViewContrastiveLoss(temperature=args.temperature)

        self.elementwise_mlp = nn.Sequential(
            nn.Linear(base_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2),
        )
        self.pair_decoder = getattr(args, 'pair_decoder', 'hybrid_ensemble')
        self.hybrid_pair_decoder = HybridPairDecoder(base_dim, dropout=args.dropout)
        self.ensemble_gate = PairEnsembleGate(base_dim, dropout=args.dropout)
        self.pair_topology = nn.Sequential(
            nn.Linear(base_dim * 2, base_dim),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(base_dim, 1),
        )
        self.topology_scale = nn.Parameter(torch.tensor(0.20))

    def _prepare_graph_dict(self, graph_input, graph_names):
        if isinstance(graph_input, dict):
            return graph_input
        return {name: graph_input for name in graph_names}

    def _encode_similarity_views(self, graph_input, encoders):
        prepared = self._prepare_graph_dict(graph_input, encoders.keys())
        fallback_graph = prepared.get('consensus', next(iter(prepared.values())))
        fallback = encoders['consensus'](fallback_graph)
        outputs = {'consensus': fallback}
        for name, encoder in encoders.items():
            graph = prepared.get(name)
            outputs[name] = encoder(graph) if graph is not None else fallback
        return outputs

    def _association_views(self, drdipr_graph, drug_feature, disease_feature, protein_feature):
        drug_feature = self.drug_linear(drug_feature)
        disease_feature = self.disease_linear(disease_feature)
        protein_feature = self.protein_linear(protein_feature)

        feature_dict = {
            'drug': drug_feature,
            'disease': disease_feature,
            'protein': protein_feature,
        }

        drdipr_graph.ndata['h'] = feature_dict
        homo_graph = dgl.to_homogeneous(drdipr_graph, ndata='h')
        features = torch.cat((drug_feature, disease_feature, protein_feature), dim=0)
        for layer in self.hgt_layers:
            features = layer(homo_graph, features, homo_graph.ndata['_TYPE'], homo_graph.edata['_TYPE'], presorted=True)

        drug_assoc = features[:self.args.drug_number, :]
        disease_assoc = features[self.args.drug_number:self.args.drug_number + self.args.disease_number, :]
        return drug_assoc, disease_assoc

    def forward(
        self,
        drdr_graph,
        didi_graph,
        drdipr_graph,
        drug_feature,
        disease_feature,
        protein_feature,
        drug_topo_feat,
        disease_topo_feat,
        sample,
        edge_stats=None,
        return_aux=False,
    ):
        drug_views = self._encode_similarity_views(drdr_graph, self.drug_view_encoders)
        disease_views = self._encode_similarity_views(didi_graph, self.disease_view_encoders)
        drug_sim_stack = torch.stack([drug_views['fingerprint'], drug_views['gip'], drug_views['consensus']], dim=1)
        disease_sim_stack = torch.stack([disease_views['phenotype'], disease_views['gip'], disease_views['consensus']], dim=1)
        drug_sim, drug_view_weights = self.drug_view_fusion(drug_sim_stack)
        disease_sim, disease_view_weights = self.disease_view_fusion(disease_sim_stack)
        drug_assoc, disease_assoc = self._association_views(drdipr_graph, drug_feature, disease_feature, protein_feature)

        drug_base = self.drug_trans(torch.stack((drug_sim, drug_assoc), dim=1)).reshape(self.args.drug_number, -1)
        disease_base = self.disease_trans(torch.stack((disease_sim, disease_assoc), dim=1)).reshape(self.args.disease_number, -1)

        drug_topology = self.drug_topology_encoder(drug_topo_feat)
        disease_topology = self.disease_topology_encoder(disease_topo_feat)

        contrastive = self.contrastive_loss(drug_sim, drug_assoc, drug_topology)
        contrastive = contrastive + self.contrastive_loss(disease_sim, disease_assoc, disease_topology)
        contrastive = contrastive / 2.0

        drug_repr = self.drug_gate(drug_base, drug_topology)
        disease_repr = self.disease_gate(disease_base, disease_topology)

        pair_drug = drug_repr[sample[:, 0]]
        pair_disease = disease_repr[sample[:, 1]]
        topology_score = self.topology_scale * torch.tanh(self.pair_topology(torch.cat([pair_drug, pair_disease], dim=-1)))
        pair_bias = None
        if edge_stats is not None:
            pair_bias = edge_stats.get('pair_bias')

        pair_mul = torch.mul(pair_drug, pair_disease)
        pair_diff = torch.abs(pair_drug - pair_disease)
        pair_sum = pair_drug + pair_disease
        pair_dot = (pair_drug * pair_disease).sum(dim=-1, keepdim=True)
        pair_cos = fn.cosine_similarity(pair_drug, pair_disease, dim=-1).unsqueeze(-1)

        if self.pair_decoder == 'elementwise':
            output = self.elementwise_mlp(pair_mul)
            if pair_bias is not None:
                output = output + torch.cat([torch.zeros_like(pair_bias), pair_bias], dim=-1)
        elif self.pair_decoder == 'hybrid_ensemble':
            elementwise_logits = self.elementwise_mlp(pair_mul)
            if pair_bias is not None:
                elementwise_logits = elementwise_logits + torch.cat([torch.zeros_like(pair_bias), pair_bias], dim=-1)
            hybrid_logits = self.hybrid_pair_decoder(pair_drug, pair_disease, topology_score=topology_score, pair_bias=pair_bias)
            gate_topology = topology_score if pair_bias is None else topology_score + pair_bias
            gate = self.ensemble_gate(pair_mul, pair_diff, pair_sum, pair_dot, pair_cos, gate_topology)
            output = gate * hybrid_logits + (1.0 - gate) * elementwise_logits
        else:
            output = self.hybrid_pair_decoder(pair_drug, pair_disease, topology_score=topology_score, pair_bias=pair_bias)

        if return_aux:
            aux = {
                'contrastive': contrastive,
                'drug_repr': drug_repr,
                'disease_repr': disease_repr,
                'topology_score': topology_score.detach(),
                'drug_view_weights': drug_view_weights.detach(),
                'disease_view_weights': disease_view_weights.detach(),
            }
            return drug_repr, output, aux

        return drug_repr, output, contrastive
