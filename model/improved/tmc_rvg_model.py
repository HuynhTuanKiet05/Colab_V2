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


class TMC_AMDGT_RVG(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device

        self.drug_linear = nn.Linear(300, args.hgt_in_dim)
        self.disease_linear = nn.Linear(64, args.hgt_in_dim)
        self.protein_linear = nn.Linear(320, args.hgt_in_dim)

        self.gt_drug = gt_net_drug.GraphTransformer(
            self.device,
            args.gt_layer,
            args.drug_number,
            args.gt_out_dim,
            args.gt_out_dim,
            args.gt_head,
            args.dropout,
        )
        self.gt_disease = gt_net_disease.GraphTransformer(
            self.device,
            args.gt_layer,
            args.disease_number,
            args.gt_out_dim,
            args.gt_out_dim,
            args.gt_head,
            args.dropout,
        )

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

        encoder_layer = nn.TransformerEncoderLayer(d_model=args.gt_out_dim, nhead=args.tr_head)
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

        self.mlp = nn.Sequential(
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
    ):
        drug_sim = self.gt_drug(drdr_graph)
        disease_sim = self.gt_disease(didi_graph)
        drug_assoc, disease_assoc = self._association_views(drdipr_graph, drug_feature, disease_feature, protein_feature)

        drug_base = self.drug_trans(torch.stack((drug_sim, drug_assoc), dim=1)).view(self.args.drug_number, -1)
        disease_base = self.disease_trans(torch.stack((disease_sim, disease_assoc), dim=1)).view(self.args.disease_number, -1)

        drug_topology = self.drug_topology_encoder(drug_topo_feat)
        disease_topology = self.disease_topology_encoder(disease_topo_feat)

        contrastive = self.contrastive_loss(drug_sim, drug_assoc, drug_topology)
        contrastive = contrastive + self.contrastive_loss(disease_sim, disease_assoc, disease_topology)
        contrastive = contrastive / 2.0

        drug_repr = self.drug_gate(drug_base, drug_topology)
        disease_repr = self.disease_gate(disease_base, disease_topology)

        pair_embedding = torch.mul(drug_repr[sample[:, 0]], disease_repr[sample[:, 1]])
        output = self.mlp(pair_embedding)

        return drug_repr, output, contrastive
