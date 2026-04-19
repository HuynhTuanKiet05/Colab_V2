import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from AMDGT_original.model import gt_net_drug, gt_net_disease
from .rlg_hgt import RLGHGT


def _valid_num_heads(dim, preferred):
    for heads in range(min(preferred, dim), 0, -1):
        if dim % heads == 0:
            return heads
    return 1


class MultiViewFusion(nn.Module):
    def __init__(self, dim, dropout):
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


class TokenMixer(nn.Module):
    def __init__(self, dim, num_tokens, num_heads, num_layers, dropout):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.token_embeddings = nn.Parameter(torch.randn(num_tokens, dim) * 0.02)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, tokens):
        mixed = self.encoder(tokens + self.token_embeddings.unsqueeze(0))
        weights = torch.softmax(self.pool(mixed).squeeze(-1), dim=1)
        pooled = (weights.unsqueeze(-1) * mixed).sum(dim=1)
        return self.out_norm(pooled), weights


class PairMixtureOfExperts(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        pair_dim = dim * 6 + 4
        gate_dim = dim * 3 + 4
        hidden = max(dim * 3, 256)
        compact = max(dim * 2, 192)

        self.bilinear = nn.Bilinear(dim, dim, 2)
        self.expert_main = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, compact),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(compact, 2),
        )
        self.expert_local = nn.Sequential(
            nn.LayerNorm(dim * 3 + 4),
            nn.Linear(dim * 3 + 4, compact),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(compact, 2),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(gate_dim),
            nn.Linear(gate_dim, compact),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(compact, 3),
        )
        self.skip = nn.Linear(pair_dim, 2)

    def forward(self, drug_repr, disease_repr, topology_score=None):
        pair_mul = drug_repr * disease_repr
        pair_diff = torch.abs(drug_repr - disease_repr)
        pair_sum = drug_repr + disease_repr
        pair_sqdiff = (drug_repr - disease_repr) ** 2
        pair_dot = (drug_repr * disease_repr).sum(dim=-1, keepdim=True)
        pair_cos = F.cosine_similarity(drug_repr, disease_repr, dim=-1).unsqueeze(-1)
        if topology_score is None:
            topology_score = torch.zeros_like(pair_dot)
        fuzzy_close = torch.exp(-pair_diff.mean(dim=-1, keepdim=True))
        fuzzy_overlap = torch.sigmoid(pair_cos)

        full_features = torch.cat(
            [drug_repr, disease_repr, pair_mul, pair_diff, pair_sum, pair_sqdiff, pair_dot, pair_cos, topology_score, fuzzy_close],
            dim=-1,
        )
        local_features = torch.cat([pair_mul, pair_diff, pair_sum, pair_cos, pair_dot, topology_score, fuzzy_close, fuzzy_overlap], dim=-1)
        gate_features = torch.cat([pair_mul, pair_diff, pair_sum, pair_cos, pair_dot, topology_score, fuzzy_close, fuzzy_overlap], dim=-1)

        experts = torch.stack(
            [
                self.bilinear(drug_repr, disease_repr),
                self.expert_main(full_features),
                self.expert_local(local_features),
            ],
            dim=1,
        )
        weights = torch.softmax(self.gate(gate_features), dim=-1).unsqueeze(-1)
        return (weights * experts).sum(dim=1) + self.skip(full_features)


class AMNTDDA(nn.Module):
    def __init__(self, args):
        super(AMNTDDA, self).__init__()
        self.args = args
        self.runtime_device = torch.device(os.environ.get('AMDGT_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.drug_feature_dim = 300
        self.disease_feature_dim = 64
        self.protein_feature_dim = 320
        self.contrastive_temperature = getattr(args, 'contrastive_temperature', 0.2)
        self.cached_aux = {}

        self.drug_linear = nn.Linear(self.drug_feature_dim, args.hgt_in_dim)
        self.disease_linear = nn.Linear(self.disease_feature_dim, args.hgt_in_dim)
        self.protein_linear = nn.Linear(self.protein_feature_dim, args.hgt_in_dim)
        self.drug_norm = nn.LayerNorm(args.hgt_in_dim)
        self.disease_norm = nn.LayerNorm(args.hgt_in_dim)
        self.protein_norm = nn.LayerNorm(args.hgt_in_dim)
        self.input_dropout = nn.Dropout(args.dropout)

        self.drug_view_encoders = nn.ModuleDict({
            'fingerprint': gt_net_drug.GraphTransformer(self.runtime_device, args.gt_layer, args.drug_number, args.gt_out_dim, args.gt_out_dim, args.gt_head, args.dropout),
            'gip': gt_net_drug.GraphTransformer(self.runtime_device, args.gt_layer, args.drug_number, args.gt_out_dim, args.gt_out_dim, args.gt_head, args.dropout),
            'consensus': gt_net_drug.GraphTransformer(self.runtime_device, args.gt_layer, args.drug_number, args.gt_out_dim, args.gt_out_dim, args.gt_head, args.dropout),
        })
        self.disease_view_encoders = nn.ModuleDict({
            'phenotype': gt_net_disease.GraphTransformer(self.runtime_device, args.gt_layer, args.disease_number, args.gt_out_dim, args.gt_out_dim, args.gt_head, args.dropout),
            'gip': gt_net_disease.GraphTransformer(self.runtime_device, args.gt_layer, args.disease_number, args.gt_out_dim, args.gt_out_dim, args.gt_head, args.dropout),
            'consensus': gt_net_disease.GraphTransformer(self.runtime_device, args.gt_layer, args.disease_number, args.gt_out_dim, args.gt_out_dim, args.gt_head, args.dropout),
        })

        canonical_etypes = [
            ('drug', 'association', 'disease'),
            ('disease', 'association_rev', 'drug'),
            ('drug', 'association', 'protein'),
            ('protein', 'association_rev', 'drug'),
            ('disease', 'association', 'protein'),
            ('protein', 'association_rev', 'disease'),
        ]
        node_types = ['drug', 'disease', 'protein']
        self.hgt = RLGHGT(
            hidden_dim=args.hgt_in_dim,
            out_dim=args.hgt_in_dim,
            num_heads=args.hgt_head,
            num_layers=args.hgt_layer,
            canonical_etypes=canonical_etypes,
            node_types=node_types,
            dropout=args.dropout,
            use_relation_attention=getattr(args, 'use_relation_attention', True),
            use_metapath=getattr(args, 'use_metapath', True),
            use_global=getattr(args, 'use_global_hgt', True),
            use_topological=getattr(args, 'use_topological', True),
        )

        self.hgt_drug_out = nn.Sequential(nn.Linear(args.hgt_in_dim, args.gt_out_dim), nn.LayerNorm(args.gt_out_dim))
        self.hgt_disease_out = nn.Sequential(nn.Linear(args.hgt_in_dim, args.gt_out_dim), nn.LayerNorm(args.gt_out_dim))
        self.drug_raw_context = nn.Sequential(
            nn.Linear(self.drug_feature_dim, args.gt_out_dim),
            nn.LayerNorm(args.gt_out_dim),
            nn.GELU(),
            nn.Dropout(args.dropout),
        )
        self.disease_raw_context = nn.Sequential(
            nn.Linear(self.disease_feature_dim, args.gt_out_dim),
            nn.LayerNorm(args.gt_out_dim),
            nn.GELU(),
            nn.Dropout(args.dropout),
        )

        mix_heads = _valid_num_heads(args.gt_out_dim, args.tr_head)
        self.drug_view_fusion = MultiViewFusion(args.gt_out_dim, args.dropout)
        self.disease_view_fusion = MultiViewFusion(args.gt_out_dim, args.dropout)
        self.drug_token_mixer = TokenMixer(args.gt_out_dim, num_tokens=5, num_heads=mix_heads, num_layers=args.tr_layer, dropout=args.dropout)
        self.disease_token_mixer = TokenMixer(args.gt_out_dim, num_tokens=5, num_heads=mix_heads, num_layers=args.tr_layer, dropout=args.dropout)

        align_dim = min(max(args.gt_out_dim // 2, 64), 256)
        self.drug_align_sim = nn.Linear(args.gt_out_dim, align_dim)
        self.drug_align_hgt = nn.Linear(args.gt_out_dim, align_dim)
        self.disease_align_sim = nn.Linear(args.gt_out_dim, align_dim)
        self.disease_align_hgt = nn.Linear(args.gt_out_dim, align_dim)

        self.topology_scorer = nn.Sequential(
            nn.Linear(args.gt_out_dim * 2, args.gt_out_dim),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.gt_out_dim, 1),
        )
        self.topology_scale = nn.Parameter(torch.tensor(0.15))
        self.pair_scorer = PairMixtureOfExperts(args.gt_out_dim, args.dropout)

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

    def _contrastive_loss(self, lhs, rhs):
        lhs = F.normalize(lhs, dim=-1)
        rhs = F.normalize(rhs, dim=-1)
        logits = lhs @ rhs.t() / self.contrastive_temperature
        labels = torch.arange(lhs.shape[0], device=lhs.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

    def forward(self, drdr_graph, didi_graph, drdipr_graph, drug_feature, disease_feature, protein_feature, sample, return_aux=False):
        drug_views = self._encode_similarity_views(drdr_graph, self.drug_view_encoders)
        disease_views = self._encode_similarity_views(didi_graph, self.disease_view_encoders)

        drug_view_stack = torch.stack([drug_views['fingerprint'], drug_views['gip'], drug_views['consensus']], dim=1)
        disease_view_stack = torch.stack([disease_views['phenotype'], disease_views['gip'], disease_views['consensus']], dim=1)

        drug_view_fused, drug_view_weights = self.drug_view_fusion(drug_view_stack)
        disease_view_fused, disease_view_weights = self.disease_view_fusion(disease_view_stack)

        raw_drug_token = self.drug_raw_context(drug_feature)
        raw_disease_token = self.disease_raw_context(disease_feature)

        hgt_drug_feature = self.input_dropout(self.drug_norm(self.drug_linear(drug_feature)))
        hgt_disease_feature = self.input_dropout(self.disease_norm(self.disease_linear(disease_feature)))
        hgt_protein_feature = self.input_dropout(self.protein_norm(self.protein_linear(protein_feature)))
        feature_dict = {
            'drug': hgt_drug_feature,
            'disease': hgt_disease_feature,
            'protein': hgt_protein_feature,
        }

        if drdipr_graph.device != hgt_drug_feature.device:
            drdipr_graph = drdipr_graph.to(hgt_drug_feature.device)
        hgt_out = self.hgt(drdipr_graph, feature_dict)
        drug_hgt = self.hgt_drug_out(hgt_out['drug'])
        disease_hgt = self.hgt_disease_out(hgt_out['disease'])

        drug_tokens = torch.stack(
            [drug_views['fingerprint'], drug_views['gip'], drug_views['consensus'], drug_hgt, raw_drug_token],
            dim=1,
        )
        disease_tokens = torch.stack(
            [disease_views['phenotype'], disease_views['gip'], disease_views['consensus'], disease_hgt, raw_disease_token],
            dim=1,
        )
        drug_repr, drug_token_weights = self.drug_token_mixer(drug_tokens)
        disease_repr, disease_token_weights = self.disease_token_mixer(disease_tokens)

        pair_drug = drug_repr[sample[:, 0]]
        pair_disease = disease_repr[sample[:, 1]]
        topology_score = self.topology_scale * torch.tanh(self.topology_scorer(torch.cat([pair_drug, pair_disease], dim=-1)))
        output = self.pair_scorer(pair_drug, pair_disease, topology_score=topology_score)

        self.cached_aux = {
            'contrastive': self._contrastive_loss(self.drug_align_sim(drug_view_fused), self.drug_align_hgt(drug_hgt))
            + self._contrastive_loss(self.disease_align_sim(disease_view_fused), self.disease_align_hgt(disease_hgt)),
            'drug_view_weights': drug_view_weights.detach(),
            'disease_view_weights': disease_view_weights.detach(),
            'drug_token_weights': drug_token_weights.detach(),
            'disease_token_weights': disease_token_weights.detach(),
            'topology_score': topology_score.detach(),
        }

        if return_aux:
            return drug_repr, output, self.cached_aux
        return drug_repr, output
