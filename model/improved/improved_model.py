import os
import torch
import torch.nn as nn
from AMDGT_original.model import gt_net_drug, gt_net_disease
from .rlg_hgt import RLGHGT


class AMNTDDA(nn.Module):
    def __init__(self, args):
        super(AMNTDDA, self).__init__()
        self.args = args
        self.runtime_device = torch.device(os.environ.get('AMDGT_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.drug_linear = nn.Linear(300, args.hgt_in_dim)
        self.disease_linear = nn.Linear(64, args.hgt_in_dim)
        self.protein_linear = nn.Linear(320, args.hgt_in_dim)
        self.drug_norm = nn.LayerNorm(args.hgt_in_dim)
        self.disease_norm = nn.LayerNorm(args.hgt_in_dim)
        self.protein_norm = nn.LayerNorm(args.hgt_in_dim)
        self.input_dropout = nn.Dropout(args.dropout)
        self.hgt_drug_out = nn.Linear(args.hgt_in_dim, args.gt_out_dim)
        self.hgt_disease_out = nn.Linear(args.hgt_in_dim, args.gt_out_dim)
        self.gt_drug = gt_net_drug.GraphTransformer(self.runtime_device, args.gt_layer, args.drug_number, args.gt_out_dim, args.gt_out_dim,
                                                    args.gt_head, args.dropout)
        self.gt_disease = gt_net_disease.GraphTransformer(self.runtime_device, args.gt_layer, args.disease_number, args.gt_out_dim,
                                                    args.gt_out_dim, args.gt_head, args.dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=args.gt_out_dim, nhead=args.tr_head, dropout=args.dropout, batch_first=True)
        self.drug_trans = nn.TransformerEncoder(encoder_layer, num_layers=args.tr_layer)
        self.disease_trans = nn.TransformerEncoder(encoder_layer, num_layers=args.tr_layer)

        self.drug_tr = nn.Transformer(d_model=args.gt_out_dim, nhead=args.tr_head, num_encoder_layers=3, num_decoder_layers=3, batch_first=True)
        self.disease_tr = nn.Transformer(d_model=args.gt_out_dim, nhead=args.tr_head, num_encoder_layers=3, num_decoder_layers=3, batch_first=True)
        self.fusion_gate_drug = nn.Sequential(nn.Linear(args.gt_out_dim * 2, args.gt_out_dim), nn.Sigmoid())
        self.fusion_gate_disease = nn.Sequential(nn.Linear(args.gt_out_dim * 2, args.gt_out_dim), nn.Sigmoid())

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

        fused_dim = args.gt_out_dim * 2
        pair_dim = fused_dim * 5
        self.fused_norm_dr = nn.LayerNorm(fused_dim)
        self.fused_norm_di = nn.LayerNorm(fused_dim)
        self.pair_norm = nn.LayerNorm(pair_dim)
        self.residual_mlp = nn.Sequential(
            nn.Linear(pair_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )
        self.residual_skip = nn.Linear(pair_dim, 2)


    def forward(self, drdr_graph, didi_graph, drdipr_graph, drug_feature, disease_feature, protein_feature, sample):
        dr_sim = self.gt_drug(drdr_graph)
        di_sim = self.gt_disease(didi_graph)

        drug_feature = self.input_dropout(self.drug_norm(self.drug_linear(drug_feature)))
        disease_feature = self.input_dropout(self.disease_norm(self.disease_linear(disease_feature)))
        protein_feature = self.input_dropout(self.protein_norm(self.protein_linear(protein_feature)))

        feature_dict = {
            'drug': drug_feature,
            'disease': disease_feature,
            'protein': protein_feature
        }

        if drdipr_graph.device != drug_feature.device:
            drdipr_graph = drdipr_graph.to(drug_feature.device)
        hgt_out = self.hgt(drdipr_graph, feature_dict)

        dr_hgt = self.hgt_drug_out(hgt_out['drug'])
        di_hgt = self.hgt_disease_out(hgt_out['disease'])

        dr_gate = self.fusion_gate_drug(torch.cat([dr_sim, dr_hgt], dim=-1))
        di_gate = self.fusion_gate_disease(torch.cat([di_sim, di_hgt], dim=-1))
        dr = dr_gate * dr_sim + (1 - dr_gate) * dr_hgt
        di = di_gate * di_sim + (1 - di_gate) * di_hgt

        # Multi-view structural refinement: combine pairwise graph view,
        # HGT view and absolute difference view before the scorer.
        dr = self.fused_norm_dr(torch.cat([dr, dr_hgt], dim=-1))
        di = self.fused_norm_di(torch.cat([di, di_hgt], dim=-1))
        pair_mul = dr[sample[:, 0]] * di[sample[:, 1]]
        pair_diff = torch.abs(dr[sample[:, 0]] - di[sample[:, 1]])
        pair_sum = dr[sample[:, 0]] + di[sample[:, 1]]
        pair_stack = torch.cat([pair_mul, pair_diff, pair_sum, dr[sample[:, 0]], di[sample[:, 1]]], dim=-1)
        drdi_embedding = self.pair_norm(pair_stack)

        output = self.residual_mlp(drdi_embedding) + self.residual_skip(drdi_embedding)

        return dr, output

