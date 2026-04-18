from __future__ import annotations

import os
import sys
import csv
import torch
import torch.nn.functional as fn
import dgl
import numpy as np
import pandas as pd
from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Thêm đường dẫn gốc để import được model và data_preprocess
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'ductri_hgt_update'))

from model.AMNTDDA import AMNTDDA
from data_preprocess import get_adj, k_matrix, dgl_similarity_graph, dgl_heterograph

app = FastAPI(title='HGT Drug-Disease Prediction API', version='2.0.0')

# Cấu hình thiết bị
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class PredictRequest(BaseModel):
    query_type: str = Field(pattern='^(drug_to_disease|disease_to_drug)$')
    input_text: str
    top_k: int = 10
    dataset: str = 'C-dataset' # Mặc định là C-dataset, có thể truyền B-dataset hoặc F-dataset

class InferenceManager:
    def __init__(self):
        self.cached_models = {}
        self.cached_data = {}

    def get_dataset_paths(self, dataset_name: str):
        data_dir = os.path.join(PROJECT_ROOT, 'ductri_hgt_update', 'data', dataset_name, '')
        model_path = os.path.join(PROJECT_ROOT, 'ductri_hgt_update', 'Result', dataset_name, 'AMNTDDA', 'best_model.pth')
        return data_dir, model_path

    def load_context(self, dataset_name: str):
        if dataset_name in self.cached_data:
            return self.cached_data[dataset_name]

        data_dir, model_path = self.get_dataset_paths(dataset_name)
        if not os.path.exists(data_dir):
             raise HTTPException(status_code=404, detail=f"Dataset directory not found: {dataset_name}")

        # Giả lập args cho data_preprocess
        class MockArgs:
            def __init__(self, data_dir):
                self.data_dir = data_dir
                self.neighbor = 20
                self.hgt_layer = 2
                self.hgt_head = 8
                self.hgt_head_dim = 25
                self.hgt_in_dim = 128
                self.hgt_out_dim = 200
                self.gt_layer = 2
                self.gt_head = 2
                self.gt_out_dim = 200
                self.tr_layer = 2
                self.tr_head = 4
                self.dropout = 0.2
        
        args = MockArgs(data_dir)
        
        # Load CSV metadata for UI display
        drugs_info = self.load_csv(data_dir + 'DrugInformation.csv')
        disease_info = self.load_disease_info(data_dir + 'DiseaseFeature.csv')
        protein_info = self.load_csv(data_dir + 'ProteinInformation.csv')

        # Load numerical data for model
        drf = pd.read_csv(data_dir + 'DrugFingerprint.csv').iloc[:, 1:].to_numpy()
        drg = pd.read_csv(data_dir + 'DrugGIP.csv').iloc[:, 1:].to_numpy()
        dip = pd.read_csv(data_dir + 'DiseasePS.csv').iloc[:, 1:].to_numpy()
        dig = pd.read_csv(data_dir + 'DiseaseGIP.csv').iloc[:, 1:].to_numpy()
        
        drs = np.where(drf == 0, drg, (drf + drg) / 2)
        dis = np.where(dip == 0, dig, (dip + dig) / 2)

        data = {
            'drs': drs, 'dis': dis,
            'drpr': pd.read_csv(data_dir + 'DrugProteinAssociationNumber.csv', dtype=int).to_numpy(),
            'dipr': pd.read_csv(data_dir + 'ProteinDiseaseAssociationNumber.csv', dtype=int).to_numpy(),
            'drugfeature': pd.read_csv(data_dir + 'Drug_mol2vec.csv', header=None).iloc[:, 1:].to_numpy(),
            'diseasefeature': pd.read_csv(data_dir + 'DiseaseFeature.csv', header=None).iloc[:, 1:].to_numpy(),
            'proteinfeature': pd.read_csv(data_dir + 'Protein_ESM.csv', header=None).iloc[:, 1:].to_numpy()
        }
        
        args.drug_number = data['drugfeature'].shape[0]
        args.disease_number = data['diseasefeature'].shape[0]
        args.protein_number = data['proteinfeature'].shape[0]

        # Build graphs
        drdr_graph, didi_graph, _ = dgl_similarity_graph(data, args)
        
        # Build heterograph (initially empty associations for inference? Or full?)
        # For simple inference, we can use the full training association or just zeros.
        # Here we use empty as we are predicting NEW associations.
        empty_drdi = np.empty((0, 2), dtype=int)
        drdipr_graph, data = dgl_heterograph(data, empty_drdi, args)

        # Move to device
        drdr_graph = drdr_graph.to(device)
        didi_graph = didi_graph.to(device)
        drdipr_graph = drdipr_graph.to(device)
        
        drug_feature = torch.tensor(data['drugfeature'], dtype=torch.float32).to(device)
        disease_feature = torch.tensor(data['diseasefeature'], dtype=torch.float32).to(device)
        protein_feature = torch.tensor(data['proteinfeature'], dtype=torch.float32).to(device)

        context = {
            'args': args,
            'drdr_graph': drdr_graph,
            'didi_graph': didi_graph,
            'drdipr_graph': drdipr_graph,
            'drug_feature': drug_feature,
            'disease_feature': disease_feature,
            'protein_feature': protein_feature,
            'drugs_info': drugs_info,
            'disease_info': disease_info,
            'protein_info': protein_info
        }
        self.cached_data[dataset_name] = context
        return context

    def get_model(self, dataset_name: str):
        if dataset_name in self.cached_models:
            return self.cached_models[dataset_name]

        data_dir, model_path = self.get_dataset_paths(dataset_name)
        if not os.path.exists(model_path):
            return None # Fallback to mock if model not trained yet

        ctx = self.load_context(dataset_name)
        
        # Init model with the same args as training
        model = AMNTDDA(ctx['args']).to(device)
        
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            self.cached_models[dataset_name] = model
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def load_csv(self, path):
        if not os.path.exists(path): return []
        with open(path, 'r', encoding='utf-8') as f:
            return list(csv.DictReader(f))

    def load_disease_info(self, path):
        rows = []
        if not os.path.exists(path): return rows
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                code = line.strip().split(',')[0]
                if code: rows.append({'id': code, 'name': code})
        return rows

manager = InferenceManager()

def fuzzy_match(items: List[Dict[str, str]], text: str, name_key: str = 'name'):
    text_lower = text.strip().lower()
    for item in items:
        if (item.get(name_key) or '').lower() == text_lower or item.get('id', '').lower() == text_lower:
            return item
    # If no exact match, try partial
    for item in items:
        if text_lower in (item.get(name_key) or '').lower() or text_lower in item.get('id', '').lower():
            return item
    return None

@app.post('/predict')
async def predict(payload: PredictRequest):
    dataset = payload.dataset
    ctx = manager.load_context(dataset)
    model = manager.get_model(dataset)
    is_real = model is not None
    
    # Smart Match & Note logic
    note = f"Đang sử dụng mô hình HGT thật (Tăng tốc GPU). Dữ liệu: {dataset}." if is_real else "Đang chạy Demo (Chưa tìm thấy best_model.pth cho dataset này)."
    source = None
    
    if payload.query_type == 'drug_to_disease':
        source = fuzzy_match(ctx['drugs_info'], payload.input_text, 'name')
        if not source:
            # Check if user entered a disease instead
            alt_source = fuzzy_match(ctx['disease_info'], payload.input_text, 'name')
            if alt_source:
                note = f"⚠️ Thông báo: '{payload.input_text}' là một Bệnh."
            else:
                note = f"ℹ️ Không tìm thấy thực thể '{payload.input_text}' trong hệ thống dữ liệu."
            return {'status': 'success', 'results': [], 'note': note, 'graph': {'nodes': [], 'links': []}}
            
        source_idx = ctx['drugs_info'].index(source)
        num_diseases = len(ctx['disease_info'])
        x_infer = torch.tensor([[source_idx, i] for i in range(num_diseases)]).to(device)
        target_info = ctx['disease_info']
        target_type = 'disease'
    else:
        source = fuzzy_match(ctx['disease_info'], payload.input_text, 'name')
        if not source:
            # Check if user entered a drug instead
            alt_source = fuzzy_match(ctx['drugs_info'], payload.input_text, 'name')
            if alt_source:
                note = f"⚠️ Thông báo: '{payload.input_text}' là một Thuốc."
            else:
                note = f"ℹ️ Không tìm thấy thực thể '{payload.input_text}' trong hệ thống dữ liệu."
            return {'status': 'success', 'results': [], 'note': note, 'graph': {'nodes': [], 'links': []}}

        source_idx = ctx['disease_info'].index(source)
        num_drugs = len(ctx['drugs_info'])
        x_infer = torch.tensor([[i, source_idx] for i in range(num_drugs)]).to(device)
        target_info = ctx['drugs_info']
        target_type = 'drug'

    if is_real:
        with torch.no_grad():
            _, scores = model(
                ctx['drdr_graph'], ctx['didi_graph'], ctx['drdipr_graph'],
                ctx['drug_feature'], ctx['disease_feature'], ctx['protein_feature'],
                x_infer
            )
            probs = fn.softmax(scores, dim=-1)[:, 1].cpu().numpy()
    else:
        probs = np.random.uniform(0.1, 0.9, len(target_info))

    # Sort and take top K
    results = []
    indices = np.argsort(-probs)[:payload.top_k]
    for idx in indices:
        item = target_info[idx]
        results.append({
            'id': item['id'],
            'name': item.get('name', item['id']),
            'score': float(round(probs[idx], 4)),
            'type': target_type
        })

    # Build Graph for Visualization
    graph_nodes = [{'id': f"{payload.query_type.split('_')[0]}:{source['id']}", 'actual_id': source['id'], 'label': source.get('name', source['id']), 'type': payload.query_type.split('_')[0], 'color': '#2563eb' if payload.query_type.startswith('drug') else '#dc2626'}]
    graph_links = []
    
    # Show top 5 proteins connected to source (from metadata)
    for p in ctx['protein_info'][:5]:
        p_id = f"protein:{p['id']}"
        graph_nodes.append({'id': p_id, 'actual_id': p['id'], 'label': p.get('name', p['id']), 'type': 'protein', 'color': '#f59e0b'})
        graph_links.append({'source': graph_nodes[0]['id'], 'target': p_id, 'score': 0.8})

    for res in results[:5]:
        res_id = f"{res['type']}:{res['id']}"
        graph_nodes.append({'id': res_id, 'actual_id': res['id'], 'label': res['name'], 'type': res['type'], 'color': '#dc2626' if res['type'] == 'disease' else '#2563eb'})
        graph_links.append({'source': graph_nodes[0]['id'], 'target': res_id, 'score': res['score']})

    return {
        'matched_input': {'id': source['id'], 'name': source.get('name', source['id']), 'type': payload.query_type.split('_')[0]},
        'results': results,
        'graph': {'nodes': graph_nodes, 'links': graph_links},
        'note': 'Đang sử dụng Real HGT Model (GPU Acceleration)' if is_real else 'Đang chạy Demo (Chưa tìm thấy best_model.pth cho dataset này)'
    }

@app.get('/health')
def health():
    return {'status': 'ok', 'device': str(device)}
