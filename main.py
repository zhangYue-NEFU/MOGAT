#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MOGAT
"""

import os
import glob
import random
import warnings
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# 引入 Scheduler 相关模块
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import to_undirected

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from lifelines.utils import concordance_index as _cindex

# === 环境变量配置 ===
warnings.filterwarnings("ignore")

# ========================= 1. 全局配置 =========================
BASE_DIR = r"/Volumes/SAMSUNG256/ComputeRisk/Cancers"
RESULT_ROOT_DIR = r"modelResult_PanCancer_Advanced"
STRING_FILE = os.path.join(BASE_DIR, "string_interactions.tsv")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_FOLDS = 5
SEED = 2025

# --- ✅ 修改点 1: 参数配置变更 ---
# TOP_K_OMICS 移除全局常量，改为在循环中定义
BATCH_SIZE = 64
EPOCHS_TRAIN = 100
LR = 5e-4  # 激进学习率，配合 Scheduler 使用
DROPOUT = 0.5
WEIGHT_DECAY = 1e-3  # ✅ 增强正则化，从 1e-4 提升到 1e-3

ES_PATIENCE = 20
ES_MIN_DELTA = 1e-4

# --- ✅ 修改点 2: 扩展搜索空间 ---
SEARCH_SPACE = {
    'TOP_K': [1000],  # 新增特征数量搜索
    'AE_DIM': [512],  # 新增 512，移除较弱的 64
    'GNN_DIM': [512],  # 新增 512
    'GNN_LAYERS': [3]
}
# SEARCH_SPACE = {
#     'TOP_K': [1000],  # 新增特征数量搜索
#     'AE_DIM': [512],  # 新增 512，移除较弱的 64
#     'GNN_DIM': [512],  # 新增 512
#     'GNN_LAYERS': [3]
# }

# === 依赖引入 ===
try:
    from utils import (
        load_and_align_all_datasets,
        build_ppi_graph,
        perform_dynamic_clustering,
        select_features_inside_fold
    )
except ImportError:
    raise ImportError("❌ Critical Error: 'allRISK2.py' not found in current directory.")


# ========================= 2. 复现性工具 =========================
# ========================= 强制复现补丁 =========================
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # 您已经有了，保持
os.environ["PYTHONHASHSEED"] = str(SEED)  # 锁死 Python 哈希


# 在 set_seed 函数中加入这一行核武器：
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # 🔥 新增：强制使用确定性算法 (如果有操作不支持会报错，但 GCN/Transformer 通常支持)
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            pass  # 老版本 PyTorch 可能没有这个


def cox_loss(risk, t, e):
    idx = torch.argsort(t, descending=True)
    r = risk[idx].reshape(-1)
    e = e[idx]
    return -torch.sum((r - torch.logcumsumexp(r, dim=0)) * e) / (torch.sum(e) + 1e-8)


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = -np.inf
        self.early_stop = False

    def __call__(self, score, model, path):
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            torch.save(model.state_dict(), path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# ========================= 3. 模型结构 =========================
class AutoPower2AE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        h = (input_dim + latent_dim) // 2
        self.enc = nn.Sequential(
            nn.Linear(input_dim, h),
            nn.BatchNorm1d(h),
            nn.LeakyReLU(0.2),
            nn.Dropout(DROPOUT),
            nn.Linear(h, latent_dim)
        )

    def forward(self, x):
        return self.enc(x)


import torch
import torch.nn as nn
import torch.nn.functional as F
# 引入 GlobalAttention
from torch_geometric.nn import GCNConv, GlobalAttention


class DynamicGNN(nn.Module):
    def __init__(self, n_node, out_dim, n_layers=2):
        super().__init__()
        # 1. 基础 Embedding 保持不变
        self.emb = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, out_dim))

        # 2. GCN 层保持不变
        # self.convs = nn.ModuleList([GCNConv(out_dim, out_dim) for _ in range(n_layers)])
        from torch_geometric.nn import GATConv

        # 在 __init__ 中：
        # heads=4 表示多头注意力，增强鲁棒性
        self.convs = nn.ModuleList([GATConv(out_dim, out_dim, heads=4, concat=False) for _ in range(n_layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(out_dim) for _ in range(n_layers)])

        # 3. 【关键修改】定义 Attention Pooling
        # gate_nn 是一个简单的线性层，用于计算每个节点的得分 (Score)
        self.pool = GlobalAttention(gate_nn=nn.Linear(out_dim, 1))

        # 4. Skip Connection (原有逻辑保留)
        self.skip_linear = nn.Linear(n_node, out_dim)
        self.skip_act = nn.ReLU()

    def forward(self, x, ei):
        B, N = x.shape
        # Input Embedding
        h = self.emb(x.view(-1, 1))

        # 处理边索引 (Batch 处理)
        if ei.size(1) > 0:
            off = torch.arange(B, device=x.device).repeat_interleave(ei.size(1)) * N
            ei_b = ei.repeat(1, B) + off

            # GCN 消息传递
            for cv, bn in zip(self.convs, self.bns):
                h = torch.relu(bn(cv(h, ei_b)))

        # 生成 Batch Index [0,0...0, 1,1...1, ...]
        batch_idx = torch.arange(B, device=x.device).repeat_interleave(N)

        # 【关键修改】使用 Attention Pooling
        # out_gnn 会自动对 batch 内的节点进行加权求和
        out_gnn = self.pool(h, batch_idx)

        # Skip Connection
        out_skip = self.skip_act(self.skip_linear(x))

        return out_gnn + out_skip


class ProjectedTransformerFusion(nn.Module):
    def __init__(self, input_dims, fusion_dim=64, num_cancers=33, use_cancer_emb=True):
        super().__init__()
        self.use_cancer_emb = use_cancer_emb
        self.projs = nn.ModuleList(
            [nn.Sequential(nn.Linear(d, fusion_dim), nn.LayerNorm(fusion_dim), nn.ReLU()) for d in input_dims]
        )
        if use_cancer_emb:
            self.cancer_emb = nn.Embedding(num_cancers, fusion_dim)
        self.tf = nn.TransformerEncoderLayer(d_model=fusion_dim, nhead=4, batch_first=True, dropout=DROPOUT)
        self.cls = nn.Parameter(torch.zeros(1, 1, fusion_dim))
        self.head = nn.Sequential(nn.Linear(fusion_dim, 32), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(32, 1))

    def forward(self, feats, cancer_ids):
        projed = [p(f).unsqueeze(1) for p, f in zip(self.projs, feats)]
        seq = [self.cls.expand(feats[0].size(0), -1, -1)]
        if self.use_cancer_emb:
            seq.append(self.cancer_emb(cancer_ids).unsqueeze(1))
        seq.extend(projed)
        x = torch.cat(seq, dim=1)
        return self.head(self.tf(x)[:, 0, :])


class PanCancerAlignedFullNet(nn.Module):
    def __init__(self, in_dims, ae_d, gnn_d, gnn_l, num_nodes, num_cancers):
        super().__init__()
        self.enc_mi = AutoPower2AE(in_dims['mi'], ae_d)
        self.enc_mr = AutoPower2AE(in_dims['mr'], ae_d)
        self.gnn_pr = DynamicGNN(num_nodes, gnn_d, gnn_l)
        self.fuse = ProjectedTransformerFusion([ae_d, gnn_d, ae_d], fusion_dim=128, num_cancers=num_cancers,
                                               use_cancer_emb=True)

    def forward(self, x_mi, x_pr, x_mr, c_idx, ei):
        f_mi = self.enc_mi(x_mi)
        f_pr = self.gnn_pr(x_pr, ei)
        f_mr = self.enc_mr(x_mr)
        return self.fuse([f_mi, f_pr, f_mr], c_idx)


# ========================= 4. 主程序 =========================
if __name__ == "__main__":
    if os.path.exists(RESULT_ROOT_DIR):
        shutil.rmtree(RESULT_ROOT_DIR)
    os.makedirs(RESULT_ROOT_DIR, exist_ok=True)

    set_seed(SEED)

    # 1) 加载数据
    data_pack = load_and_align_all_datasets()
    if not data_pack:
        raise RuntimeError("❌ load_and_align_all_datasets() returned None.")
    X_dict_raw, Y_dict, cancer_labels, *rest = data_pack
    common_genes_dict = rest[0]
    tier1_genes_list = rest[1]
    all_mr_gene_names = common_genes_dict['mr']

    # 2) PPI
    ei_base = build_ppi_graph(STRING_FILE, common_genes_dict['pr']).to(DEVICE)
    num_prot_nodes = len(common_genes_dict['pr'])

    # 3) Cancer IDs
    le = {c: i for i, c in enumerate(np.unique(cancer_labels))}
    cancer_ids = np.array([le[c] for c in cancer_labels])
    num_cancers = len(le)

    # 4) Subtype labels
    subtype_labels = perform_dynamic_clustering(X_dict_raw['mr'], cancer_labels, common_genes_dict['mr'])

    global_detailed_records = []

    # === ✅ 修改点 1 & 2: 循环 TopK 和 扩展的搜索空间 ===
    for top_k in SEARCH_SPACE['TOP_K']:
        for ae_d in SEARCH_SPACE['AE_DIM']:
            for gnn_d in SEARCH_SPACE['GNN_DIM']:
                for gnn_l in SEARCH_SPACE['GNN_LAYERS']:

                    set_seed(SEED)
                    # Config 名字中加入 TopK 标识
                    p_str = f"K{top_k}_AE{ae_d}_GNN{gnn_d}_L{gnn_l}"
                    print(f"\n{'=' * 20} Config: {p_str} {'=' * 20}")

                    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                    try:
                        split_gen = skf.split(np.zeros(len(subtype_labels)), subtype_labels)
                    except Exception:
                        split_gen = skf.split(np.zeros(len(cancer_labels)), cancer_labels)

                    for fold, (tr_i, te_i) in enumerate(split_gen):
                        set_seed(SEED + fold)
                        print(f"  👉 Fold {fold + 1}/{N_FOLDS} (TopK={top_k})...")

                        tr_sub, val_sub = train_test_split(
                            tr_i, test_size=0.2,
                            stratify=cancer_labels[tr_i],
                            random_state=SEED + fold
                        )

                        indices_map = {'tr': tr_sub, 'val': val_sub, 'te': te_i}
                        data_processed = {'tr': {}, 'val': {}, 'te': {}}
                        unscaled_tr_mr = None

                        # --- Impute + Log ---
                        for m in ['mr', 'mi', 'pr']:
                            imp = SimpleImputer(strategy='mean')
                            imp.fit(X_dict_raw[m][tr_sub])
                            for split in ['tr', 'val', 'te']:
                                imp_data = imp.transform(X_dict_raw[m][indices_map[split]])
                                imp_data = np.nan_to_num(imp_data)
                                if m == 'mr':
                                    imp_data = np.log1p(np.maximum(imp_data, 0))
                                    if split == 'tr':
                                        unscaled_tr_mr = imp_data.copy()
                                data_processed[split][m] = imp_data

                        # --- Scale ---
                        for m in ['mr', 'mi', 'pr']:
                            sc = StandardScaler()
                            sc.fit(data_processed['tr'][m])
                            for split in ['tr', 'val', 'te']:
                                data_processed[split][m] = np.nan_to_num(sc.transform(data_processed[split][m]))

                        # --- Feature Selection (使用当前的 top_k) ---
                        y_tr_sub_t = Y_dict['time'][tr_sub]
                        y_tr_sub_e = Y_dict['event'][tr_sub]

                        selected_mr_indices = select_features_inside_fold(
                            unscaled_tr_mr,
                            y_tr_sub_t, y_tr_sub_e,
                            all_mr_gene_names, tier1_genes_list,
                            target_num=top_k  # ✅ 传入动态 top_k
                        )

                        # --- Pack Tensors ---
                        tensors = {}
                        for split in ['tr', 'val', 'te']:
                            idx = indices_map[split]
                            tensors[split] = {
                                'mr': torch.tensor(data_processed[split]['mr'][:, selected_mr_indices],
                                                   dtype=torch.float32),
                                'mi': torch.tensor(data_processed[split]['mi'], dtype=torch.float32),
                                'pr': torch.tensor(data_processed[split]['pr'], dtype=torch.float32),
                                'c': torch.tensor(cancer_ids[idx], dtype=torch.long),
                                't': torch.tensor(Y_dict['time'][idx], dtype=torch.float32),
                                'e': torch.tensor(Y_dict['event'][idx], dtype=torch.float32),
                                'labels': cancer_labels[idx]
                            }

                        # --- Train Setup ---
                        in_dims = {'mi': tensors['tr']['mi'].shape[1], 'mr': tensors['tr']['mr'].shape[1]}
                        model = PanCancerAlignedFullNet(in_dims, ae_d, gnn_d, gnn_l, num_prot_nodes, num_cancers).to(
                            DEVICE)

                        op = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

                        # === ✅ 修改点 3: 学习率调度器 (Warmup + Cosine) ===
                        # 前 10 epochs 线性增加 LR，之后 Cosine 衰减
                        warmup_epochs = 10
                        scheduler_warmup = LinearLR(op, start_factor=0.01, total_iters=warmup_epochs)
                        scheduler_cosine = CosineAnnealingLR(op, T_max=EPOCHS_TRAIN - warmup_epochs, eta_min=1e-5)
                        scheduler = SequentialLR(op, schedulers=[scheduler_warmup, scheduler_cosine],
                                                 milestones=[warmup_epochs])

                        stopper = EarlyStopping(patience=ES_PATIENCE, min_delta=ES_MIN_DELTA)
                        temp_path = os.path.join(RESULT_ROOT_DIR, f"temp_{p_str}_fold{fold}.pth")

                        dl_tr = DataLoader(
                            TensorDataset(tensors['tr']['mi'], tensors['tr']['pr'], tensors['tr']['mr'],
                                          tensors['tr']['c'],
                                          tensors['tr']['t'], tensors['tr']['e']),
                            batch_size=BATCH_SIZE, shuffle=True, drop_last=True,num_workers=0
                        )

                        # --- Training Loop ---
                        for epoch in range(EPOCHS_TRAIN):
                            model.train()
                            for b_mi, b_pr, b_mr, b_c, b_t, b_e in dl_tr:
                                op.zero_grad()
                                r = model(b_mi.to(DEVICE), b_pr.to(DEVICE), b_mr.to(DEVICE), b_c.to(DEVICE), ei_base)
                                loss = cox_loss(r, b_t.to(DEVICE), b_e.to(DEVICE))
                                loss.backward()

                                # ✅ 梯度裁剪 (Gradient Clipping)，防止 1e-3 LR 导致梯度爆炸
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                                op.step()

                            # ✅ 更新学习率
                            scheduler.step()

                            # Validation
                            model.eval()
                            with torch.no_grad():
                                dl_val = DataLoader(
                                    TensorDataset(tensors['val']['mi'], tensors['val']['pr'], tensors['val']['mr'],
                                                  tensors['val']['c']),
                                    batch_size=BATCH_SIZE, shuffle=False
                                )
                                pred_val = []
                                for b_mi, b_pr, b_mr, b_c in dl_val:
                                    p = model(b_mi.to(DEVICE), b_pr.to(DEVICE), b_mr.to(DEVICE), b_c.to(DEVICE),
                                              ei_base)
                                    pred_val.append(p.cpu().numpy())
                                pred_val = np.concatenate(pred_val).reshape(-1)

                                c_val = 0.5
                                try:
                                    c_val = _cindex(
                                        tensors['val']['t'].cpu().numpy(),
                                        -pred_val,
                                        tensors['val']['e'].cpu().numpy()
                                    )
                                except Exception:
                                    pass

                                stopper(c_val, model, temp_path)

                            if stopper.early_stop:
                                break

                        # --- Test Phase ---
                        if os.path.exists(temp_path):
                            model.load_state_dict(torch.load(temp_path, map_location=DEVICE))

                        model.eval()
                        with torch.no_grad():
                            dl_te = DataLoader(
                                TensorDataset(tensors['te']['mi'], tensors['te']['pr'], tensors['te']['mr'],
                                              tensors['te']['c']),
                                batch_size=BATCH_SIZE, shuffle=False
                            )
                            pred_te = []
                            for b_mi, b_pr, b_mr, b_c in dl_te:
                                p = model(b_mi.to(DEVICE), b_pr.to(DEVICE), b_mr.to(DEVICE), b_c.to(DEVICE), ei_base)
                                pred_te.append(p.cpu().numpy())
                            pred_te = np.concatenate(pred_te).reshape(-1)

                        # Detailed Recording
                        te_labels = tensors['te']['labels']
                        te_time = tensors['te']['t'].cpu().numpy()
                        te_event = tensors['te']['e'].cpu().numpy()

                        for c_type in np.unique(te_labels):
                            idx = (te_labels == c_type)
                            if np.sum(idx) > 10:
                                try:
                                    c = _cindex(te_time[idx], -pred_te[idx], te_event[idx])
                                    global_detailed_records.append({
                                        'Config': p_str,
                                        'Fold': fold + 1,
                                        'Cancer': c_type,
                                        'C-index': float(c)
                                    })
                                except Exception:
                                    pass

                        del model, tensors
                        torch.cuda.empty_cache()
                        if os.path.exists(temp_path):
                            os.remove(temp_path)

                    # Save intermediate results
                    df_tmp = pd.DataFrame(global_detailed_records)
                    df_tmp.to_csv(os.path.join(RESULT_ROOT_DIR, 'MOGAT_Advanced.csv'), index=False)
                    print(f"  💾 Saved progress: {len(df_tmp)} records")

    print("\n" + "=" * 80)
    print("✅ Advanced Optimization Finished!")
    print("=" * 80)
