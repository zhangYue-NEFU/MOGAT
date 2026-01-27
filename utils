import os
import glob
import random
import warnings
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from lifelines import CoxPHFitter
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import to_undirected
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GlobalAttention
from lifelines.utils import concordance_index as _cindex
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-whitegrid')
BASE_DIR = r"/Volumes/SAMSUNG256/ComputeRisk/Cancers"  # 请确保路径正确
RESULT_ROOT_DIR = r"modelResult_PanCancer_Final"
STRING_FILE = os.path.join(BASE_DIR, "string_interactions.tsv")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_FOLDS = 5
SEED = 2025

TOP_K_OMICS = 1000  # 最终保留特征数
BATCH_SIZE = 32
EPOCHS_TRAIN = 100
LR = 2e-4
DROPOUT = 0.5

# 搜索空间
SEARCH_SPACE = {
    'AE_DIM': [64,128,256,512,1024],
    'GNN_DIM': [64,128,256,512,1024],
    'GNN_LAYERS': [2,3]
}

# ========================= 2. 先验知识库 =========================

SYMBOL_TO_ENSG = {
    'TP53': 'ENSG00000141510', 'APC': 'ENSG00000134982', 'VHL': 'ENSG00000134086', 'WT1': 'ENSG00000184937',
    'RB1': 'ENSG00000139687', 'CDKN2A': 'ENSG00000147889', 'PTEN': 'ENSG00000171862', 'NF1': 'ENSG00000196712',
    'NF2': 'ENSG00000186575', 'TSC1': 'ENSG00000165699', 'TSC2': 'ENSG00000103197', 'STK11': 'ENSG00000038427',
    'KEAP1': 'ENSG00000079999', 'SMAD4': 'ENSG00000141646', 'FBXW7': 'ENSG00000109670', 'BAP1': 'ENSG00000163930',
    'PBRM1': 'ENSG00000163939', 'SETD2': 'ENSG00000181555', 'ARID1A': 'ENSG00000117713', 'GATA3': 'ENSG00000107485',
    'CDH1': 'ENSG00000039068', 'RUNX1': 'ENSG00000159216', 'NFE2L2': 'ENSG00000116044', 'NOTCH1': 'ENSG00000148400',
    'KMT2C': 'ENSG00000055609', 'KMT2D': 'ENSG00000167548', 'ATRX': 'ENSG00000085224', 'CIC': 'ENSG00000079432',
    'KRAS': 'ENSG00000133703', 'NRAS': 'ENSG00000213281', 'HRAS': 'ENSG00000174775', 'BRAF': 'ENSG00000157764',
    'RAF1': 'ENSG00000132155', 'MAP2K1': 'ENSG00000169032', 'MAPK1': 'ENSG00000100030', 'PIK3CA': 'ENSG00000121879',
    'PIK3CB': 'ENSG00000051382', 'PIK3R1': 'ENSG00000145675', 'AKT1': 'ENSG00000142208', 'AKT2': 'ENSG00000105221',
    'MTOR': 'ENSG00000198793', 'EGFR': 'ENSG00000146648', 'ERBB2': 'ENSG00000141736', 'ERBB3': 'ENSG00000065361',
    'MET': 'ENSG00000105976', 'ALK': 'ENSG00000171094', 'ROS1': 'ENSG00000047936', 'RET': 'ENSG00000165731',
    'KIT': 'ENSG00000157404', 'PDGFRA': 'ENSG00000134853', 'FLT3': 'ENSG00000122025', 'FGFR3': 'ENSG00000068078',
    'CD274': 'ENSG00000120217', 'PDCD1': 'ENSG00000188389', 'CTLA4': 'ENSG00000163599', 'LAG3': 'ENSG00000089692',
    'HAVCR2': 'ENSG00000135077', 'TIGIT': 'ENSG00000181847', 'CD8A': 'ENSG00000153563', 'GZMB': 'ENSG00000100453',
    'PRF1': 'ENSG00000180644', 'IFNG': 'ENSG00000111537', 'TNF': 'ENSG00000232810', 'IL6': 'ENSG00000136244',
    'CXCL8': 'ENSG00000169429', 'IL1B': 'ENSG00000125538', 'TGFB1': 'ENSG00000105329', 'IDO1': 'ENSG00000131203',
    'FOXP3': 'ENSG00000049768', 'HK2': 'ENSG00000159399', 'PKM': 'ENSG00000067225', 'LDHA': 'ENSG00000134333',
    'GAPDH': 'ENSG00000111640', 'SLC2A1': 'ENSG00000117394', 'HIF1A': 'ENSG00000100644', 'VEGFA': 'ENSG00000112715',
    'FASN': 'ENSG00000169710', 'VIM': 'ENSG00000026025', 'FN1': 'ENSG00000115414', 'CDH2': 'ENSG00000170558',
    'ZEB1': 'ENSG00000148516', 'TWIST1': 'ENSG00000122691', 'SNAI1': 'ENSG00000124216', 'SNAI2': 'ENSG00000019991',
    'MMP9': 'ENSG00000100985', 'MMP2': 'ENSG00000087245', 'BRCA1': 'ENSG00000012048', 'BRCA2': 'ENSG00000139618',
    'ATM': 'ENSG00000149311', 'ATR': 'ENSG00000171617', 'PARP1': 'ENSG00000143799', 'MLH1': 'ENSG00000076242',
    'MSH2': 'ENSG00000095002', 'MSH6': 'ENSG00000116062', 'RAD51': 'ENSG00000051180', 'MKI67': 'ENSG00000148773',
    'PCNA': 'ENSG00000132646', 'MYC': 'ENSG00000136997', 'SOX2': 'ENSG00000181449', 'NANOG': 'ENSG00000111704',
    'POU5F1': 'ENSG00000204531', 'CCND1': 'ENSG00000110092', 'CCNE1': 'ENSG00000105173', 'BCL2': 'ENSG00000171791',
    'BAX': 'ENSG00000087088', 'CASP3': 'ENSG00000164305', 'TERT': 'ENSG00000164362', 'ESR1': 'ENSG00000091831',
    'PGR': 'ENSG00000082175', 'IDH2': 'ENSG00000182054', 'AR': 'ENSG00000169083', 'ERG': 'ENSG00000157554',
    'NKX2-1': 'ENSG00000136352', 'TP63': 'ENSG00000073282', 'KLK3': 'ENSG00000142515'
}

SUBTYPE_MARKERS = {
    'BRCA': ['MKI67', 'CCNB1', 'BIRC5', 'ESR1', 'PGR', 'BCL2', 'FOXA1', 'GATA3', 'ERBB2', 'GRB7', 'EGFR', 'KRT5',
             'KRT14', 'MMP11'],
    'UCEC': ['PTEN', 'PIK3CA', 'KRAS', 'CTNNB1', 'TP53', 'POLE', 'MSH2', 'MLH1', 'CCNE1'],
    'CESC': ['CDKN2A', 'TP63', 'KRT5', 'VEGFA', 'ERBB2', 'PIK3CA'],
    'OV': ['TP53', 'BRCA1', 'BRCA2', 'PAX8', 'WT1', 'MKI67', 'CCNE1', 'CDH1'],
    'COAD': ['MLH1', 'MSH2', 'MSH6', 'APC', 'BRAF', 'KRAS', 'CDX2', 'VIM', 'TGFB1'],
    'STAD': ['MLH1', 'MSH2', 'CDH1', 'ERBB2', 'VEGFA', 'TP53', 'KRAS'],
    'ESCA': ['TP63', 'SOX2', 'KRT5', 'CCND1', 'ERBB2', 'VEGFA', 'GATA4'],
    'PAAD': ['KRAS', 'GATA6', 'FOXA1', 'KRT81', 'KRT17', 'TP53', 'SMAD4'],
    'LIHC': ['AFP', 'GPC3', 'EPCAM', 'KRT19', 'CTNNB1', 'TP53', 'AXIN1'],
    'LUAD': ['NKX2-1', 'NAPSA', 'KRT7', 'KRAS', 'EGFR', 'ALK', 'ROS1', 'STK11'],
    'LUSC': ['TP63', 'KRT5', 'KRT6A', 'SOX2', 'NFE2L2', 'PIK3CA', 'CDKN2A'],
    'HNSC': ['CDKN2A', 'TP53', 'CCND1', 'EGFR', 'PIK3CA', 'FAT1', 'NOTCH1', 'KRT5'],
    'LGG': ['IDH1', 'IDH2', 'TP53', 'ATRX', 'CIC', 'FUBP1', 'EGFR', 'PTEN'],
    'SKCM': ['BRAF', 'NRAS', 'MITF', 'AXL', 'CD274', 'CTLA4', 'TYR', 'MLANA'],
    'BLCA': ['KRT5', 'KRT14', 'GATA3', 'FOXA1', 'UPK1A', 'UPK2', 'PPARG', 'EGFR'],
    'KIRP': ['MET', 'HNF1B', 'UMOD', 'CDH1', 'SETD2', 'NFE2L2'],
    'PRAD': ['AR', 'KLK3', 'ERG', 'ETV1', 'PTEN', 'TP53', 'SPOP', 'FOXA1'],
    'THCA': ['BRAF', 'HRAS', 'NRAS', 'KRAS', 'RET', 'PAX8', 'NKX2-1', 'TG'],
    'GENERAL': ['CD8A', 'CD4', 'FOXP3', 'CD274', 'PDCD1', 'MKI67', 'PCNA', 'CDH1', 'VIM', 'CD44']
}
def set_seed(seed):
    """设置所有随机种子以保证复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def find_file(folder_path, suffix_pattern):
    search_path = os.path.join(folder_path, suffix_pattern)
    files = sorted(glob.glob(search_path))
    if not files:
        files = sorted(glob.glob(search_path + ".gz"))
    if not files and "Cancer" in suffix_pattern:
        simple_pattern = suffix_pattern.replace("Cancer", "").replace("cancer", "")
        search_path = os.path.join(folder_path, simple_pattern)
        files = sorted(glob.glob(search_path))
        if not files:
            files = sorted(glob.glob(search_path + ".gz"))
    if files: return files[0]
    return None


def clean_gene_name(name):
    name = str(name).upper().strip()
    if '|' in name: name = name.split('|')[0]
    if '.' in name and name.startswith('ENS'): name = name.split('.')[0]
    return name
def get_prior_knowledge_gene_list(use_ensembl=False):
    if use_ensembl:
        return sorted(list(set(SYMBOL_TO_ENSG.values())))
    else:
        return sorted(list(set(SYMBOL_TO_ENSG.keys())))
def load_single_cancer(cancer_type):
    print(f"   Loading {cancer_type}...", end="", flush=True)
    cancer_dir = os.path.join(BASE_DIR, cancer_type)
    if not os.path.isdir(cancer_dir): return None

    path_mi = find_file(cancer_dir, f"*{cancer_type}*mirna.tsv*")
    path_pr = find_file(cancer_dir, f"*{cancer_type}*protein.tsv*")
    path_mr = find_file(cancer_dir, f"*{cancer_type}*star_tpm*")
    path_su = find_file(cancer_dir, f"*{cancer_type}*survival.tsv*")

    if not (path_mi and path_pr and path_mr and path_su): print(" [Missing Files]"); return None

    try:
        cleaned = {}
        df_mi = pd.read_csv(path_mi, sep='\t', index_col=0, engine='c').T
        df_pr = pd.read_csv(path_pr, sep='\t', index_col=0, engine='c').T
        df_mr = pd.read_csv(path_mr, sep='\t', index_col=0, engine='c').T
        df_sur = pd.read_csv(path_su, sep='\t', index_col=0, engine='c')

        df_mr.columns = [clean_gene_name(c) for c in df_mr.columns]

        for k, df in {'mi': df_mi, 'pr': df_pr, 'mr': df_mr}.items():
            df.index = df.index.astype(str).str.strip()
            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.groupby(level=0, axis=1).mean()
            cleaned[k] = df

        df_sur.index = df_sur.index.astype(str).str.strip()
        df_sur = df_sur.loc[:, ~df_sur.columns.duplicated()]
        if 'OS' not in df_sur.columns: return None
        cleaned['sur'] = df_sur[['OS', 'OS.time']]

        common = sorted(list(set(cleaned['mi'].index) & set(cleaned['pr'].index) &
                             set(cleaned['mr'].index) & set(cleaned['sur'].index)))

        if len(common) < 30: return None
        print(f" [OK, N={len(common)}]")
        return {'mi': cleaned['mi'].loc[common], 'pr': cleaned['pr'].loc[common],
                'mr': cleaned['mr'].loc[common], 'sur': cleaned['sur'].loc[common], 'type': cancer_type}
    except Exception as e:
        print(f" [Error: {e}]");
        return None
def load_and_align_all_datasets():
    print("\n[Step 1] Loading all cancer datasets (Raw Data)...")
    cancers = sorted([d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))])

    raw_data_list = []
    common_genes = {'mi': None, 'pr': None, 'mr': None}

    # --- 第一部分：加载数据并找交集基因 (保持不变) ---
    for c in cancers:
        res = load_single_cancer(c)
        if res:
            raw_data_list.append(res)
            for mod in ['mi', 'pr', 'mr']:
                curr = set(res[mod].columns)
                if common_genes[mod] is None:
                    common_genes[mod] = curr
                else:
                    common_genes[mod] = common_genes[mod] & curr

    if not raw_data_list: return None

    # --- 第二部分：确定先验基因列表 (保持不变) ---
    print(f"\n[Step 2] Identifying Prior Knowledge Genes (Tier 1)...")
    all_marker_genes = set()
    for c_type, markers in SUBTYPE_MARKERS.items():
        all_marker_genes.update(markers)

    sample_gene = list(common_genes['mr'])[0]
    is_ensembl = sample_gene.startswith('ENS')
    prior_genes = get_prior_knowledge_gene_list(use_ensembl=is_ensembl)

    marker_genes_list = []
    if is_ensembl:
        for m in all_marker_genes:
            if m in SYMBOL_TO_ENSG: marker_genes_list.append(SYMBOL_TO_ENSG[m])
    else:
        marker_genes_list = list(all_marker_genes)

    must_keep_genes = set(prior_genes) | set(marker_genes_list)
    all_common_mr = sorted(list(common_genes['mr']))
    matched_must_keep = sorted(list(must_keep_genes & set(all_common_mr)))

    print(f"  -> Found {len(matched_must_keep)} Tier 1 genes (will be forced kept).")
    print(f"  -> Total available mRNA candidates: {len(all_common_mr)}")

    # --- 第三部分：数据对齐 (此处有修改) ---
    common_genes['mr'] = all_common_mr
    common_genes['pr'] = sorted(list(common_genes['pr']))
    common_genes['mi'] = sorted(list(common_genes['mi']))

    print("\n[Step 3] Stacking Raw Data (NaNs are preserved for fold-internal handling)...")
    X_list = {'mi': [], 'pr': [], 'mr': []}
    y_list = {'time': [], 'event': []}
    cancer_labels = []

    # ✅ 修改 1: 初始化一个列表用来存 ID
    sample_ids = []

    for item in raw_data_list:
        n = len(item['sur'])
        cancer_labels.extend([item['type']] * n)
        y_list['time'].extend(item['sur']['OS.time'].values)
        y_list['event'].extend(item['sur']['OS'].values)

        # ✅ 修改 2: 提取样本 ID (假设 survival data 的 index 是 ID)
        # 这通常是 TCGA-XX-XXXX-01 这种格式
        sample_ids.extend(item['sur'].index.tolist())

        for mod in ['mi', 'pr', 'mr']:
            df_aligned = item[mod].reindex(columns=common_genes[mod])
            data = df_aligned.values

            all_nan_mask = np.isnan(data).all(axis=0)
            if np.any(all_nan_mask): data[:, all_nan_mask] = 0.0

            X_list[mod].append(data)

    try:
        final_data = {mod: np.concatenate(X_list[mod], axis=0) for mod in ['mi', 'pr', 'mr']}
    except ValueError:
        return None

    final_y = {'time': np.array(y_list['time']), 'event': np.array(y_list['event'])}

    # ✅ 修改 3: 在返回值里加上 sample_ids
    return final_data, final_y, np.array(cancer_labels), np.array(sample_ids), common_genes, matched_must_keep
def perform_dynamic_clustering(X_mr, cancer_labels, gene_names):
    print(f"\n[Step 4] Performing Knowledge-Guided Molecular Subtyping (Stratification Only)...")
    # 为了聚类，这里临时做一个简单的插补（不影响后续训练数据）
    imp_temp = SimpleImputer(strategy='mean')
    # 处理大数据集时，为防内存溢出或计算太慢，可先用0填充做聚类
    X_temp = np.nan_to_num(X_mr, nan=0.0)

    unique_cancers = np.unique(cancer_labels)
    subtype_labels = np.array(["Unassigned"] * len(cancer_labels), dtype=object)

    X_log = np.log1p(np.maximum(X_temp, 0))

    for cancer in unique_cancers:
        indices = np.where(cancer_labels == cancer)[0]
        if len(indices) < 30:
            for idx in indices: subtype_labels[idx] = f"{cancer}_0"
            continue

        # 使用 SEED 保证聚类结果一致
        pca = PCA(n_components=min(10, len(indices) // 2), random_state=SEED)
        X_subset = pca.fit_transform(X_log[indices])

        best_k = 2
        best_score = -1
        max_k = min(5, len(indices) // 15)
        if max_k < 2: max_k = 2

        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=SEED, n_init=10)
            labels = kmeans.fit_predict(X_subset)
            score = silhouette_score(X_subset, labels)
            if score > best_score:
                best_score = score
                best_k = k

        final_kmeans = KMeans(n_clusters=best_k, random_state=SEED, n_init=10)
        final_clusters = final_kmeans.fit_predict(X_subset)
        for i, idx in enumerate(indices):
            subtype_labels[idx] = f"{cancer}_{final_clusters[i]}"

    print(f"  ✅ Subtyping Complete for Stratification.")
    return subtype_labels
def select_features_inside_fold(X_train, y_time, y_event, all_genes, tier1_genes, target_num=1000):
    """
    完全基于训练集进行的特征筛选
    """
    # 1. 识别 Tier 1 的索引
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}
    tier1_indices = [gene_to_idx[g] for g in tier1_genes if g in gene_to_idx]

    if len(tier1_indices) >= target_num:
        return sorted(tier1_indices)

    needed = target_num - len(tier1_indices)

    # 2. 识别候选基因
    candidate_indices = [i for i in range(len(all_genes)) if i not in tier1_indices]

    # 3. 快速预筛选 (Variance) - 只用训练集数据
    X_cand = X_train[:, candidate_indices]
    variances = np.var(X_cand, axis=0)
    n_pre_select = min(2000, len(candidate_indices))
    top_var_idx_local = np.argsort(variances)[-n_pre_select:]

    # 映射回原始索引
    candidates_for_cox = [candidate_indices[i] for i in top_var_idx_local]

    # 4. Cox 筛选
    cph = CoxPHFitter(penalizer=0.1)
    scores = []

    X_cox_subset = X_train[:, candidates_for_cox]

    for i in range(len(candidates_for_cox)):
        try:
            df_tmp = pd.DataFrame({
                'T': y_time + 1e-5,
                'E': y_event,
                'G': X_cox_subset[:, i]
            })
            cph.fit(df_tmp, duration_col='T', event_col='E', show_progress=False)
            p_val = cph.summary.loc['G', 'p']
            if p_val < 0.05:
                scores.append((p_val, candidates_for_cox[i]))
        except:
            pass

    # 5. 排序选出最好的 Tier 2
    scores.sort(key=lambda x: x[0])
    best_tier2 = [idx for p, idx in scores[:needed]]

    final_indices = sorted(list(set(tier1_indices + best_tier2)))
    return final_indices
class AutoPower2AE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        hidden_dim = (input_dim + latent_dim) // 2
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.LeakyReLU(0.2), nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.LeakyReLU(0.2), nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z
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
    def __init__(self, input_dims, fusion_dim=64, num_cancers=33):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Sequential(nn.Linear(d, fusion_dim), nn.LayerNorm(fusion_dim), nn.ReLU()) for d in input_dims
        ])
        self.cancer_emb = nn.Embedding(num_cancers, fusion_dim)
        self.tf = nn.TransformerEncoderLayer(d_model=fusion_dim, nhead=4, batch_first=True, dropout=DROPOUT)
        self.cls = nn.Parameter(torch.zeros(1, 1, fusion_dim))
        self.head = nn.Sequential(nn.Linear(fusion_dim, 32), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(32, 1))

    def forward(self, feats, cancer_ids):
        projed = [p(f).unsqueeze(1) for p, f in zip(self.projs, feats)]
        c_feat = self.cancer_emb(cancer_ids).unsqueeze(1)
        x = torch.cat([self.cls.expand(feats[0].size(0), -1, -1), c_feat] + projed, dim=1)
        return self.head(self.tf(x)[:, 0, :])


class PanCancerOriginalNet(nn.Module):
    def __init__(self, in_dims, ae_d, gnn_d, gnn_l, num_nodes, num_cancers):
        super().__init__()
        self.ae_mi = AutoPower2AE(in_dims['mi'], ae_d)
        self.ae_mr = AutoPower2AE(in_dims['mr'], ae_d)
        self.gnn_pr = DynamicGNN(num_nodes, gnn_d, gnn_l)
        self.fuse = ProjectedTransformerFusion([ae_d, gnn_d, ae_d], fusion_dim=128, num_cancers=num_cancers)

    def forward(self, x_mi, x_pr, x_mr, c_idx, ei):
        _, f_mi = self.ae_mi(x_mi)
        f_pr = self.gnn_pr(x_pr, ei)
        _, f_mr = self.ae_mr(x_mr)
        return self.fuse([f_mi, f_pr, f_mr], c_idx)


def cox_loss(risk, t, e):
    idx = torch.argsort(t, descending=True)
    r = risk[idx].reshape(-1)
    e = e[idx]
    return -torch.sum((r - torch.logcumsumexp(r, dim=0)) * e) / (torch.sum(e) + 1e-8)


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0001):
        self.patience = patience;
        self.min_delta = min_delta;
        self.counter = 0;
        self.best_score = -np.inf;
        self.early_stop = False

    def __call__(self, score):
        if score > self.best_score + self.min_delta:
            self.best_score = score; self.counter = 0
        else:
            self.counter += 1;
        if self.counter >= self.patience: self.early_stop = True
def build_ppi_graph(string_file, protein_names):
    print(f"  -> Building PPI Graph for {len(protein_names)} proteins...")
    prot2idx = {p: i for i, p in enumerate(protein_names)}

    # RPPA Map
    rppa_map = {
        '1433BETA': 'YWHAB', '1433EPSILON': 'YWHAE', '1433ZETA': 'YWHAZ', '1433SIGMA': 'SFN',
        '4EBP1': 'EIF4EBP1', '53BP1': 'TP53BP1', 'HER2': 'ERBB2', 'HER3': 'ERBB3',
        'LKB1': 'STK11', 'MEK1': 'MAP2K1', 'P27': 'CDKN1B', 'P21': 'CDKN1A',
        'P53': 'TP53', 'DJ1': 'PARK7', 'CHK1': 'CHEK1', 'CHK2': 'CHEK2',
        'CRAF': 'RAF1', 'BRAF': 'BRAF', 'BIM': 'BCL2L11', 'BAD': 'BAD',
        'BAK': 'BAK1', 'BAX': 'BAX', 'BCL2': 'BCL2', 'BCLXL': 'BCL2L1',
        'CATENINBETA': 'CTNNB1', 'CATENINDELTA1': 'CTNND1', 'COX2': 'PTGS2',
        'CYCLINB1': 'CCNB1', 'CYCLIND1': 'CCND1', 'CYCLINE1': 'CCNE1',
        'EGFR': 'EGFR', 'ERALPHA': 'ESR1', 'FOXH1': 'FOXH1', 'FOXM1': 'FOXM1',
        'GATA3': 'GATA3', 'GSK3ALPHA': 'GSK3A', 'GSK3BETA': 'GSK3B',
        'IGFBP2': 'IGFBP2', 'INPP4B': 'INPP4B', 'IRS1': 'IRS1', 'JNK2': 'MAPK9',
        'LCK': 'LCK', 'MAPK': 'MAPK1', 'MTOR': 'MTOR', 'MYH11': 'MYH11',
        'NHERF1': 'SLC9A3R1', 'NRF2': 'NFE2L2', 'P38': 'MAPK14', 'PAI1': 'SERPINE1',
        'PCNA': 'PCNA', 'PDK1': 'PDK1', 'PEA15': 'PEA15', 'PKCALPHA': 'PRKCA',
        'PKCDELTA': 'PRKCD', 'PRAS40': 'AKT1S1', 'PREX1': 'PREX1', 'PTEN': 'PTEN',
        'RAD50': 'RAD50', 'RAD51': 'RAD51', 'RAPTOR': 'RPTOR', 'RICTOR': 'RICTOR',
        'S6': 'RPS6', 'SRC': 'SRC', 'STAT3': 'STAT3', 'STAT5ALPHA': 'STAT5A',
        'TAZ': 'WWTR1', 'TFRC': 'TFRC', 'TRANSGLUTAMINASE': 'TGM2', 'TSC1': 'TSC1',
        'TSC2': 'TSC2', 'VHL': 'VHL', 'XBP1': 'XBP1', 'YAP': 'YAP1', 'YB1': 'YBX1'
    }

    simple_map = {}
    for p in protein_names:
        clean_name = p.split('_')[0].upper()
        simple_map[clean_name] = p
        if clean_name in rppa_map:
            std_name = rppa_map[clean_name]
            simple_map[std_name] = p

    try:
        df = pd.read_csv(string_file, sep='\t')
        df.columns = [c.replace('#', '').strip() for c in df.columns]
        if 'node1' in df.columns:
            c1, c2 = 'node1', 'node2'
        elif 'protein1' in df.columns:
            c1, c2 = 'protein1', 'protein2'
        else:
            c1, c2 = df.columns[0], df.columns[1]

        score_col = next((c for c in df.columns if 'score' in c.lower()), None)
        if score_col:
            df[score_col] = pd.to_numeric(df[score_col], errors='coerce')
            df = df[df[score_col] >= 0.95]

        edges = []
        for _, r in df.iterrows():
            n1 = str(r[c1]).upper().replace('9606.', '')
            n2 = str(r[c2]).upper().replace('9606.', '')
            idx1 = idx2 = None
            if n1 in prot2idx:
                idx1 = prot2idx[n1]
            elif n1 in simple_map:
                idx1 = prot2idx[simple_map[n1]]
            if n2 in prot2idx:
                idx2 = prot2idx[n2]
            elif n2 in simple_map:
                idx2 = prot2idx[simple_map[n2]]
            if idx1 is not None and idx2 is not None:
                edges.append([idx1, idx2])

        if not edges: return torch.tensor([[], []], dtype=torch.long)
        return to_undirected(torch.tensor(edges, dtype=torch.long).t().contiguous())
    except:
        return torch.tensor([[], []], dtype=torch.long)
