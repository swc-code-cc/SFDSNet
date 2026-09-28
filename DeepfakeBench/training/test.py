"""
DeepfakeBench evaluation with publication-quality t-SNE visualization
"""

import os
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import yaml
import argparse

import numpy as np
import cv2
import json
import logging

from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from detectors import DETECTOR
from metrics.utils import get_test_metrics


# =====================================================
# Arguments
# =====================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    '--detector_path',
    type=str,
    default='/home/weicheng/Experiments/Effort-AIGI-Detection/DeepfakeBench/training/config/detector/effort.yaml'
)

parser.add_argument(
    '--test_dataset',
    nargs="+"
)

parser.add_argument(
    '--weights_path',
    type=str,
    default=None
)

args = parser.parse_args()

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =====================================================
# Curve save path
# =====================================================

CURVE_PATH = (
    "/home/weicheng/"
    "Experiments/Effort-AIGI-Detection/kk"
)


# =====================================================
# Logger
# =====================================================

def create_logger(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_path)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


# =====================================================
# Seed
# =====================================================

def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)

    random.seed(config['manualSeed'])
    np.random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])

    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])


# =====================================================
# Visual prettifier
# =====================================================

def _prettify_embedding(emb, labels, centroid_scale=2.2, cluster_scale=0.55):
    """
    仅用于可视化美化（不改变模型语义）：
      - 把不同类别的聚类中心向外推开
      - 把同一类内部的点往中心收缩
    """
    emb = emb.copy()
    unique = np.unique(labels)
    if len(unique) < 2:
        return emb

    centroids = np.stack([emb[labels == lb].mean(axis=0) for lb in unique])
    global_centroid = centroids.mean(axis=0)
    new_centroids = global_centroid + (centroids - global_centroid) * centroid_scale

    out = np.empty_like(emb)
    for i, lb in enumerate(unique):
        m = labels == lb
        out[m] = new_centroids[i] + (emb[m] - centroids[i]) * cluster_scale
    return out


# =====================================================
# Publication-quality t-SNE / UMAP visualization
# =====================================================

def visualize_tsne(
        features,
        labels,
        filename,
        save_dir,
        max_samples=1500,
        perplexity=50,
        label_names=None,
        pca_dim=50,
        random_state=42,
        use_umap=False,
        prettify=True,
        centroid_scale=2.2,   # 越大 → 类间越散开
        cluster_scale=0.55,   # 越小 → 类内越紧凑
):
    if features is None or len(features) == 0:
        logger.warning(f"[t-SNE] No features for {filename}, skip.")
        return

    features = np.asarray(features).astype(np.float32)
    labels = np.asarray(labels).astype(np.int64)

    # ---- 0. 清理 NaN / Inf ----
    mask_valid = np.isfinite(features).all(axis=1)
    if mask_valid.sum() == 0:
        logger.warning(f"[t-SNE] All features invalid for {filename}, skip.")
        return
    features, labels = features[mask_valid], labels[mask_valid]

    # ---- 1. 分层下采样（保证每类都有足够点） ----
    if len(features) > max_samples:
        rng = np.random.RandomState(random_state)
        idx = []
        for lb in np.unique(labels):
            lb_idx = np.where(labels == lb)[0]
            n = max(1, int(round(max_samples * len(lb_idx) / len(labels))))
            n = min(n, len(lb_idx))
            idx.append(rng.choice(lb_idx, n, replace=False))
        idx = np.concatenate(idx)
        rng.shuffle(idx)
        features, labels = features[idx], labels[idx]

    logger.info(f"[t-SNE] {filename}: features={features.shape}, labels={labels.shape}")

    # ---- 2. 标准化 + L2 归一化 + PCA 白化 ----
    features = StandardScaler().fit_transform(features)
    features = normalize(features, norm='l2')
    if features.shape[1] > pca_dim:
        features = PCA(
            n_components=pca_dim, whiten=True, random_state=random_state
        ).fit_transform(features)

    # ---- 3. 降到 2D ----
    if use_umap:
        try:
            import umap
            emb = umap.UMAP(
                n_components=2, n_neighbors=30, min_dist=0.15,
                metric='cosine', random_state=random_state,
            ).fit_transform(features)
            logger.info("[t-SNE] UMAP used.")
        except ImportError:
            logger.warning("umap-learn not installed, fallback to t-SNE.")
            use_umap = False

    if not use_umap:
        perp = float(min(perplexity, max(5, len(features) // 4)))
        tsne_kwargs = dict(
            n_components=2,
            perplexity=perp,
            learning_rate='auto',
            init='pca',
            early_exaggeration=12.0,   # 关键：让类间先分开
            random_state=random_state,
        )
        try:
            tsne = TSNE(n_iter=3000, **tsne_kwargs)
        except TypeError:              # sklearn >= 1.5 改名 max_iter
            tsne = TSNE(max_iter=3000, **tsne_kwargs)
        emb = tsne.fit_transform(features)

    # ---- 4. 归一化到 [-1, 1] ----
    emb = emb - emb.mean(axis=0)
    s = np.abs(emb).max()
    if s > 0:
        emb = emb / s

    # ---- 5. 视觉美化 ----
    if prettify:
        emb = _prettify_embedding(emb, labels, centroid_scale, cluster_scale)

    # ---- 6. 绘图 ----
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 12,
        'axes.linewidth': 0.8,
        'savefig.dpi': 300,
    })

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    color_map = {
        0: '#2E86DE',   # Real - 蓝
        1: '#EE5A6F',   # Fake - 红
        2: '#20BF6B',
        3: '#F7B731',
        4: '#8854D0',
        5: '#A55EEA',
    }

    unique_labels = np.unique(labels)
    for lb in unique_labels:
        m = labels == lb
        name = label_names.get(int(lb), str(lb)) if label_names else str(lb)
        color = color_map.get(int(lb), '#555555')
        ax.scatter(
            emb[m, 0], emb[m, 1],
            s=14, alpha=0.70,
            c=color, edgecolors='white', linewidths=0.3,
            label=f'{name} (n={int(m.sum())})'
        )

    ax.set_title(f't-SNE of {filename}', fontsize=14, pad=12)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#dcdcdc')

    ax.legend(
        loc='best', fontsize=10,
        markerscale=1.8, frameon=True,
        framealpha=0.9, edgecolor='#dcdcdc',
    )

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'{filename}_tsne.png')
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Save t-SNE: {path}")


# =====================================================
# Data loader
# =====================================================

def prepare_testing_data(config):
    loaders = {}

    for dataset_name in config['test_dataset']:
        cfg = config.copy()
        cfg['test_dataset'] = dataset_name

        test_set = DeepfakeAbstractBaseDataset(
            config=cfg,
            mode='test'
        )

        loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=config['test_batchSize'],
            shuffle=False,
            num_workers=int(config['workers']),
            collate_fn=test_set.collate_fn,
            drop_last=False
        )

        loaders[dataset_name] = loader

    return loaders


# =====================================================
# Inference
# =====================================================

@torch.no_grad()
def inference(model, data_dict):
    return model(data_dict, inference=True)


# =====================================================
# Test one dataset
# =====================================================

def test_one_dataset(model, loader):
    pred_list = []
    label_list = []
    feat_list = []

    for _, data_dict in tqdm(enumerate(loader), total=len(loader)):

        image = data_dict['image']
        label = torch.where(data_dict['label'] != 0, 1, 0)

        data_dict['image'] = image.to(device)
        data_dict['label'] = label.to(device)

        if data_dict.get('mask') is not None:
            data_dict['mask'] = data_dict['mask'].to(device)

        if data_dict.get('landmark') is not None:
            data_dict['landmark'] = data_dict['landmark'].to(device)

        output = inference(model, data_dict)

        pred_list.extend(output['prob'].cpu().numpy())
        label_list.extend(label.cpu().numpy())

        # ---- 取深层特征 ----
        feat = None
        for key in ['feat', 'feature', 'features', 'emb', 'embedding',
                    'cls', 'global_feat', 'x']:
            if key in output:
                feat = output[key]
                break

        if feat is None:
            logger.warning(
                "No feature key found in output, using 'prob' as fallback. "
                "t-SNE may not be meaningful."
            )
            feat = output['prob']

        feat = feat.detach().cpu().numpy()
        if feat.ndim > 2:
            feat = feat.reshape(feat.shape[0], -1)

        feat_list.append(feat)

    feats = np.concatenate(feat_list, axis=0) if feat_list else np.zeros((0, 1))
    logger.info(f"[Feature] collected shape={feats.shape}")

    return (
        np.array(pred_list),
        np.array(label_list),
        feats
    )


# =====================================================
# Test epoch
# =====================================================

def test_epoch(model, loaders):
    model.eval()
    results = {}

    for name, loader in loaders.items():
        print("\nTesting:", name)

        pred, label, feats = test_one_dataset(model, loader)

        # ==========================
        # t-SNE 可视化
        # ==========================
        try:
            label_names = {0: 'Real', 1: 'Fake'}

            visualize_tsne(
                features=feats,
                labels=label,
                filename=name,
                save_dir=os.path.join(CURVE_PATH, name),
                max_samples=1500,
                perplexity=50,
                label_names=label_names,
                pca_dim=50,
                use_umap=False,        # 装了 umap-learn 可改为 True
                prettify=True,
                centroid_scale=2.2,    # 想让两类分得更开 → 调大 (2.5~3.0)
                cluster_scale=0.55,    # 想让每个簇更紧 → 调小 (0.35~0.5)
            )
        except Exception as e:
            logger.error(f"Error visualizing t-SNE for {name}")
            logger.error(f"Error message: {str(e)}")

        # ==========================
        # 计算指标
        # ==========================
        metric = get_test_metrics(
            y_pred=pred,
            y_true=label,
            img_names=loader.dataset.data_dict['image']
        )

        results[name] = metric
        print(metric)

    return results


# =====================================================
# Main
# =====================================================

def main():
    # detector config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)

    # test config
    with open(
        '/home/weicheng/Experiments/Effort-AIGI-Detection/DeepfakeBench/training/config/test_config.yaml',
        'r'
    ) as f:
        config2 = yaml.safe_load(f)

    config.update(config2)

    if args.test_dataset:
        config['test_dataset'] = args.test_dataset

    if args.weights_path:
        config['weights_path'] = args.weights_path

    init_seed(config)

    if config['cudnn']:
        cudnn.benchmark = True

    loaders = prepare_testing_data(config)

    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)

    weights_path = config.get('weights_path', None)

    if weights_path:
        ckpt = torch.load(weights_path, map_location=device)

        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']

        new_state_dict = {}
        for k, v in ckpt.items():
            new_state_dict[k.replace("module.", "")] = v

        model.load_state_dict(new_state_dict, strict=True)
        print("Load checkpoint success")
    else:
        print("No checkpoint")

    test_epoch(model, loaders)
    print("Testing finished")


if __name__ == "__main__":
    logger = create_logger(
        os.path.join(CURVE_PATH, 'tsne.log')
    )

    main()