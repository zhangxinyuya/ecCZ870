#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧬 Jaccard-Visual 融合版:基因工程目标识类分析工具
核心算法:Jaccard Typed Distance (逻辑更优)
可视化引擎:完全采用t-SNE风格的输出格式
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import os
import glob
from pathlib import Path
import warnings

# ==================== 绘图风格设置 (与t-SNE完全一致) ====================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 18
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 600
plt.style.use('default')
warnings.filterwarnings('ignore')


# ==================== 智能标签管理器 (完全采用t-SNE版本) ====================
class LabelManager:
    """智能标签管理器,解决重叠问题 (与t-SNE版本完全一致)"""

    def __init__(self, ax, margin=0.02):
        self.ax = ax
        self.margin = margin
        self.placed_labels = []
        self.label_boxes = []

    def get_text_bbox(self, text, x, y, fontsize=10):
        """获取文本的准确边界框"""
        temp_text = self.ax.text(x, y, text, fontsize=fontsize, alpha=0)
        renderer = self.ax.figure.canvas.get_renderer()
        bbox = temp_text.get_window_extent(renderer)
        temp_text.remove()
        bbox_data = bbox.transformed(self.ax.transData.inverted())
        return bbox_data

    def calculate_label_positions(self, points, labels, fontsize=10):
        """计算所有标签的最佳位置"""
        n_points = len(points)
        label_positions = []

        x_coords = points[:, 0]
        y_coords = points[:, 1]
        x_range = x_coords.max() - x_coords.min()
        y_range = y_coords.max() - y_coords.min()
        safe_distance = max(x_range, y_range) * 0.1

        for i, (point, label) in enumerate(zip(points, labels)):
            px, py = point
            angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
            distances = [safe_distance * 1, safe_distance * 2, safe_distance * 3]

            best_position = None
            best_score = float('inf')

            for distance in distances:
                for angle in angles:
                    candidate_x = px + distance * np.cos(angle)
                    candidate_y = py + distance * np.sin(angle)

                    if not self._is_within_bounds(candidate_x, candidate_y):
                        continue

                    bbox = self.get_text_bbox(label, candidate_x, candidate_y, fontsize)
                    score = self._calculate_conflict_score(bbox, points, label_positions, px, py)

                    if score < best_score:
                        best_score = score
                        best_position = (candidate_x, candidate_y, bbox)

                    if score == 0:
                        break

                if best_score == 0:
                    break

            if best_position is None:
                angle = (i / n_points) * 2 * np.pi
                best_position = (
                    px + safe_distance * np.cos(angle),
                    py + safe_distance * np.sin(angle),
                    None
                )

            label_positions.append(best_position)

        return label_positions

    def _is_within_bounds(self, x, y):
        """检查位置是否在绘图区域内"""
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        margin_x = (xlim[1] - xlim[0]) * self.margin
        margin_y = (ylim[1] - ylim[0]) * self.margin
        return (xlim[0] + margin_x <= x <= xlim[1] - margin_x and
                ylim[0] + margin_y <= y <= ylim[1] - margin_y)

    def _calculate_conflict_score(self, bbox, points, existing_positions, origin_x, origin_y):
        """计算位置冲突得分"""
        score = 0
        if bbox is None:
            return float('inf')

        for point in points:
            if self._bbox_contains_point(bbox, point):
                score += 100

        for pos in existing_positions:
            if len(pos) >= 3 and pos[2] is not None:
                if self._bbox_overlap(bbox, pos[2]):
                    score += 50

        center_x = (bbox.x0 + bbox.x1) / 2
        center_y = (bbox.y0 + bbox.y1) / 2
        distance = np.sqrt((center_x - origin_x) ** 2 + (center_y - origin_y) ** 2)
        score += distance * 10

        return score

    def _bbox_contains_point(self, bbox, point):
        """检查边界框是否包含点"""
        return (bbox.x0 <= point[0] <= bbox.x1 and bbox.y0 <= point[1] <= bbox.y1)

    def _bbox_overlap(self, bbox1, bbox2):
        """检查两个边界框是否重叠"""
        return not (bbox1.x1 < bbox2.x0 or bbox2.x1 < bbox1.x0 or
                    bbox1.y1 < bbox2.y0 or bbox2.y1 < bbox1.y0)

    def place_labels(self, points, labels, colors=None, fontsize=10):
        """放置所有标签 (与t-SNE样式完全一致)"""
        positions = self.calculate_label_positions(points, labels, fontsize)
        annotations = []

        for i, (point, label, position) in enumerate(zip(points, labels, positions)):
            px, py = point
            lx, ly = position[0], position[1]

            if colors is not None and i < len(colors):
                color = colors[i]
            else:
                color = f'C{i % 10}'

            annotation = self.ax.annotate(
                label, (px, py),
                xytext=(lx, ly),
                fontsize=fontsize,
                fontweight='bold',
                color='black',
                ha='center',
                va='center',
                bbox=dict(
                    boxstyle='round,pad=0.5',
                    facecolor='white',
                    alpha=0.9,
                    edgecolor=color,
                    linewidth=2
                ),
                arrowprops=dict(
                    arrowstyle='->',
                    color=color,
                    alpha=0.7,
                    lw=1.5,
                    shrinkA=3,
                    shrinkB=3
                ),
                zorder=10
            )
            annotations.append(annotation)

        return annotations


# ==================== 核心分析器 ====================
class JaccardVisualAnalyzer:
    """
    融合版分析器:
    1. 逻辑:使用 Jaccard Typed Distance
    2. 视觉:完全采用t-SNE风格的输出格式
    """

    def __init__(self, output_dir="jaccard_visual_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # CPD 映射表
        self.cpd_to_name_mapping = {
            'cpd00036': 'succinate', 'cpd00039': 'L-Lysine', 'cpd00051': 'L-arginine',
            'cpd00054': 'L-serine', 'cpd00060': 'L-Methionine', 'cpd00064': 'L-ornithine',
            'cpd00065': 'L-Tryptophan', 'cpd00066': 'L-phenylalanine', 'cpd00069': 'L-Tyrosine',
            'cpd00084': 'L-cysteine', 'cpd00107': 'L-leucine', 'cpd00119': 'L-histidine',
            'cpd00129': 'L-proline', 'cpd00156': 'L-Valine', 'cpd00161': 'L-Threonine',
            'cpd00227': 'L-homoserine', 'cpd00274': 'L-citrulline', 'cpd00322': 'L-Isoleucine'
        }

        self.datasets = {}
        self.products = []
        self.distance_matrix = None
        self.tsne_results = {}
        self.cluster_results = None
        self.random_state = 42

        # 与t-SNE完全一致的专业配色方案
        self.colors = [
            '#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6',
            '#1ABC9C', '#E67E22', '#34495E', '#8E44AD', '#16A085',
            '#F1C40F', '#E91E63', '#00BCD4', '#FF5722', '#607D8B'
        ]

    def load_data(self, data_dir=".", file_pattern="*.csv"):
        """加载数据"""
        files = glob.glob(os.path.join(data_dir, file_pattern))
        print(f"📂 在 {data_dir} 找到 {len(files)} 个文件")

        for file_path in files:
            fname = Path(file_path).stem
            # 简单的名称清洗
            for prefix in ['targets_', 'target_', 'gene_', 'cpd_']:
                fname = fname.replace(prefix, '')

            p_name = self.cpd_to_name_mapping.get(fname, fname)

            try:
                df = pd.read_csv(file_path)
                if 'gene_id' in df.columns and not df.empty:
                    self.datasets[p_name] = df
            except Exception as e:
                print(f"  ❌ 跳过 {fname}: {e}")

        self.products = sorted(list(self.datasets.keys()))
        print(f"✅ 成功加载 {len(self.products)} 个产品")


    def _build_gene_type_sets(self):
        """为每个产品构建 {gene_id||TYPE} 集合（用于相似度/距离计算的底层数据结构）"""
        product_sets = {}
        for p in self.products:
            df = self.datasets.get(p)
            s = set()

            if df is None or df.empty or 'gene_id' not in df.columns:
                product_sets[p] = s
                continue

            for _, row in df.iterrows():
                gid = row.get('gene_id', None)
                gtype = str(row.get('type', 'OE')).upper().strip()
                if pd.notna(gid):
                    s.add(f"{gid}||{gtype}")
            product_sets[p] = s
        return product_sets

    def _get_high_frequency_keys(self, product_sets, freq_threshold=15):
        """
        统计所有产品中高频出现的“同基因同策略”(gene_id||TYPE) 组合。

        计数口径：某个组合在多少个产品（代谢物策略）中出现（每个产品最多计 1 次）。
        """
        counts = {}
        for _, key_set in product_sets.items():
            for key in key_set:
                counts[key] = counts.get(key, 0) + 1

        high_freq = {k for k, v in counts.items() if v >= freq_threshold}
        return high_freq, counts


    def calculate_jaccard_typed(self, clean_high_freq=True, freq_threshold=15):
        """核心算法:计算类型敏感的Jaccard距离（可选：相似度计算前做高频“同基因同策略”清洗）

        参数
        ----
        clean_high_freq : bool
            是否在计算两两相似度/距离前，剔除在全体产品中高频出现的 gene_id||TYPE 组合（仅影响距离矩阵，不改动原始数据集）。
        freq_threshold : int
            高频阈值：某个 gene_id||TYPE 组合在 >= freq_threshold 个产品中出现，则在相似度计算中被剔除。
        """
        print("\n🧮 计算 Jaccard Typed 距离矩阵...")

        n = len(self.products)
        dist_mat = np.zeros((n, n))

        # 预处理:构建集合（Gene + Type 组合键）
        raw_product_sets = self._build_gene_type_sets()

        # 仅用于相似度计算的数据清洗：剔除全局高频的“同基因同策略”
        if clean_high_freq:
            high_freq_keys, counts = self._get_high_frequency_keys(raw_product_sets, freq_threshold=freq_threshold)

            # 记录下来，方便你后续追溯/排查（不影响 Excel 输出）
            self.high_freq_keys_ = high_freq_keys
            self.high_freq_key_counts_ = counts

            print(f"  🧹 相似度清洗开启：freq_threshold={freq_threshold}")
            print(f"     • 将剔除高频组合数: {len(high_freq_keys)}")

            # 展示出现频次最高的前几个（可选的提示信息）
            if len(counts) > 0:
                top_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
                top_preview = ", ".join([f"{k}({v})" for k, v in top_items])
                print(f"     • Top频次预览(前10): {top_preview}")
        else:
            high_freq_keys = set()
            self.high_freq_keys_ = set()
            self.high_freq_key_counts_ = {}

        product_sets = {p: (s - high_freq_keys) for p, s in raw_product_sets.items()}

        # 计算距离
        for i in range(n):
            for j in range(i + 1, n):
                p1, p2 = self.products[i], self.products[j]
                set1, set2 = product_sets[p1], product_sets[p2]

                union_len = len(set1 | set2)
                if union_len == 0:
                    d = 1.0
                else:
                    inter_len = len(set1 & set2)
                    d = 1.0 - (inter_len / union_len)  # Jaccard 距离

                dist_mat[i, j] = dist_mat[j, i] = d

        self.distance_matrix = pd.DataFrame(dist_mat, index=self.products, columns=self.products)
        return self.distance_matrix

    def run_tsne_and_cluster(self, perplexity=15, n_clusters='auto'):
        """运行t-SNE和聚类"""
        print(f"\n🔄 运行 t-SNE (Perplexity={perplexity})...")

        # t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, metric='precomputed',
                    random_state=self.random_state, init='random', max_iter=1000)
        coords = tsne.fit_transform(self.distance_matrix.values)
        print(f"  📌 t-SNE KL Divergence (perplexity={perplexity}): {tsne.kl_divergence_:.6f}")
        # KMeans 聚类
        if n_clusters == 'auto':
            print("  🤖 自动寻找最佳聚类数...")
            best_score = -1
            best_k = 3
            for k in range(2, min(10, len(self.products))):
                km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                labels = km.fit_predict(coords)
                score = silhouette_score(coords, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
            n_clusters = best_k
            print(f"  🎯 最佳聚类数: {n_clusters} (Score: {best_score:.3f})")

        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(coords)
        final_sil = silhouette_score(coords, labels)
        print(f"  ✅ Final KMeans clustering result -> K={n_clusters}, silhouette={final_sil:.4f}")
        # 整理结果
        df_res = pd.DataFrame({
            'product': self.products,
            'cluster': labels,
            'tsne_1': coords[:, 0],
            'tsne_2': coords[:, 1]
        })

        clusters = {}
        for c in range(n_clusters):
            clusters[c] = df_res[df_res['cluster'] == c]['product'].tolist()

        self.cluster_results = {
            'df': df_res,
            'clusters': clusters,
            'n_clusters': n_clusters,
            'perplexity': perplexity,
            'kl_divergence': float(tsne.kl_divergence_),
            'silhouette': float(final_sil)
        }

        return self.cluster_results

    def plot_perplexity_metrics(self, perplexity_range=None, n_clusters_eval=8, figsize=(16, 6)):
        """
        额外输出一张「组图」: Perplexity 指标评估图（KL散度 / 轮廓系数 / 聚类数）

        - KL散度: 使用 sklearn TSNE 的 tsne.kl_divergence_（每个 perplexity 训练结束后的最终值）
        - 轮廓系数/聚类数: 在 t-SNE 2D 坐标上做 KMeans（固定聚类数规则）后计算

        只会保存 1 个文件: {output_dir}/perplexity_metrics.png
        """
        if self.distance_matrix is None or self.distance_matrix.empty:
            raise ValueError("❌ 请先计算距离矩阵（calculate_jaccard_typed）")

        n = len(self.products)
        if n < 3:
            print("⚠️ 产品数量过少（<3），无法进行perplexity指标评估图绘制")
            return None

        # 默认perplexity范围：2 到 min(25, n-1)（与常见经验一致）
        if perplexity_range is None:
            max_p = min(25, n - 1)
            perplexity_range = range(2, max_p + 1)

        perplexities = []
        kl_divergences = []
        silhouette_scores = []
        cluster_counts = []

        # 固定聚类数规则：最多 n_clusters_eval，但不能超过 n-1，且至少为2
        k_fixed = min(n_clusters_eval, n - 1)
        if k_fixed < 2:
            print("⚠️ 无法设定有效聚类数（<2），跳过perplexity指标评估图绘制")
            return None

        print("\n📊 生成 Perplexity 指标评估组图（将额外保存 1 张图片）...")
        print(f"   • Perplexity 范围: {min(perplexity_range)} - {max(perplexity_range)}")
        print(f"   • KMeans 聚类数（用于评估）: {k_fixed}")

        for p in list(perplexity_range):
            # sklearn TSNE 要求 perplexity < n_samples
            if p >= n:
                continue

            try:
                tsne = TSNE(
                    n_components=2,
                    perplexity=p,
                    metric='precomputed',
                    random_state=self.random_state,
                    init='random',
                    max_iter=1000
                )
                coords = tsne.fit_transform(self.distance_matrix.values)
                kl = float(getattr(tsne, "kl_divergence_", np.nan))

                # KMeans + silhouette
                kmeans = KMeans(n_clusters=k_fixed, random_state=self.random_state, n_init=10)
                labels = kmeans.fit_predict(coords)
                uniq = np.unique(labels)

                if len(uniq) > 1:
                    sil = float(silhouette_score(coords, labels))
                else:
                    sil = -1.0

                perplexities.append(p)
                kl_divergences.append(kl)
                silhouette_scores.append(sil)
                cluster_counts.append(int(len(uniq)))

            except Exception:
                # 某些perplexity可能不稳定，跳过即可
                continue

        if len(perplexities) < 2:
            print("⚠️ 可用perplexity结果不足（<2），跳过perplexity指标评估图绘制")
            return None

        # 绘图（参考你上面 t-SNE 版本的 1×3 组图样式）
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.patch.set_facecolor('white')

        # 子图1：KL散度
        ax1 = axes[0]
        ax1.set_facecolor('white')
        ax1.grid(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_linewidth(1.2)
        ax1.spines['bottom'].set_linewidth(1.2)

        ax1.plot(perplexities, kl_divergences, 'b-o', linewidth=2, markersize=8)
        ax1.fill_between(perplexities, kl_divergences, alpha=0.3)
        ax1.set_xlabel('Perplexity', fontsize=12, fontweight='bold')
        ax1.set_ylabel('KL Divergence', fontsize=12, fontweight='bold')
        ax1.set_title('KL Divergence vs Perplexity', fontsize=14, fontweight='bold')

        best_kl_idx = int(np.nanargmin(kl_divergences))
        ax1.scatter(perplexities[best_kl_idx], kl_divergences[best_kl_idx],
                    color='red', s=200, zorder=5, edgecolors='darkred', linewidth=2)
        ax1.annotate(f'Best: {perplexities[best_kl_idx]}',
                     xy=(perplexities[best_kl_idx], kl_divergences[best_kl_idx]),
                     xytext=(10, 10), textcoords='offset points',
                     fontsize=10, fontweight='bold', color='red',
                     arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

        # 子图2：轮廓系数（只画有效值）
        ax2 = axes[1]
        ax2.set_facecolor('white')
        ax2.grid(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_linewidth(1.2)
        ax2.spines['bottom'].set_linewidth(1.2)

        valid_idx = [i for i, s in enumerate(silhouette_scores) if s > 0]
        if valid_idx:
            v_perp = [perplexities[i] for i in valid_idx]
            v_sil = [silhouette_scores[i] for i in valid_idx]
            ax2.plot(v_perp, v_sil, 'g-o', linewidth=2, markersize=8)
            ax2.fill_between(v_perp, v_sil, alpha=0.3, color='green')

            best_sil_idx = int(np.argmax(v_sil))
            ax2.scatter(v_perp[best_sil_idx], v_sil[best_sil_idx],
                        color='red', s=200, zorder=5, edgecolors='darkred', linewidth=2)
            ax2.annotate(f'Best: {v_perp[best_sil_idx]}',
                         xy=(v_perp[best_sil_idx], v_sil[best_sil_idx]),
                         xytext=(10, -10), textcoords='offset points',
                         fontsize=10, fontweight='bold', color='red',
                         arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

        ax2.set_xlabel('Perplexity', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
        ax2.set_title('Clustering Quality vs Perplexity', fontsize=14, fontweight='bold')
        ax2.set_ylim([-0.1, 1.0])

        # 子图3：聚类数
        ax3 = axes[2]
        ax3.set_facecolor('white')
        ax3.grid(False)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.spines['left'].set_linewidth(1.2)
        ax3.spines['bottom'].set_linewidth(1.2)

        ax3.bar(perplexities, cluster_counts, color='orange', alpha=0.7,
                edgecolor='darkorange', linewidth=2)
        ax3.set_xlabel('Perplexity', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Number of Clusters', fontsize=12, fontweight='bold')
        ax3.set_title('Cluster Count vs Perplexity', fontsize=14, fontweight='bold')

        for p, nclu in zip(perplexities, cluster_counts):
            if nclu > 0:
                ax3.text(p, nclu + 0.1, str(nclu), ha='center', va='bottom', fontweight='bold')

        fig.suptitle('Perplexity Analysis: Finding the Optimal Value',
                     fontsize=16, fontweight='bold', y=1.05)

        plt.tight_layout()

        save_path = self.output_dir / 'perplexity_metrics.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)

        print(f"  ✅ Perplexity评估指标图已保存: {save_path}")
        return save_path

    def _get_plot_data(self):
        """获取绘图数据,确保顺序一致 (与t-SNE完全一致)"""
        cluster_df = self.cluster_results['df']
        unique_clusters = sorted([c for c in np.unique(cluster_df['cluster']) if c >= 0])

        # 按聚类分组构建数据,保证顺序一致
        points = []
        point_colors = []
        labels = []

        for i, cluster_id in enumerate(unique_clusters):
            mask = cluster_df['cluster'] == cluster_id
            cluster_data = cluster_df[mask]
            color = self.colors[i % len(self.colors)]

            for _, row in cluster_data.iterrows():
                points.append([row['tsne_1'], row['tsne_2']])
                point_colors.append(color)
                labels.append(row['product'])

        return {
            'points': np.array(points),
            'point_colors': point_colors,
            'labels': labels,
            'cluster_df': cluster_df,
            'unique_clusters': unique_clusters
        }

    def _create_tsne_plot(self, plot_data, perplexity, show_labels=True):
        """创建单个聚类图 (与t-SNE完全一致的样式)"""
        # 与t-SNE完全一致的图片大小
        fig, ax = plt.subplots(figsize=(20, 14))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        # 移除顶部和右侧边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)

        # 关闭网格线
        ax.grid(False)

        # 设置绘图区域
        x_coords = plot_data['cluster_df']['tsne_1'].values
        y_coords = plot_data['cluster_df']['tsne_2'].values
        x_span = x_coords.max() - x_coords.min()
        y_span = y_coords.max() - y_coords.min()
        margin_ratio = 0.15

        ax.set_xlim(x_coords.min() - x_span * margin_ratio,
                    x_coords.max() + x_span * margin_ratio)
        ax.set_ylim(y_coords.min() - y_span * margin_ratio,
                    y_coords.max() + y_span * margin_ratio)

        # 绘制散点图 (与t-SNE完全一致的样式)
        for i, cluster_id in enumerate(plot_data['unique_clusters']):
            mask = plot_data['cluster_df']['cluster'] == cluster_id
            cluster_data = plot_data['cluster_df'][mask]
            color = self.colors[i % len(self.colors)]

            ax.scatter(
                cluster_data['tsne_1'], cluster_data['tsne_2'],
                c=color, label=f'Cluster {cluster_id + 1}',
                alpha=0.85, s=800, edgecolors='white', linewidth=4,
                zorder=5
            )

        # 添加标签(根据参数决定是否显示)
        if show_labels:
            label_manager = LabelManager(ax)
            label_manager.place_labels(
                plot_data['points'],
                plot_data['labels'],
                plot_data['point_colors'],
                fontsize=16
            )

        # 设置标题和标签 (与t-SNE完全一致的样式)
        ax.set_xlabel('t-SNE Component 1', fontsize=20, fontweight='bold', labelpad=20)
        ax.set_ylabel('t-SNE Component 2', fontsize=20, fontweight='bold', labelpad=20)
        ax.set_title(
            f'Clustering Analysis of 18 Metabolic Products',
            fontsize=30, fontweight='bold', pad=30
        )

        # 图例放在坐标轴内 (与t-SNE完全一致的样式)
        legend = ax.legend(
            loc='upper right', frameon=True,
            fancybox=True, shadow=True, fontsize=16,
            title='Strategy Clusters', title_fontsize=20,
            bbox_to_anchor=(0.95, 0.98),
            framealpha=0.95, edgecolor='#333333',
            borderpad=1.5, columnspacing=1.5, handletextpad=1
        )
        legend.get_title().set_fontweight('bold')

        ax.tick_params(axis='both', which='major', labelsize=16,
                       colors='#333333', width=2, length=8)

        plt.tight_layout()

        return fig, ax

    def plot_visuals(self):
        """生成美观的图表 (与t-SNE完全一致的输出格式)"""
        print("\n🎨 生成可视化图表...")
        res = self.cluster_results
        perplexity = res['perplexity']

        # 获取绘图数据
        plot_data = self._get_plot_data()

        # 生成带标签的图
        fig1, ax1 = self._create_tsne_plot(plot_data, perplexity, show_labels=True)

        # 生成不带标签的图
        fig2, ax2 = self._create_tsne_plot(plot_data, perplexity, show_labels=False)

        # 保存图片 (与t-SNE完全一致的文件名和格式)
        save_path_with_labels = self.output_dir / 'tsne_clustering_with_labels.png'
        plt.figure(fig1.number)
        plt.savefig(save_path_with_labels, dpi=600, bbox_inches='tight',
                    facecolor='white', edgecolor='none', pad_inches=0.5)

        save_path_no_labels = self.output_dir / 'tsne_clustering_no_labels.png'
        plt.figure(fig2.number)
        plt.savefig(save_path_no_labels, dpi=600, bbox_inches='tight',
                    facecolor='white', edgecolor='none', pad_inches=0.5)

        plt.close('all')

        print(f"  ✅ 带标签图片已保存: {save_path_with_labels}")
        print(f"  ✅ 无标签图片已保存: {save_path_no_labels}")

    def export_analysis_to_excel(self, output_file="Gene_Cluster_Analysis_Report.xlsx"):
        """
        将聚类分析结果和共同基因目标导出为 Excel 表格。
        """
        # 寻找共同基因
        print("\n🔍 寻找聚类内共同基因目标...")
        common_targets = {}
        clusters = self.cluster_results['clusters']

        for cluster_id, products in clusters.items():
            if len(products) < 2:
                print(f"  ⚠️ 聚类 {cluster_id + 1} 只有1个产品,跳过共同目标分析")
                continue

            print(f"  📊 分析聚类 {cluster_id + 1} ({len(products)}个产品)...")

            # 收集该聚类所有产品的基因目标
            cluster_genes = {'OE': None, 'KD': None, 'KO': None}

            for i, product in enumerate(products):
                df = self.datasets[product]

                if 'gene_id' in df.columns:
                    # 按基因工程类型分组
                    for gene_type in ['OE', 'KD', 'KO']:
                        if 'type' in df.columns:
                            genes = set(
                                df[df['type'].str.upper().str.strip() == gene_type]['gene_id'].dropna().tolist())
                        else:
                            # 如果没有type列,假设都是OE
                            if gene_type == 'OE':
                                genes = set(df['gene_id'].dropna().tolist())
                            else:
                                genes = set()

                        if i == 0:  # 第一个产品
                            cluster_genes[gene_type] = genes
                        else:  # 求交集
                            if cluster_genes[gene_type] is not None:
                                cluster_genes[gene_type] &= genes

            # 保存有共同基因的类型
            cluster_common = {}
            for gene_type, genes in cluster_genes.items():
                if genes:  # 有共同基因
                    cluster_common[gene_type] = list(genes)
                    print(f"    {gene_type}: {len(genes)} 个共同基因")

            if cluster_common:
                common_targets[f"Cluster_{cluster_id + 1}"] = cluster_common
            else:
                print(f"    ❌ 没有发现共同基因目标")

        # 准备Excel数据
        excel_data = []

        # 基因名称映射表
        gene_name_map = {
            'CEY17_RS14115': 'fbaA', 'CEY17_RS03780': 'pyc', 'CEY17_RS08495': 'gap',
            'CEY17_RS08485': 'tpiA', 'CEY17_RS08450': 'opcA', 'CEY17_RS08435': 'tkt',
            'CEY17_RS08545': 'rpe', 'CEY17_RS01410': 'aspB', 'CEY17_RS10245': 'pyk',
            'CEY17_RS15645': 'ME2', 'CEY17_RS10200': 'gdhA', 'CEY17_RS08440': 'tal',
            'CEY17_RS08445': 'zwf', 'CEY17_RS01500': 'lysC', 'CEY17_RS01505': 'asd',
            'CEY17_RS08455': 'pgl', 'CEY17_RS07800': 'gndA', 'CEY17_RS05085': 'prsA',
            'CEY17_RS10780': 'asnB', 'CEY17_RS08615': 'pyrB', 'CEY17_RS06790': 'pfkA',
            'CEY17_RS11015': 'aceE'
        }

        # 按聚类整合数据
        cluster_df = self.cluster_results['df']
        sorted_clusters = sorted(common_targets.keys(), key=lambda x: int(x.split('_')[1]))

        for cluster_key in sorted_clusters:
            cluster_num = int(cluster_key.split('_')[1])

            # 从聚类结果中筛选出属于该聚类的所有产品名称
            products = cluster_df[cluster_df['cluster'] == (cluster_num - 1)]['product'].tolist()
            product_str = ", ".join(products)

            # 添加聚类标题装饰行
            excel_data.append({"Gene Name": f"== {cluster_key} ==", "Gene ID": f"Products: {product_str}", "Type": ""})
            # 添加表头行
            excel_data.append({"Gene Name": "Gene Name", "Gene ID": "Gene ID", "Type": "Type"})

            # 添加具体的基因数据
            genes_found = False
            for gtype, gids in common_targets[cluster_key].items():
                for gid in gids:
                    excel_data.append({
                        "Gene Name": gene_name_map.get(gid, "unknown"),
                        "Gene ID": gid,
                        "Type": gtype
                    })
                    genes_found = True

            if not genes_found:
                excel_data.append({"Gene Name": "No common genes found", "Gene ID": "-", "Type": "-"})

            # 添加一个空行作为聚类间的间隔
            excel_data.append({"Gene Name": "", "Gene ID": "", "Type": ""})

        # 导出到 Excel 并美化格式
        full_output_path = self.output_dir / output_file

        try:
            final_df = pd.DataFrame(excel_data)

            # 使用 xlsxwriter 引擎以支持样式设置
            with pd.ExcelWriter(full_output_path, engine='xlsxwriter') as writer:
                final_df.to_excel(writer, index=False, header=False, sheet_name='Cluster Analysis')

                workbook = writer.book
                worksheet = writer.sheets['Cluster Analysis']

                # 设置列宽,让内容不被遮挡
                worksheet.set_column('A:A', 25)  # 基因名列
                worksheet.set_column('B:B', 35)  # 基因 ID 列
                worksheet.set_column('C:C', 12)  # 类型列

            print(f"✅ 成功!分析报告已导出至: {full_output_path}")

        except Exception as e:
            print(f"❌ 导出 Excel 失败: {e}")

    def run(self, data_dir, perplexity=15, n_clusters='auto'):
        """一键运行"""
        print("🚀 开始分析...")
        self.load_data(data_dir, file_pattern="targets_*.csv")
        self.calculate_jaccard_typed()
        self.plot_perplexity_metrics()
        self.run_tsne_and_cluster(perplexity, n_clusters)
        self.plot_visuals()
        self.export_analysis_to_excel()

        # 列出生成的关键文件
        print("\n📂 生成的关键文件:")
        print(f"   • {self.output_dir}/tsne_clustering_with_labels.png - 带标签的可视化图表")
        print(f"   • {self.output_dir}/tsne_clustering_no_labels.png - 不带标签的可视化图表")
        print(f"   • {self.output_dir}/Gene_Cluster_Analysis_Report.xlsx - Excel分析报告")

        print("\n✨ 全部完成!")


# ==================== 主程序入口 ====================

if __name__ == "__main__":
    # 配置
    DATA_DIR = "."  # 你的csv文件夹路径
    OUTPUT_DIR = "final_analysis_results"

    # 初始化并运行
    analyzer = JaccardVisualAnalyzer(output_dir=OUTPUT_DIR)

    # 运行
    # perplexity 建议:数据量<50用5-15,数据量>100用30+
    analyzer.run(data_dir=DATA_DIR, perplexity=12, n_clusters='auto')