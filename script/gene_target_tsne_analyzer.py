#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的基因工程目标t-SNE聚类分析工具
包含HTML报告生成和错误修复
精简版输出 - 只保留关键文件
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.manifold import TSNE
import os
import glob
from pathlib import Path
import warnings
from datetime import datetime
import json

# 设置字体和样式 - 调整为更简洁的风格
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 18
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 600
# 移除seaborn网格样式，使用更简洁的默认样式
plt.style.use('default')
warnings.filterwarnings('ignore')


def convert_to_serializable(obj):
    """转换numpy类型为Python原生类型以支持JSON序列化"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


class LabelManager:
    """智能标签管理器，解决重叠问题"""

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
        """放置所有标签"""
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


class OptimizedGeneTargetTSNEAnalyzer:
    """优化的基因工程目标t-SNE聚类分析器"""

    def __init__(self, oe_weight=4, kd_weight=0.25, ko_weight=0, normal_weight=1,
                 output_dir="tsne_analyzer_results", minimize_output=True):
        """初始化分析器"""
        self.oe_weight = oe_weight
        self.kd_weight = kd_weight
        self.ko_weight = ko_weight
        self.normal_weight = normal_weight
        self.minimize_output = minimize_output

        # CPD编号到产品名称的映射字典
        self.cpd_to_name_mapping = {
            'cpd00036': 'succinate',
            'cpd00039': 'L-Lysine',
            'cpd00051': 'L-arginine',
            'cpd00054': 'L-serine',
            'cpd00060': 'L-Methionine',
            'cpd00064': 'L-ornithine',
            'cpd00065': 'L-Tryptophan',
            'cpd00066': 'L-phenylalanine',
            'cpd00069': 'L-Tyrosine',
            'cpd00084': 'L-cysteine',
            'cpd00107': 'L-leucine',
            'cpd00119': 'L-histidine',
            'cpd00129': 'L-proline',
            'cpd00156': 'L-Valine',
            'cpd00161': 'L-Threonine',
            'cpd00227': 'L-homoserine',
            'cpd00274': 'L-citrulline',
            'cpd00322': 'L-Isoleucine'
        }

        # 创建输出目录
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # 数据存储容器
        self.datasets = {}
        self.gene_target_matrix = None
        self.distance_matrix = None
        self.distance_matrix_normalized = None
        self.tsne_results = {}
        self.cluster_results = None
        self.all_genes = []
        self.products = []

        # 分析参数设置
        self.perplexity_range = range(2, 26)
        self.random_state = 42

    def _map_cpd_to_product_names(self):
        """将CPD编号映射为产品名称"""
        mapped_products = []
        name_changes = {}

        for product in self.products:
            if product in self.cpd_to_name_mapping:
                new_name = self.cpd_to_name_mapping[product]
                mapped_products.append(new_name)
                name_changes[product] = new_name
            else:
                mapped_products.append(product)

        self.products = mapped_products

        if name_changes:
            new_datasets = {}
            for old_name, df in self.datasets.items():
                if old_name in name_changes:
                    new_name = name_changes[old_name]
                    new_datasets[new_name] = df
                else:
                    new_datasets[old_name] = df
            self.datasets = new_datasets

    def load_data_from_files(self, file_pattern="*.csv", data_dir="."):
        """从CSV文件批量加载数据"""
        file_pattern_path = os.path.join(data_dir, file_pattern)
        files = glob.glob(file_pattern_path)

        if not files:
            raise FileNotFoundError(f"未找到匹配模式 {file_pattern_path} 的文件")

        for file_path in files:
            filename = Path(file_path).stem
            product_name = self._extract_product_name(filename)

            try:
                df = pd.read_csv(file_path)
                if self._validate_dataframe(df, file_path):
                    self.datasets[product_name] = df
            except Exception as e:
                continue

        self.products = list(self.datasets.keys())

        if not self.products:
            raise ValueError("没有成功加载任何有效数据文件")

        self._map_cpd_to_product_names()

    def _extract_product_name(self, filename):
        """从文件名提取产品名称"""
        product_name = filename
        prefixes_to_remove = ['targets_', 'target_', 'gene_', 'cpd_', 'compound_']
        for prefix in prefixes_to_remove:
            product_name = product_name.replace(prefix, '')
        return product_name if product_name else filename

    def _validate_dataframe(self, df, file_path):
        """验证DataFrame的数据格式"""
        required_columns = ['gene_id']
        if not all(col in df.columns for col in required_columns):
            return False
        if df.empty or df['gene_id'].isna().all():
            return False
        return True

    def vectorize_gene_targets(self):
        """将基因目标向量化为矩阵表示"""
        all_genes_set = set()
        for product, df in self.datasets.items():
            if 'gene_id' in df.columns:
                valid_genes = df['gene_id'].dropna().unique()
                all_genes_set.update(valid_genes)

        self.all_genes = sorted(list(all_genes_set))
        matrix_data = []

        for product in self.products:
            df = self.datasets[product]
            gene_vector = {gene: self.normal_weight for gene in self.all_genes}

            if 'gene_id' in df.columns:
                for _, row in df.iterrows():
                    gene_id = row['gene_id']
                    if 'type' in df.columns:
                        gene_type = str(row['type']).upper().strip()
                    else:
                        gene_type = 'OE'

                    if pd.notna(gene_id) and gene_id in gene_vector:
                        if gene_type == 'OE':
                            gene_vector[gene_id] = self.oe_weight
                        elif gene_type == 'KD':
                            gene_vector[gene_id] = self.kd_weight
                        elif gene_type == 'KO':
                            gene_vector[gene_id] = self.ko_weight

            matrix_data.append(list(gene_vector.values()))

        self.gene_target_matrix = pd.DataFrame(
            matrix_data,
            index=self.products,
            columns=self.all_genes
        )

        # 仅在非精简模式下保存矩阵
        if not self.minimize_output:
            matrix_path = self.output_dir / 'gene_target_matrix.csv'
            self.gene_target_matrix.to_csv(matrix_path)
        return self.gene_target_matrix

    def calculate_distance_matrix(self, metric='euclidean'):
        """计算产品间的距离矩阵"""
        if self.gene_target_matrix is None:
            raise ValueError("请先执行 vectorize_gene_targets()")

        distance_matrix = pairwise_distances(
            self.gene_target_matrix.values,
            metric=metric
        )

        self.distance_matrix = pd.DataFrame(
            distance_matrix,
            index=self.products,
            columns=self.products
        )

        max_distance = self.distance_matrix.values.max()
        if max_distance > 0:
            self.distance_matrix_normalized = self.distance_matrix / max_distance
        else:
            self.distance_matrix_normalized = self.distance_matrix.copy()

        # 仅在非精简模式下保存距离矩阵
        if not self.minimize_output:
            self.distance_matrix.to_csv(self.output_dir / 'distance_matrix.csv')
            self.distance_matrix_normalized.to_csv(self.output_dir / 'distance_matrix_normalized.csv')
        return self.distance_matrix

    def perform_tsne_analysis(self, perplexity_range=None, n_components=2):
        """执行t-SNE分析，测试不同perplexity值"""
        if self.gene_target_matrix is None:
            raise ValueError("请先执行 vectorize_gene_targets()")

        if perplexity_range is None:
            perplexity_range = self.perplexity_range

        perplexity_list = list(perplexity_range)
        self.tsne_results = {}

        for perplexity in perplexity_list:
            try:
                max_perplexity = len(self.products) - 1
                if perplexity >= max_perplexity:
                    perplexity = max_perplexity - 1
                    if perplexity < 1:
                        continue

                tsne = TSNE(
                    n_components=n_components,
                    perplexity=perplexity,
                    random_state=self.random_state,
                    max_iter=1000,
                    n_iter_without_progress=300,
                    learning_rate='auto',
                    init='random'
                )

                tsne_result = tsne.fit_transform(self.gene_target_matrix.values)

                self.tsne_results[perplexity] = {
                    'coordinates': tsne_result,
                    'kl_divergence': tsne.kl_divergence_
                }

            except Exception as e:
                continue

        # 仅在非精简模式下保存详细的t-SNE结果
        if not self.minimize_output:
            self._save_tsne_results()
        return self.tsne_results

    def _save_tsne_results(self):
        """保存所有t-SNE结果到文件"""
        tsne_dir = self.output_dir / 'tsne_results'
        tsne_dir.mkdir(exist_ok=True)

        for perplexity, result in self.tsne_results.items():
            df = pd.DataFrame(
                result['coordinates'],
                index=self.products,
                columns=['tSNE_1', 'tSNE_2']
            )
            df['KL_divergence'] = result['kl_divergence']
            df.to_csv(tsne_dir / f'tsne_perplexity_{perplexity}.csv')

    def identify_clusters_from_tsne(self, perplexity=15, method='visual', n_clusters=8):
        """基于t-SNE结果识别聚类"""
        available_perplexities = list(self.tsne_results.keys())
        if perplexity not in available_perplexities:
            if available_perplexities:
                perplexity = available_perplexities[0]
            else:
                raise ValueError("没有可用的t-SNE结果，请先执行t-SNE分析")

        tsne_coords = self.tsne_results[perplexity]['coordinates']

        if method == 'visual':
            cluster_labels = self._visual_clustering_simulation(tsne_coords, n_clusters)
        elif method == 'kmeans':
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(tsne_coords)
        else:
            raise ValueError(f"不支持的聚类方法: {method}")

        cluster_df = pd.DataFrame({
            'product': self.products,
            'cluster': cluster_labels,
            'tsne_1': tsne_coords[:, 0],
            'tsne_2': tsne_coords[:, 1]
        })

        clusters = {}
        unique_labels = np.unique(cluster_labels)
        for cluster_id in unique_labels:
            if cluster_id >= 0:
                cluster_products = cluster_df[cluster_df['cluster'] == cluster_id]['product'].tolist()
                clusters[cluster_id] = cluster_products

        self.cluster_results = {
            'clusters': clusters,
            'labels': cluster_labels,
            'cluster_df': cluster_df,
            'perplexity_used': perplexity,
            'method': method,
            'n_clusters': len(clusters)
        }

        # 始终保存聚类结果（这是关键文件）
        cluster_df.to_csv(self.output_dir / 'cluster_results.csv', index=False)
        return self.cluster_results

    def _visual_clustering_simulation(self, coords, n_clusters):
        """模拟视觉聚类"""
        if n_clusters == 'auto':
            best_score = -1
            best_k = 2
            for k in range(2, min(len(coords), 10)):
                try:
                    kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                    labels = kmeans.fit_predict(coords)
                    score = silhouette_score(coords, labels)
                    if score > best_score:
                        best_score = score
                        best_k = k
                except:
                    continue
            n_clusters = best_k

        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        return kmeans.fit_predict(coords)

    def plot_perplexity_comparison(self, perplexities_to_plot=None, figsize=(24, 16)):
        """绘制不同perplexity值的t-SNE结果对比图"""
        if not self.tsne_results:
            raise ValueError("请先执行t-SNE分析")

        if perplexities_to_plot is None:
            # 选择代表性的perplexity值进行展示
            available = sorted(self.tsne_results.keys())
            if len(available) <= 6:
                perplexities_to_plot = available
            else:
                # 选择6个代表性的值：最小、最大和4个中间值
                indices = [0, len(available) // 5, 2 * len(available) // 5,
                           3 * len(available) // 5, 4 * len(available) // 5, -1]
                perplexities_to_plot = [available[i] for i in indices]

        n_plots = len(perplexities_to_plot)
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        # 使用一致的颜色映射
        unique_products = self.products
        color_map = {product: f'C{i % 10}' for i, product in enumerate(unique_products)}

        plot_idx = 0
        for row in range(n_rows):
            for col in range(n_cols):
                if plot_idx >= n_plots:
                    if row < n_rows and col < n_cols:
                        axes[row, col].axis('off')
                    continue

                perplexity = perplexities_to_plot[plot_idx]
                ax = axes[row, col]

                # 设置白色背景和移除网格线
                ax.set_facecolor('white')
                ax.grid(False)

                # 移除顶部和右侧边框
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_linewidth(1.2)
                ax.spines['bottom'].set_linewidth(1.2)

                # 获取t-SNE坐标
                coords = self.tsne_results[perplexity]['coordinates']
                kl_div = self.tsne_results[perplexity]['kl_divergence']

                # 绘制散点图
                for i, product in enumerate(unique_products):
                    ax.scatter(coords[i, 0], coords[i, 1],
                               c=color_map[product], s=300, alpha=0.7,
                               edgecolors='white', linewidth=2)

                # 添加标签（简化版，避免重叠）
                for i, product in enumerate(unique_products):
                    ax.annotate(product,
                                (coords[i, 0], coords[i, 1]),
                                fontsize=8, ha='center', va='center')

                # 设置标题和标签
                ax.set_title(f'Perplexity = {perplexity}\nKL Divergence = {kl_div:.3f}',
                             fontsize=12, fontweight='bold')
                ax.set_xlabel('t-SNE 1', fontsize=10)
                ax.set_ylabel('t-SNE 2', fontsize=10)

                plot_idx += 1

        # 添加总标题
        fig.suptitle('t-SNE Results Comparison Across Different Perplexity Values',
                     fontsize=20, fontweight='bold', y=1.02)

        plt.tight_layout()

        # 保存图表
        if not self.minimize_output:
            save_path = self.output_dir / 'perplexity_comparison.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            print(f"  ✅ Perplexity比较图已保存: {save_path}")

        plt.show()
        return fig, axes

    def plot_perplexity_metrics(self, figsize=(16, 6)):
        """绘制不同perplexity值的评估指标对比图"""
        if not self.tsne_results:
            raise ValueError("请先执行t-SNE分析")

        # 计算各个perplexity的评估指标
        consistency_results = self.evaluate_cluster_consistency()

        perplexities = sorted(self.tsne_results.keys())
        kl_divergences = [self.tsne_results[p]['kl_divergence'] for p in perplexities]
        silhouette_scores = []
        n_clusters_list = []

        for p in perplexities:
            if p in consistency_results and consistency_results[p]['silhouette_score'] > -1:
                silhouette_scores.append(consistency_results[p]['silhouette_score'])
                n_clusters_list.append(consistency_results[p]['n_clusters'])
            else:
                silhouette_scores.append(0)
                n_clusters_list.append(0)

        # 创建子图
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.patch.set_facecolor('white')  # 设置整个图表背景为白色

        # 图1: KL散度
        ax1 = axes[0]
        ax1.set_facecolor('white')
        ax1.grid(False)
        # 移除顶部和右侧边框
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_linewidth(1.2)
        ax1.spines['bottom'].set_linewidth(1.2)

        ax1.plot(perplexities, kl_divergences, 'b-o', linewidth=2, markersize=8)
        ax1.fill_between(perplexities, kl_divergences, alpha=0.3)
        ax1.set_xlabel('Perplexity', fontsize=12, fontweight='bold')
        ax1.set_ylabel('KL Divergence', fontsize=12, fontweight='bold')
        ax1.set_title('KL Divergence vs Perplexity', fontsize=14, fontweight='bold')

        # 标记最优值
        best_kl_idx = kl_divergences.index(min(kl_divergences))
        ax1.scatter(perplexities[best_kl_idx], kl_divergences[best_kl_idx],
                    color='red', s=200, zorder=5, edgecolors='darkred', linewidth=2)
        ax1.annotate(f'Best: {perplexities[best_kl_idx]}',
                     xy=(perplexities[best_kl_idx], kl_divergences[best_kl_idx]),
                     xytext=(10, 10), textcoords='offset points',
                     fontsize=10, fontweight='bold', color='red',
                     arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

        # 图2: 轮廓系数
        ax2 = axes[1]
        ax2.set_facecolor('white')
        ax2.grid(False)
        # 移除顶部和右侧边框
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_linewidth(1.2)
        ax2.spines['bottom'].set_linewidth(1.2)

        valid_indices = [i for i, s in enumerate(silhouette_scores) if s > 0]
        if valid_indices:
            valid_perplexities = [perplexities[i] for i in valid_indices]
            valid_silhouettes = [silhouette_scores[i] for i in valid_indices]

            ax2.plot(valid_perplexities, valid_silhouettes, 'g-o', linewidth=2, markersize=8)
            ax2.fill_between(valid_perplexities, valid_silhouettes, alpha=0.3, color='green')

            # 标记最优值
            if valid_silhouettes:
                best_sil_idx = valid_silhouettes.index(max(valid_silhouettes))
                ax2.scatter(valid_perplexities[best_sil_idx], valid_silhouettes[best_sil_idx],
                            color='red', s=200, zorder=5, edgecolors='darkred', linewidth=2)
                ax2.annotate(f'Best: {valid_perplexities[best_sil_idx]}',
                             xy=(valid_perplexities[best_sil_idx], valid_silhouettes[best_sil_idx]),
                             xytext=(10, -10), textcoords='offset points',
                             fontsize=10, fontweight='bold', color='red',
                             arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

        ax2.set_xlabel('Perplexity', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
        ax2.set_title('Clustering Quality vs Perplexity', fontsize=14, fontweight='bold')
        ax2.set_ylim([-0.1, 1.0])

        # 图3: 聚类数量
        ax3 = axes[2]
        ax3.set_facecolor('white')
        ax3.grid(False)
        # 移除顶部和右侧边框
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.spines['left'].set_linewidth(1.2)
        ax3.spines['bottom'].set_linewidth(1.2)

        ax3.bar(perplexities, n_clusters_list, color='orange', alpha=0.7, edgecolor='darkorange', linewidth=2)
        ax3.set_xlabel('Perplexity', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Number of Clusters', fontsize=12, fontweight='bold')
        ax3.set_title('Cluster Count vs Perplexity', fontsize=14, fontweight='bold')

        # 添加数值标签
        for i, (p, n) in enumerate(zip(perplexities, n_clusters_list)):
            if n > 0:
                ax3.text(p, n + 0.1, str(n), ha='center', va='bottom', fontweight='bold')

        # 总标题
        fig.suptitle('Perplexity Analysis: Finding the Optimal Value',
                     fontsize=16, fontweight='bold', y=1.05)

        plt.tight_layout()

        # 保存图表
        save_path = self.output_dir / 'perplexity_metrics.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"  ✅ Perplexity评估指标图已保存: {save_path}")

        plt.show()

        # 打印推荐
        print("\n📊 Perplexity分析结果:")
        print(f"  🎯 最低KL散度: perplexity = {perplexities[best_kl_idx]}")
        if valid_indices and valid_silhouettes:
            print(f"  🎯 最佳聚类质量: perplexity = {valid_perplexities[best_sil_idx]}")
        print(f"  💡 建议: 综合考虑KL散度和聚类质量，选择合适的perplexity值")

        return fig, axes

    def plot_optimized_tsne_clusters(self, perplexity=15, save_formats=['png'], figsize=(20, 14)):
        """绘制优化的t-SNE聚类结果图（带标签和不带标签两个版本）"""
        plot_data = self._get_tsne_plot_data(perplexity)

        # 生成带标签的图
        fig1, ax1 = self._create_tsne_plot(plot_data, perplexity, show_labels=True)

        # 生成不带标签的图
        fig2, ax2 = self._create_tsne_plot(plot_data, perplexity, show_labels=False)

        # 保存图片（关键文件）
        for fmt in save_formats:
            save_path_with_labels = self.output_dir / f'tsne_clustering_with_labels.{fmt}'
            plt.figure(fig1.number)
            plt.savefig(save_path_with_labels, dpi=600, bbox_inches='tight',
                        facecolor='white', edgecolor='none', pad_inches=0.5)

            save_path_no_labels = self.output_dir / f'tsne_clustering_no_labels.{fmt}'
            plt.figure(fig2.number)
            plt.savefig(save_path_no_labels, dpi=600, bbox_inches='tight',
                        facecolor='white', edgecolor='none', pad_inches=0.5)

        plt.show()
        return (fig1, ax1), (fig2, ax2)

    def _create_tsne_plot(self, plot_data, perplexity, show_labels=True):
        """创建单个t-SNE聚类图（可选择是否显示标签）"""
        fig, ax = plt.subplots(figsize=(20, 14))
        fig.patch.set_facecolor('white')  # 设置整个图表背景为白色
        ax.set_facecolor('white')  # 设置绘图区域背景为白色

        # 移除顶部和右侧边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)

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

        # 绘制散点图
        for i, cluster_id in enumerate(plot_data['unique_clusters']):
            mask = plot_data['cluster_df']['cluster'] == cluster_id
            cluster_data = plot_data['cluster_df'][mask]
            color = plot_data['colors'][i % len(plot_data['colors'])]

            ax.scatter(
                cluster_data['tsne_1'], cluster_data['tsne_2'],
                c=color, label=f'Cluster {cluster_id + 1}',
                alpha=0.85, s=800, edgecolors='white', linewidth=4,
                zorder=5
            )

        # 添加标签（根据参数决定是否显示）
        if show_labels:
            label_manager = LabelManager(ax)
            label_manager.place_labels(
                plot_data['points'],
                plot_data['labels'],
                plot_data['point_colors'],
                fontsize=16
            )

        # 设置标题和标签
        ax.set_xlabel('t-SNE Component 1', fontsize=20, fontweight='bold', labelpad=20)
        ax.set_ylabel('t-SNE Component 2', fontsize=20, fontweight='bold', labelpad=20)
        ax.set_title(
            f't-SNE Clustering Results\nPerplexity = {perplexity}',
            fontsize=30, fontweight='bold', pad=30
        )

        # 图例放在坐标轴内
        legend = ax.legend(
            loc='upper right', frameon=True,
            fancybox=True, shadow=True, fontsize=16,
            title='Cluster Groups', title_fontsize=20,
            bbox_to_anchor=(0.95, 0.98),
            framealpha=0.95, edgecolor='#333333',
            borderpad=1.5, columnspacing=1.5, handletextpad=1
        )
        legend.get_title().set_fontweight('bold')

        ax.tick_params(axis='both', which='major', labelsize=16,
                       colors='#333333', width=2, length=8)

        plt.tight_layout()

        return fig, ax

    def _get_tsne_plot_data(self, perplexity):
        """获取t-SNE绘图数据，确保顺序一致"""
        if self.cluster_results is None:
            available_perplexities = list(self.tsne_results.keys())
            if available_perplexities:
                perplexity = available_perplexities[0]
            self.identify_clusters_from_tsne(perplexity=perplexity)

        cluster_df = self.cluster_results['cluster_df']
        unique_clusters = sorted([c for c in np.unique(cluster_df['cluster']) if c >= 0])

        # 专业的科学配色方案
        colors = [
            '#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6',
            '#1ABC9C', '#E67E22', '#34495E', '#8E44AD', '#16A085',
            '#F1C40F', '#E91E63', '#00BCD4', '#FF5722', '#607D8B'
        ]

        # 按聚类分组构建数据，保证顺序一致
        points = []
        point_colors = []
        labels = []

        for i, cluster_id in enumerate(unique_clusters):
            mask = cluster_df['cluster'] == cluster_id
            cluster_data = cluster_df[mask]
            color = colors[i % len(colors)]

            for _, row in cluster_data.iterrows():
                points.append([row['tsne_1'], row['tsne_2']])
                point_colors.append(color)
                labels.append(row['product'])

        return {
            'points': np.array(points),
            'point_colors': point_colors,
            'labels': labels,
            'cluster_df': cluster_df,
            'unique_clusters': unique_clusters,
            'colors': colors
        }

    def evaluate_cluster_consistency(self):
        """评估不同perplexity值下聚类的一致性"""
        if not self.tsne_results or len(self.tsne_results) < 2:
            print("⚠️ 需要至少2个perplexity值的t-SNE结果来评估一致性")
            return {}

        print("🔄 评估聚类一致性...")

        consistency_results = {}
        perplexity_values = list(self.tsne_results.keys())

        for perplexity in perplexity_values:
            print(f"  📊 分析perplexity={perplexity}...")

            # 为每个perplexity值计算聚类
            tsne_coords = self.tsne_results[perplexity]['coordinates']

            try:
                # 使用固定的聚类数进行比较
                n_clusters = min(8, len(self.products) - 1)
                kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
                labels = kmeans.fit_predict(tsne_coords)

                # 计算聚类质量指标
                if len(np.unique(labels)) > 1:
                    silhouette = silhouette_score(tsne_coords, labels)
                else:
                    silhouette = -1

                consistency_results[perplexity] = {
                    'labels': labels,
                    'silhouette_score': silhouette,
                    'n_clusters': len(np.unique(labels)),
                    'kl_divergence': self.tsne_results[perplexity]['kl_divergence']
                }

                print(f"    轮廓系数: {silhouette:.3f}")

            except Exception as e:
                print(f"    ❌ 计算失败: {e}")
                consistency_results[perplexity] = {
                    'labels': None,
                    'silhouette_score': -1,
                    'n_clusters': 0,
                    'kl_divergence': self.tsne_results[perplexity]['kl_divergence']
                }

        # 仅在非精简模式下保存详细一致性分析
        if not self.minimize_output:
            consistency_df = pd.DataFrame({
                'perplexity': perplexity_values,
                'silhouette_score': [consistency_results[p]['silhouette_score'] for p in perplexity_values],
                'n_clusters': [consistency_results[p]['n_clusters'] for p in perplexity_values],
                'kl_divergence': [consistency_results[p]['kl_divergence'] for p in perplexity_values]
            })
            consistency_df.to_csv(self.output_dir / 'cluster_consistency.csv', index=False)

        print(f"  ✅ 一致性分析完成")

        return consistency_results

    def find_common_targets(self):
        """寻找聚类内的共同基因目标"""
        if self.cluster_results is None:
            raise ValueError("❌ 请先执行聚类分析")

        print("🔄 寻找聚类内共同基因目标...")
        common_targets = {}

        for cluster_id, products in self.cluster_results['clusters'].items():
            if len(products) < 2:
                print(f"  ⚠️ 聚类 {cluster_id + 1} 只有1个产品，跳过共同目标分析")
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
                            genes = set(df[df['type'] == gene_type]['gene_id'].dropna().tolist())
                        else:
                            # 如果没有type列，假设都是OE
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

        # 保存共同目标到JSON文件（关键文件）
        if common_targets:
            common_targets_path = self.output_dir / 'common_targets.json'
            with open(common_targets_path, 'w', encoding='utf-8') as f:
                json.dump(common_targets, f, ensure_ascii=False, indent=2)
            print(f"  ✅ 共同目标分析完成，结果已保存: {common_targets_path}")
        else:
            print("  ⚠️ 未发现任何聚类内的共同基因目标")

        return common_targets

    def get_cluster_statistics(self):
        """获取详细的聚类统计信息"""
        if self.cluster_results is None:
            raise ValueError("❌ 请先执行聚类分析")

        print("📊 计算聚类统计信息...")
        stats_data = []

        for product in self.products:
            df = self.datasets[product]

            # 获取聚类ID
            cluster_row = self.cluster_results['cluster_df'][
                self.cluster_results['cluster_df']['product'] == product
                ]

            if not cluster_row.empty:
                cluster_id = cluster_row.iloc[0]['cluster']
                cluster_name = f"Cluster {cluster_id + 1}" if cluster_id >= 0 else "Noise"
            else:
                cluster_name = "Unknown"

            # 统计基因修饰类型
            total_genes = len(df)
            if 'type' in df.columns:
                type_counts = df['type'].value_counts()
                oe_count = type_counts.get('OE', 0)
                kd_count = type_counts.get('KD', 0)
                ko_count = type_counts.get('KO', 0)
            else:
                # 如果没有type列，假设都是OE
                oe_count = total_genes
                kd_count = ko_count = 0

            stats_data.append({
                'product': product,
                'cluster': cluster_name,
                'total_genes': total_genes,
                'OE_count': oe_count,
                'KD_count': kd_count,
                'KO_count': ko_count,
                'OE_ratio': oe_count / total_genes if total_genes > 0 else 0,
                'modification_diversity': len([x for x in [oe_count, kd_count, ko_count] if x > 0])
            })

        return pd.DataFrame(stats_data)

    def generate_comprehensive_report(self):
        """生成综合HTML报告"""
        print("📝 生成综合分析报告...")

        # 获取各种统计信息
        stats_df = self.get_cluster_statistics()
        common_targets = self.find_common_targets()
        consistency_results = self.evaluate_cluster_consistency()

        # 生成HTML报告
        report_content = self._create_enhanced_html_report(stats_df, common_targets, consistency_results)

        # 保存报告（关键文件）
        report_path = self.output_dir / 'analysis_report.html'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"  ✅ 综合分析报告已生成: {report_path}")

        # 生成分析摘要
        self._generate_analysis_summary()

        return report_path

    def _create_enhanced_html_report(self, stats_df, common_targets, consistency_results):
        """创建增强版HTML报告"""
        # 计算一些额外的统计信息
        total_genes_modified = sum(
            (self.gene_target_matrix != self.normal_weight).sum(axis=1)
        )

        avg_genes_per_product = total_genes_modified / len(self.products) if self.products else 0

        # 获取最佳perplexity（基于KL散度）
        best_perplexity = None
        if self.tsne_results:
            best_perplexity = min(self.tsne_results.keys(),
                                  key=lambda p: self.tsne_results[p]['kl_divergence'])

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Gene Engineering Target Cluster Analysis Report</title>
            <style>
                body {{ 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    margin: 0; padding: 0; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: #333;
                }}
                .container {{ 
                    max-width: 1400px; margin: 20px auto; 
                    background: white; border-radius: 15px; 
                    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                    overflow: hidden;
                }}
                .header {{
                    background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
                    color: white; padding: 30px; text-align: center;
                }}
                .content {{ padding: 30px; }}
                h1 {{ margin: 0; font-size: 2.5em; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }}
                h2 {{ 
                    color: #2c3e50; border-bottom: 3px solid #3498db; 
                    padding-bottom: 10px; margin-top: 40px;
                    font-size: 1.5em;
                }}
                h3 {{ color: #34495e; margin-top: 25px; }}
                .meta-info {{ 
                    background: #ecf0f1; padding: 15px; border-radius: 10px; 
                    margin: 20px 0; font-size: 0.95em;
                }}
                .metrics-grid {{ 
                    display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                    gap: 20px; margin: 25px 0; 
                }}
                .metric {{ 
                    background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
                    color: white; padding: 20px; border-radius: 10px; text-align: center;
                    box-shadow: 0 5px 15px rgba(52, 152, 219, 0.3);
                }}
                .metric-value {{ font-size: 2em; font-weight: bold; margin-bottom: 5px; }}
                .metric-label {{ font-size: 0.9em; opacity: 0.9; }}
                table {{ 
                    border-collapse: collapse; width: 100%; margin: 20px 0;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ 
                    background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
                    color: white; font-weight: bold;
                }}
                tr:nth-child(even) {{ background-color: #f8f9fa; }}
                tr:hover {{ background-color: #e3f2fd; }}
                .cluster-card {{ 
                    margin: 15px 0; padding: 20px; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; border-radius: 10px;
                    box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
                }}
                .cluster-title {{ font-size: 1.2em; font-weight: bold; margin-bottom: 10px; }}
                .products-list {{ 
                    background: rgba(255,255,255,0.1); padding: 10px; 
                    border-radius: 5px; margin-top: 10px;
                }}
                .parameter-grid {{
                    display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 15px; margin: 20px 0;
                }}
                .parameter-item {{
                    background: #f8f9fa; padding: 15px; border-radius: 8px;
                    border-left: 4px solid #3498db;
                }}
                .files-list {{ 
                    background: #f8f9fa; padding: 20px; border-radius: 10px; 
                    margin: 20px 0;
                }}
                .files-list ul {{ list-style-type: none; padding: 0; }}
                .files-list li {{ 
                    padding: 8px 0; border-bottom: 1px solid #dee2e6;
                    position: relative; padding-left: 25px;
                }}
                .files-list li:before {{
                    content: "📄"; position: absolute; left: 0; top: 8px;
                }}
                .footer {{
                    background: #2c3e50; color: white; padding: 20px; text-align: center;
                    font-style: italic;
                }}
                .highlight {{ 
                    background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
                    color: white; padding: 2px 6px; border-radius: 4px;
                }}
                .warning {{ 
                    background: #fff3cd; border: 1px solid #ffeaa7; 
                    color: #856404; padding: 10px; border-radius: 5px; margin: 10px 0;
                }}
                .success {{ 
                    background: #d4edda; border: 1px solid #c3e6cb; 
                    color: #155724; padding: 10px; border-radius: 5px; margin: 10px 0;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🧬 Gene Engineering Target Cluster Analysis</h1>
                    <p style="font-size: 1.2em; margin-top: 10px;">
                        Comprehensive Analysis Based on t-SNE Methodology
                    </p>
                </div>

                <div class="content">
                    <div class="meta-info">
                        <strong>📅 Analysis Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
                        <strong>🔬 Method:</strong> t-SNE based clustering | 
                        <strong>📊 Output Mode:</strong> {'Minimized' if self.minimize_output else 'Full'}
                    </div>

                    <h2>📈 Executive Summary</h2>
                    <div class="metrics-grid">
                        <div class="metric">
                            <div class="metric-value">{len(self.products)}</div>
                            <div class="metric-label">Products Analyzed</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{len(self.all_genes)}</div>
                            <div class="metric-label">Unique Genes</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{len(self.cluster_results['clusters']) if self.cluster_results else 0}</div>
                            <div class="metric-label">Clusters Identified</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{len(self.tsne_results)}</div>
                            <div class="metric-label">Perplexity Values Tested</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{avg_genes_per_product:.1f}</div>
                            <div class="metric-label">Avg. Modified Genes/Product</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{best_perplexity or 'N/A'}</div>
                            <div class="metric-label">Optimal Perplexity</div>
                        </div>
                    </div>

                    <h2>⚙️ Analysis Parameters</h2>
                    <div class="parameter-grid">
                        <div class="parameter-item">
                            <strong>Gene Engineering Weights:</strong><br>
                            • Overexpression (OE): <span class="highlight">{self.oe_weight}</span><br>
                            • Knockdown (KD): <span class="highlight">{self.kd_weight}</span><br>
                            • Knockout (KO): <span class="highlight">{self.ko_weight}</span><br>
                            • Normal: <span class="highlight">{self.normal_weight}</span>
                        </div>
                        <div class="parameter-item">
                            <strong>t-SNE Parameters:</strong><br>
                            • Perplexity Range: <span class="highlight">{list(self.perplexity_range)[0]} - {list(self.perplexity_range)[-1]}</span><br>
                            • Components: <span class="highlight">2</span><br>
                            • Random State: <span class="highlight">{self.random_state}</span><br>
                            • Max Iterations: <span class="highlight">1000</span>
                        </div>
                        <div class="parameter-item">
                            <strong>Analysis Method:</strong><br>
                            • Clustering: <span class="highlight">{self.cluster_results.get('method', 'N/A') if self.cluster_results else 'N/A'}</span><br>
                            • Distance Metric: <span class="highlight">Euclidean</span><br>
                            • Evaluation: <span class="highlight">Silhouette Score</span>
                        </div>
                    </div>
        """

        # 添加聚类一致性分析
        if consistency_results:
            html_content += f"""
                    <h2>🔄 Clustering Consistency Analysis</h2>
                    <div class="success">
                        <strong>✅ Consistency evaluation completed across {len(consistency_results)} perplexity values</strong>
                    </div>
                    <table>
                        <tr>
                            <th>Perplexity</th>
                            <th>Silhouette Score</th>
                            <th>Number of Clusters</th>
                            <th>KL Divergence</th>
                            <th>Quality Assessment</th>
                        </tr>
            """

            for perplexity, result in consistency_results.items():
                silhouette = result['silhouette_score']
                if silhouette > 0.5:
                    quality = "🟢 Excellent"
                elif silhouette > 0.3:
                    quality = "🟡 Good"
                elif silhouette > 0:
                    quality = "🟠 Fair"
                else:
                    quality = "🔴 Poor"

                html_content += f"""
                        <tr>
                            <td>{perplexity}</td>
                            <td>{silhouette:.3f}</td>
                            <td>{result['n_clusters']}</td>
                            <td>{result['kl_divergence']:.3f}</td>
                            <td>{quality}</td>
                        </tr>
                """

            html_content += "</table>"

        # 添加聚类统计表
        html_content += f"""
                    <h2>📊 Detailed Cluster Statistics</h2>
                    {stats_df.to_html(index=False, classes='table', escape=False)}
        """

        # 添加聚类详情
        if self.cluster_results:
            html_content += "<h2>🎯 Cluster Composition</h2>"
            for cluster_id, products in self.cluster_results['clusters'].items():
                html_content += f"""
                <div class="cluster-card">
                    <div class="cluster-title">Cluster {cluster_id + 1} ({len(products)} products)</div>
                    <div class="products-list">
                        <strong>Products:</strong> {', '.join(products)}
                    </div>
                </div>
                """

        # 添加共同基因目标
        if common_targets:
            html_content += "<h2>🔗 Common Gene Targets within Clusters</h2>"
            for cluster_name, targets in common_targets.items():
                html_content += f"""
                <div class="cluster-card">
                    <div class="cluster-title">{cluster_name}</div>
                """
                for gene_type, genes in targets.items():
                    if genes:
                        html_content += f"""
                        <div style="margin: 10px 0;">
                            <strong>{gene_type} Genes ({len(genes)}):</strong><br>
                            <div class="products-list">{', '.join(genes)}</div>
                        </div>
                        """
                html_content += "</div>"
        else:
            html_content += """
                    <h2>🔗 Common Gene Targets within Clusters</h2>
                    <div class="warning">
                        <strong>⚠️ No common gene targets found within clusters.</strong><br>
                        This could indicate high diversity in gene engineering strategies within clusters.
                    </div>
            """

        # 添加生成的文件列表
        if self.minimize_output:
            html_content += f"""
                    <h2>📁 Generated Analysis Files (Minimized)</h2>
                    <div class="files-list">
                        <h3>Key Files Only:</h3>
                        <ul>
                            <li><strong>cluster_results.csv</strong> - Final clustering assignments</li>
                            <li><strong>tsne_clustering_with_labels.png</strong> - Clustering visualization with labels</li>
                            <li><strong>tsne_clustering_no_labels.png</strong> - Clustering visualization without labels</li>
                            <li><strong>analysis_report.html</strong> - This comprehensive report</li>
                            <li><strong>common_targets.json</strong> - Common gene targets by cluster (if found)</li>
                            <li><strong>analysis_summary.json</strong> - Machine-readable analysis summary</li>
                        </ul>
                    </div>
            """
        else:
            html_content += f"""
                    <h2>📁 Generated Analysis Files (Full)</h2>
                    <div class="files-list">
                        <h3>Core Analysis Files:</h3>
                        <ul>
                            <li><strong>gene_target_matrix.csv</strong> - Gene target weight matrix</li>
                            <li><strong>distance_matrix.csv</strong> - Product distance matrix</li>
                            <li><strong>cluster_results.csv</strong> - Final clustering assignments</li>
                            <li><strong>cluster_consistency.csv</strong> - Consistency analysis across perplexity values</li>
                        </ul>

                        <h3>t-SNE Results:</h3>
                        <ul>
                            <li><strong>tsne_results/</strong> - Individual t-SNE results for each perplexity value</li>
                            <li><strong>tsne_clustering_with_labels.png</strong> - Clustering visualization with labels</li>
                            <li><strong>tsne_clustering_no_labels.png</strong> - Clustering visualization without labels</li>
                        </ul>

                        <h3>Reports:</h3>
                        <ul>
                            <li><strong>analysis_report.html</strong> - This detailed report</li>
                            <li><strong>common_targets.json</strong> - Common gene targets by cluster</li>
                            <li><strong>analysis_summary.json</strong> - Machine-readable analysis summary</li>
                        </ul>
                    </div>
            """

        html_content += f"""
                </div>

                <div class="footer">
                    <p>Analysis completed with {len(self.products)} products, {len(self.all_genes)} genes, and {len(self.tsne_results)} perplexity configurations.</p>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            </div>
        </body>
        </html>
        """

        return html_content

    def _generate_analysis_summary(self):
        """生成机器可读的分析摘要（修复JSON序列化问题）"""
        best_kl_perplexity = min(self.tsne_results.keys(),
                                 key=lambda p: self.tsne_results[p]['kl_divergence']) if self.tsne_results else None

        # 评估聚类一致性
        consistency_results = self.evaluate_cluster_consistency()
        best_silhouette_perplexity = None
        if consistency_results:
            valid_scores = {p: r['silhouette_score'] for p, r in consistency_results.items()
                            if r['silhouette_score'] > 0}
            if valid_scores:
                best_silhouette_perplexity = max(valid_scores.keys(), key=lambda p: valid_scores[p])

        # 创建摘要报告
        summary = {
            'analysis_metadata': {
                'date': datetime.now().isoformat(),
                'method': 't-SNE based clustering',
                'n_products': len(self.products),
                'n_genes': len(self.all_genes),
                'n_clusters': len(self.cluster_results['clusters']) if self.cluster_results else 0
            },
            'perplexity_analysis': {
                'tested_range': list(self.tsne_results.keys()),
                'best_kl_divergence': {
                    'perplexity': best_kl_perplexity,
                    'value': float(
                        self.tsne_results[best_kl_perplexity]['kl_divergence']) if best_kl_perplexity else None
                },
                'best_clustering_quality': {
                    'perplexity': best_silhouette_perplexity,
                    'silhouette_score': float(consistency_results[best_silhouette_perplexity][
                                                  'silhouette_score']) if best_silhouette_perplexity else None
                }
            },
            'cluster_results': {
                'n_clusters': len(self.cluster_results['clusters']) if self.cluster_results else 0,
                'cluster_sizes': {f'cluster_{k + 1}': len(v) for k, v in
                                  self.cluster_results['clusters'].items()} if self.cluster_results else {},
                'common_targets_found': len(self.find_common_targets()) > 0
            }
        }

        # 转换为可序列化格式
        summary = convert_to_serializable(summary)

        # 保存摘要（关键文件）
        summary_path = self.output_dir / 'analysis_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"✅ 分析摘要已保存: {summary_path}")

        # 打印关键发现
        print("\n📊 关键发现:")
        if best_kl_perplexity:
            print(f"   🎯 最佳KL散度困惑度: {best_kl_perplexity}")
        if best_silhouette_perplexity:
            print(f"   🎯 最佳聚类质量困惑度: {best_silhouette_perplexity}")
        if self.cluster_results:
            print(f"   📈 发现 {len(self.cluster_results['clusters'])} 个聚类")

        return summary

    def run_complete_analysis(self, perplexity_range=None, target_perplexity=15, n_clusters=8):
        """运行完整的分析流程"""
        print("🚀 开始完整分析流程...")

        # 步骤1: 基因目标向量化
        print("📊 步骤1: 基因目标向量化...")
        self.vectorize_gene_targets()

        # 步骤2: 计算距离矩阵
        print("📏 步骤2: 计算距离矩阵...")
        self.calculate_distance_matrix()

        # 步骤3: t-SNE分析
        print("🔄 步骤3: 执行t-SNE分析...")
        self.perform_tsne_analysis(perplexity_range)

        # 步骤4: 比较不同perplexity值（新增）
        print("📊 步骤4: 比较不同perplexity值...")
        if len(self.tsne_results) > 1:
            # 绘制perplexity比较图
            if not self.minimize_output:
                self.plot_perplexity_comparison()
            # 绘制评估指标图
            self.plot_perplexity_metrics()

        # 步骤5: 聚类识别
        print("🎯 步骤5: 识别聚类...")
        self.identify_clusters_from_tsne(
            perplexity=target_perplexity,
            n_clusters=n_clusters
        )

        # 步骤6: 生成可视化图表
        print("📈 步骤6: 生成可视化图表...")
        self.plot_optimized_tsne_clusters(perplexity=target_perplexity)

        # 步骤7: 生成HTML报告
        print("📝 步骤7: 生成综合报告...")
        self.generate_comprehensive_report()

        print("✅ 完整分析流程完成!")

        # 列出生成的关键文件
        if self.minimize_output:
            print("\n📂 生成的关键文件（精简模式）:")
            print(f"   • {self.output_dir}/cluster_results.csv - 聚类结果")
            print(f"   • {self.output_dir}/tsne_clustering_with_labels.png - 带标签的可视化图表")
            print(f"   • {self.output_dir}/tsne_clustering_no_labels.png - 不带标签的可视化图表")
            print(f"   • {self.output_dir}/perplexity_metrics.png - perplexity评估图")
            print(f"   • {self.output_dir}/analysis_report.html - 综合报告")
            print(f"   • {self.output_dir}/analysis_summary.json - 分析摘要")
            if (self.output_dir / 'common_targets.json').exists():
                print(f"   • {self.output_dir}/common_targets.json - 共同基因目标")
        else:
            print("\n📂 生成的所有文件（完整模式）:")
            print(f"   • {self.output_dir}/cluster_results.csv - 聚类结果")
            print(f"   • {self.output_dir}/tsne_clustering_with_labels.png - 带标签的可视化图表")
            print(f"   • {self.output_dir}/tsne_clustering_no_labels.png - 不带标签的可视化图表")
            print(f"   • {self.output_dir}/perplexity_comparison.png - perplexity比较图")
            print(f"   • {self.output_dir}/perplexity_metrics.png - perplexity评估图")
            print(f"   • {self.output_dir}/analysis_report.html - 综合报告")
            print(f"   • {self.output_dir}/gene_target_matrix.csv - 基因目标矩阵")
            print(f"   • {self.output_dir}/distance_matrix.csv - 距离矩阵")
            print(f"   • {self.output_dir}/cluster_consistency.csv - 一致性分析")
            print(f"   • {self.output_dir}/tsne_results/ - 详细t-SNE结果")
            print(f"   • {self.output_dir}/analysis_summary.json - 分析摘要")
            if (self.output_dir / 'common_targets.json').exists():
                print(f"   • {self.output_dir}/common_targets.json - 共同基因目标")

        return self.cluster_results


class GeneClusteringRunner:
    """基因聚类分析运行器 - 提供灵活的调用接口"""

    def __init__(self):
        self.analyzer = None

    def run_analysis_from_directory(self,
                                    data_directory,
                                    file_pattern="*.csv",
                                    output_dir="my_analysis_results",
                                    oe_weight=5,
                                    kd_weight=-1,
                                    ko_weight=-2,
                                    normal_weight=0.0,
                                    target_perplexity=15,
                                    n_clusters=8,
                                    minimize_output=True):
        """
        从指定目录运行聚类分析

        Args:
            minimize_output: 如果为True，只生成关键文件
        """

        print(f"🔍 从目录加载数据: {data_directory}")
        print(f"📁 文件模式: {file_pattern}")
        print(f"📂 输出目录: {output_dir}")
        print(f"💾 输出模式: {'精简' if minimize_output else '完整'}")

        # 检查目录是否存在
        if not os.path.exists(data_directory):
            raise FileNotFoundError(f"数据目录不存在: {data_directory}")

        # 创建分析器
        self.analyzer = OptimizedGeneTargetTSNEAnalyzer(
            oe_weight=oe_weight,
            kd_weight=kd_weight,
            ko_weight=ko_weight,
            normal_weight=normal_weight,
            output_dir=output_dir,
            minimize_output=minimize_output
        )

        # 加载数据
        self.analyzer.load_data_from_files(
            file_pattern=file_pattern,
            data_dir=data_directory
        )

        print(f"✅ 成功加载 {len(self.analyzer.products)} 个产品数据")

        # 运行完整分析
        results = self.analyzer.run_complete_analysis(
            perplexity_range=range(2, min(26, len(self.analyzer.products))),
            target_perplexity=target_perplexity,
            n_clusters=n_clusters
        )

        return results


def quick_analysis(data_dir, output_dir="quick_results", minimize_output=True, **kwargs):
    """快速运行分析的便捷函数"""
    runner = GeneClusteringRunner()
    return runner.run_analysis_from_directory(
        data_directory=data_dir,
        output_dir=output_dir,
        minimize_output=minimize_output, **kwargs
    )
