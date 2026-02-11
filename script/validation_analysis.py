import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import re

# Publication-ready font settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 15
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

STRATEGY_MAPPING = {'OE': ['OE'], 'KD': ['KD'], 'KO': ['KD']}


def extract_cpd_from_product(product_name):
    match = re.search(r'cpd\d{5}', product_name)
    return match.group() if match else None


def read_cgproduct_file(filepath):
    try:
        df = pd.read_excel(filepath)
        if 'Gene in 14067' not in df.columns:
            return None

        df['Product'] = df['Product'].ffill()
        product_targets = {}

        for product in df['Product'].unique():
            if pd.notna(product):
                cpd_id = extract_cpd_from_product(product)
                if not cpd_id:
                    continue

                product_df = df[df['Product'] == product]
                product_targets[cpd_id] = {'product_name': product, 'strategy_targets': {}}

                for strategy in product_df['Strategy'].dropna().unique():
                    strategy = str(strategy).strip().upper()
                    strategy_df = product_df[product_df['Strategy'] == strategy]
                    targets = set()

                    for _, row in strategy_df.iterrows():
                        gene_id = row.get('Gene in 14067')
                        if pd.notna(gene_id) and str(gene_id).strip():
                            gene_str = str(gene_id).strip()
                            if ';' in gene_str:
                                targets.update(g.strip() for g in gene_str.split(';') if g.strip())
                            elif ',' in gene_str:
                                targets.update(g.strip() for g in gene_str.split(',') if g.strip())
                            else:
                                targets.add(gene_str)

                    if targets:
                        product_targets[cpd_id]['strategy_targets'][strategy] = targets

        return product_targets
    except Exception:
        return None


def read_prediction_csv(csv_file):
    try:
        df = pd.read_csv(csv_file)
        strategy_targets = {}

        gene_col = next((col for col in ['gene_id', 'gene_name', 'target', 'Target', 'Gene'] if col in df.columns),
                        None)
        strategy_col = next((col for col in ['type', 'strategy', 'Strategy', 'modification'] if col in df.columns),
                            None)

        if gene_col and strategy_col:
            for strategy, strategy_group in df.groupby(strategy_col):
                if pd.notna(strategy) and str(strategy).strip():
                    strategy = str(strategy).strip().upper()
                    targets = set(strategy_group[gene_col].dropna().astype(str).str.strip())
                    if targets:
                        strategy_targets[strategy] = targets

        return strategy_targets
    except Exception:
        return {}


def combine_targets_by_strict_strategy_rules(literature_strategy_targets, csv_strategy_targets):
    combined_literature_targets = set()
    combined_csv_targets = set()

    for lit_strategy, lit_targets in literature_strategy_targets.items():
        if not lit_targets:
            continue

        allowed_csv_strategies = STRATEGY_MAPPING.get(lit_strategy, [])
        has_matching_csv = False

        for csv_strategy in allowed_csv_strategies:
            if csv_strategy in csv_strategy_targets:
                has_matching_csv = True
                combined_csv_targets.update(csv_strategy_targets[csv_strategy])

        if has_matching_csv:
            combined_literature_targets.update(lit_targets)

    return combined_literature_targets, combined_csv_targets


def analyze_validation_data(cgproduct_file, folder_paths, custom_names, filename_patterns=None):
    product_targets = read_cgproduct_file(cgproduct_file)
    if not product_targets:
        return None, None

    if filename_patterns is None:
        filename_patterns = ['targets_{cpd_id}.csv'] * len(folder_paths)

    if len(filename_patterns) != len(folder_paths):
        raise ValueError("Number of filename_patterns must match number of folder_paths")

    hit_rates_data = []
    coverage_rates_data = []

    for cpd_id, target_info in product_targets.items():
        hit_rates = []
        coverage_rates = []

        for i, (folder_path, pattern) in enumerate(zip(folder_paths, filename_patterns)):
            filename = pattern.format(cpd_id=cpd_id)
            csv_file = Path(folder_path) / filename

            if csv_file.exists():
                csv_strategy_targets = read_prediction_csv(csv_file)

                if csv_strategy_targets:
                    literature_strategy_targets = target_info['strategy_targets']
                    combined_lit_targets, combined_csv_targets = combine_targets_by_strict_strategy_rules(
                        literature_strategy_targets, csv_strategy_targets
                    )

                    if combined_lit_targets and combined_csv_targets:
                        overlap_targets = combined_lit_targets.intersection(combined_csv_targets)
                        overlap_count = len(overlap_targets)
                        model_count = len(combined_csv_targets)
                        literature_count = len(combined_lit_targets)

                        hit_rate = (overlap_count / model_count * 100) if model_count > 0 else 0
                        coverage_rate = (overlap_count / literature_count * 100) if literature_count > 0 else 0
                    else:
                        hit_rate = coverage_rate = 0
                else:
                    hit_rate = coverage_rate = 0
            else:
                hit_rate = coverage_rate = 0

            hit_rates.append(hit_rate)
            coverage_rates.append(coverage_rate)

        if any(rate > 0 for rate in hit_rates):
            hit_rates_data.append(hit_rates)
            coverage_rates_data.append(coverage_rates)

    return np.array(hit_rates_data), np.array(coverage_rates_data)


def create_publication_boxplots(cgproduct_file, folder_paths, custom_names,
                                filename_patterns=None, save_dir='validation_plots'):
    hit_rates_array, coverage_rates_array = analyze_validation_data(
        cgproduct_file, folder_paths, custom_names, filename_patterns
    )

    if hit_rates_array is None:
        return None

    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = ['#3498DB', '#F39C12', '#2ECC71', '#E74C3C'][:len(custom_names)]

    # Plot Hit Rates
    hit_rates_by_method = [hit_rates_array[:, i] for i in range(len(custom_names))]
    bp1 = ax1.boxplot(hit_rates_by_method, labels=custom_names, patch_artist=True,
                      showmeans=True, meanline=True)

    # Plot Coverage Rates
    coverage_rates_by_method = [coverage_rates_array[:, i] for i in range(len(custom_names))]
    bp2 = ax2.boxplot(coverage_rates_by_method, labels=custom_names, patch_artist=True,
                      showmeans=True, meanline=True)

    for bp in [bp1, bp2]:
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
        for meanline in bp['means']:
            meanline.set_color('red')
            meanline.set_linewidth(2)
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(2)

    # Add data points and refine axes
    for ax, data_by_method in zip([ax1, ax2], [hit_rates_by_method, coverage_rates_by_method]):
        for i, data in enumerate(data_by_method):
            y = data
            x = np.random.normal(i + 1, 0.04, size=len(y))
            ax.scatter(x, y, alpha=0.6, s=25, color='darkred', edgecolors='black', linewidth=0.3)

        # --- 修复后的坐标轴逻辑 ---
        for edge in ['left', 'bottom']:
            ax.spines[edge].set_visible(True)
            ax.spines[edge].set_color('black')
            ax.spines[edge].set_linewidth(1.5)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 显式开启刻度线
        ax.tick_params(axis='both', which='major', direction='out', length=6,
                       width=1.5, colors='black', bottom=True, left=True)

        ax.set_xticklabels(custom_names, rotation=45, ha='right', fontweight='bold')
        ax.grid(False)

    ax1.set_ylabel('Ratio of Validated targets', fontweight='bold')
    ax1.set_title('Validated targets Distribution', fontweight='bold')
    ax2.set_ylabel('Literature Coverage (%)', fontweight='bold')
    ax2.set_title('Literature Coverage Distribution', fontweight='bold')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightgray', alpha=0.7, edgecolor='black', label='Interquartile range'),
        plt.Line2D([0], [0], color='black', linewidth=2, label='Median'),
        plt.Line2D([0], [0], color='red', linewidth=2, label='Mean'),
        plt.scatter([], [], color='darkred', s=25, alpha=0.6, label='Data Points')
    ]
    fig.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.06, 0.93), fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'validation_boxplots.png')
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.show()

    return fig


if __name__ == "__main__":
    # 该部分仅在脚本被直接运行时触发，调用代码请在外部脚本执行
    pass