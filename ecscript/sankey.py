import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)

# ─── 配置 ──────────────────────────────────────────────────────────────────
BATCH_DIR      = "./ecfb_results/batch_20260404_010743"
EXCEL_LIT_FILE = "Cgproduct.xlsx"
OUTPUT_PNG     = "./conflict_sankey_ecfb.png"
OUTPUT_CSV     = "./conflict_genes_trace_ecfb.csv"

PRODUCT_COL  = "Product"
GENE_COL_LIT = "Gene in 14067"
STRATEGY_COL = "Strategy"

# ─── 工具函数 ────────────────────────────────────────────────────────────────
def extract_cpd_code(text):
    m = re.search(r'cpd(\d{5})', str(text).lower())
    return f"cpd{m.group(1)}" if m else None

def normalize_strategy(s):
    if pd.isna(s): return None
    s = str(s).strip().upper()
    if s in ['OE', 'UP']:         return 'OE'
    if s in ['KD', 'KO', 'DOWN']: return 'KD'
    return None

def split_genes(gene_str):
    if pd.isna(gene_str): return []
    parts = re.split(r'\s+and\s+|;|,', str(gene_str).strip(), flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]

def find_gene_col(df):
    for col in df.columns:
        if any(k in col.lower() for k in ['gene_id', 'gene', 'target']):
            return col
    return df.columns[0]

def find_type_col(df):
    for col in df.columns:
        if any(k in col.lower() for k in ['regulation_type', 'type', 'strategy', 'mod']):
            return col
    return None

# ─── 冲突类型分类 ─────────────────────────────────────────────────────────────
def classify_conflict(row):
    is_conf = str(row.get('is_conflict', '')).strip().lower()
    if is_conf not in ('true', '1', 'yes'):
        return None

    oe = str(row.get('oe_sources', '') or '').strip()
    kd = str(row.get('kd_sources', '') or '').strip()

    oe_ecb = 'ecBridge' in oe;  oe_efs = 'ecFSEOF' in oe
    kd_ecb = 'ecBridge' in kd;  kd_efs = 'ecFSEOF' in kd

    cross = ((oe_ecb and not oe_efs) and (kd_efs and not kd_ecb)) or \
            ((oe_efs and not oe_ecb) and (kd_ecb and not kd_efs))
    if cross:          return 'cross_tool'
    if oe_ecb and kd_ecb: return 'ecb_internal'
    if oe_efs and kd_efs: return 'efs_internal'
    return None

# ─── 加载文献 ────────────────────────────────────────────────────────────────
def load_literature(excel_file):
    df = pd.read_excel(excel_file, sheet_name=0, header=0)
    df[PRODUCT_COL] = df[PRODUCT_COL].ffill()
    lit_dict = {}
    for product_name, grp in df.groupby(PRODUCT_COL):
        cpd = extract_cpd_code(str(product_name).strip())
        if cpd is None: continue
        targets = {}
        for _, row in grp.iterrows():
            strat = normalize_strategy(row.get(STRATEGY_COL))
            if strat is None: continue
            for g in split_genes(row.get(GENE_COL_LIT)):
                targets[g] = strat
        if targets:
            lit_dict[cpd] = targets
    print(f"[文献] 共 {len(lit_dict)} 个产品: {sorted(lit_dict.keys())}")
    return lit_dict

# ─── 追踪冲突基因 ─────────────────────────────────────────────────────────────
def collect_conflict_data(batch_dir, lit_dict):
    rows = []
    cpd_dirs = sorted(glob.glob(os.path.join(batch_dir, 'cpd*')))
    if not cpd_dirs:
        print(f"[警告] 在 {batch_dir} 下未找到 cpd* 子目录")
        return pd.DataFrame()

    for cpd_folder in cpd_dirs:
        cpd = os.path.basename(cpd_folder)

        raw_files  = (glob.glob(os.path.join(cpd_folder, 'ecfb', '*_ecfb_merged_input.csv')) or
                      glob.glob(os.path.join(cpd_folder, '*_ecfb_merged_input.csv')))
        filt_files = (glob.glob(os.path.join(cpd_folder, 'ecfb', '*_singletest_filtered.csv')) or
                      glob.glob(os.path.join(cpd_folder, '*_singletest_filtered.csv')))

        if not raw_files: continue
        try:
            raw_df = pd.read_csv(raw_files[0])
        except Exception as e:
            print(f"  [跳过] {cpd}: {e}"); continue

        gene_col = find_gene_col(raw_df)

        filt_map = {}
        if filt_files:
            try:
                filt_df = pd.read_csv(filt_files[0])
                fg, ft = find_gene_col(filt_df), find_type_col(filt_df)
                for _, row in filt_df.iterrows():
                    g = str(row.get(fg, '')).strip()
                    t = normalize_strategy(row.get(ft)) if ft else None
                    if g and t: filt_map[g] = t
            except Exception as e:
                print(f"  [警告] {cpd} filtered: {e}")

        lit_targets = lit_dict.get(cpd, {})

        for _, row in raw_df.iterrows():
            conflict_type = classify_conflict(row)
            if conflict_type is None: continue
            gene = str(row.get(gene_col, '')).strip()
            if not gene: continue

            if gene in filt_map:
                fate    = filt_map[gene]
                lit_dir = lit_targets.get(gene)
                if gene in lit_targets:
                    # ✅ 关键：只有方向一致才算 consistent
                    lit_status = 'consistent' if lit_dir == fate else 'inconsistent'
                else:
                    lit_status = 'not_in_lit'
            else:
                fate, lit_status = 'removed', 'removed'   # 统一归入 removed

            rows.append({'cpd': cpd, 'gene': gene,
                         'conflict_type': conflict_type,
                         'fate': fate, 'lit_status': lit_status})

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"[追踪结果] 共 {len(df)} 条记录，已保存: {OUTPUT_CSV}")
        print(df.groupby(['conflict_type', 'lit_status']).size().to_string())
    return df

# ─── 贝塞尔流带 & 节点绘制 ────────────────────────────────────────────────────
def _flow(ax, x0, yb0, yt0, x1, yb1, yt1, color, alpha=0.42):
    if yt0 <= yb0 or yt1 <= yb1: return
    xm = (x0 + x1) / 2
    verts = [(x0,yb0),(xm,yb0),(xm,yb1),(x1,yb1),
             (x1,yt1),(xm,yt1),(xm,yt0),(x0,yt0),(x0,yb0)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
    ax.add_patch(mpatches.PathPatch(
        Path(verts, codes), facecolor=color, alpha=alpha,
        edgecolor='none', zorder=2))

def _node(ax, x, yb, h, nw, color, label, n, side='right'):
    if h <= 0: return
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, yb), nw, h, boxstyle="round,pad=0.02",
        facecolor=color, edgecolor='white', linewidth=1.4, zorder=3))
    tx = x + nw + 0.15 if side == 'right' else x - 0.15
    ax.text(tx, yb + h / 2, f'{label}\nn={n}',
            ha='left' if side == 'right' else 'right',
            va='center', fontsize=9, color='#2c2c2a', zorder=5,
            linespacing=1.5, fontweight='medium')

# ─── 主绘图函数（两列桑基图）─────────────────────────────────────────────────
def build_sankey(df, output_file=OUTPUT_PNG):
    if df.empty:
        print("[错误] 无数据"); return

    # ── 分类定义 ─────────────────────────────────────────────────────────
    # 左列：冲突类型（上→下）
    CT   = ['ecb_internal', 'cross_tool', 'efs_internal']
    # 右列：最终结果（上→下）；removed 排最下
    LIT  = ['consistent', 'inconsistent', 'not_in_lit', 'removed']

    CT_LABELS  = {
        'ecb_internal': 'ecBridge\ninternal conflict',
        'cross_tool':   'Cross-tool conflict\n(ecBridge ↔ ecFSEOF)',
        'efs_internal': 'ecFSEOF\ninternal conflict',
    }
    LIT_LABELS = {
        'consistent':   'Consistent\nwith literature',
        'inconsistent': 'Inconsistent\nwith literature',
        'not_in_lit':   'Not in\nliterature',
        'removed':      'Removed\nby filter',
    }
    CT_COLORS  = {
        'ecb_internal': '#97c459',
        'cross_tool':   '#f0997b',
        'efs_internal': '#378add',
    }
    LIT_COLORS = {
        'consistent':   '#1d9e75',
        'inconsistent': '#e24b4a',
        'not_in_lit':   '#ef9f27',
        'removed':      '#b4b2a9',
    }

    # ── 计数 ─────────────────────────────────────────────────────────────
    cnt = {}
    for ct in CT:
        for lit in LIT:
            cnt[(ct, lit)] = len(df[(df.conflict_type == ct) & (df.lit_status == lit)])

    ct_totals  = {ct:  sum(cnt[(ct, l)]  for l in LIT) for ct in CT}
    lit_totals = {lit: sum(cnt[(ct, lit)] for ct in CT) for lit in LIT}
    n_total    = sum(ct_totals.values())
    if n_total == 0:
        print("[错误] 冲突基因数为0"); return

    # ── 布局 ─────────────────────────────────────────────────────────────
    H   = 8.0
    GAP = 0.25
    NW  = 0.40
    X1, X2 = 1.2, 6.8          # 左列节点 x，右列节点 x
    sc  = H / n_total           # 每个基因对应的高度

    def positions(items, totals):
        """从顶部往下排列节点，返回 {item: (y_bottom, height)}"""
        pos = {}; y = H
        for item in items:
            h = totals[item] * sc
            y -= h
            pos[item] = (y, h)
            y -= GAP
        return pos

    ct_pos  = positions(CT,  ct_totals)
    lit_pos = positions(LIT, lit_totals)

    # ── 绘图 ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(-0.3, H + 0.3)
    ax.axis('off')

    # 流带：左 → 右，按冲突类型着色
    src_cursor = {ct:  ct_pos[ct][0]  + ct_pos[ct][1]  for ct in CT}
    tgt_cursor = {lit: lit_pos[lit][0] + lit_pos[lit][1] for lit in LIT}

    for ct in CT:
        for lit in LIT:
            h = cnt[(ct, lit)] * sc
            if h <= 0: continue
            sb, st = src_cursor[ct]  - h, src_cursor[ct]
            tb, tt = tgt_cursor[lit] - h, tgt_cursor[lit]
            _flow(ax, X1 + NW, sb, st, X2, tb, tt, CT_COLORS[ct])
            src_cursor[ct]  -= h
            tgt_cursor[lit] -= h

    # 左列节点（标注在左侧）
    for ct in CT:
        yb, h = ct_pos[ct]
        _node(ax, X1, yb, h, NW, CT_COLORS[ct], CT_LABELS[ct], ct_totals[ct], side='left')

    # 右列节点（标注在右侧）
    for lit in LIT:
        yb, h = lit_pos[lit]
        _node(ax, X2, yb, h, NW, LIT_COLORS[lit], LIT_LABELS[lit], lit_totals[lit], side='right')



    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"[桑基图] 已保存: {output_file}")
    plt.show()

# ─── 入口 ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    lit_dict = load_literature(EXCEL_LIT_FILE)
    df       = collect_conflict_data(BATCH_DIR, lit_dict)
    build_sankey(df, OUTPUT_PNG)