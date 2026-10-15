#!/usr/bin/env python
# Author: Zhang xinyu
# Date: 2026-10-15
"""Compare ecBridge, ecFSEOF, and ecFB batch predictions with literature.

Hit-rate denominators include all unique genes, including BOTH/conflict.
Matches require an OE/KD gene-direction pair agreeing with literature.
Coverage divides matches by the number of literature gene-direction pairs."""

import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family':        'Arial',
    'axes.unicode_minus': False,
    'font.size': 12, 'axes.titlesize': 16, 'axes.labelsize': 13,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11,
})


def _extract_cpd(text):
    m = re.search(r'cpd(\d{5})', str(text).lower())
    return f"cpd{m.group(1)}" if m else str(text).strip()


def _norm_reg(v):
    """Normalize directions to OE/KD; return None for ambiguous or unknown values."""
    if pd.isna(v):
        return None
    v = str(v).strip().upper()
    if v in ('KD', 'KO', 'DOWN', 'POS_TO_NEG'):
        return 'KD'
    if v in ('OE', 'UP', 'NEG_TO_POS'):
        return 'OE'
    return None


def _detect_col(df, aliases):
    low = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a.lower() in low:
            return low[a.lower()]
    return None


def _is_rxn_id(v):
    return bool(re.match(r'^rxn\d+', v.strip().lower()))


def load_rxn_gene_map(map_file):
    """Load reaction-to-gene mappings from CSV."""
    if not map_file or not Path(map_file).exists():
        return {}
    try:
        df = pd.read_csv(map_file)
        rc = _detect_col(df, ['rxn_id', 'reaction_id', 'reaction', 'target_reactions']) or df.columns[0]
        gc = _detect_col(df, ['gene_id', 'gene']) or df.columns[1]
        m = {}
        for _, row in df.iterrows():
            r, g = str(row[rc]).strip(), str(row[gc]).strip()
            if r and g and r.lower() != 'nan' and g.lower() != 'nan':
                m.setdefault(r, []).append(g)
        print(f"rxn→gene map loaded: {len(m)} reactions")
        return m
    except Exception as e:
        print(f"  [WARN] rxn_gene_map load failed: {e}")
        return {}


def _read_all_genes(csv_path, rxn_gene_map=None):
    """Count unique genes, including BOTH/conflict, for the hit-rate denominator."""
    if not csv_path or not Path(csv_path).exists():
        return np.nan
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return 0
        gc = _detect_col(df, ['gene_id', 'gene', 'target']) or df.columns[0]


        is_rxn = False
        if rxn_gene_map:
            for v in df[gc].dropna().head(5):
                s = str(v).strip()
                if _is_rxn_id(s) or s in rxn_gene_map:
                    is_rxn = True
                    break

        genes = set()
        for val in df[gc].dropna():
            for tok in re.split(r'[;,]', str(val).strip()):
                tok = tok.strip()
                if not tok or tok.lower() in ('nan', 'none'):
                    continue
                if is_rxn and rxn_gene_map:
                    genes.update(rxn_gene_map.get(tok, []))
                    base = re.sub(r'_num\d+$', '', re.sub(r'_reverse$', '', tok))
                    genes.update(rxn_gene_map.get(base, []))
                else:
                    genes.add(tok)
        return len(genes)
    except Exception:
        return np.nan


def _read_clean_targets(csv_path, threshold=0.1, rxn_gene_map=None):
    """Return OE/KD gene-direction pairs, applying improvement filters when available."""
    if not csv_path or not Path(csv_path).exists():
        return set()
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return set()

        if 'production_improvement' in df.columns or 'yield_improvement' in df.columns:
            mask = pd.Series([False] * len(df))
            if 'production_improvement' in df.columns:
                mask |= df['production_improvement'] > threshold
            if 'yield_improvement' in df.columns:
                mask |= df['yield_improvement'] > threshold
            df = df[mask]

        gc = _detect_col(df, ['gene_id', 'gene', 'target']) or df.columns[0]
        tc = _detect_col(df, ['regulation_type', 'regulation', 'strategy', 'mod', 'type'])

        is_rxn = False
        if rxn_gene_map:
            for v in df[gc].dropna().head(5):
                s = str(v).strip()
                if _is_rxn_id(s) or s in rxn_gene_map:
                    is_rxn = True
                    break

        rows = []
        for _, row in df.iterrows():
            target_str = str(row[gc]).strip() if pd.notna(row[gc]) else ''
            if not target_str or target_str.lower() in ('nan', 'none'):
                continue

            raw_type = str(row[tc]).strip() if (tc and pd.notna(row.get(tc))) else ''
            reg = _norm_reg(raw_type)
            if reg is None:
                continue

            for tok in re.split(r'[;,]', target_str):
                tok = tok.strip()
                if not tok:
                    continue
                if is_rxn and rxn_gene_map:
                    genes = rxn_gene_map.get(tok, [])
                    if not genes:
                        base = re.sub(r'_num\d+$', '', re.sub(r'_reverse$', '', tok))
                        genes = rxn_gene_map.get(base, [])
                    for g in genes:
                        rows.append((g, reg))
                else:
                    rows.append((tok, reg))

        return set(rows)
    except Exception as e:
        print(f"  [WARN] Failed to read {csv_path}: {e}")
        return set()


def load_literature(excel_file, literature_col='Gene in 14067',
                    product_col='Product', strategy_col='Strategy'):
    """Load literature targets as compound-to-gene-direction sets."""
    if not excel_file or not Path(excel_file).exists():
        print("No literature file provided — skipping hit/coverage calculation")
        return {}
    try:
        df = pd.read_excel(excel_file, sheet_name=0, header=None)
        for idx, row in df.iterrows():
            if any(str(c).lower().startswith('references') for c in row):
                df = df.iloc[:idx]
                break
        df.columns = df.iloc[0]
        df = df[1:]
        df[product_col] = df[product_col].ffill()

        lit = {}
        for prod, grp in df.groupby(product_col):
            if pd.isna(prod) or not str(prod).strip():
                continue
            cpd = _extract_cpd(str(prod))
            tgts = set()
            for _, row in grp.iterrows():
                g, s = row.get(literature_col), row.get(strategy_col)
                if pd.notna(g) and pd.notna(s):
                    reg = _norm_reg(s)
                    if reg:
                        for gene in re.split(r'[;,]', str(g)):
                            gene = gene.strip()
                            if gene:
                                tgts.add((gene, reg))
            if tgts:
                lit[cpd] = tgts
        print(f"Literature loaded: {len(lit)} compounds")
        return lit
    except Exception as e:
        print(f"Literature load failed: {e}")
        return {}


def _find_csv(directory, keywords, recursive=False):
    if not Path(directory).exists():
        return None
    for f in Path(directory).glob('**/*.csv' if recursive else '*.csv'):
        if all(k in f.name.lower() for k in keywords):
            return str(f)
    return None


def scan_batch_dir(batch_dir, method_name, lit_data=None,
                   threshold=0.1, rxn_gene_map=None):
    batch_dir = Path(batch_dir)
    if not batch_dir.exists():
        print(f"  [ERROR] Directory not found: {batch_dir}")
        return []

    skip = {'__pycache__', '.ipynb_checkpoints', 'debug'}
    cpd_dirs = sorted([d for d in batch_dir.iterdir()
                       if d.is_dir() and d.name not in skip])
    print(f"\n[{method_name}] Scanning {batch_dir}  ({len(cpd_dirs)} compounds)")

    use_map = rxn_gene_map if method_name == 'ecBridge' else None

    records = []
    for cpd_dir in cpd_dirs:
        cpd = _extract_cpd(cpd_dir.name)

        if method_name == 'ecBridge':
            sub = cpd_dir / 'ecbridge'
            search_dir = sub if sub.is_dir() else cpd_dir
        elif method_name == 'ecFSEOF':
            sub = cpd_dir / 'ecfseof'
            search_dir = sub if sub.is_dir() else cpd_dir
        else:
            search_dir = cpd_dir

        raw_path = (_find_csv(search_dir, ['ecfb_merged_input'])
                    or _find_csv(search_dir, ['raw'])
                    or _find_csv(search_dir, ['merged'])
                    or _find_csv(cpd_dir,    ['raw'], recursive=True))

        filt_path = (_find_csv(search_dir, ['singletest_filtered'])
                     or _find_csv(search_dir, ['filtered'])
                     or _find_csv(cpd_dir,    ['filtered'], recursive=True))


        raw_count  = _read_all_genes(raw_path,  use_map)
        filt_count = _read_all_genes(filt_path, use_map)

        rec = {'cpd': cpd, 'method': method_name,
               'raw': raw_count, 'filtered': filt_count}

        raw_cov = raw_hit = filt_cov = filt_hit = np.nan

        if lit_data and lit_data.get(cpd):
            lit = lit_data[cpd]

            if raw_path and not np.isnan(raw_count) and raw_count > 0:
                pred = _read_clean_targets(raw_path, threshold, use_map)
                ov = len(pred & lit)
                raw_cov = ov / len(lit)       * 100
                raw_hit = ov / raw_count      * 100

            if filt_path and not np.isnan(filt_count) and filt_count > 0:
                pred = _read_clean_targets(filt_path, threshold, use_map)
                ov = len(pred & lit)
                filt_cov = ov / len(lit)      * 100
                filt_hit = ov / filt_count    * 100

        rec['raw_coverage']      = round(raw_cov,  2) if not np.isnan(raw_cov)  else np.nan
        rec['raw_hit_rate']      = round(raw_hit,  2) if not np.isnan(raw_hit)  else np.nan
        rec['filtered_coverage'] = round(filt_cov, 2) if not np.isnan(filt_cov) else np.nan
        rec['filtered_hit_rate'] = round(filt_hit, 2) if not np.isnan(filt_hit) else np.nan
        rec['coverage'] = rec['filtered_coverage']
        rec['hit_rate'] = rec['filtered_hit_rate']
        records.append(rec)

    ok = sum(1 for r in records if not np.isnan(r['filtered']))
    print(f"  Valid compounds: {ok}/{len(records)}")
    return records


METHOD_ORDER  = ['ecBridge', 'ecFSEOF_Kp', 'ecFB']
METHOD_COLORS = {
    'ecBridge':    ('#4C72B0', '#7BA3D4'),
    'ecFSEOF_Kp':  ('#DD8452', '#F0B88A'),
    'ecFB':        ('#55A868', '#8FD4A0'),
}


def plot_target_counts(df_all, output_dir):
    """Plot mean raw and filtered gene counts with SEM."""
    methods = [m for m in METHOD_ORDER if m in df_all['method'].values]
    n = len(methods)

    raw_means  = [df_all[df_all['method'] == m]['raw'].mean()      for m in methods]
    raw_sems   = [df_all[df_all['method'] == m]['raw'].sem()       for m in methods]
    filt_means = [df_all[df_all['method'] == m]['filtered'].mean() for m in methods]
    filt_sems  = [df_all[df_all['method'] == m]['filtered'].sem()  for m in methods]

    x, w = np.arange(n), 0.35
    fig, ax = plt.subplots(figsize=(max(8, n * 3), 6))
    fig.subplots_adjust(left=0.10, right=0.96, top=0.88, bottom=0.14)

    b_raw  = ax.bar(x - w/2, raw_means,  w, yerr=raw_sems,
                    capsize=5, alpha=0.85, edgecolor='white',
                    label='Raw', error_kw=dict(elinewidth=1.2, ecolor='#444'))
    b_filt = ax.bar(x + w/2, filt_means, w, yerr=filt_sems,
                    capsize=5, alpha=0.85, edgecolor='white',
                    label='Filtered', error_kw=dict(elinewidth=1.2, ecolor='#444'))
    for i, bar in enumerate(b_raw):
        bar.set_facecolor(METHOD_COLORS.get(methods[i], ('#888', '#bbb'))[0])
    for i, bar in enumerate(b_filt):
        bar.set_facecolor(METHOD_COLORS.get(methods[i], ('#888', '#bbb'))[1])

    for bars, means, sems in [(b_raw, raw_means, raw_sems),
                               (b_filt, filt_means, filt_sems)]:
        for bar, m, s in zip(bars, means, sems):
            if not np.isnan(m):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (s or 0) + 0.5,
                        f'{m:.1f}', ha='center', va='bottom',
                        fontsize=10, fontweight='bold')

    tops = [m + (s or 0) for m, s in
            zip(raw_means + filt_means, raw_sems + filt_sems) if not np.isnan(m)]
    ax.set_ylim(0, (max(tops) if tops else 10) * 1.30)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Gene Targets (± SEM)', fontsize=12, fontweight='bold')
    ax.set_title('Target Count',
                 fontsize=14, fontweight='bold', pad=12)
    ax.legend(loc='upper right', frameon=True, framealpha=0.9,
              edgecolor='#ccc', fontsize=10)
    for s in ['left', 'bottom']:
        ax.spines[s].set_linewidth(1.2)
        ax.spines[s].set_color('#333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(length=4, width=1.2, color='#333')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    out = Path(output_dir) / 'comparison_target_counts.png'
    fig.savefig(out, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"Fig 1 saved: {out}")
    plt.close(fig)


def plot_coverage_hitrate(df_all, output_dir):
    """Plot raw and filtered coverage and hit rates with SEM."""
    df_lit = df_all.dropna(
        subset=['raw_coverage', 'filtered_coverage',
                'raw_hit_rate', 'filtered_hit_rate'], how='all')
    if df_lit.empty:
        print("No literature data — skipping Fig 2")
        return

    methods = [m for m in METHOD_ORDER if m in df_all['method'].values]
    n = len(methods)

    stats = {}
    for m in methods:
        sub = df_lit[df_lit['method'] == m]
        stats[m] = {k: sub[c].agg(f) for k, c, f in [
            ('rc_m', 'raw_coverage',      'mean'), ('rc_s', 'raw_coverage',      'sem'),
            ('fc_m', 'filtered_coverage', 'mean'), ('fc_s', 'filtered_coverage', 'sem'),
            ('rh_m', 'raw_hit_rate',      'mean'), ('rh_s', 'raw_hit_rate',      'sem'),
            ('fh_m', 'filtered_hit_rate', 'mean'), ('fh_s', 'filtered_hit_rate', 'sem'),
        ]}

    x, w = np.arange(n), 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(14, n * 4), 6),
                                    sharey=True)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.14, wspace=0.08)

    def _bars(ax, c_raw, c_filt, rm, rs, fm, fs):
        br = ax.bar(x - w/2, rm, w, yerr=rs, capsize=5, alpha=0.85, color=c_raw,
                    edgecolor='white', label='Raw',
                    error_kw=dict(elinewidth=1.2, ecolor='#444'))
        bf = ax.bar(x + w/2, fm, w, yerr=fs, capsize=5, alpha=0.85, color=c_filt,
                    edgecolor='white', label='Filtered',
                    error_kw=dict(elinewidth=1.2, ecolor='#444'))
        for bars, means, sems in [(br, rm, rs), (bf, fm, fs)]:
            for bar, m, s in zip(bars, means, sems):
                if not np.isnan(m):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + (s or 0) + 0.3,
                            f'{m:.1f}%', ha='center', va='bottom',
                            fontsize=9, fontweight='bold')
                else:
                    ax.text(bar.get_x() + bar.get_width() / 2, 0.3,
                            'N/A', ha='center', va='bottom', fontsize=9, color='#999')

    _bars(ax1, '#A8C8E8', '#4C72B0',
          [stats[m]['rc_m'] for m in methods], [stats[m]['rc_s'] for m in methods],
          [stats[m]['fc_m'] for m in methods], [stats[m]['fc_s'] for m in methods])
    ax1.set_xticks(x); ax1.set_xticklabels(methods, fontsize=12, fontweight='bold')
    ax1.set_ylabel('Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Coverage Rate (%)', fontsize=14, fontweight='bold', pad=10)
    ax1.legend(loc='upper left', frameon=True, framealpha=0.9,
               edgecolor='#ccc', fontsize=10)

    _bars(ax2, '#F0C8A0', '#DD8452',
          [stats[m]['rh_m'] for m in methods], [stats[m]['rh_s'] for m in methods],
          [stats[m]['fh_m'] for m in methods], [stats[m]['fh_s'] for m in methods])
    ax2.set_xticks(x); ax2.set_xticklabels(methods, fontsize=12, fontweight='bold')
    ax2.set_title('Hit Rate (%)', fontsize=14, fontweight='bold', pad=10)
    ax2.legend(loc='upper left', frameon=True, framealpha=0.9,
               edgecolor='#ccc', fontsize=10)

    all_tops = [stats[m][k] + (stats[m][ks] or 0)
                for m in methods
                for k, ks in [('rc_m','rc_s'),('fc_m','fc_s'),
                               ('rh_m','rh_s'),('fh_m','fh_s')]
                if not np.isnan(stats[m][k])]
    ax1.set_ylim(0, (max(all_tops) if all_tops else 100) * 1.30)

    for ax in (ax1, ax2):
        for s in ['left', 'bottom']:
            ax.spines[s].set_linewidth(1.2)
            ax.spines[s].set_color('#333')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(length=4, width=1.2, color='#333')
        ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    fig.suptitle('Coverage & Hit Rate (Raw vs Filtered)\n'
                 'Numerator: OE/KD matches only  |  Denominator: all genes',
                 fontsize=13, fontweight='bold', y=0.99)

    out = Path(output_dir) / 'comparison_coverage_hitrate.png'
    fig.savefig(out, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"Fig 2 saved: {out}")
    plt.close(fig)


def plot_combined_vertical(df_all, output_dir):
    """Save a three-panel figure of target counts, coverage, and hit rates."""
    methods = [m for m in METHOD_ORDER if m in df_all['method'].values]
    n = len(methods)
    x = np.arange(n)
    w = 0.35


    raw_means = [df_all[df_all['method'] == m]['raw'].mean() for m in methods]
    raw_sems = [df_all[df_all['method'] == m]['raw'].sem() for m in methods]
    filt_means = [df_all[df_all['method'] == m]['filtered'].mean() for m in methods]
    filt_sems = [df_all[df_all['method'] == m]['filtered'].sem() for m in methods]

    df_lit = df_all.dropna(
        subset=['raw_coverage', 'filtered_coverage',
                'raw_hit_rate', 'filtered_hit_rate'], how='all')
    has_lit = not df_lit.empty

    stats = {}
    if has_lit:
        for m in methods:
            sub = df_lit[df_lit['method'] == m]
            stats[m] = {k: sub[c].agg(f) for k, c, f in [
                ('rc_m', 'raw_coverage', 'mean'), ('rc_s', 'raw_coverage', 'sem'),
                ('fc_m', 'filtered_coverage', 'mean'), ('fc_s', 'filtered_coverage', 'sem'),
                ('rh_m', 'raw_hit_rate', 'mean'), ('rh_s', 'raw_hit_rate', 'sem'),
                ('fh_m', 'filtered_hit_rate', 'mean'), ('fh_s', 'filtered_hit_rate', 'sem'),
            ]}


    fig_w = max(7, n * 2.5)
    fig, axes = plt.subplots(3, 1, figsize=(fig_w, 18))
    fig.subplots_adjust(left=0.13, right=0.95, top=0.96, bottom=0.05, hspace=0.30)


    def _style(ax):
        for s in ['left', 'bottom']:
            ax.spines[s].set_linewidth(1.2)
            ax.spines[s].set_color('#333')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(length=4, width=1.2, color='#333', labelsize=20)
        ax.set_facecolor('white')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=24, fontweight='bold')


    def _label_bars(ax, bars, means, sems, fmt='{:.1f}', offset=0.5):
        for bar, m, s in zip(bars, means, sems):
            if not np.isnan(m):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (s or 0) + offset,
                        fmt.format(m), ha='center', va='bottom',
                        fontsize=20, fontweight='bold')


    ax0 = axes[0]
    b_raw = ax0.bar(x - w / 2, raw_means, w, yerr=raw_sems, capsize=5,
                    alpha=0.85, edgecolor='white', label='Raw',
                    error_kw=dict(elinewidth=1.2, ecolor='#444'))
    b_filt = ax0.bar(x + w / 2, filt_means, w, yerr=filt_sems, capsize=5,
                     alpha=0.85, edgecolor='white', label='Filtered',
                     error_kw=dict(elinewidth=1.2, ecolor='#444'))
    for i, bar in enumerate(b_raw):
        bar.set_facecolor(METHOD_COLORS.get(methods[i], ('#888', '#bbb'))[0])
    for i, bar in enumerate(b_filt):
        bar.set_facecolor(METHOD_COLORS.get(methods[i], ('#888', '#bbb'))[1])
    _label_bars(ax0, b_raw, raw_means, raw_sems, fmt='{:.1f}', offset=0.5)
    _label_bars(ax0, b_filt, filt_means, filt_sems, fmt='{:.1f}', offset=0.5)
    tops0 = [m + (s or 0) for m, s in zip(raw_means + filt_means,
                                          raw_sems + filt_sems) if not np.isnan(m)]
    ax0.set_ylim(0, (max(tops0) if tops0 else 10) * 1.30)
    ax0.set_ylabel('Average Gene Targets (± SEM)', fontsize=24, fontweight='bold')
    ax0.set_title('Target Count', fontsize=28, fontweight='bold',
                  loc='center', pad=10)
    ax0.legend(loc='upper left', frameon=True, framealpha=0.9,
               edgecolor='#ccc', fontsize=15)
    _style(ax0)


    ax1 = axes[1]
    if has_lit:
        rc_raw = [stats[m]['rc_m'] for m in methods]
        rc_rs = [stats[m]['rc_s'] for m in methods]
        rc_filt = [stats[m]['fc_m'] for m in methods]
        rc_fs = [stats[m]['fc_s'] for m in methods]
        br = ax1.bar(x - w / 2, rc_raw, w, yerr=rc_rs, capsize=5, alpha=0.85,
                     color='#A8C8E8', edgecolor='white', label='Raw',
                     error_kw=dict(elinewidth=1.2, ecolor='#444'))
        bf = ax1.bar(x + w / 2, rc_filt, w, yerr=rc_fs, capsize=5, alpha=0.85,
                     color='#4C72B0', edgecolor='white', label='Filtered',
                     error_kw=dict(elinewidth=1.2, ecolor='#444'))
        _label_bars(ax1, br, rc_raw, rc_rs, fmt='{:.1f}%', offset=0.3)
        _label_bars(ax1, bf, rc_filt, rc_fs, fmt='{:.1f}%', offset=0.3)
        tops1 = [v for v in
                 [m + (s or 0) for m, s in zip(rc_raw + rc_filt, rc_rs + rc_fs)]
                 if not np.isnan(v)]
        ax1.set_ylim(0, (max(tops1) if tops1 else 100) * 1.30)
        ax1.legend(loc='upper left', frameon=True, framealpha=0.9,
                   edgecolor='#ccc', fontsize=15)
    else:
        ax1.text(0.5, 0.5, 'No literature data', transform=ax1.transAxes,
                 ha='center', va='center', fontsize=13, color='#999')
    ax1.set_ylabel('Coverage Rate (%)', fontsize=24, fontweight='bold')
    ax1.set_title('Coverage Rate (%)', fontsize=28, fontweight='bold',
                  loc='center', pad=10)
    _style(ax1)


    ax2 = axes[2]
    if has_lit:
        rh_raw = [stats[m]['rh_m'] for m in methods]
        rh_rs = [stats[m]['rh_s'] for m in methods]
        rh_filt = [stats[m]['fh_m'] for m in methods]
        rh_fs = [stats[m]['fh_s'] for m in methods]
        br2 = ax2.bar(x - w / 2, rh_raw, w, yerr=rh_rs, capsize=5, alpha=0.85,
                      color='#F0C8A0', edgecolor='white', label='Raw',
                      error_kw=dict(elinewidth=1.2, ecolor='#444'))
        bf2 = ax2.bar(x + w / 2, rh_filt, w, yerr=rh_fs, capsize=5, alpha=0.85,
                      color='#DD8452', edgecolor='white', label='Filtered',
                      error_kw=dict(elinewidth=1.2, ecolor='#444'))
        _label_bars(ax2, br2, rh_raw, rh_rs, fmt='{:.1f}%', offset=0.3)
        _label_bars(ax2, bf2, rh_filt, rh_fs, fmt='{:.1f}%', offset=0.3)
        tops2 = [v for v in
                 [m + (s or 0) for m, s in zip(rh_raw + rh_filt, rh_rs + rh_fs)]
                 if not np.isnan(v)]
        ax2.set_ylim(0, (max(tops2) if tops2 else 100) * 1.30)
        ax2.legend(loc='upper left', frameon=True, framealpha=0.9,
                   edgecolor='#ccc', fontsize=15)
    else:
        ax2.text(0.5, 0.5, 'No literature data', transform=ax2.transAxes,
                 ha='center', va='center', fontsize=13, color='#999')
    ax2.set_ylabel('Hit Rate (%)', fontsize=24, fontweight='bold')
    ax2.set_title('Hit Rate (%)', fontsize=28, fontweight='bold',
                  loc='center', pad=10)
    _style(ax2)

    fig.patch.set_facecolor('white')

    out = Path(output_dir) / 'comparison_combined_vertical.png'
    fig.savefig(out, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"Fig 3 (combined vertical) saved: {out}")
    plt.close(fig)


def run_comparison(
    ecbridge_batch_dir,
    ecfb_batch_dir,
    ecfseof_batch_dir,
    excel_file=None,
    rxn_gene_map_file=None,
    literature_col='Gene in 14067',
    product_col='Product',
    strategy_col='Strategy',
    improvement_threshold=0.1,
    output_dir='comparison_results',
):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Three-Source Comparison Analysis")
    print("=" * 60)

    rxn_gene_map = load_rxn_gene_map(rxn_gene_map_file) if rxn_gene_map_file else {}
    lit_data     = load_literature(excel_file, literature_col,
                                   product_col, strategy_col) if excel_file else {}

    all_records = []
    for batch_dir, method in [
        (ecbridge_batch_dir, 'ecBridge'),
        (ecfseof_batch_dir,  'ecFSEOF_Kp'),
        (ecfb_batch_dir,     'ecFB'),
    ]:
        all_records.extend(
            scan_batch_dir(batch_dir, method, lit_data,
                           improvement_threshold, rxn_gene_map))

    df_all = pd.DataFrame(all_records)

    print("\n" + "=" * 60)
    print("Summary  (denominator = all genes incl. conflict/BOTH)")
    print("=" * 60)
    for method in ['ecBridge', 'ecFSEOF_Kp', 'ecFB']:
        sub = df_all[df_all['method'] == method]
        if sub.empty:
            print(f"\n  {method}: no data"); continue
        print(f"\n  {method}:  {len(sub)} compounds")
        print(f"    Raw avg genes     : {sub['raw'].mean():.1f} ± {sub['raw'].sem():.1f}")
        print(f"    Filtered avg genes: {sub['filtered'].mean():.1f} ± {sub['filtered'].sem():.1f}")
        if lit_data:
            for label, col in [('Raw coverage',      'raw_coverage'),
                                ('Raw hit rate',      'raw_hit_rate'),
                                ('Filtered coverage', 'filtered_coverage'),
                                ('Filtered hit rate', 'filtered_hit_rate')]:
                val = sub[col].mean()
                print(f"    {label:20s}: {f'{val:.1f}%' if not np.isnan(val) else 'N/A'}")

    plot_target_counts(df_all, output_dir)
    plot_coverage_hitrate(df_all, output_dir)
    plot_combined_vertical(df_all, output_dir)

    csv_out = output_dir / 'comparison_detail.csv'
    df_all.to_csv(csv_out, index=False)
    print(f"\nDetail CSV saved: {csv_out}")
    return df_all


if __name__ == '__main__':

    df = run_comparison(
        ecbridge_batch_dir = './ecbridge_results/batch_20260328_012821',
        ecfb_batch_dir     = './ecfb_results/batch_20260328_003354',
        ecfseof_batch_dir  = './ecfseof_results/batch_20260328_015249',

        excel_file        = 'Cgproduct.xlsx',
        rxn_gene_map_file = 'rxn_gene_map.csv',

        literature_col = 'Gene in 14067',
        product_col    = 'Product',
        strategy_col   = 'Strategy',

        improvement_threshold = 0.1,
        output_dir            = 'comparison_results',
    )

    print("\n" + df.to_string(index=False))
