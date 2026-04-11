#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
batch_predict_parallel.py
=========================
Batch parallel prediction script for metabolic engineering target identification.
Fully based on ecfb.py — no dependency on ecbridge.py / ecfseof.py.

Supported methods per compound:
    'ecbridge' — run ecBridge only
    'ecfseof'  — run ecFSEOF only
    'ecfb'     — run both, merge + resolve conflicts, unified single-target test
    'both'     — alias for 'ecfb'

Output directory structure (ecfb mode — FLAT, no sub-folders per compound):
    OUTPUT_BASE/
        batch_YYYYMMDD_HHMMSS/
            cpd00039/
                cpd00039_ecfb_merged_input.csv   ← merged gene pool before testing
                cpd00039_singletest_filtered.csv ← final validated targets
            cpd00322/
                ...
            batch_summary.csv
            all_targets_combined.csv
            error_log.txt

The 'sources' column in *_singletest_filtered.csv records which tool(s) correctly
predicted the retained regulation direction:
    'ecBridge'            — only ecBridge predicted this gene/direction
    'ecFSEOF'             — only ecFSEOF predicted this gene/direction
    'ecBridge+ecFSEOF'    — both tools predicted the same direction
    (for conflict genes the source reflects the WINNING direction after testing)
"""

import os
import sys
import time
import warnings
import traceback
from datetime import datetime
from multiprocessing import Pool, cpu_count

import pandas as pd

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

os.chdir(PROJECT_ROOT)

# Automatically find all candidate directories and verify ecfb.py is reachable
_candidate_dirs = [
    SCRIPT_DIR,
    PROJECT_ROOT,
    os.path.join(SCRIPT_DIR, 'MEResource'),
    os.path.join(SCRIPT_DIR, 'script'),
    os.path.join(SCRIPT_DIR, 'ecscript'),
    os.path.join(PROJECT_ROOT, 'ecscript'),
    os.path.join(PROJECT_ROOT, 'script'),
]

PATHS_TO_ADD = [p for p in _candidate_dirs if os.path.exists(p)]

# Fail fast if ecfb.py is not findable — avoids silent subprocess errors
_found_ecfb = any(
    os.path.exists(os.path.join(p, 'ecfb.py')) for p in PATHS_TO_ADD
)
if not _found_ecfb:
    raise FileNotFoundError(
        "Cannot find ecfb.py in any of:\n"
        + "\n".join(f"  {p}" for p in PATHS_TO_ADD)
        + "\nAdd the correct directory to PATHS_TO_ADD manually."
    )

for _p in PATHS_TO_ADD:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Optional tqdm progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Tip: install tqdm for progress bars (pip install tqdm)")


# =============================================================================
# Subprocess initialiser
# =============================================================================

def _init_worker(paths):
    """Ensure sys.path is correct in every worker process."""
    import warnings
    warnings.filterwarnings('ignore')
    for p in paths:
        if p not in sys.path:
            sys.path.insert(0, p)


# =============================================================================
# Per-compound worker function
# =============================================================================

def _process_single_product(args):
    """
    Run the full prediction pipeline for one compound inside a subprocess.

    Parameters (packed as a tuple)
    --------------------------------
    cpd, model_file, batch_output_dir, paths, method,
    biomass_rxn, c_source_rxn,
    ecb_num_step, ecb_bio_fraction, ecb_threshold, ecb_top_score_percentage,
    efs_fseof_steps, efs_top_percentage,
    st_improvement_threshold, st_exclusion_threshold,
    debug_output, naming

    Returns
    -------
    dict with keys: compound, method, ecfb_passed, ecfb_path, time, status, error
    """
    (cpd, model_file, batch_output_dir, paths, method,
     biomass_rxn, c_source_rxn,
     ecb_num_step, ecb_bio_fraction, ecb_threshold, ecb_top_score_percentage,
     efs_fseof_steps, efs_top_percentage,
     st_improvement_threshold, st_exclusion_threshold,
     debug_output, naming) = args

    for p in paths:
        if p not in sys.path:
            sys.path.insert(0, p)

    start   = time.time()
    # Each compound gets its own flat folder directly under the batch directory
    cpd_dir = os.path.join(batch_output_dir, cpd)
    os.makedirs(cpd_dir, exist_ok=True)

    if method == 'both':
        method = 'ecfb'

    result = {
        'compound':        cpd,
        'method':          method,
        'ecfb_passed':     None,
        'ecfb_path':       None,
        'ecbridge_passed': None,
        'ecbridge_path':   None,
        'ecfseof_passed':  None,
        'ecfseof_path':    None,
        'time':   None,
        'status': 'failed',
        'error':  '',
    }

    try:
        from ecfb import run_prediction

        run_result = run_prediction(
            mode                     = method,
            model                    = model_file,
            product_id               = cpd,
            biomass_rxn              = biomass_rxn,
            output_dir               = cpd_dir,
            ecb_num_step             = ecb_num_step,
            ecb_bio_fraction         = ecb_bio_fraction,
            ecb_threshold            = ecb_threshold,
            ecb_top_score_percentage = ecb_top_score_percentage,
            efs_fseof_steps          = efs_fseof_steps,
            efs_top_percentage       = efs_top_percentage,
            st_improvement_threshold = st_improvement_threshold,
            st_exclusion_threshold   = st_exclusion_threshold,
            c_source_rxn             = c_source_rxn,
            debug_output             = debug_output,
            naming                   = naming,
        )

        def _count(path):
            """Row count of a CSV, or None if missing."""
            if path and os.path.exists(path):
                try:
                    return len(pd.read_csv(path))
                except Exception:
                    return None
            return None

        if method == 'ecfb':
            filt_path = run_result.get('ecfb_filtered')
            result['ecfb_passed'] = _count(filt_path)
            result['ecfb_path']   = filt_path
        elif method == 'ecbridge':
            filt_path = run_result.get('ecb_filtered')
            result['ecbridge_passed'] = _count(filt_path)
            result['ecbridge_path']   = filt_path
        elif method == 'ecfseof':
            filt_path = run_result.get('efs_filtered')
            result['ecfseof_passed'] = _count(filt_path)
            result['ecfseof_path']   = filt_path

        result['time']   = round(time.time() - start, 2)
        result['status'] = 'success'

    except Exception as e:
        result['time']  = round(time.time() - start, 2)
        result['error'] = f"{e}\n{traceback.format_exc()}"

    return result


# =============================================================================
# Summary / aggregation helpers
# =============================================================================

def _save_summary(new_rows, existing_rows, batch_output_dir,
                  summary_file, method, final=False):
    """
    Write batch_summary.csv and (when final=True) all_targets_combined.csv.
    Returns path of all_targets_combined.csv or None.
    """
    all_rows = existing_rows + new_rows
    pd.DataFrame(all_rows).to_csv(summary_file, index=False)

    if not final:
        return None

    combined_rows = []
    for row in all_rows:
        if row['status'] != 'success':
            continue

        if method == 'ecfb':
            path = row.get('ecfb_path')
        elif method == 'ecbridge':
            path = row.get('ecbridge_path')
        else:
            path = row.get('ecfseof_path')

        if path and os.path.exists(path):
            try:
                df = pd.read_csv(path)
                df.insert(0, 'compound', row['compound'])
                combined_rows.append(df)
            except Exception:
                pass

    if combined_rows:
        all_combined  = pd.concat(combined_rows, ignore_index=True)
        combined_path = os.path.join(batch_output_dir, 'all_targets_combined.csv')
        all_combined.to_csv(combined_path, index=False)
        return combined_path

    return None


def _print_result_line(result, idx, total, method):
    """Print a one-line progress summary for a completed compound."""
    ok   = result['status'] == 'success'
    icon = '✅' if ok else '❌'

    if method == 'ecfb':
        detail = f"passed={result['ecfb_passed']}"
    elif method == 'ecbridge':
        detail = f"EB_passed={result['ecbridge_passed']}"
    else:
        detail = f"FS_passed={result['ecfseof_passed']}"

    print(f"{icon} [{idx}/{total}] {result['compound']}  {detail}  {result['time']}s")
    if not ok:
        short_err = str(result['error']).split('\n')[0][:100]
        print(f"   ↳ {short_err}")


# =============================================================================
# Main parallel batch function
# =============================================================================

def batch_predict(
        model_file,
        product_ids,
        method                   = 'ecfb',
        output_base_dir          = './batch_predict_results',
        biomass_rxn              = 'Bio_cgATCC14067_c',
        c_source_rxn             = 'EX_cpd00027_e_reverse',
        ecb_num_step             = 20,
        ecb_bio_fraction         = 0.1,
        ecb_threshold            = 0.01,
        ecb_top_score_percentage = 1.0,
        efs_fseof_steps          = 10,
        efs_top_percentage       = 1.0,
        st_improvement_threshold = 0.1,
        st_exclusion_threshold   = -5.0,
        n_workers                = None,
        resume_from              = None,
        show_progress            = True,
        save_interval            = 3,
        debug_output             = False,
        naming                   = 'modelseed',
):
    """
    Parallel batch prediction using ecfb.run_prediction().

    ecfb mode output per compound (FLAT — no sub-folders):
        batch_XXXX/cpd00039/cpd00039_ecfb_merged_input.csv
        batch_XXXX/cpd00039/cpd00039_singletest_filtered.csv

    sources column values in singletest_filtered.csv:
        'ecBridge'            — ecBridge only predicted this gene/direction
        'ecFSEOF'             — ecFSEOF only predicted this gene/direction
        'ecBridge+ecFSEOF'    — both tools agreed on the same direction
        conflict genes        — sources = tool(s) that predicted the WINNING direction
    """
    if method == 'both':
        method = 'ecfb'
    assert method in ('ecbridge', 'ecfseof', 'ecfb'), \
        "method must be 'ecbridge', 'ecfseof', 'ecfb', or 'both'"

    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    else:
        n_workers = max(1, min(n_workers, cpu_count()))

    model_file = os.path.abspath(model_file)

    if resume_from and os.path.exists(resume_from):
        batch_output_dir = os.path.abspath(resume_from)
        print(f"Resuming from: {batch_output_dir}")
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_output_dir = os.path.abspath(
            os.path.join(output_base_dir, f"batch_{ts}")
        )
        os.makedirs(batch_output_dir, exist_ok=True)

    summary_file   = os.path.join(batch_output_dir, 'batch_summary.csv')
    existing_rows  = []
    done_compounds = set()

    if os.path.exists(summary_file):
        existing_df    = pd.read_csv(summary_file)
        existing_rows  = existing_df.to_dict('records')
        done_compounds = set(existing_df[existing_df['status'] == 'success']['compound'])
        if done_compounds:
            print(f"Already completed {len(done_compounds)} compounds — skipping")

    remaining = [c for c in product_ids if c not in done_compounds]

    if not remaining:
        print("All compounds already processed!")
        return pd.read_csv(summary_file), batch_output_dir

    method_label = {
        'ecbridge': 'ecBridge only',
        'ecfseof':  'ecFSEOF only',
        'ecfb':     'ecBridge + ecFSEOF (ecFB combined)',
    }
    print(f"\n{'=' * 70}")
    print(f"Batch prediction  [{method_label[method]}]")
    print(f"{'=' * 70}")
    print(f"Total compounds : {len(product_ids)}  "
          f"Done: {len(done_compounds)}  Remaining: {len(remaining)}")
    print(f"Parallel workers: {n_workers}")
    print(f"Model           : {model_file}")
    print(f"Naming          : {naming}")
    print(f"Output          : {batch_output_dir}")
    if method in ('ecbridge', 'ecfb'):
        print(f"  ecBridge — MOMA steps: {ecb_num_step}  "
              f"SoC top: {ecb_top_score_percentage*100:.0f}%")
    if method in ('ecfseof', 'ecfb'):
        print(f"  ecFSEOF  — FSEOF steps: {efs_fseof_steps}  "
              f"top %: {efs_top_percentage*100:.0f}%")
    print(f"  Single-target — improvement: ≥{st_improvement_threshold}%  "
          f"exclusion: ≤{st_exclusion_threshold}%")
    if method == 'ecfb':
        print(f"  Output per compound (FLAT):")
        print(f"    <cpd>_ecfb_merged_input.csv   — merged gene pool (pre-test)")
        print(f"    <cpd>_singletest_filtered.csv — validated targets + source labels")
    print(f"{'=' * 70}\n")

    task_args = [
        (cpd, model_file, batch_output_dir, PATHS_TO_ADD, method,
         biomass_rxn, c_source_rxn,
         ecb_num_step, ecb_bio_fraction, ecb_threshold, ecb_top_score_percentage,
         efs_fseof_steps, efs_top_percentage,
         st_improvement_threshold, st_exclusion_threshold,
         debug_output, naming)
        for cpd in remaining
    ]

    pool_cfg = dict(
        processes   = n_workers,
        initializer = _init_worker,
        initargs    = (PATHS_TO_ADD,),
    )

    new_rows  = []
    start_all = time.time()

    def _handle(result, idx):
        new_rows.append(result)
        _print_result_line(result, idx, len(remaining), method)
        if idx % save_interval == 0:
            _save_summary(new_rows, existing_rows,
                          batch_output_dir, summary_file, method, final=False)

    with Pool(**pool_cfg) as pool:
        if HAS_TQDM and show_progress:
            with tqdm(total=len(remaining), desc="Batch prediction",
                      unit="cpd", ncols=100) as pbar:
                for idx, result in enumerate(
                    pool.imap_unordered(_process_single_product, task_args), 1
                ):
                    _handle(result, idx)
                    pbar.set_postfix_str(
                        f"{'OK' if result['status']=='success' else 'FAIL'} "
                        f"{result['compound']}"
                    )
                    pbar.update(1)
        else:
            for idx, result in enumerate(
                pool.imap_unordered(_process_single_product, task_args), 1
            ):
                _handle(result, idx)

    combined_path = _save_summary(
        new_rows, existing_rows, batch_output_dir, summary_file, method, final=True
    )

    total_time    = time.time() - start_all
    success_count = sum(1 for r in new_rows if r['status'] == 'success')
    failed_count  = len(new_rows) - success_count
    avg_time      = (sum(r['time'] for r in new_rows) / len(new_rows)
                     if new_rows else 0)
    speedup       = ((len(new_rows) * avg_time) / total_time
                     if total_time > 0 and n_workers > 1 else 1)

    print(f"\n{'=' * 70}")
    print(f"Batch complete!")
    print(f"  Total time : {total_time:.1f}s ({total_time/60:.1f} min)  "
          f"Avg: {avg_time:.1f}s/cpd  Speedup: {speedup:.1f}x")
    print(f"  Success: {success_count}  Failed: {failed_count}")

    if failed_count > 0:
        error_log = os.path.join(batch_output_dir, 'error_log.txt')
        print(f"\nFailed compounds:")
        with open(error_log, 'w', encoding='utf-8') as f:
            for r in new_rows:
                if r['status'] == 'failed':
                    short = str(r['error']).split('\n')[0][:80]
                    print(f"   - {r['compound']}: {short}")
                    f.write(f"{'='*60}\nCompound: {r['compound']}\n{r['error']}\n\n")
        print(f"  Full error log: {error_log}")

    print(f"\nOutput:")
    print(f"  Summary  : {summary_file}")
    if combined_path:
        print(f"  Combined : {combined_path}")
    print(f"{'=' * 70}\n")

    return pd.read_csv(summary_file), batch_output_dir


# =============================================================================
# Serial version — for debugging
# =============================================================================

def batch_predict_serial(
        model_file,
        product_ids,
        method                   = 'ecfb',
        output_base_dir          = './batch_predict_results',
        biomass_rxn              = 'Bio_cgATCC14067_c',
        c_source_rxn             = 'EX_cpd00027_e_reverse',
        ecb_num_step             = 20,
        ecb_bio_fraction         = 0.1,
        ecb_threshold            = 0.01,
        ecb_top_score_percentage = 1.0,
        efs_fseof_steps          = 10,
        efs_top_percentage       = 1.0,
        st_improvement_threshold = 0.1,
        st_exclusion_threshold   = -5.0,
        debug_output             = False,
        naming                   = 'modelseed',
):
    """
    Serial (single-process) batch prediction — recommended for debugging.
    Exceptions propagate immediately with full stack traces.
    """
    if method == 'both':
        method = 'ecfb'
    assert method in ('ecbridge', 'ecfseof', 'ecfb')

    from ecfb import run_prediction

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_output_dir = os.path.join(output_base_dir, f"batch_{ts}")
    os.makedirs(batch_output_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"Serial batch (debug mode) — {len(product_ids)} compounds  method: {method}")
    print(f"{'=' * 70}\n")

    rows = []
    for i, cpd in enumerate(product_ids, 1):
        print(f"\n[{i}/{len(product_ids)}] {cpd}")
        cpd_dir = os.path.join(batch_output_dir, cpd)
        os.makedirs(cpd_dir, exist_ok=True)
        t0 = time.time()

        row = {
            'compound':        cpd,
            'method':          method,
            'ecfb_passed':     None,
            'ecfb_path':       None,
            'ecbridge_passed': None,
            'ecbridge_path':   None,
            'ecfseof_passed':  None,
            'ecfseof_path':    None,
            'time':   None,
            'status': 'failed',
            'error':  '',
        }

        try:
            run_result = run_prediction(
                mode                     = method,
                model                    = model_file,
                product_id               = cpd,
                biomass_rxn              = biomass_rxn,
                output_dir               = cpd_dir,
                ecb_num_step             = ecb_num_step,
                ecb_bio_fraction         = ecb_bio_fraction,
                ecb_threshold            = ecb_threshold,
                ecb_top_score_percentage = ecb_top_score_percentage,
                efs_fseof_steps          = efs_fseof_steps,
                efs_top_percentage       = efs_top_percentage,
                st_improvement_threshold = st_improvement_threshold,
                st_exclusion_threshold   = st_exclusion_threshold,
                c_source_rxn             = c_source_rxn,
                debug_output             = debug_output,
                naming                   = naming,
            )

            def _count(path):
                if path and os.path.exists(path):
                    try:
                        return len(pd.read_csv(path))
                    except Exception:
                        return None
                return None

            if method == 'ecfb':
                filt_path = run_result.get('ecfb_filtered')
                row['ecfb_passed'] = _count(filt_path)
                row['ecfb_path']   = filt_path
                detail = f"passed={row['ecfb_passed']}"
            elif method == 'ecbridge':
                filt_path = run_result.get('ecb_filtered')
                row['ecbridge_passed'] = _count(filt_path)
                row['ecbridge_path']   = filt_path
                detail = f"EB_passed={row['ecbridge_passed']}"
            else:
                filt_path = run_result.get('efs_filtered')
                row['ecfseof_passed'] = _count(filt_path)
                row['ecfseof_path']   = filt_path
                detail = f"FS_passed={row['ecfseof_passed']}"

            row['time']   = round(time.time() - t0, 2)
            row['status'] = 'success'
            print(f"  OK  {detail}  {row['time']}s")

        except Exception as e:
            row['time']  = round(time.time() - t0, 2)
            row['error'] = str(e)[:300]
            print(f"  FAILED: {e}")
            traceback.print_exc()

        rows.append(row)

    summary_df   = pd.DataFrame(rows)
    summary_file = os.path.join(batch_output_dir, 'batch_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    _save_summary(rows, [], batch_output_dir, summary_file, method, final=True)

    print(f"\nSummary saved: {summary_file}")
    return summary_df, batch_output_dir


# =============================================================================
# Main entry point
# =============================================================================

if __name__ == "__main__":

    # ==================================================================
    # Example A: ModelSEED model (your own model)
    # ==================================================================
    MODEL_FILE  = './model_Calibration/DLKcat_Calibration/ecCZ870_DLKcat_Calibration_modify.json'
    OUTPUT_DIR  = './batch_predict_results'
    BIOMASS_RXN = 'Bio_cgATCC14067_c'
    C_SOURCE    = 'EX_cpd00027_e_reverse'
    NAMING      = 'modelseed'

    PRODUCT_IDS = [
        "cpd00039", "cpd00322", "cpd00060", "cpd00107",
        "cpd00156", "cpd00161", "cpd00227", "cpd00129",
        "cpd00051", "cpd00064", "cpd00274", "cpd00069",
        "cpd00065", "cpd00054", "cpd00084", "cpd00119",
        "cpd00066",
    ]

    # ------------------------------------------------------------------
    # Option A: Parallel prediction (production)
    # ------------------------------------------------------------------
    summary_df, output_dir = batch_predict(
        model_file               = MODEL_FILE,
        product_ids              = PRODUCT_IDS,
        method                   = 'ecfb',
        output_base_dir          = OUTPUT_DIR,
        biomass_rxn              = BIOMASS_RXN,
        c_source_rxn             = C_SOURCE,
        ecb_num_step             = 20,
        ecb_bio_fraction         = 0.1,
        ecb_threshold            = 0.01,
        ecb_top_score_percentage = 0.9,
        efs_fseof_steps          = 10,
        efs_top_percentage       = 0.9,
        st_improvement_threshold = 0.1,
        st_exclusion_threshold   = -5.0,
        n_workers                = 4,
        show_progress            = True,
        save_interval            = 3,
        debug_output             = False,
        naming                   = NAMING,
        # resume_from = './batch_predict_results/batch_20240101_120000',
    )

    print(summary_df.to_string(index=False))
    print(f"\nAll results saved in: {output_dir}")

    # ==================================================================
    # Example B: BiGG model (e.g. E. coli core model)
    # Uncomment and adapt for BiGG-format models
    # ==================================================================
    # MODEL_FILE  = './models/e_coli_core.json'
    # OUTPUT_DIR  = './batch_predict_results_bigg'
    # BIOMASS_RXN = 'BIOMASS_Ecoli_core_w_GAM'
    # C_SOURCE    = 'EX_glc__D_e'       # glucose exchange in BiGG
    # NAMING      = 'bigg'
    #
    # PRODUCT_IDS = [
    #     "succ",      # succinate (BiGG ID without compartment or EX_ prefix)
    #     "lac__D",    # D-lactate
    # ]
    #
    # summary_df, output_dir = batch_predict(
    #     model_file               = MODEL_FILE,
    #     product_ids              = PRODUCT_IDS,
    #     method                   = 'ecfb',
    #     output_base_dir          = OUTPUT_DIR,
    #     biomass_rxn              = BIOMASS_RXN,
    #     c_source_rxn             = C_SOURCE,
    #     naming                   = NAMING,
    #     n_workers                = 4,
    # )