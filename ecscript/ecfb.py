#!/usr/bin/env python
# Author: Zhang xinyu
# Date: 2026-10-15

"""Metabolic engineering target prediction with ecBridge, ecFSEOF, or combined ecFB."""

import os
import re
import json
import hashlib
import logging
import warnings
import tempfile
from collections import Counter
from datetime import datetime
from math import isnan

import numpy as np
import pandas as pd
import cobra
from cobra.io import load_json_model
from cobra.flux_analysis import moma, pfba

warnings.filterwarnings('ignore')
os.environ['NUMEXPR_NUM_THREADS'] = '4'


# NAMING CONVENTION SUPPORT (ModelSEED / BiGG)


from naming_convention import NamingConvention


_DEFAULT_NAMING = NamingConvention('modelseed')


# SHARED CONSTANTS (convention-independent)


COMMON_COFACTORS     = _DEFAULT_NAMING.COMMON_COFACTORS
EXCLUDED_METABOLITES = _DEFAULT_NAMING.EXCLUDED_METABOLITES
EXCLUDED_REACTIONS   = _DEFAULT_NAMING.EXCLUDED_REACTIONS


METABOLIC_EXCLUSION_KDdirection = []


OEF            = 5.0
KDF            = 0.2
GROWTH_FRACTION = 0.25


# SHARED UTILITY HELPERS


def _get_nc(naming):
    """Return a NamingConvention, accepting None / str / NamingConvention."""
    if naming is None:
        return _DEFAULT_NAMING
    if isinstance(naming, str):
        return NamingConvention(naming)
    return naming


def is_cofactor(metabolite_id, naming=None):
    """Return True if the metabolite is a common cofactor."""
    return _get_nc(naming).is_cofactor(metabolite_id)


def is_excluded_metabolite(metabolite_id, naming=None):
    """Return True if the metabolite is in the hard exclusion list."""
    return _get_nc(naming).is_excluded_metabolite(metabolite_id)


def is_bio_reaction(reaction_id, naming=None):
    """Return True if the reaction is a biomass assembly reaction."""
    return _get_nc(naming).is_bio_reaction(reaction_id)


def is_excluded_reaction(reaction_id, naming=None):
    """Return True if the reaction is globally excluded."""
    return _get_nc(naming).is_excluded_reaction(reaction_id)


def normalize_reaction_id(reaction_id, naming=None):
    """Strip isozyme suffix to obtain the base reaction ID."""
    return _get_nc(naming).normalize_reaction_id(reaction_id)


def get_unique_reaction_count(reaction_list, naming=None):
    """Return the number of distinct base reaction IDs in *reaction_list*."""
    if not reaction_list:
        return 0
    nc = _get_nc(naming)
    return len(set(nc.normalize_reaction_id(r) for r in reaction_list))


def check_enzyme_constraints(model):
    """Return True when the model carries enzyme-constraint information."""
    if hasattr(model, 'constraints'):
        return True
    if hasattr(model, 'json') and isinstance(model.json, dict):
        if 'enzyme_constraint' in model.json:
            return True
    for reaction in model.reactions:
        if hasattr(reaction, 'kcat_MW') and reaction.kcat_MW:
            try:
                if float(reaction.kcat_MW) > 0:
                    return True
            except (ValueError, TypeError):
                continue
    return False


def _load_model(model_or_file):
    """Accept a cobra.Model object or a model file path; return a cobra.Model."""
    if isinstance(model_or_file, str):
        try:
            from ECMpy_function import get_enzyme_constraint_model_with_reaction_set_exclusion
            return get_enzyme_constraint_model_with_reaction_set_exclusion(model_or_file)
        except Exception:
            return load_json_model(model_or_file)
    return model_or_file


# ecBRIDGE CORE ALGORITHM


# minpool optimiser

def _ecb_optimize_minimize_enzyme_pool(model, objective_rxn, fraction_of_optimum=1.0):
    """Maximize the target, then minimize enzyme cost at the requested target fraction."""
    model.objective = objective_rxn
    model.objective_direction = 'max'
    max_objective = model.slim_optimize()

    if max_objective is None or isnan(max_objective) or max_objective <= 1e-9:
        return model.optimize()

    target_flux = max_objective * fraction_of_optimum
    tolerance   = abs(target_flux) * 0.001

    obj_rxn = model.reactions.get_by_id(objective_rxn)
    original_bounds = obj_rxn.bounds
    obj_rxn.bounds  = (target_flux - tolerance, target_flux + tolerance)

    default_penalty = 1e-9
    enzyme_cost_objective = {}
    for rxn in model.reactions:
        if hasattr(rxn, 'kcat_MW') and rxn.kcat_MW:
            try:
                kcat_mw = float(rxn.kcat_MW)
                enzyme_cost_objective[rxn] = -1.0 / kcat_mw if kcat_mw > 0 else -default_penalty
            except (ValueError, TypeError):
                enzyme_cost_objective[rxn] = -default_penalty
        else:
            enzyme_cost_objective[rxn] = -default_penalty

    model.objective = enzyme_cost_objective
    model.objective_direction = 'max'
    solution = model.optimize()
    obj_rxn.bounds = original_bounds
    return solution


# Active-reaction identification

def _ecb_find_active_reactions(model, target_rxn, biomass_rxn, bio_fraction=0.1, naming=None):
    """Identify reactions carrying non-zero flux under production conditions.
    Uses minpool when enzyme constraints are present; otherwise plain FBA."""
    nc = _get_nc(naming)
    print("Analysing active reactions...")
    has_ec = check_enzyme_constraints(model)

    if not has_ec:
        print("  No enzyme constraints — using standard FBA")
        return _ecb_find_active_reactions_standard(model, target_rxn, biomass_rxn, bio_fraction, naming=naming)

    print("  Enzyme constraints detected — applying minpool")
    try:
        with model as m:
            m.objective = biomass_rxn
            max_bio = m.slim_optimize()
            if isnan(max_bio) or max_bio < 1e-6:
                raise ValueError("Cannot achieve growth target")
            m.reactions.get_by_id(biomass_rxn).lower_bound = max_bio * bio_fraction
            solution = _ecb_optimize_minimize_enzyme_pool(m, target_rxn, fraction_of_optimum=1.0)
            if solution.status != 'optimal':
                raise ValueError("minpool optimisation failed")
            active = [
                rxn_id for rxn_id, flux in solution.fluxes.items()
                if abs(flux) > 1e-6 and not nc.is_bio_reaction(rxn_id)
            ]
            print(f"  minpool: {len(active)} active reactions")
            return active
    except Exception as e:
        print(f"  minpool failed ({e}) — falling back to standard FBA")
        return _ecb_find_active_reactions_standard(model, target_rxn, biomass_rxn, bio_fraction, naming=naming)


def _ecb_find_active_reactions_standard(model, target_rxn, biomass_rxn, bio_fraction=0.1, naming=None):
    """Standard active-reaction identification via plain FBA."""
    nc = _get_nc(naming)
    with model as m:
        m.objective = biomass_rxn
        max_bio = m.slim_optimize()
        m.reactions.get_by_id(biomass_rxn).lower_bound = max_bio * bio_fraction
        m.objective = target_rxn
        solution = m.optimize()
        if solution.status != 'optimal':
            raise ValueError("Cannot find feasible solution")
        active = [
            rxn_id for rxn_id, flux in solution.fluxes.items()
            if abs(flux) > 1e-6 and not nc.is_bio_reaction(rxn_id)
        ]
    print(f"  Standard FBA: {len(active)} active reactions")
    return active


# MOMA analysis

def _ecb_run_moma_analysis(model, biomass_rxn, target_rxn, num_pts=10,
                            c_source_rxn='EX_cpd00027_e_reverse'):
    """Sample fluxes with MOMA using a shared carbon uptake limit.

    Return target covariances and the reaction-by-sample flux matrix."""
    print(f"Running MOMA analysis ({num_pts} sampling points)...")
    has_ec = check_enzyme_constraints(model)


    optimal_c_bound = None
    if c_source_rxn in [r.id for r in model.reactions]:
        c_rxn = model.reactions.get_by_id(c_source_rxn)
        orig_bounds = (c_rxn.lower_bound, c_rxn.upper_bound)
        c_rxn.upper_bound = 1000
        try:
            with model as m:
                m.objective = target_rxn
                sol = m.optimize()
                if sol.status == 'optimal':
                    optimal_c_bound = abs(sol.fluxes.get(c_source_rxn, 0))
                    print(f"  Unified carbon bound: {optimal_c_bound:.6f}")
        except Exception as e:
            print(f"  Warning: carbon bound calc failed ({e}), skipping")
        finally:
            c_rxn.lower_bound, c_rxn.upper_bound = orig_bounds

    if optimal_c_bound is not None:
        c_rxn = model.reactions.get_by_id(c_source_rxn)
        saved_ub = c_rxn.upper_bound
        c_rxn.upper_bound = optimal_c_bound

    try:
        with model.copy() as m:
            m.objective = biomass_rxn
            if has_ec:
                wt_dist = _ecb_optimize_minimize_enzyme_pool(m, biomass_rxn, fraction_of_optimum=1.0)
            else:
                wt_dist = m.optimize()

        with model as m:
            m.objective = target_rxn
            max_const = m.slim_optimize()
            m.objective_direction = 'min'
            min_const = m.slim_optimize()

        extended_max = max_const * 1.1 if max_const > 0 else max_const * 0.9
        extended_min = min_const * 1.1 if min_const < 0 else min_const * 0.9

        results  = {}
        successful = 0

        for i, f_target in enumerate(np.linspace(extended_min, extended_max, num_pts)):
            try:
                with model as m:
                    m.reactions.get_by_id(target_rxn).bounds = (f_target, f_target)
                    m.objective = biomass_rxn
                    f_bio = m.slim_optimize()
                    if isnan(f_bio) or f_bio < 1e-8:
                        continue
                    m.reactions.get_by_id(biomass_rxn).lower_bound = f_bio * 0.95
                    sol_moma = moma(m, solution=wt_dist, linear=True)
                    if sol_moma.status == 'optimal':
                        results[f"sample_{i+1}"] = sol_moma.fluxes
                        successful += 1
            except Exception:
                continue

    finally:
        if optimal_c_bound is not None:
            model.reactions.get_by_id(c_source_rxn).upper_bound = saved_ub

    if successful < 3:
        raise ValueError("Too few successful MOMA sampling points")

    df_flux = pd.DataFrame.from_dict(results)
    df_cov  = df_flux.abs().T.cov()

    print(f"  Successful samples: {successful}/{num_pts}")
    return df_cov[target_rxn], df_flux

def _ecb_run_moma_analysis_qy(model, biomass_rxn, target_rxn, num_pts=10,
                            c_source_rxn='EX_cpd00027_e_reverse'):
    print(f"Running MOMA analysis ({num_pts} sampling points)...")
    has_ec = check_enzyme_constraints(model)

    with model.copy() as m:
        m.objective = biomass_rxn
        if has_ec:
            wt_dist = _ecb_optimize_minimize_enzyme_pool(m, biomass_rxn, fraction_of_optimum=1.0)
        else:
            wt_dist = m.optimize()

    with model as m:
        m.objective = target_rxn
        max_const = m.slim_optimize()
        m.objective_direction = 'min'
        min_const = m.slim_optimize()

    extended_max = max_const * 1.1 if max_const > 0 else max_const * 0.9
    extended_min = min_const * 1.1 if min_const < 0 else min_const * 0.9

    results   = {}
    successful = 0

    for i, f_target in enumerate(np.linspace(extended_min, extended_max, num_pts)):
        try:
            with model as m:
                m.reactions.get_by_id(target_rxn).bounds = (f_target, f_target)
                m.objective = biomass_rxn
                f_bio = m.slim_optimize()
                if isnan(f_bio) or f_bio < 1e-8:
                    continue
                m.reactions.get_by_id(biomass_rxn).lower_bound = f_bio * 0.95
                sol_moma = moma(m, solution=wt_dist, linear=True)
                if sol_moma.status == 'optimal':
                    results[f"sample_{i+1}"] = sol_moma.fluxes
                    successful += 1
        except Exception:
            continue

    if successful < 3:
        raise ValueError("Too few successful MOMA sampling points")

    df_flux = pd.DataFrame.from_dict(results)
    df_cov  = df_flux.abs().T.cov()

    print(f"  Successful samples: {successful}/{num_pts}")
    return df_cov[target_rxn], df_flux

# Metabolite direction extraction

def _ecb_extract_metabolite_directions(model):
    """Return (dict_reactant, dict_product) for every metabolite in *model*."""
    dict_reactant, dict_product = {}, {}
    for met in model.metabolites:
        dict_reactant[met.id] = []
        dict_product[met.id]  = []
        for rxn in met.reactions:
            coeff = rxn.get_coefficient(met.id)
            if coeff < 0:
                dict_reactant[met.id].append(rxn.id)
            elif coeff > 0:
                dict_product[met.id].append(rxn.id)
    return dict_reactant, dict_product


# Metabolite scoring

def _ecb_calculate_metabolite_scores(model, df_cov, active_reactions, threshold=0.01, naming=None):
    """Score every metabolite using MOMA covariance signal (no topology weighting).
    Score_Weighted == Score_Normalized."""
    nc = _get_nc(naming)
    print("Calculating metabolite scores...")

    df_pcov  = df_cov[df_cov.abs() > threshold].dropna()
    pos_rxns = list(df_pcov[df_pcov > 0].index)
    neg_rxns = list(df_pcov[df_pcov < 0].index)

    active_set = set(active_reactions)
    metabolite_scores = {}

    for met in model.metabolites:
        if nc.is_excluded_metabolite(met.id) or nc.is_cofactor(met.id):
            metabolite_scores[met.id] = {
                'Metabolite_ID':    met.id,
                'Score_Positive':   0.0,
                'Score_Negative':   0.0,
                'Score_Normalized': 0.0,
                'Score_Weighted':   0.0,
                'Total_Reactions':  0,
                'Active_Reactions': 0,
                'Positive_Reactions': 0,
                'Negative_Reactions': 0,
                'Is_Excluded': True,
            }
            continue

        rxn_candidates = [rxn.id for rxn in met.reactions]
        rxn_pos = list(set(pos_rxns) & set(rxn_candidates))
        rxn_neg = list(set(neg_rxns) & set(rxn_candidates))

        score_pos = df_pcov[rxn_pos].abs().sum() if rxn_pos else 0
        score_neg = df_pcov[rxn_neg].abs().sum() if rxn_neg else 0

        active_in_met = list(set(rxn_candidates) & active_set)
        norm_factor   = np.sqrt(1 + get_unique_reaction_count(active_in_met, naming=naming))
        score_norm    = (score_pos - score_neg) / norm_factor

        metabolite_scores[met.id] = {
            'Metabolite_ID':    met.id,
            'Score_Positive':   float(score_pos),
            'Score_Negative':   float(score_neg),
            'Score_Normalized': float(score_norm),
            'Score_Weighted':   float(score_norm),
            'Total_Reactions':  len(rxn_candidates),
            'Active_Reactions': len(active_in_met),
            'Positive_Reactions': len(rxn_pos),
            'Negative_Reactions': len(rxn_neg),
            'Is_Excluded': False,
        }

    return metabolite_scores


# Target reaction identification

def _ecb_find_target_reactions(model, dict_reactant, dict_product,
                                metabolite_scores, exclude_met_file=None, naming=None):
    """Rank targets by metabolite conversion: negative-to-positive (Up),
    positive-to-negative (Down), positive optimization (Up), then negative
    optimization (Up). Higher priorities take precedence."""
    nc = _get_nc(naming)
    print("Identifying target reactions...")

    excluded_mets = set()
    if exclude_met_file:
        try:
            with open(exclude_met_file, 'r') as f:
                excluded_mets = {line.strip() for line in f}
        except FileNotFoundError:
            print(f"  Warning: exclusion file not found — {exclude_met_file}")

    met_pos, met_neg = [], []
    for met_id, scores in metabolite_scores.items():
        if nc.is_cofactor(met_id):
            continue
        if not nc.is_cytoplasmic_metabolite(met_id) or met_id in excluded_mets or scores['Is_Excluded']:
            continue
        if scores['Score_Weighted'] > 0:
            met_pos.append(met_id)
        elif scores['Score_Weighted'] < 0:
            met_neg.append(met_id)

    target_reactions = {}


    print("  Processing Neg_to_Pos (highest priority)")
    neg_to_pos_count = excluded_count_1 = 0

    for met1 in met_neg:
        s1 = abs(metabolite_scores[met1]['Score_Weighted'])
        for met2 in met_pos:
            s2 = abs(metabolite_scores[met2]['Score_Weighted'])
            r_np = set(dict_product[met2]) & set(dict_reactant[met1])
            for rxn_id in r_np:
                if nc.is_excluded_reaction(rxn_id):
                    excluded_count_1 += 1
                    continue
                new_soc = s1 + s2
                if rxn_id not in target_reactions:
                    target_reactions[rxn_id] = {
                        'reaction_id':       rxn_id,
                        'source_metabolite': met1,
                        'target_metabolite': met2,
                        'regulation_type':   'Up',
                        'conversion_type':   'Neg_to_Pos',
                        'soc_score':         new_soc,
                        'source_score':      -s1,
                        'target_score':      s2,
                        'priority_reason':   'Primary_Bridge_Reaction',
                    }
                    neg_to_pos_count += 1
                elif target_reactions[rxn_id]['conversion_type'] == 'Neg_to_Pos':
                    if new_soc > target_reactions[rxn_id]['soc_score']:
                        target_reactions[rxn_id].update({
                            'source_metabolite': met1, 'target_metabolite': met2,
                            'soc_score': new_soc, 'source_score': -s1, 'target_score': s2,
                        })

    print(f"    {neg_to_pos_count} Neg_to_Pos reactions found")
    if excluded_count_1:
        print(f"    {excluded_count_1} reactions excluded")


    print("  Processing Pos_to_Neg (secondary priority)")

    active_rules = []
    for rule in METABOLIC_EXCLUSION_KDdirection:
        active_sources = rule['source_metabolites'] & set(met_pos)
        if active_sources:
            active_rules.append({
                'sources': active_sources,
                'targets': rule['target_metabolites'],
                'description': rule['description'],
            })

    pos_to_neg_count = 0
    skipped_priority = skipped_rule = skipped_better_prod = excluded_count_2 = 0

    for met1 in met_neg:
        s1 = abs(metabolite_scores[met1]['Score_Weighted'])
        for met2 in met_pos:
            s2 = abs(metabolite_scores[met2]['Score_Weighted'])

            skip_rule = any(
                met2 in rule['sources'] and met1 in rule['targets']
                for rule in active_rules
            )
            if skip_rule:
                skipped_rule += 1
                continue

            r_pn = set(dict_reactant[met2]) & set(dict_product[met1])
            for rxn_id in r_pn:
                if nc.is_excluded_reaction(rxn_id):
                    excluded_count_2 += 1
                    continue

                should_skip = False
                try:
                    rxn = model.reactions.get_by_id(rxn_id)
                    met2_score = metabolite_scores[met2]['Score_Weighted']
                    for product_met, coeff in rxn.metabolites.items():
                        if (coeff > 0 and product_met.id in metabolite_scores
                                and not nc.is_cofactor(product_met.id)):
                            ps = metabolite_scores[product_met.id]['Score_Weighted']
                            if ps > 0 and ps > met2_score:
                                should_skip = True
                                skipped_better_prod += 1
                                break
                except KeyError:
                    pass
                if should_skip:
                    continue

                new_soc = s1 + s2
                if rxn_id not in target_reactions:
                    target_reactions[rxn_id] = {
                        'reaction_id':       rxn_id,
                        'source_metabolite': met2,
                        'target_metabolite': met1,
                        'regulation_type':   'Down',
                        'conversion_type':   'Pos_to_Neg',
                        'soc_score':         new_soc,
                        'source_score':      s2,
                        'target_score':      -s1,
                        'priority_reason':   'Secondary_Prevention',
                    }
                    pos_to_neg_count += 1
                elif target_reactions[rxn_id]['conversion_type'] == 'Pos_to_Neg':
                    if new_soc > target_reactions[rxn_id]['soc_score']:
                        target_reactions[rxn_id].update({
                            'source_metabolite': met2, 'target_metabolite': met1,
                            'soc_score': new_soc, 'source_score': s2, 'target_score': -s1,
                        })
                else:
                    skipped_priority += 1

    print(f"    {pos_to_neg_count} Pos_to_Neg reactions found")


    print("  Processing Pos_Optimization (third priority)")

    sorted_pos = sorted(
        [(m, abs(metabolite_scores[m]['Score_Weighted'])) for m in met_pos],
        key=lambda x: x[1], reverse=True
    )

    pos_opt_count = excluded_count_3 = 0
    for i in range(len(sorted_pos)):
        for j in range(i):
            met_lower,  s_lower  = sorted_pos[i]
            met_higher, s_higher = sorted_pos[j]
            r_lh = set(dict_reactant[met_lower]) & set(dict_product[met_higher])
            if not (r_lh and s_higher > s_lower):
                continue
            for rxn_id in r_lh:
                if nc.is_excluded_reaction(rxn_id):
                    excluded_count_3 += 1
                    continue
                new_soc = s_higher - s_lower
                if rxn_id not in target_reactions:
                    target_reactions[rxn_id] = {
                        'reaction_id':       rxn_id,
                        'source_metabolite': met_lower,
                        'target_metabolite': met_higher,
                        'regulation_type':   'Up',
                        'conversion_type':   'Pos_Optimization',
                        'soc_score':         new_soc,
                        'source_score':      s_lower,
                        'target_score':      s_higher,
                        'priority_reason':   'Tertiary_Optimization',
                    }
                    pos_opt_count += 1
                elif target_reactions[rxn_id]['conversion_type'] == 'Pos_Optimization':
                    if new_soc > target_reactions[rxn_id]['soc_score']:
                        target_reactions[rxn_id].update({
                            'source_metabolite': met_lower, 'target_metabolite': met_higher,
                            'soc_score': new_soc, 'source_score': s_lower, 'target_score': s_higher,
                        })

    print(f"    {pos_opt_count} Pos_Optimization reactions found")
    print(f"  Total target reactions: {len(target_reactions)}")
    return target_reactions


# Isozyme helper

def _ecb_find_isozyme_reactions(model, target_reaction_id, naming=None):
    """Return all reactions sharing the same base ID (ignoring _numN suffix)."""
    nc = _get_nc(naming)
    return nc.find_isozyme_reactions(model, target_reaction_id)


# Result processing

def _ecb_process_target_reactions(target_reactions, metabolite_scores, active_reactions,
                                   model, top_score_percentage=0.4, return_all=False, naming=None):
    """Build result DataFrames and apply the SoC-based percentile filter."""
    print("Processing target reactions...")
    if not target_reactions:
        empty = pd.DataFrame()
        return (empty, empty) if return_all else empty

    active_set     = set(active_reactions)
    priority_order = {'Neg_to_Pos': 1, 'Pos_to_Neg': 2, 'Pos_Optimization': 3}
    results        = []

    for rxn_data in target_reactions.values():
        rxn_id    = rxn_data['reaction_id']
        is_active = rxn_id in active_set

        try:
            rxn = model.reactions.get_by_id(rxn_id)
            reaction_name     = rxn.name or rxn_id
            reaction_equation = str(rxn.reaction)
        except Exception:
            reaction_name, reaction_equation = rxn_id, "Unknown"

        isozymes      = _ecb_find_isozyme_reactions(model, rxn_id, naming=naming)
        isozyme_count = len(isozymes)
        active_iso    = sum(1 for iso in isozymes if iso in active_set)

        results.append({
            'Target_reactions':        rxn_id,
            'Reaction_name':           reaction_name,
            'Reaction_equation':       reaction_equation,
            'Regulation_type':         rxn_data['regulation_type'],
            'Conversion_type':         rxn_data['conversion_type'],
            'Priority_reason':         rxn_data.get('priority_reason', 'Unknown'),
            'SoC':                     rxn_data['soc_score'],
            'Source_metabolite':       rxn_data['source_metabolite'],
            'Destination_metabolite':  rxn_data['target_metabolite'],
            'Source_score':            rxn_data['source_score'],
            'Target_score':            rxn_data['target_score'],
            'Is_Active':               is_active,
            'Isozyme_count':           isozyme_count,
            'Active_isozymes':         active_iso,
            'All_isozymes':            ';'.join(isozymes) if len(isozymes) > 1 else rxn_id,
            'Selection_reason':        '',
            'Type_rank':               0,
            'Global_SoC_rank':         0,
        })

    df = pd.DataFrame(results)
    selected_reactions = []

    for conv_type in df['Conversion_type'].unique():
        type_df     = df[df['Conversion_type'] == conv_type].copy()
        type_sorted = type_df.sort_values('SoC', ascending=False)

        for rank, idx in enumerate(type_sorted.index):
            df.loc[idx, 'Type_rank'] = rank + 1

        ideal_count = max(1, int(len(type_sorted) * top_score_percentage))
        if ideal_count >= len(type_sorted):
            top_df = type_sorted
        else:
            cutoff = type_sorted.iloc[ideal_count - 1]['SoC']
            top_df = type_sorted[type_sorted['SoC'] >= cutoff]

        for idx in top_df.index:
            rank = df.loc[idx, 'Type_rank']
            soc  = df.loc[idx, 'SoC']
            act  = df.loc[idx, 'Is_Active']
            tag  = 'Active' if act else 'Inactive'
            df.loc[idx, 'Selection_reason'] = (
                f'Top_{top_score_percentage:.0%}_SoC_{conv_type}_{tag}_Rank_{rank}_Score_{soc:.3f}'
            )

        remaining  = type_sorted[~type_sorted.index.isin(top_df.index)]
        active_sup = remaining[remaining['Is_Active'] == True]

        for idx in active_sup.index:
            rank = df.loc[idx, 'Type_rank']
            soc  = df.loc[idx, 'SoC']
            df.loc[idx, 'Selection_reason'] = (
                f'Active_supplement_{conv_type}_SoC_Rank_{rank}_Score_{soc:.3f}'
            )

        type_selected = pd.concat([top_df, active_sup])
        selected_reactions.append(type_selected)
        print(f"    {conv_type}: {len(type_selected)} reactions selected")

    if selected_reactions:
        final_df = pd.concat(selected_reactions, ignore_index=True)
        final_df['_prio'] = final_df['Conversion_type'].map(priority_order).fillna(999)
        final_df = final_df.sort_values(['_prio', 'SoC'], ascending=[True, False])
        final_df = final_df.reset_index(drop=True)
        final_df['Global_Rank'] = range(1, len(final_df) + 1)
        soc_sorted = final_df.sort_values('SoC', ascending=False)
        for gr, idx in enumerate(soc_sorted.index):
            final_df.loc[idx, 'Global_SoC_rank'] = gr + 1
        final_df.drop('_prio', axis=1, inplace=True)
    else:
        final_df = pd.DataFrame()

    df_all = df.copy()
    df_all['_prio'] = df_all['Conversion_type'].map(priority_order).fillna(999)
    df_all = df_all.sort_values(['_prio', 'SoC'], ascending=[True, False]).reset_index(drop=True)
    df_all['Global_Rank'] = range(1, len(df_all) + 1)
    soc_sorted2 = df_all.sort_values('SoC', ascending=False)
    for gr, idx in enumerate(soc_sorted2.index):
        df_all.loc[idx, 'Global_SoC_rank'] = gr + 1
    df_all.drop('_prio', axis=1, inplace=True)

    print(f"  Final selection: {len(final_df)} reactions")
    return (df_all, final_df) if return_all else final_df


# ecBridge entry point

def predict_ecbridge_targets(model,
                              exclude_met_file=None,
                              target_compound=None,
                              biomass_rxn='Bio_cgATCC14067_c',
                              num_step=20,
                              bio_fraction=0.1,
                              threshold=0.01,
                              top_score_percentage=0.4,
                              c_source_rxn='EX_cpd00027_e_reverse',
                              verbose=True,
                              output_dir=None,
                              naming=None):
    """Predict ecBridge targets using MOMA covariance and metabolite scores.

    Apply exclusions and the top-SoC filter; return all and selected targets."""
    if verbose:
        print("=" * 60)
        print("ecBridge — topology-free version")
        print("Priority: Neg_to_Pos > Pos_to_Neg > Pos_Optimization")
        print("=" * 60)
        print(f"MOMA samples: {num_step}  |  selection: {top_score_percentage:.0%}")

    nc = _get_nc(naming)

    if target_compound:
        target_chem_id = target_compound
    else:
        target_chem_id = '_'.join(model.id.split('_')[1:])
        target_chem_id = target_chem_id.split('_path_')[0]

    target_rxn = nc.make_exchange_rxn(target_chem_id)
    if verbose:
        print(f"Target reaction: {target_rxn}")
        nc.summary()

    with model as m:
        m.objective = biomass_rxn
        max_bio  = m.slim_optimize()
        m.objective = target_rxn
        max_prod = m.slim_optimize()
        if isnan(max_bio) or max_bio < 1e-6:
            raise ValueError("Model cannot grow")
        if isnan(max_prod) or max_prod < 1e-6:
            raise ValueError("Model cannot produce target compound")

    dict_reactant, dict_product = _ecb_extract_metabolite_directions(model)
    active_reactions = _ecb_find_active_reactions(model, target_rxn, biomass_rxn, bio_fraction, naming=naming)
    df_cov, df_flux  = _ecb_run_moma_analysis(model, biomass_rxn, target_rxn, num_step, c_source_rxn)

    model_rxn_ids = [rxn.id for rxn in model.reactions]
    df_cov = df_cov.reindex(model_rxn_ids).dropna()

    metabolite_scores = _ecb_calculate_metabolite_scores(model, df_cov, active_reactions, threshold, naming=naming)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        tag = target_chem_id
        cov_path = os.path.join(output_dir, f'{tag}_reaction_covariance.csv')
        df_cov_out = df_cov.reset_index()
        df_cov_out.columns = ['Reaction_ID', 'Covariance_with_target']
        df_cov_out = df_cov_out.sort_values('Covariance_with_target',
                                            key=lambda x: x.abs(), ascending=False)
        df_cov_out.to_csv(cov_path, index=False)
        print(f"  [debug] Reaction covariance saved → {cov_path}")

        met_path = os.path.join(output_dir, f'{tag}_metabolite_scores.csv')
        met_df = pd.DataFrame.from_dict(metabolite_scores, orient='index')
        met_df = met_df.sort_values('Score_Weighted', key=lambda x: x.abs(), ascending=False)
        met_df.to_csv(met_path, index=False)
        print(f"  [debug] Metabolite scores saved  → {met_path}")

        flux_path = os.path.join(output_dir, f'{tag}_moma_flux_matrix.csv')
        df_flux.to_csv(flux_path)
        print(f"  [debug] MOMA flux matrix saved   → {flux_path}")

    target_reactions = _ecb_find_target_reactions(
        model, dict_reactant, dict_product, metabolite_scores, exclude_met_file, naming=naming
    )

    all_results, final_results = _ecb_process_target_reactions(
        target_reactions, metabolite_scores, active_reactions,
        model, top_score_percentage, return_all=True, naming=naming
    )

    if verbose:
        print("\n" + "=" * 60)
        print("ecBridge analysis complete")
        print("=" * 60)
        print(f"Total candidate reactions : {len(all_results)}")
        print(f"Selected reactions        : {len(final_results)}")
        if not final_results.empty:
            print(f"  Active   : {int(final_results['Is_Active'].sum())}")
            print(f"  Up (OE)  : {(final_results['Regulation_type']=='Up').sum()}")
            print(f"  Down (KD): {(final_results['Regulation_type']=='Down').sum()}")

    return all_results, final_results


# ecFSEOF CORE ALGORITHM


class ecFSEOF:
    """Predict gene targets from product-growth scans and enzyme-cost minimization.

    Aggregate reaction V-scores into gene K-scores, classify OE/KD/BOTH,
    and filter by enzyme data, essentiality, and score percentage."""

    def __init__(self,
                 model,
                 product_rxn,
                 biomass_rxn=None,
                 fseof_steps=10,
                 top_percentage=1,
                 separate_oe_kd_filtering=True,
                 filtering_strategy='include_ties',
                 essential_genes_file=None,
                 auto_identify_essential=True,
                 additional_essential_genes=None,
                 essential_genes_cache_dir='essential_genes_cache',
                 force_reidentify_essential=False,
                 c_source_rxn='EX_cpd00027_e_reverse',
                 output_dir=None,
                 naming=None):
        """Parameters
        ----------
        model : cobra.Model Enzyme-constrained metabolic model.
        product_rxn : str Target product exchange reaction ID.
        biomass_rxn : str, optional Auto-detected from objective if None.
        fseof_steps : int Pareto points (default 10).
        top_percentage : float Fraction of top-ranked genes to retain (0–1).
        separate_oe_kd_filtering : bool Separate percentage filters for OE/KD.
        filtering_strategy : str 'include_ties' or 'strict_rank'.
        essential_genes_file : str, optional Path to essential gene ID list.
        auto_identify_essential : bool Run in-silico knockout (default True).
        additional_essential_genes : list, optional Extra essential gene IDs.
        essential_genes_cache_dir : str Cache directory.
        force_reidentify_essential : bool Ignore cache.
        c_source_rxn : str Carbon-source reaction ID.
        output_dir : str, optional Debug CSV output directory.
        naming : str or NamingConvention, optional 'modelseed' or 'bigg'."""
        self.model         = model.copy()
        self.product_rxn   = product_rxn
        self.nc            = _get_nc(naming)
        self.fseof_steps   = fseof_steps
        self.top_percentage = top_percentage
        self.oe_percentage  = top_percentage
        self.kd_percentage  = top_percentage
        self.separate_oe_kd_filtering = separate_oe_kd_filtering
        self.filtering_strategy       = filtering_strategy
        self.essential_genes_cache_dir = essential_genes_cache_dir
        self.constraint_rxn_id = c_source_rxn
        self.output_dir    = output_dir
        self.tol           = 1e-13

        if biomass_rxn is None:
            self.biomass_rxn = list(self.model.objective.keys())[0].id
        else:
            self.biomass_rxn = biomass_rxn

        self.targets           = {'OE': [], 'KD': []}
        self.flux_matrix       = None
        self.reaction_scores   = None
        self.gene_scores       = None
        self.bidirectional_genes = []
        self.optimal_constraint_bound = None

        os.makedirs(self.essential_genes_cache_dir, exist_ok=True)
        self._set_quiet_mode(True)

        if auto_identify_essential:
            print("Loading essential genes (with cache)...")
            self.essential_genes = self._load_or_identify_essential_genes(
                essential_genes_file, force_reidentify_essential
            )
        else:
            self.essential_genes = self._load_essential_genes(essential_genes_file)

        if additional_essential_genes:
            before = len(self.essential_genes)
            self.essential_genes = list(set(self.essential_genes + additional_essential_genes))
            print(f"  Added {len(self.essential_genes) - before} user-specified essential genes")

        self.has_enzyme_constraints = self._check_enzyme_constraints()
        self._calculate_optimal_constraint_bound()
        self._close_unwanted_reactions()
        self._print_init_info()


    def run(self):
        """Run the full EC-FSEOF pipeline."""
        print("\n" + "=" * 60)
        print("EC-FSEOF Analysis")
        print("=" * 60)
        print("\n[Step 1] EC-FSEOF scan")
        self._ec_fseof()

        print("\n" + "=" * 60)
        print(f"EC-FSEOF complete — OE: {len(self.targets['OE'])}, KD: {len(self.targets['KD'])}")
        print("=" * 60)

        all_df   = self._build_output_dataframe(filtered=False)
        final_df = self._build_output_dataframe(filtered=True)
        return all_df, final_df


    def _ec_fseof(self):
        """Main EC-FSEOF routine."""
        print("Strategy: Real enzyme cost minimisation (minpool)")
        original_bounds = self._apply_optimal_constraint()
        constraint_tolerance   = 0.0001
        min_absolute_tolerance = 1e-6

        try:
            print("\nStage 1: Calculating Pareto-optimal points...")
            pareto_points = self._calculate_pareto_points()

            print(f"\nStage 2: Batch minpool optimisation "
                  f"(tolerance {constraint_tolerance * 100:.1f}%)...")
            flux_matrix = self._batch_optimize_with_enzyme_cost(
                pareto_points, constraint_tolerance, min_absolute_tolerance
            )

            print("\nStage 3: Analysing flux trends...")
            self._process_flux_matrix(flux_matrix)
            self._analyze_flux_trends()
            self._apply_filtering()

            print("\nEC-FSEOF scan complete!")
        finally:
            self._restore_constraint_bounds(original_bounds)

    def _optimize_minimize_enzyme_pool(self, model, objective_rxn, fraction_of_optimum=1.0):
        """Two-stage minpool optimisation (internal to ecFSEOF)."""
        model.objective = objective_rxn
        model.objective_direction = 'max'
        max_objective = model.slim_optimize()

        if max_objective is None or max_objective <= 1e-9:
            return model.optimize()

        target_flux = max_objective * fraction_of_optimum
        tolerance   = abs(target_flux) * 0.0001

        obj_rxn = model.reactions.get_by_id(objective_rxn)
        original_bounds = obj_rxn.bounds
        obj_rxn.bounds  = (target_flux - tolerance, target_flux + tolerance)

        default_penalty = 1e-9
        enzyme_cost_obj = {}
        for rxn in model.reactions:
            if hasattr(rxn, 'kcat_MW') and rxn.kcat_MW:
                try:
                    kcat_mw = float(rxn.kcat_MW)
                    enzyme_cost_obj[rxn] = -1.0 / kcat_mw if kcat_mw > 0 else -default_penalty
                except (ValueError, TypeError):
                    enzyme_cost_obj[rxn] = -default_penalty
            else:
                enzyme_cost_obj[rxn] = -default_penalty

        model.objective = enzyme_cost_obj
        model.objective_direction = 'max'
        solution = model.optimize()
        obj_rxn.bounds = original_bounds
        return solution

    def _calculate_pareto_points(self):
        """Calculate Pareto-optimal (product, biomass) pairs."""
        print("Calculating maximum theoretical product yield...")
        with self.model as m:
            m.objective = self.product_rxn
            sol = m.optimize()
            max_product = sol.objective_value * 0.9
            print(f"  Max product flux : {sol.objective_value:.6f}")
            print(f"  FSEOF upper limit: {max_product:.6f}")

        if max_product <= 0:
            raise ValueError(f"Product flux is not positive: {max_product}")

        print("Calculating wild-type baseline...")
        with self.model as m:
            m.objective = self.biomass_rxn
            sol = m.optimize()
            print(f"  Wild-type growth  : {sol.objective_value:.6f}")
            print(f"  Wild-type product : {sol.fluxes[self.product_rxn]:.6f}")

        product_constraints = np.linspace(max_product / 9, max_product, self.fseof_steps)

        pareto_points = []
        for i, pval in enumerate(product_constraints):
            try:
                with self.model.copy() as pm:
                    pm.reactions.get_by_id(self.product_rxn).bounds = (pval, pval)
                    pm.objective = self.biomass_rxn
                    sol = pm.optimize()
                    if sol.status != 'optimal':
                        print(f"  Point {i+1}: infeasible ({sol.status})")
                        pareto_points.append((pval, 0))
                        continue
                    actual_p = sol.fluxes[self.product_rxn]
                    pareto_points.append((actual_p, sol.objective_value))
                    print(f"  Point {i+1}: product={actual_p:.4f}, "
                          f"biomass={sol.objective_value:.4f}")
            except Exception as e:
                print(f"  Point {i+1}: error — {e}")
                pareto_points.append((pval, 0))

        print(f"Pareto points calculated: {len(pareto_points)}")
        return pareto_points

    def _batch_optimize_with_enzyme_cost(self, pareto_points,
                                          constraint_tolerance,
                                          min_absolute_tolerance):
        """Run minpool at every Pareto point and collect the flux matrix."""
        rxn_ids     = [r.id for r in self.model.reactions]
        flux_matrix = np.zeros((len(rxn_ids), len(pareto_points)))

        for i, (product_flux, biomass_flux) in enumerate(pareto_points):
            try:
                print(f"  Optimising point {i+1}/{len(pareto_points)}: "
                      f"product={product_flux:.4f}, biomass={biomass_flux:.4f}")

                if biomass_flux <= 1e-6:
                    print("    Skipped: biomass too low")
                    continue

                with self.model.copy() as om:
                    if (self.product_rxn not in om.reactions or
                            self.biomass_rxn not in om.reactions):
                        continue

                    ptol = (abs(product_flux) * constraint_tolerance
                            if abs(product_flux) > min_absolute_tolerance
                            else min_absolute_tolerance)
                    om.reactions.get_by_id(self.product_rxn).bounds = (
                        max(0, product_flux - ptol), product_flux
                    )

                    btol = (abs(biomass_flux) * constraint_tolerance
                            if abs(biomass_flux) > min_absolute_tolerance
                            else min_absolute_tolerance)
                    om.reactions.get_by_id(self.biomass_rxn).bounds = (
                        max(0, biomass_flux - btol), biomass_flux
                    )

                    sol = self._optimize_minimize_enzyme_pool(
                        om, self.product_rxn, fraction_of_optimum=1.0
                    )

                    if sol.status != 'optimal':
                        print(f"    Failed: {sol.status}")
                        continue

                    for j, rid in enumerate(rxn_ids):
                        if rid in sol.fluxes:
                            flux_matrix[j, i] = sol.fluxes[rid]

                    fp = sol.fluxes.get(self.product_rxn, 0)
                    fb = sol.fluxes.get(self.biomass_rxn, 0)
                    print(f"    Result: product={fp:.4f}, biomass={fb:.4f}")

            except Exception as e:
                print(f"    Error at point {i+1}: {e}")

        print("Batch optimisation complete")
        return flux_matrix


    def _process_flux_matrix(self, flux_matrix):
        """Convert raw flux_matrix array into a filtered DataFrame."""
        rxn_ids = [r.id for r in self.model.reactions]
        self.flux_matrix = pd.DataFrame(
            flux_matrix,
            index=rxn_ids,
            columns=[f"alpha_{i}" for i in range(flux_matrix.shape[1])]
        )
        print(f"Flux matrix shape: {self.flux_matrix.shape}")

        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
            tag      = self.nc.extract_product_tag(self.product_rxn)
            raw_path = os.path.join(self.output_dir, f'{tag}_flux_matrix_raw.csv')
            self.flux_matrix.to_csv(raw_path)
            print(f"  [debug] Raw flux matrix saved → {raw_path}")

        keep = [
            rid for rid in self.flux_matrix.index
            if rid in self.model.reactions
            and self.model.reactions.get_by_id(rid).gene_reaction_rule
            and 'standard' not in self.model.reactions.get_by_id(rid).gene_reaction_rule
            and not np.all(np.abs(self.flux_matrix.loc[rid].values) < 1e-6)
        ]
        self.flux_matrix = self.flux_matrix.loc[keep]
        print(f"Filtered reactions: {len(self.flux_matrix)}")

        if self.output_dir is not None:
            filt_path = os.path.join(self.output_dir, f'{tag}_flux_matrix_filtered.csv')
            self.flux_matrix.to_csv(filt_path)
            print(f"  [debug] Filtered flux matrix saved → {filt_path}")

    def _analyze_flux_trends(self):
        """Calculate V-scores, then K-scores, then classify genes."""
        print("Calculating V-scores...")
        if self.flux_matrix is None or self.flux_matrix.empty:
            raise ValueError("Flux matrix not available")

        biomax_fluxes = self.flux_matrix.iloc[:, 0]
        target_rxns, v_scores, target_types, has_kcat_list = [], [], [], []

        for rxn_id in self.flux_matrix.index:
            try:
                rxn = self.model.reactions.get_by_id(rxn_id)
                has_kcat  = self._has_valid_kcat_mw(rxn)
                v_biomax  = abs(biomax_fluxes[rxn_id])
                flux_vals = np.abs(self.flux_matrix.loc[rxn_id].values)
                avg_flux  = np.mean(flux_vals)

                if v_biomax < 1e-12:
                    v_score = 1000 if avg_flux > 1e-12 else None
                else:
                    v_score = min(avg_flux / v_biomax, 1000)

                if v_score is None:
                    continue
                if v_score > 1.0:
                    t = 'OE'
                elif 0.05 < v_score < 1.0:
                    t = 'KD'
                else:
                    continue

                target_rxns.append(rxn_id)
                v_scores.append(v_score)
                target_types.append(t)
                has_kcat_list.append(has_kcat)

            except Exception as e:
                print(f"  Warning: reaction {rxn_id} — {e}")

        self.reaction_scores = pd.DataFrame(
            {'v_score': v_scores, 'type': target_types, 'has_kcat_mw': has_kcat_list},
            index=target_rxns
        ).sort_values('v_score', ascending=False)

        print(f"V-scores: {len(self.reaction_scores)} candidate reactions")
        self._calculate_gene_scores()
        self._classify_genes()

    def _calculate_gene_scores(self):
        """Aggregate V-scores into per-gene K-scores."""
        print("Calculating K-scores...")
        gene_scores     = {}
        gene_rxn_counts = {}
        gene_rxn_types  = {}
        gene_has_kcat   = {}

        for rxn_id, row in self.reaction_scores.iterrows():
            try:
                rxn = self.model.reactions.get_by_id(rxn_id)
                if not rxn.genes:
                    continue
                for gene in rxn.genes:
                    gid = gene.id
                    gene_scores.setdefault(gid, 0)
                    gene_rxn_counts.setdefault(gid, 0)
                    gene_rxn_types.setdefault(gid, {})
                    gene_has_kcat.setdefault(gid, False)
                    gene_scores[gid]     += row['v_score']
                    gene_rxn_counts[gid] += 1
                    if row['has_kcat_mw']:
                        gene_has_kcat[gid] = True
                    rtype = row['type']
                    gene_rxn_types[gid][rtype] = gene_rxn_types[gid].get(rtype, 0) + 1
            except Exception as e:
                print(f"  Warning: gene aggregation for {rxn_id} — {e}")

        k_scores_data = {}
        self.bidirectional_genes = []

        for gid in gene_scores:
            k = gene_scores[gid] / gene_rxn_counts[gid]
            has_oe = 'OE' in gene_rxn_types[gid]
            has_kd = 'KD' in gene_rxn_types[gid]

            if has_oe and has_kd:
                self.bidirectional_genes.append(gid)
                gtype = 'BOTH'
            elif k > 1.0:
                gtype = 'OE'
            elif k < 1.0:
                gtype = 'KD'
            else:
                gtype = 'NEUTRAL'

            k_scores_data[gid] = {
                'k_score':          k,
                'v_score_sum':      gene_scores[gid],
                'reaction_count':   gene_rxn_counts[gid],
                'type':             gtype,
                'reaction_types':   gene_rxn_types[gid],
                'is_bidirectional': gid in self.bidirectional_genes,
                'has_kcat_mw':      gene_has_kcat.get(gid, False),
            }

        self.gene_scores = pd.DataFrame.from_dict(
            k_scores_data, orient='index'
        ).sort_values('k_score', ascending=False)
        print(f"K-scores for {len(self.gene_scores)} genes "
              f"({len(self.bidirectional_genes)} bidirectional)")

    def _classify_genes(self):
        """Populate self.targets from self.gene_scores."""
        self.targets = {'OE': [], 'KD': []}
        for gid, row in self.gene_scores.iterrows():
            gtype = row['type']
            if gtype == 'BOTH':
                self.targets['OE'].append(gid)
                self.targets['KD'].append(gid)
            elif gtype == 'OE':
                self.targets['OE'].append(gid)
            elif gtype == 'KD':
                self.targets['KD'].append(gid)
        for k in self.targets:
            self.targets[k] = list(set(self.targets[k]))
        print(f"Classification: OE={len(self.targets['OE'])}, KD={len(self.targets['KD'])}")


    def _apply_filtering(self):
        """Three-stage filter:
        1. Remove genes without kcat_MW.
        2. Remove essential genes from KD pool.
        3. Apply percentage filter per OE/KD pool."""
        if self.gene_scores is None or self.gene_scores.empty:
            self.targets = {'OE': [], 'KD': []}
            return

        no_kcat = self.gene_scores[~self.gene_scores['has_kcat_mw']]
        if len(no_kcat):
            print(f"Removing {len(no_kcat)} genes without kcat_MW")
        self.gene_scores = self.gene_scores[self.gene_scores['has_kcat_mw'] == True]

        if self.gene_scores.empty:
            self.targets = {'OE': [], 'KD': []}
            return

        essential_set = set(self.essential_genes)
        for rxn in self.model.reactions:
            if self.nc.is_bio_reaction(rxn.id):
                for g in rxn.genes:
                    essential_set.add(g.id)

        kd_candidates = [
            gid for gid in self.gene_scores.index
            if self.gene_scores.loc[gid, 'type'] in ('KD', 'BOTH')
        ]
        kd_essential = [gid for gid in kd_candidates if gid in essential_set]
        if kd_essential:
            print(f"Removing {len(kd_essential)} essential genes from KD pool")
            for gid in kd_essential:
                if self.gene_scores.loc[gid, 'type'] == 'BOTH':
                    self.gene_scores.at[gid, 'type'] = 'OE'
                    self.gene_scores.at[gid, 'is_bidirectional'] = False
                elif self.gene_scores.loc[gid, 'type'] == 'KD':
                    self.gene_scores = self.gene_scores.drop(index=gid)

        print(f"After essential gene filter: {len(self.gene_scores)} genes remain")
        if self.gene_scores.empty:
            self.targets = {'OE': [], 'KD': []}
            return

        print(f"Filtering strategy: {self.filtering_strategy}")
        if self.separate_oe_kd_filtering:
            self._apply_separate_filtering()
        else:
            self._apply_overall_filtering()

    def _apply_overall_filtering(self):
        both   = self.gene_scores[self.gene_scores['type'] == 'BOTH']
        normal = self.gene_scores[self.gene_scores['type'] != 'BOTH']
        filtered_normal = (
            self._filter_by_percentage(normal, self.top_percentage, 'ALL')
            if not normal.empty else pd.DataFrame()
        )
        parts    = [p for p in [both, filtered_normal] if not p.empty]
        filtered = pd.concat(parts).sort_values('k_score', ascending=False) if parts else pd.DataFrame()
        self._rebuild_targets_from_df(filtered)

    def _apply_separate_filtering(self):
        print(f"Separate filter: OE top {self.oe_percentage*100:.1f}%, "
              f"KD top {self.kd_percentage*100:.1f}%")
        both    = self.gene_scores[self.gene_scores['type'] == 'BOTH']
        oe_only = self.gene_scores[self.gene_scores['type'] == 'OE']
        kd_only = self.gene_scores[self.gene_scores['type'] == 'KD']

        parts = []
        if not both.empty:
            parts.append(both)
            print(f"  Bidirectional: {len(both)} (all kept)")
        if not oe_only.empty:
            parts.append(self._filter_by_percentage(oe_only, self.oe_percentage, 'OE'))
        if not kd_only.empty:
            parts.append(self._filter_by_percentage(kd_only, self.kd_percentage, 'KD'))

        filtered = pd.concat(parts) if parts else pd.DataFrame()
        self.gene_scores = filtered
        self._rebuild_targets_from_df(filtered)

    def _filter_by_percentage(self, df, percentage, label):
        n         = max(1, int(len(df) * percentage))
        ascending = (label == 'KD')
        sorted_df = df.sort_values('k_score', ascending=ascending)

        if self.filtering_strategy == 'include_ties':
            cutoff   = sorted_df.iloc[n - 1]['k_score']
            selected = sorted_df[sorted_df['k_score'] <= cutoff] if ascending else \
                       sorted_df[sorted_df['k_score'] >= cutoff]
        else:
            selected = sorted_df.head(n)

        print(f"  {label}: {len(df)} → {len(selected)}")
        return selected

    def _rebuild_targets_from_df(self, df):
        self.targets = {'OE': [], 'KD': []}
        if df.empty:
            return
        for gid, row in df.iterrows():
            gtype = row['type']
            if gtype == 'BOTH':
                self.targets['OE'].append(gid)
                self.targets['KD'].append(gid)
            elif gtype == 'OE':
                self.targets['OE'].append(gid)
            elif gtype == 'KD':
                self.targets['KD'].append(gid)
        print(f"After filtering: OE={len(self.targets['OE'])}, KD={len(self.targets['KD'])}")


    def _build_output_dataframe(self, filtered=True):
        """Build a clean gene-level DataFrame.
        filtered=True → genes that passed percentage filtering.
        filtered=False → all genes that passed kcat_MW and essential-gene filters."""
        if self.gene_scores is None or self.gene_scores.empty:
            return pd.DataFrame(columns=[
                'gene_id', 'gene_name', 'type', 'k_score',
                'v_score_sum', 'reaction_count', 'is_bidirectional', 'Target_reactions'
            ])

        if filtered:
            selected_genes = set(self.targets['OE'] + self.targets['KD'])
            scores_df = self.gene_scores[self.gene_scores.index.isin(selected_genes)]
        else:
            scores_df = self.gene_scores

        rows = []
        for gid, row in scores_df.iterrows():
            try:
                gene = self.model.genes.get_by_id(gid)
                gene_name = gene.name if (hasattr(gene, 'name') and gene.name) else ''
                metabolic_rxns = sorted(
                    r.id for r in gene.reactions if not self.nc.is_bio_reaction(r.id)
                )
                rxn_ids = ';'.join(metabolic_rxns)
            except Exception:
                gene_name = ''
                rxn_ids   = ''

            rows.append({
                'gene_id':          gid,
                'gene_name':        gene_name,
                'type':             row['type'],
                'k_score':          float(row['k_score']),
                'v_score_sum':      float(row['v_score_sum']),
                'reaction_count':   int(row['reaction_count']),
                'is_bidirectional': bool(row['is_bidirectional']),
                'Target_reactions': rxn_ids,
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values('k_score', ascending=False).reset_index(drop=True)
        return df


    def _load_or_identify_essential_genes(self, essential_genes_file, force_reidentify):
        essential_genes = []
        if not force_reidentify:
            cached = self._load_essential_genes_cache()
            if cached is not None:
                essential_genes.extend(cached)
                if essential_genes_file:
                    file_genes = self._load_essential_genes(essential_genes_file)
                    before = len(essential_genes)
                    essential_genes = list(set(essential_genes + file_genes))
                    if len(essential_genes) > before:
                        self._save_essential_genes_cache(essential_genes)
                return essential_genes

        identified = self._identify_essential_genes()
        if essential_genes_file:
            identified = list(set(identified + self._load_essential_genes(essential_genes_file)))
        self._save_essential_genes_cache(identified)
        return identified

    def _identify_essential_genes(self, growth_fraction=0.001):
        essential_genes = []
        for rxn in self.model.reactions:
            if self.nc.is_bio_reaction(rxn.id):
                for g in rxn.genes:
                    essential_genes.append(g.id)
        bio_genes = set(essential_genes)

        self.model.objective = self.biomass_rxn
        sol = self.model.optimize()
        wt_growth = sol.objective_value

        if wt_growth <= 0:
            print("  Warning: model shows no growth, only Bio_ genes marked essential")
            return list(bio_genes)

        threshold = wt_growth * growth_fraction
        print(f"  WT growth: {wt_growth:.6f}, threshold: {threshold:.6f}")
        print(f"  Testing {len(self.model.genes)} genes...")

        for i, gene in enumerate(self.model.genes):
            if gene.id in bio_genes:
                continue
            if i % 100 == 0 and i > 0:
                print(f"  Progress: {i}/{len(self.model.genes)}")
            with self.model as ko:
                gene.knock_out()
                try:
                    ko_sol = ko.optimize()
                    if ko_sol.objective_value < threshold:
                        essential_genes.append(gene.id)
                except Exception:
                    essential_genes.append(gene.id)

        essential_genes = list(set(essential_genes))
        print(f"  Total essential genes: {len(essential_genes)}")
        return essential_genes

    def _load_essential_genes(self, file_path=None):
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path) as f:
                    return [line.strip() for line in f if line.strip()]
            except Exception as e:
                print(f"  Warning: could not load essential genes file — {e}")
        return []

    def _generate_model_signature(self):
        info = {
            'reactions':   sorted(r.id for r in self.model.reactions),
            'metabolites': sorted(m.id for m in self.model.metabolites),
            'genes':       sorted(g.id for g in self.model.genes),
            'model_id':    self.model.id,
            'biomass_rxn': self.biomass_rxn,
        }
        return hashlib.md5(json.dumps(info, sort_keys=True).encode()).hexdigest()

    def _get_cache_path(self):
        sig = self._generate_model_signature()
        return os.path.join(self.essential_genes_cache_dir, f'essential_genes_{sig}.json')

    def _save_essential_genes_cache(self, genes):
        path = self._get_cache_path()
        data = {
            'essential_genes':  genes,
            'model_id':         self.model.id,
            'biomass_rxn':      self.biomass_rxn,
            'model_signature':  self._generate_model_signature(),
            'saved_at':         datetime.now().isoformat(),
        }
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"  Essential genes cached: {path}")
        except Exception as e:
            print(f"  Warning: cache write failed — {e}")

    def _load_essential_genes_cache(self):
        path = self._get_cache_path()
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            if data.get('model_signature') != self._generate_model_signature():
                print("  Cache signature mismatch — re-identifying")
                return None
            genes = data.get('essential_genes', [])
            print(f"  Loaded {len(genes)} essential genes from cache")
            return genes
        except Exception as e:
            print(f"  Warning: cache read failed — {e}")
            return None


    def _has_valid_kcat_mw(self, reaction):
        return (hasattr(reaction, 'kcat_MW') and
                reaction.kcat_MW is not None and
                reaction.kcat_MW != '' and
                reaction.kcat_MW != 0)

    def _check_enzyme_constraints(self):
        if hasattr(self.model, 'constraints'):
            return True
        if hasattr(self.model, 'json') and isinstance(self.model.json, dict):
            if 'enzyme_constraint' in self.model.json:
                return True
        for rxn in self.model.reactions:
            if self._has_valid_kcat_mw(rxn):
                return True
        return False

    def _calculate_optimal_constraint_bound(self):
        """Set an appropriate upper bound for the carbon-source reaction."""
        if self.constraint_rxn_id not in [r.id for r in self.model.reactions]:
            print(f"  Warning: constraint reaction {self.constraint_rxn_id} not found")
            self.optimal_constraint_bound = None
            return
        rxn  = self.model.reactions.get_by_id(self.constraint_rxn_id)
        orig = (rxn.lower_bound, rxn.upper_bound)
        rxn.upper_bound = 1000
        try:
            with self.model as m:
                m.objective = self.product_rxn
                sol = m.optimize()
                if sol.status == 'optimal':
                    self.optimal_constraint_bound = abs(
                        sol.fluxes.get(self.constraint_rxn_id, 0)
                    )
                    print(f"  Optimal carbon constraint: {self.optimal_constraint_bound:.6f}")
                else:
                    self.optimal_constraint_bound = 2.0
        except Exception as e:
            print(f"  Warning: constraint calculation failed ({e}), using 2.0")
            self.optimal_constraint_bound = 2.0
        finally:
            rxn.lower_bound, rxn.upper_bound = orig

    def _apply_optimal_constraint(self):
        if self.optimal_constraint_bound is None:
            return None
        if self.constraint_rxn_id not in [r.id for r in self.model.reactions]:
            return None
        rxn  = self.model.reactions.get_by_id(self.constraint_rxn_id)
        orig = (rxn.lower_bound, rxn.upper_bound)
        rxn.upper_bound = self.optimal_constraint_bound
        return orig

    def _restore_constraint_bounds(self, original_bounds):
        if original_bounds is None:
            return
        if self.constraint_rxn_id in [r.id for r in self.model.reactions]:
            rxn = self.model.reactions.get_by_id(self.constraint_rxn_id)
            rxn.lower_bound, rxn.upper_bound = original_bounds

    def _close_unwanted_reactions(self):
        """Block reactions that should never carry flux (convention-dependent)."""
        to_close = self.nc.CLOSE_REACTIONS
        for rid in to_close:
            try:
                self.model.reactions.get_by_id(rid).bounds = (0, 0)
                print(f"  Closed reaction: {rid}")
            except KeyError:
                pass

    def _set_quiet_mode(self, quiet=True):
        level = logging.ERROR if quiet else logging.INFO
        logging.getLogger('cobra').setLevel(level)
        try:
            self.model.solver.configuration.verbosity = 0 if quiet else 1
        except Exception:
            pass

    def _print_init_info(self):
        print("\n" + "=" * 60)
        print("ecFSEOF Initialisation Complete")
        print("=" * 60)
        print(f"  Model          : {self.model.id}")
        print(f"  Naming         : {self.nc.name}")
        print(f"  Product rxn    : {self.product_rxn}")
        print(f"  Biomass rxn    : {self.biomass_rxn}")
        print(f"  FSEOF steps    : {self.fseof_steps}")
        print(f"  Top %          : {self.top_percentage * 100:.1f}%")
        print(f"  Strategy       : {self.filtering_strategy}")
        print(f"  Essential genes: {len(self.essential_genes)}")
        print(f"  Enzyme constrd : {'Yes' if self.has_enzyme_constraints else 'No'}")
        print("=" * 60)


# SINGLE-TARGET VALIDATION (ecBridge / ecFSEOF compatible)


def _st_detect_source(row):
    """Read the source label from sources, then source_tag, then column names."""

    if 'sources' in row and pd.notna(row.get('sources', None)):
        src_val = str(row['sources']).strip()
        if src_val and src_val.lower() != 'nan':
            return src_val


    if 'source_tag' in row and pd.notna(row.get('source_tag', None)):
        tag_val = str(row['source_tag']).strip()
        if tag_val and tag_val.lower() != 'nan':
            return tag_val


    if 'Regulation_type' in row and pd.notna(row.get('Regulation_type', None)):
        return 'ecBridge'
    if 'type' in row and pd.notna(row.get('type', None)):
        return 'ecFSEOF'

    return 'Unknown'


def _st_parse_regulation_type(row):
    """Normalize Up/Down or OE/KD/BOTH to OE, KD, or BOTH."""
    if 'Regulation_type' in row and pd.notna(row['Regulation_type']):
        reg_raw = str(row['Regulation_type']).strip().lower()
        return 'OE' if 'up' in reg_raw else 'KD'

    if 'type' in row and pd.notna(row['type']):
        t = str(row['type']).strip().upper()
        if t == 'BOTH':
            return 'BOTH'
        return 'OE' if t == 'OE' else 'KD'

    return 'KD'


def _st_map_reactions_to_genes(model, ecbridge_df):
    """Map reactions to gene directions, reaction lists, input rows, and sources.

    Mark contradictory directions as conflict for testing both OE and KD."""
    gene_reg_temp = {}
    gene_rxn_map  = {}
    gene_row_map  = {}
    gene_source_map = {}

    for _, row in ecbridge_df.iterrows():
        reg_type  = _st_parse_regulation_type(row)
        src_label = _st_detect_source(row)

        rxn_field = row.get('Target_reactions', '')
        rxn_ids   = [r.strip() for r in str(rxn_field).split(';') if r.strip()]

        for rxn_id in rxn_ids:
            try:
                rxn   = model.reactions.get_by_id(rxn_id)
                genes = [g.id for g in rxn.genes]
            except KeyError:
                print(f"  [WARN] Reaction {rxn_id} not found in model, skipping")
                genes = []

            for gid in genes:
                gene_rxn_map.setdefault(gid, []).append((rxn_id, reg_type))
                gene_row_map.setdefault(gid, []).append(row.to_dict())
                gene_source_map.setdefault(gid, set()).add(src_label)


                if reg_type == 'BOTH':
                    gene_reg_temp.setdefault(gid, set()).update({'OE', 'KD'})
                else:
                    gene_reg_temp.setdefault(gid, set()).add(reg_type)

    gene_reg_map = {
        gid: ('conflict' if len(regs) > 1 else list(regs)[0])
        for gid, regs in gene_reg_temp.items()
    }

    gene_source_str = {
        gid: '+'.join(sorted(srcs))
        for gid, srcs in gene_source_map.items()
    }

    return gene_reg_map, gene_rxn_map, gene_row_map, gene_source_str


def _st_get_original_kcat_values(model):
    """Extract kcat_MW values for every reaction that has one."""
    original = {}
    for rxn in model.reactions:
        if hasattr(rxn, 'kcat_MW') and rxn.kcat_MW:
            try:
                original[rxn.id] = float(rxn.kcat_MW)
            except Exception:
                pass
    return original


def _st_calculate_baseline(model, biomass_rxn, product_rxn, cs_rxn):
    """Calculate baseline production and yield with the minimum growth constraint."""
    with model as m:
        m.objective = biomass_rxn
        max_growth  = m.slim_optimize()
        if max_growth is None or np.isnan(max_growth) or max_growth < 1e-6:
            raise ValueError("Model cannot grow — check biomass_rxn parameter")

        m.reactions.get_by_id(biomass_rxn).lower_bound = GROWTH_FRACTION * max_growth
        m.objective = product_rxn
        sol = pfba(m)

        production = sol.fluxes[product_rxn]
        cs_uptake  = abs(sol.fluxes.get(cs_rxn, 1.0))
        yield_val  = production / cs_uptake if cs_uptake > 1e-9 else 0.0

    return float(production), float(yield_val)


def _st_apply_gene_modification_and_test(model, gene_id, mod_type,
                                          original_kcat_values,
                                          biomass_rxn, product_rxn, cs_rxn):
    """Scale gene-associated kcat_MW by OEF or KDF, reload constraints,
    and return pFBA production and yield."""
    modified_kcat_data = {}
    temp_filename = None

    try:
        with model as m:
            gene = m.genes.get_by_id(gene_id)
            for rxn in gene.reactions:
                if hasattr(rxn, 'kcat_MW') and rxn.kcat_MW:
                    try:
                        orig   = original_kcat_values.get(rxn.id, float(rxn.kcat_MW))
                        factor = OEF if mod_type == 'OE' else KDF
                        modified_kcat_data[rxn.id] = orig * factor
                    except Exception:
                        pass

        with tempfile.NamedTemporaryFile(suffix='.json', mode='w', delete=False) as tmp:
            temp_filename = tmp.name
            cobra.io.save_json_model(model, temp_filename)

        with open(temp_filename, 'r') as f:
            model_dict = json.load(f)

        if 'enzyme_constraint' not in model_dict:
            model_dict['enzyme_constraint'] = {
                'enzyme_mass_fraction': 0.46,
                'total_protein_fraction': 0.56,
                'average_saturation': 0.5,
                'lowerbound': 0,
                'upperbound': 0.129,
            }

        for rxn_entry in model_dict['reactions']:
            rid = rxn_entry['id']
            if rid in modified_kcat_data:
                rxn_entry['kcat_MW'] = modified_kcat_data[rid]
                rxn_entry['kcat']    = ''
            else:
                rxn_entry['kcat_MW'] = original_kcat_values.get(rid, rxn_entry.get('kcat_MW', ''))
                rxn_entry['kcat']    = ''

        with open(temp_filename, 'w') as f:
            json.dump(model_dict, f)

        try:
            from ECMpy_function import get_enzyme_constraint_model
            updated_model = get_enzyme_constraint_model(temp_filename)
        except Exception:
            updated_model = cobra.io.load_json_model(temp_filename)

        updated_model.objective = biomass_rxn
        grow_sol = updated_model.optimize()
        if grow_sol.status == 'optimal' and grow_sol.objective_value > 1e-6:
            updated_model.reactions.get_by_id(biomass_rxn).lower_bound = (
                GROWTH_FRACTION * grow_sol.objective_value
            )

        updated_model.objective = product_rxn
        sol        = pfba(updated_model)
        production = float(sol.fluxes[product_rxn])
        cs_uptake  = abs(sol.fluxes.get(cs_rxn, 1.0))
        yield_val  = production / cs_uptake if cs_uptake > 1e-9 else 0.0

        return production, yield_val

    except Exception as e:
        print(f"    [ERROR] Gene {gene_id} {mod_type} test failed: {e}")
        return 0.0, 0.0

    finally:
        if temp_filename and os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except Exception:
                pass


def _st_build_result_row(model, gene_id, mod_type, prod_imp, yield_imp,
                          gene_rxn_map, is_conflict, source,
                          oe_sources='', kd_sources=''):
    """Build a test result with sources supporting the retained direction.

    Preserve all original source labels in predicted_by."""
    try:
        gene      = model.genes.get_by_id(gene_id)
        gene_name = gene.name if (hasattr(gene, 'name') and gene.name) else ''
        reactions = ';'.join([r.id for r in gene.reactions])
    except Exception:
        gene_name = ''
        reactions = ';'.join([rxn_id for rxn_id, _ in gene_rxn_map.get(gene_id, [])])

    return {
        'gene_id':                gene_id,
        'gene_name':              gene_name,
        'type':                   mod_type,

        'sources':                source,
        'is_conflict':            is_conflict,
        'reactions':              reactions,
        'production_improvement': round(prod_imp, 6),
        'yield_improvement':      round(yield_imp, 6),
    }


def _ecb_reactions_to_gene_df(model, ecb_df, naming=None):
    """Aggregate ecBridge reaction targets into gene targets; mark mixed directions as conflict."""
    if ecb_df is None or ecb_df.empty:
        return pd.DataFrame()


    if 'gene_id' in ecb_df.columns:
        return ecb_df.copy()

    gene_data = {}

    for _, row in ecb_df.iterrows():

        rxn_field = str(row.get('All_isozymes', '') or row.get('Target_reactions', ''))
        rxn_ids   = [r.strip() for r in rxn_field.split(';') if r.strip()]
        if not rxn_ids:
            rxn_ids = [str(row.get('Target_reactions', '')).strip()]

        reg_raw  = str(row.get('Regulation_type', '')).strip().lower()
        reg_type = 'OE' if 'up' in reg_raw else 'KD'

        soc       = float(row.get('SoC', 0.0))
        conv_type = str(row.get('Conversion_type', ''))
        src_met   = str(row.get('Source_metabolite', ''))
        dst_met   = str(row.get('Destination_metabolite', ''))
        is_active = bool(row.get('Is_Active', False))

        for rxn_id in rxn_ids:
            try:
                rxn   = model.reactions.get_by_id(rxn_id)
                genes = [g.id for g in rxn.genes]
            except KeyError:
                genes = []

            for gid in genes:
                if gid not in gene_data:
                    try:
                        gene      = model.genes.get_by_id(gid)
                        gene_name = gene.name if (hasattr(gene, 'name') and gene.name) else ''
                        all_rxns  = ';'.join(r.id for r in gene.reactions)
                    except Exception:
                        gene_name = ''
                        all_rxns  = rxn_id
                    gene_data[gid] = {
                        'gene_id':   gid,
                        'gene_name': gene_name,
                        'reg_types': set(),
                        'reactions': all_rxns,
                        'soc_values':  [],
                        'conv_types':  set(),
                        'src_mets':    set(),
                        'dst_mets':    set(),
                        'is_active':   False,
                    }

                gene_data[gid]['reg_types'].add(reg_type)
                gene_data[gid]['soc_values'].append(soc)
                gene_data[gid]['conv_types'].add(conv_type)
                gene_data[gid]['src_mets'].add(src_met)
                gene_data[gid]['dst_mets'].add(dst_met)
                gene_data[gid]['is_active'] |= is_active

    rows = []
    for gid, d in gene_data.items():
        regs = d['reg_types']
        regulation_type = 'conflict' if len(regs) > 1 else list(regs)[0]
        rows.append({
            'gene_id':               gid,
            'gene_name':             d['gene_name'],
            'regulation_type':       regulation_type,
            'reactions':             d['reactions'],
            'SoC_max':               round(max(d['soc_values']), 6),
            'SoC_mean':              round(sum(d['soc_values']) / len(d['soc_values']), 6),
            'Conversion_type':       ';'.join(sorted(d['conv_types'])),
            'Source_metabolite':     ';'.join(sorted(d['src_mets'])),
            'Destination_metabolite':';'.join(sorted(d['dst_mets'])),
            'Is_Active':             d['is_active'],
        })

    if not rows:
        return pd.DataFrame()

    df = (pd.DataFrame(rows)
            .sort_values('SoC_max', ascending=False)
            .reset_index(drop=True))
    return df


def run_ecbridge_single_target_test(model,
                                     ecbridge_raw_df,
                                     product_id,
                                     biomass_rxn,
                                     output_dir='./ecbridge_results',
                                     improvement_threshold=0.1,
                                     exclusion_threshold=-3.0,
                                     c_source_rxn=None,
                                     output_prefix=None,
                                     naming=None):
    """Validate gene targets by OE/KD testing and improvement filtering.

    Test conflict genes in both directions. The sources column records tools
    supporting the retained direction; predicted_by preserves original sources."""
    print("\n" + "=" * 65)
    print("Single-Target Validation")
    print("=" * 65)

    nc          = _get_nc(naming)
    model       = _load_model(model)
    product_rxn = nc.make_exchange_rxn(product_id)
    prefix      = output_prefix or product_id
    cs_rxn      = c_source_rxn or 'EX_cpd00027_e_reverse'

    if cs_rxn not in [r.id for r in model.reactions]:
        cs_rxn = nc.find_carbon_source_fallback(model, cs_rxn)
        print(f"  [INFO] Carbon-source reaction redirected to: {cs_rxn}")

    os.makedirs(output_dir, exist_ok=True)

    raw_path = os.path.join(output_dir, f'{prefix}_raw_ecbridge.csv')


    print("\n[Step A] Computing baseline production ...")
    baseline_prod, baseline_yield = _st_calculate_baseline(
        model, biomass_rxn, product_rxn, cs_rxn
    )
    print(f"  Baseline production = {baseline_prod:.6f},  yield = {baseline_yield:.6f}")


    print("\n[Step B] Mapping reactions to genes ...")
    gene_reg_map, gene_rxn_map, gene_row_map, gene_source_str = _st_map_reactions_to_genes(
        model, ecbridge_raw_df
    )
    print(f"  Mapped to {len(gene_reg_map)} genes")

    conflict_genes = [g for g, r in gene_reg_map.items() if r == 'conflict']
    print(f"  Conflict genes (bi-directional testing): {len(conflict_genes)}")


    raw_gene_df = _ecb_reactions_to_gene_df(model, ecbridge_raw_df, naming=naming)
    raw_gene_df.to_csv(raw_path, index=False)
    print(f"\n[Output 1] Raw gene-level results saved: {raw_path}  ({len(raw_gene_df)} genes)")

    src_counter = Counter(gene_source_str.values())
    for src_label, cnt in sorted(src_counter.items()):
        print(f"  Source [{src_label}]: {cnt} genes")


    gene_oe_src = {}
    gene_kd_src = {}
    if 'oe_sources' in ecbridge_raw_df.columns and 'kd_sources' in ecbridge_raw_df.columns:

        if 'gene_id' in ecbridge_raw_df.columns:
            for _, row in ecbridge_raw_df.iterrows():
                gid = str(row['gene_id'])
                gene_oe_src[gid] = str(row.get('oe_sources', ''))
                gene_kd_src[gid] = str(row.get('kd_sources', ''))


    original_kcat = _st_get_original_kcat_values(model)
    print(f"  Collected kcat_MW for {len(original_kcat)} reactions")


    print("\n[Step D] Per-gene single-target testing ...")
    results        = []
    excluded_genes = []
    filtered_genes = []

    for gid, reg_type in gene_reg_map.items():

        if reg_type == 'conflict':


            print(f"\n  [Conflict] {gid}, testing OE and KD ...")
            oe_prod, oe_yield = _st_apply_gene_modification_and_test(
                model, gid, 'OE', original_kcat, biomass_rxn, product_rxn, cs_rxn
            )
            kd_prod, kd_yield = _st_apply_gene_modification_and_test(
                model, gid, 'KD', original_kcat, biomass_rxn, product_rxn, cs_rxn
            )
            oe_pi = (oe_prod - baseline_prod) / baseline_prod * 100 if baseline_prod > 0 else 0
            oe_yi = (oe_yield - baseline_yield) / baseline_yield * 100 if baseline_yield > 0 else 0
            kd_pi = (kd_prod - baseline_prod) / baseline_prod * 100 if baseline_prod > 0 else 0
            kd_yi = (kd_yield - baseline_yield) / baseline_yield * 100 if baseline_yield > 0 else 0

            oe_excl = oe_pi <= exclusion_threshold or oe_yi <= exclusion_threshold
            kd_excl = kd_pi <= exclusion_threshold or kd_yi <= exclusion_threshold

            if oe_excl and kd_excl:
                excluded_genes.append(gid)
                print(f"  ✗ Excluded {gid}: both directions below exclusion threshold")
                continue

            if oe_excl:
                best_type, best_pi, best_yi = 'KD', kd_pi, kd_yi
            elif kd_excl:
                best_type, best_pi, best_yi = 'OE', oe_pi, oe_yi
            else:
                if (oe_pi + oe_yi) >= (kd_pi + kd_yi):
                    best_type, best_pi, best_yi = 'OE', oe_pi, oe_yi
                else:
                    best_type, best_pi, best_yi = 'KD', kd_pi, kd_yi

            if best_pi >= improvement_threshold or best_yi >= improvement_threshold:

                if gene_oe_src or gene_kd_src:

                    if best_type == 'OE':
                        resolved_source = gene_oe_src.get(gid, '') or gene_source_str.get(gid, 'Unknown')
                    else:
                        resolved_source = gene_kd_src.get(gid, '') or gene_source_str.get(gid, 'Unknown')
                else:

                    resolved_source = gene_source_str.get(gid, 'Unknown')

                print(f"  ✓ Retained {gid} ({best_type}) "
                      f"[source: {resolved_source}]: "
                      f"prod+{best_pi:.3f}%, yield+{best_yi:.3f}%")
                results.append(_st_build_result_row(
                    model, gid, best_type, best_pi, best_yi,
                    gene_rxn_map, is_conflict=True,
                    source=resolved_source,
                    oe_sources=gene_oe_src.get(gid, ''),
                    kd_sources=gene_kd_src.get(gid, ''),
                ))
            else:
                filtered_genes.append(gid)
                print(f"  ✗ Filtered {gid}: improvement insufficient "
                      f"(prod {best_pi:.3f}%, yield {best_yi:.3f}%)")

        else:

            prod, yld   = _st_apply_gene_modification_and_test(
                model, gid, reg_type, original_kcat, biomass_rxn, product_rxn, cs_rxn
            )
            prod_imp  = (prod - baseline_prod)  / baseline_prod  * 100 if baseline_prod  > 0 else 0
            yield_imp = (yld  - baseline_yield) / baseline_yield * 100 if baseline_yield > 0 else 0

            if reg_type == 'KD' and (
                    prod_imp <= exclusion_threshold or yield_imp <= exclusion_threshold):
                excluded_genes.append(gid)
                print(f"  ✗ Excluded {gid} ({reg_type}): "
                      f"prod {prod_imp:.3f}%, yield {yield_imp:.3f}% "
                      f"(below exclusion threshold {exclusion_threshold}%)")
                continue

            if prod_imp >= improvement_threshold or yield_imp >= improvement_threshold:


                if gene_oe_src or gene_kd_src:

                    if reg_type == 'OE':
                        src = gene_oe_src.get(gid, '') or gene_source_str.get(gid, 'Unknown')
                    else:
                        src = gene_kd_src.get(gid, '') or gene_source_str.get(gid, 'Unknown')
                else:

                    src = gene_source_str.get(gid, 'Unknown')
                print(f"  ✓ Retained {gid} ({reg_type}) "
                      f"[source: {src}]: "
                      f"prod+{prod_imp:.3f}%, yield+{yield_imp:.3f}%")
                results.append(_st_build_result_row(
                    model, gid, reg_type, prod_imp, yield_imp,
                    gene_rxn_map, is_conflict=False,
                    source=src,
                ))
            else:
                filtered_genes.append(gid)
                print(f"  ✗ Filtered {gid} ({reg_type}): "
                      f"prod {prod_imp:.3f}%, yield {yield_imp:.3f}% "
                      f"(below improvement threshold {improvement_threshold}%)")


    filtered_path = os.path.join(output_dir, f'{prefix}_singletest_filtered.csv')
    col_order = [
        'gene_id', 'gene_name', 'type', 'sources',
        'is_conflict', 'reactions',
        'production_improvement', 'yield_improvement'
    ]

    if results:
        df_filtered   = pd.DataFrame(results)
        existing_cols = [c for c in col_order if c in df_filtered.columns]
        df_filtered   = df_filtered[existing_cols]
        df_filtered.to_csv(filtered_path, index=False)
    else:
        pd.DataFrame(columns=col_order).to_csv(filtered_path, index=False)

    print(f"\n[Output 2] Single-target filtered results saved: {filtered_path}")

    print("\n" + "=" * 65)
    print("Single-Target Validation Summary")
    print("=" * 65)
    print(f"  Total genes mapped   : {len(gene_reg_map)}")
    print(f"  Conflict genes       : {len(conflict_genes)}")
    print(f"  ✓ Retained genes     : {len(results)}")
    print(f"    OE                 : {sum(1 for r in results if r['type'] == 'OE')}")
    print(f"    KD                 : {sum(1 for r in results if r['type'] == 'KD')}")
    print(f"  ✗ Filtered genes     : {len(filtered_genes)}")
    print(f"  ✗ Excluded genes     : {len(excluded_genes)}")

    if results:
        df_tmp = pd.DataFrame(results)
        print("\n  Top 10 retained genes:")
        preview_cols = ['gene_id', 'type', 'sources', 'production_improvement', 'yield_improvement']
        avail = [c for c in preview_cols if c in df_tmp.columns]
        print(df_tmp[avail].head(10).to_string(index=False))

    return raw_path, filtered_path


# ecFB MERGER


def _merge_ecbridge_ecfseof(ecb_df, efs_df):
    """Normalize ecBridge and ecFSEOF targets into a shared schema and concatenate them."""
    rows = []


    for _, row in ecb_df.iterrows():
        reg = 'OE' if str(row.get('Regulation_type', '')).strip().lower() == 'up' else 'KD'
        rows.append({
            'Target_reactions': str(row.get('Target_reactions', '')),
            'Regulation_type':  reg,
            'type':             reg,
            'source_tag':       'ecBridge',
            'SoC':              float(row.get('SoC', 0)),
        })


    for _, row in efs_df.iterrows():
        ftype = str(row.get('type', '')).strip().upper()
        rows.append({
            'Target_reactions': str(row.get('Target_reactions', '')),
            'Regulation_type':  ftype,
            'type':             ftype,
            'source_tag':       'ecFSEOF',
            'k_score':          float(row.get('k_score', 0)),
        })

    merged = pd.DataFrame(rows)
    merged['source_tag'] = merged['source_tag'].fillna('Unknown')
    return merged


def _resolve_conflicts_in_merged(model, merged_df):
    """Aggregate targets by gene and record OE/KD directions and their sources.

    Mark contradictory directions as BOTH for later single-target testing."""
    gene_directions  = {}
    gene_reactions   = {}
    gene_sources     = {}
    gene_oe_sources  = {}
    gene_kd_sources  = {}

    for _, row in merged_df.iterrows():
        rxn_field = str(row.get('Target_reactions', ''))
        rxn_ids   = [r.strip() for r in rxn_field.split(';') if r.strip()]
        reg       = str(row.get('type', '')).strip().upper()
        src       = str(row.get('source_tag', 'Unknown'))

        for rxn_id in rxn_ids:

            try:
                rxn   = model.reactions.get_by_id(rxn_id)
                genes = [g.id for g in rxn.genes]
            except KeyError:

                genes = [rxn_id]

            for gid in genes:
                gene_reactions.setdefault(gid, set()).add(rxn_id)
                gene_sources.setdefault(gid, set()).add(src)

                if reg == 'BOTH':

                    gene_directions.setdefault(gid, set()).update({'OE', 'KD'})
                    gene_oe_sources.setdefault(gid, set()).add(src)
                    gene_kd_sources.setdefault(gid, set()).add(src)
                elif reg == 'OE':
                    gene_directions.setdefault(gid, set()).add('OE')
                    gene_oe_sources.setdefault(gid, set()).add(src)
                elif reg == 'KD':
                    gene_directions.setdefault(gid, set()).add('KD')
                    gene_kd_sources.setdefault(gid, set()).add(src)

    resolved_rows = []
    for gid, dirs in gene_directions.items():
        if dirs == {'OE'}:
            final_type  = 'OE'
            is_conflict = False
        elif dirs == {'KD'}:
            final_type  = 'KD'
            is_conflict = False
        else:
            final_type  = 'BOTH'
            is_conflict = True


        all_srcs  = gene_sources.get(gid, {'Unknown'})
        oe_srcs   = gene_oe_sources.get(gid, set())
        kd_srcs   = gene_kd_sources.get(gid, set())

        resolved_rows.append({
            'gene_id':          gid,
            'Target_reactions': ';'.join(sorted(gene_reactions.get(gid, set()))),
            'type':             final_type,
            'sources':          '+'.join(sorted(all_srcs)),
            'is_conflict':      is_conflict,

            'oe_sources':       '+'.join(sorted(oe_srcs))  if oe_srcs  else '',
            'kd_sources':       '+'.join(sorted(kd_srcs))  if kd_srcs  else '',
        })

    resolved_df = pd.DataFrame(resolved_rows)
    n_conflict   = resolved_df['is_conflict'].sum()
    print(f"  Merged gene pool: {len(resolved_df)} genes  "
          f"({n_conflict} conflicts requiring bi-directional testing)")
    return resolved_df


# MAIN ENTRY POINT


def run_prediction(mode,
                   model,
                   product_id,
                   biomass_rxn,
                   output_dir='./results',

                   ecb_num_step=20,
                   ecb_bio_fraction=0.1,
                   ecb_threshold=0.01,
                   ecb_top_score_percentage=1,
                   ecb_exclude_met_file=None,

                   efs_fseof_steps=10,
                   efs_top_percentage=1.0,
                   efs_auto_identify_essential=True,
                   efs_essential_genes_file=None,
                   efs_additional_essential_genes=None,
                   efs_force_reidentify=False,

                   st_improvement_threshold=0.1,
                   st_exclusion_threshold=-5.0,

                   c_source_rxn='EX_cpd00027_e_reverse',
                   debug_output=False,
                   naming='modelseed'):
    """Run ecbridge, ecfseof, or combined ecfb target prediction.

    Accept a model or JSON path. Use ecb_*, efs_*, and st_* for algorithm
    and validation options. Single modes write to their own subdirectories;
    ecFB writes merged input and validated targets directly to output_dir.
    Return a dictionary of output paths."""
    mode = mode.lower().strip()
    if mode not in ('ecbridge', 'ecfseof', 'ecfb'):
        raise ValueError(f"Unknown mode '{mode}'. Choose from: ecbridge, ecfseof, ecfb")

    nc    = _get_nc(naming)
    model = _load_model(model)

    product_rxn    = nc.make_exchange_rxn(product_id)

    ecb_dir        = os.path.join(output_dir, 'ecbridge')
    efs_dir        = os.path.join(output_dir, 'ecfseof')

    ecfb_dir       = output_dir
    debug_ecb_dir  = os.path.join(output_dir, 'debug', 'ecbridge') if debug_output else None
    debug_efs_dir  = os.path.join(output_dir, 'debug', 'ecfseof')  if debug_output else None

    if mode == 'ecbridge':
        os.makedirs(ecb_dir, exist_ok=True)
    elif mode == 'ecfseof':
        os.makedirs(efs_dir, exist_ok=True)
    else:
        os.makedirs(ecfb_dir, exist_ok=True)

    results = {}


    if mode in ('ecbridge', 'ecfb'):
        print("\n" + "#" * 70)
        print("# Running ecBridge")
        print("#" * 70)

        ecb_all, ecb_final = predict_ecbridge_targets(
            model                = model,
            exclude_met_file     = ecb_exclude_met_file,
            target_compound      = product_id,
            biomass_rxn          = biomass_rxn,
            num_step             = ecb_num_step,
            bio_fraction         = ecb_bio_fraction,
            threshold            = ecb_threshold,
            top_score_percentage = ecb_top_score_percentage,
            c_source_rxn         = c_source_rxn,
            verbose              = True,
            output_dir           = debug_ecb_dir,
            naming               = nc,
        )

        if mode == 'ecbridge':

            ecb_raw_path, ecb_filt_path = run_ecbridge_single_target_test(
                model                 = model,
                ecbridge_raw_df       = ecb_all,
                product_id            = product_id,
                biomass_rxn           = biomass_rxn,
                output_dir            = ecb_dir,
                improvement_threshold = st_improvement_threshold,
                exclusion_threshold   = st_exclusion_threshold,
                c_source_rxn          = c_source_rxn,
                output_prefix         = product_id,
                naming                = nc,
            )
            ecb_raw_named = os.path.join(ecb_dir, f'{product_id}_raw_ecbridge.csv')
            if ecb_raw_path != ecb_raw_named:
                ecb_all.to_csv(ecb_raw_named, index=False)
            results['ecb_raw']      = ecb_raw_named
            results['ecb_filtered'] = ecb_filt_path

            print("\n" + "=" * 70)
            print("ecBridge prediction complete")
            print(f"  Raw output       : {results['ecb_raw']}")
            print(f"  Filtered output  : {results['ecb_filtered']}")
            print("=" * 70)
            return results


        results['_ecb_all'] = ecb_all


    if mode in ('ecfseof', 'ecfb'):
        print("\n" + "#" * 70)
        print("# Running ecFSEOF")
        print("#" * 70)

        fseof = ecFSEOF(
            model                      = model,
            product_rxn                = product_rxn,
            biomass_rxn                = biomass_rxn,
            fseof_steps                = efs_fseof_steps,
            top_percentage             = efs_top_percentage,
            auto_identify_essential    = efs_auto_identify_essential,
            essential_genes_file       = efs_essential_genes_file,
            additional_essential_genes = efs_additional_essential_genes,
            force_reidentify_essential = efs_force_reidentify,
            c_source_rxn               = c_source_rxn,
            output_dir                 = debug_efs_dir,
            naming                     = nc,
        )
        efs_all, efs_final = fseof.run()

        if mode == 'ecfseof':
            efs_raw_path = os.path.join(efs_dir, f'{product_id}_raw_ecfseof.csv')
            efs_all.to_csv(efs_raw_path, index=False)
            print(f"\n[ecFSEOF] Raw results saved: {efs_raw_path}")

            _, efs_filt_path = run_ecbridge_single_target_test(
                model                 = model,
                ecbridge_raw_df       = efs_all,
                product_id            = product_id,
                biomass_rxn           = biomass_rxn,
                output_dir            = efs_dir,
                improvement_threshold = st_improvement_threshold,
                exclusion_threshold   = st_exclusion_threshold,
                c_source_rxn          = c_source_rxn,
                output_prefix         = product_id,
                naming                = nc,
            )
            efs_raw_std = os.path.join(efs_dir, f'{product_id}_raw_ecbridge.csv')
            if os.path.exists(efs_raw_std) and efs_raw_std != efs_raw_path:
                efs_all.to_csv(efs_raw_path, index=False)
                try:
                    os.remove(efs_raw_std)
                except Exception:
                    pass

            results['efs_raw']      = efs_raw_path
            results['efs_filtered'] = efs_filt_path

            print("\n" + "=" * 70)
            print("ecFSEOF prediction complete")
            print(f"  Raw output      : {results['efs_raw']}")
            print(f"  Filtered output : {results['efs_filtered']}")
            print("=" * 70)
            return results


        results['_efs_all'] = efs_all


    if mode == 'ecfb':
        print("\n" + "#" * 70)
        print("# ecFB: Merging ecBridge + ecFSEOF predictions")
        print("#" * 70)

        ecb_all = results.pop('_ecb_all')
        efs_all = results.pop('_efs_all')


        print("\n[ecFB] Step 1: Normalising and merging target lists ...")
        merged_schema = _merge_ecbridge_ecfseof(ecb_all, efs_all)


        print("\n[ecFB] Step 2: Resolving directional conflicts at gene level ...")
        resolved_df = _resolve_conflicts_in_merged(model, merged_schema)

        print(f"\n[ecFB] Merged gene pool: {len(resolved_df)} unique genes")


        print("\n[ecFB] Step 3: Unified single-target validation ...")
        _, ecfb_filt_path = run_ecbridge_single_target_test(
            model                 = model,
            ecbridge_raw_df       = resolved_df,
            product_id            = product_id,
            biomass_rxn           = biomass_rxn,
            output_dir            = ecfb_dir,
            improvement_threshold = st_improvement_threshold,
            exclusion_threshold   = st_exclusion_threshold,
            c_source_rxn          = c_source_rxn,
            output_prefix         = product_id,
            naming                = nc,
        )


        auto_raw  = os.path.join(ecfb_dir, f'{product_id}_raw_ecbridge.csv')
        final_raw = os.path.join(ecfb_dir, f'{product_id}_ecfb_merged_input.csv')
        if os.path.exists(auto_raw):
            try:
                os.rename(auto_raw, final_raw)
            except Exception:
                final_raw = auto_raw

        results['ecfb_merged_input'] = final_raw
        results['ecfb_filtered']     = ecfb_filt_path

        print("\n" + "=" * 70)
        print("ecFB prediction complete")
        print(f"  Merged input (pre-test) : {results['ecfb_merged_input']}")
        print(f"  Filtered (post-test)    : {results['ecfb_filtered']}")
        print("=" * 70)
        return results


# COMMAND-LINE ENTRY POINT


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            'Unified metabolic engineering target prediction.\n'
            'Modes: ecbridge | ecfseof | ecfb\n\n'
            'Output directory structure:\n'
            '  ecbridge/  *_raw_ecbridge.csv  *_singletest_filtered.csv\n'
            '  ecfseof/   *_raw_ecfseof.csv   *_singletest_filtered.csv\n'
            '  ecfb/      *_merged_targets.csv  *_singletest_filtered.csv\n'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--mode',       required=True,
                        choices=['ecbridge', 'ecfseof', 'ecfb'],
                        help='Prediction mode')
    parser.add_argument('--model',      required=True,
                        help='Metabolic model JSON file path')
    parser.add_argument('--product',    required=True,
                        help='Target compound metabolite ID (e.g. cpd00103)')
    parser.add_argument('--biomass',    required=True,
                        help='Biomass reaction ID')
    parser.add_argument('--output-dir', default='./prediction_results',
                        help='Root output directory (default: ./prediction_results)')
    parser.add_argument('--c-source-rxn', default='EX_cpd00027_e_reverse',
                        help='Carbon-source uptake reaction ID')


    parser.add_argument('--ecb-num-step',             type=int,   default=20)
    parser.add_argument('--ecb-bio-fraction',         type=float, default=0.1)
    parser.add_argument('--ecb-threshold',            type=float, default=0.01)
    parser.add_argument('--ecb-top-score-percentage', type=float, default=1)
    parser.add_argument('--ecb-exclude-met-file',     default=None)


    parser.add_argument('--efs-fseof-steps',            type=int,   default=10)
    parser.add_argument('--efs-top-percentage',         type=float, default=1.0)
    parser.add_argument('--efs-no-auto-essential',      action='store_true',
                        help='Disable automatic essential-gene identification')
    parser.add_argument('--efs-essential-genes-file',   default=None)
    parser.add_argument('--efs-force-reidentify',       action='store_true')


    parser.add_argument('--st-improvement-threshold', type=float, default=0.1,
                        help='Minimum improvement %% to retain a gene (default 0.1)')
    parser.add_argument('--st-exclusion-threshold',   type=float, default=-5.0,
                        help='KD exclusion threshold %% (default -5.0)')

    parser.add_argument('--debug', action='store_true',
                        help='Save intermediate debug CSV files')
    parser.add_argument('--naming', default='modelseed',
                        choices=['modelseed', 'bigg'],
                        help='Naming convention: modelseed or bigg (default: modelseed)')

    args = parser.parse_args()

    run_prediction(
        mode                      = args.mode,
        model                     = args.model,
        product_id                = args.product,
        biomass_rxn               = args.biomass,
        output_dir                = args.output_dir,
        ecb_num_step              = args.ecb_num_step,
        ecb_bio_fraction          = args.ecb_bio_fraction,
        ecb_threshold             = args.ecb_threshold,
        ecb_top_score_percentage  = args.ecb_top_score_percentage,
        ecb_exclude_met_file      = args.ecb_exclude_met_file,
        efs_fseof_steps           = args.efs_fseof_steps,
        efs_top_percentage        = args.efs_top_percentage,
        efs_auto_identify_essential = not args.efs_no_auto_essential,
        efs_essential_genes_file  = args.efs_essential_genes_file,
        efs_force_reidentify      = args.efs_force_reidentify,
        st_improvement_threshold  = args.st_improvement_threshold,
        st_exclusion_threshold    = args.st_exclusion_threshold,
        c_source_rxn              = args.c_source_rxn,
        debug_output              = args.debug,
        naming                    = args.naming,
    )

if __name__ == '__main__':
    main()
