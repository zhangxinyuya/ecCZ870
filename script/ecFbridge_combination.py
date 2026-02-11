#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ecFbridge: Integrated Enzyme-Constrained Target Identification System
with Enhanced Flux Tracking for Combination Optimization

Modified version using original ecFbridge evaluation logic before combination optimization.
"""

import os
import sys
import json
import time
import warnings
import tempfile
import pandas as pd
import numpy as np
import cobra
from cobra.io import load_json_model
from cobra.flux_analysis import pfba
from contextlib import contextmanager
import copy
from datetime import datetime

warnings.filterwarnings('ignore')


class IntegratedTargetIdentifierWithOptimization:
    """
    Complete integrated system with enhanced flux tracking during combination optimization.
    Modified to use original ecFbridge evaluation logic for single targets.
    """

    def __init__(self,
                 model_file,
                 product_id,
                 biomass_rxn='EX_biomass_c',
                 essential_genes_file=None,
                 enable_ibridge=True,
                 enable_distance_filter=True,
                 exclude_met_dir="seed_compounds.txt",
                 c_source="cpd00027",
                 air="aerobic",
                 o2='EX_cpd00007_e_reverse',
                 blocked_reactions=None,
                 improvement_threshold=0.1,
                 exclusion_threshold=-5.0,
                 eta_threshold=1,
                 enable_combination_optimization=True,
                 output_dir='./prediction_target',
                 verbose_flux=True):
        """
        Initialize with enhanced flux tracking capabilities.
        """
        # Store configuration
        self.model_file = model_file
        self.product_id = product_id
        self.product_rxn = f'EX_{product_id}_e'
        self.essential_genes_file = essential_genes_file
        self.enable_ibridge = enable_ibridge
        self.exclude_met_dir = exclude_met_dir
        self.c_source = c_source
        self.air = air
        self.o2 = o2
        self.blocked_reactions = blocked_reactions or []
        self.output_dir = output_dir
        self.enable_distance_filter = enable_distance_filter
        self.verbose_flux = verbose_flux

        # Threshold settings
        self.improvement_threshold = improvement_threshold
        self.exclusion_threshold = exclusion_threshold
        self.eta_threshold = eta_threshold
        self.enable_combination_optimization = enable_combination_optimization

        # Initialize data structures
        self._initialize_data_structures()

        # Load and configure model
        self._load_and_configure_model(biomass_rxn)

        # 保存原始模型和原始kcat值（用于combination优化）
        self.original_model = self.model.copy()
        self.original_kcat_values = self._get_original_kcat_values_from_model(self.original_model)

        # Pre-check glucose usage
        self._precheck_glucose_usage()

        # Initialize constraint parameters
        self.constraint_rxn_id = 'EX_cpd00027_e_reverse'
        self.optimal_constraint_bound = None

        try:
            rxn = self.model.reactions.get_by_id('EX_cpd00197_e_reverse')
            rxn.bounds = (0, 0)
            print("Closed reaction: EX_cpd00197_e_reverse")
        except KeyError:
            print("Warning: EX_cpd00197_e_reverse not found in model")
        # Print initialization summary
        self._print_initialization_summary()

    def _get_original_kcat_values_from_model(self, model):
        """从指定模型中提取原始kcat值"""
        original_kcat_values = {}
        for reaction in model.reactions:
            if hasattr(reaction, 'kcat_MW'):
                try:
                    kcat_value = float(reaction.kcat_MW)
                    original_kcat_values[reaction.id] = kcat_value
                except (ValueError, TypeError):
                    continue
        return original_kcat_values

    def _initialize_data_structures(self):
        """Initialize internal data storage structures with enhanced flux tracking."""
        # Original ecFbridge structures
        self.ecfactory_targets = {'OE': [], 'KD': []}
        self.ibridge_targets = pd.DataFrame()
        self.combined_targets = {'OE': [], 'KD': [], 'Test': []}
        self.essential_genes = []
        self.ecfactory_instance = None
        self.test_results = {}
        self.baseline_calculated = False

        # Conflict resolution tracking
        self.conflicted_genes = []
        self.conflict_resolution_results = {}
        self.gene_source_mapping = {}
        self.ibridge_gene_regulations = {}

        # Filtering tracking
        self.excluded_genes = []
        self.filtered_genes = []

        # Combination optimization tracking
        self.combination_results = {}
        self.minimal_gene_set = {'OE': [], 'KD': []}
        self.removed_genes = []
        self.eta_values = {}

        # Enhanced flux tracking
        self.flux_tracking = []
        self.optimization_steps = []
        self.flux_history = []
        self.gene_impact_summary = {}

        # Enzyme complex tracking
        self.enzyme_complexes = {}  # 这必须是实际的赋值语句
        self.gene_to_complexes = {}
        self.complex_groups = {}
        self.complex_eta_values = {}
        self.candidate_complexes = {'OE': {}, 'KD': {}}
        self.candidate_complex_groups = {'OE': {}, 'KD': {}}

    def _block_reactions(self):
        """Block specified reactions in the model."""
        blocked_count = 0
        for rxn_id in self.blocked_reactions:
            try:
                rxn = self.model.reactions.get_by_id(rxn_id)
                rxn.bounds = (0, 0)
                blocked_count += 1
            except KeyError:
                pass

        if blocked_count > 0:
            print(f"Blocked {blocked_count} reactions")

    def _determine_biomass_reaction(self, biomass_rxn):
        """Determine the biomass reaction ID."""
        if biomass_rxn is not None:
            return biomass_rxn

        try:
            if hasattr(self.model.objective, 'keys'):
                return list(self.model.objective.keys())[0].id
        except:
            pass

        for rxn in self.model.reactions:
            if any(rxn.id.startswith(prefix) for prefix in ['Bio_', 'BIOMASS', 'biomass']):
                return rxn.id

        raise ValueError("Cannot detect biomass reaction. Please specify biomass_rxn parameter.")

    def _set_quiet_mode(self, quiet=True):
        """Set solver quiet mode."""
        import logging
        logging.getLogger('cobra').setLevel(logging.ERROR if quiet else logging.INFO)
        try:
            self.model.solver.configuration.verbosity = 0 if quiet else 1
        except:
            pass

    def _print_initialization_summary(self):
        """Print initialization summary."""
        print(f"\nInitialization complete:")
        print(f"  iBridge analysis: {'Enabled' if self.enable_ibridge else 'Disabled'}")
        print(f"  Combination optimization: {'Enabled' if self.enable_combination_optimization else 'Disabled'}")
        print(f"  Improvement threshold: {self.improvement_threshold}%")
        print(f"  Exclusion threshold: {self.exclusion_threshold}%")
        if self.enable_combination_optimization:
            print(f"  Eta threshold: {self.eta_threshold}")
        print(f"  Distance filter: {'Enabled' if self.enable_distance_filter else 'Disabled'}")
        print(f"  Verbose flux output: {'Enabled' if self.verbose_flux else 'Disabled'}")

    def _precheck_glucose_usage(self):
        """Pre-check glucose usage and close rxn00006_c if needed."""
        glc_bio = self._get_glucose_uptake(self.biomass_rxn)
        glc_prod = self._get_glucose_uptake(self.product_rxn)

        if glc_bio is None or glc_prod is None:
            return

        print(f"Glucose uptake check: biomass={glc_bio:.6f} vs product={glc_prod:.6f}")

        if glc_bio > glc_prod:
            try:
                rxn = self.model.reactions.get_by_id('rxn00006_c')
                rxn.bounds = (0, 0)
                print("Closed rxn00006_c (biomass uses more glucose)")
            except KeyError:
                pass

    def _get_glucose_uptake(self, objective_rxn_id):
        """Get glucose uptake for given objective."""
        glc_id = 'EX_cpd00027_e_reverse'
        if glc_id not in self.model.reactions:
            return None

        with self.model as m:
            m.objective = objective_rxn_id
            try:
                sol = pfba(m)
            except:
                sol = m.optimize()

            if sol.status != "optimal":
                return None

            return float(sol.fluxes.get(glc_id, 0.0))

    def _identify_enzyme_complexes(self):
        """
        从模型中识别酶复合体关系
        通过解析gene_reaction_rule中的AND关系来识别
        """
        print("\nIdentifying enzyme complexes from model...")

        # 清空现有的复合体数据
        self.enzyme_complexes = {}
        self.gene_to_complexes = {}
        self.complex_groups = {}

        complex_count = 0

        for reaction in self.model.reactions:
            # 跳过没有基因关联的反应
            if not reaction.gene_reaction_rule:
                continue

            # 解析gene_reaction_rule
            gpr = reaction.gene_reaction_rule.strip()

            # 检查是否包含AND关系（表示酶复合体）
            if ' and ' in gpr.lower():
                # 提取基因ID
                # 处理可能的括号
                gpr_clean = gpr.replace('(', '').replace(')', '')

                # 分割基因
                genes = []
                for part in gpr_clean.split(' and '):
                    gene_id = part.strip()
                    if gene_id and gene_id != '':
                        genes.append(gene_id)

                # 如果有多个基因，则认为是酶复合体
                if len(genes) > 1:
                    complex_id = f"complex_{reaction.id}"
                    self.enzyme_complexes[complex_id] = genes
                    complex_count += 1

                    # 映射每个基因到复合体
                    for gene_id in genes:
                        if gene_id not in self.gene_to_complexes:
                            self.gene_to_complexes[gene_id] = []
                        self.gene_to_complexes[gene_id].append(complex_id)

        # 识别共享亚基的复合体组
        self._identify_complex_groups()

        print(f"  Found {complex_count} enzyme complexes")
        print(f"  Found {len(self.complex_groups)} complex groups with shared subunits")

        # 打印一些例子
        if complex_count > 0 and self.verbose_flux:
            examples = list(self.enzyme_complexes.items())[:3]
            for complex_id, genes in examples:
                print(f"    Example: {complex_id} -> {', '.join(genes)}")

    def _identify_complex_groups(self):
        """
        识别具有共享亚基的复合体组
        """
        # 找出参与多个复合体的基因
        shared_genes = {}
        for gene_id, complexes in self.gene_to_complexes.items():
            if len(complexes) > 1:
                shared_genes[gene_id] = complexes

        # 基于共享基因创建复合体组
        processed_complexes = set()
        group_id = 0

        for gene_id, complexes in shared_genes.items():
            # 创建包含这些复合体的组
            unprocessed = [c for c in complexes if c not in processed_complexes]

            if unprocessed:
                group_id += 1
                group_key = f"group_{group_id}"

                # 收集组中所有复合体的信息
                group_complexes = {}
                group_shared = set([gene_id])

                # 扩展组以包含所有相关复合体
                to_process = set(unprocessed)
                while to_process:
                    current_complex = to_process.pop()
                    if current_complex in processed_complexes:
                        continue

                    processed_complexes.add(current_complex)
                    complex_genes = set(self.enzyme_complexes[current_complex])

                    # 查找其他共享基因的复合体
                    for g in complex_genes:
                        if g in self.gene_to_complexes:
                            for related_complex in self.gene_to_complexes[g]:
                                if related_complex not in processed_complexes:
                                    to_process.add(related_complex)
                                    if g != gene_id and related_complex in unprocessed:
                                        group_shared.add(g)

                    # 计算该复合体的独特基因
                    unique_genes = complex_genes - group_shared

                    group_complexes[current_complex] = {
                        'all_genes': list(complex_genes),
                        'unique': list(unique_genes)
                    }

                if len(group_complexes) > 1:
                    self.complex_groups[group_key] = {
                        'shared': list(group_shared),
                        'complexes': group_complexes
                    }

    def _load_and_configure_model(self, biomass_rxn):
        """Load metabolic model and apply configuration."""
        print("Loading metabolic model...")

        try:
            from ECMpy_function import get_enzyme_constraint_model_with_reaction_set_exclusion
            self.model = get_enzyme_constraint_model_with_reaction_set_exclusion(
                self.model_file
            )
        except:
            self.model = load_json_model(self.model_file)

        # 识别酶复合体（在模型加载后立即进行）
        self._identify_enzyme_complexes()

        self._block_reactions()
        self.biomass_rxn = self._determine_biomass_reaction(biomass_rxn)
        print(f"Using biomass reaction: {self.biomass_rxn}")
        self._set_quiet_mode(True)
    # ==================== ecFactory Methods (Step 1) ====================

    def run_ecfactory_without_evaluation(self):
        """Run ecFactory analysis without individual evaluation."""
        print("\n" + "=" * 60)
        print("Step 1: Running ecFactory Analysis")
        print("=" * 60)

        from ecFactory_light_minpool import ecFactory_light_minpool

        # Initialize ecFactory
        self.ecfactory_instance = ecFactory_light_minpool(
            model=self.model,
            product_rxn=self.product_rxn,
            biomass_rxn=self.biomass_rxn,
            essential_genes_file=self.essential_genes_file,
            fseof_steps=10
        )

        # Configure filtering
        self.ecfactory_instance.set_filtering_percentages(
            separate=True,
            oe_percentage=1.0,
            kd_percentage=0.4
        )

        # Calculate optimal constraint
        self.ecfactory_instance.calculate_optimal_constraint_bound()
        self.optimal_constraint_bound = self.ecfactory_instance.optimal_constraint_bound

        print("Running ecFactory pipeline (skipping final evaluation)...")

        # Run pipeline steps
        print("\n=== EC-FSEOF Analysis ===")
        self.ecfactory_instance.ec_fseof()

        print("\n=== Leak Reaction Identification ===")
        self.ecfactory_instance.identify_leak_reactions()

        print("\n=== Enzyme Group Consistency ===")
        self.ecfactory_instance.ensure_enzyme_group_consistency()

        print("\n=== Enzyme Usage Variability Filter ===")
        self.ecfactory_instance.apply_enzyme_usage_filter()

        if self.enable_distance_filter:
            print("\n=== Metabolic Distance Filter ===")
            self.ecfactory_instance.apply_topology_filter(max_depth=20)

        # Store results
        self.ecfactory_targets = self.ecfactory_instance.targets.copy()
        self.essential_genes = self.ecfactory_instance.essential_genes.copy()

        # Record gene sources
        for target_type in ['OE', 'KD']:
            for gene_id in self.ecfactory_targets.get(target_type, []):
                if gene_id not in self.gene_source_mapping:
                    self.gene_source_mapping[gene_id] = {'sources': set(), 'predictions': set()}
                self.gene_source_mapping[gene_id]['sources'].add('ecFactory')
                self.gene_source_mapping[gene_id]['predictions'].add(target_type)

        print(f"\necFactory analysis complete:")
        print(f"  OE targets: {len(self.ecfactory_targets['OE'])}")
        print(f"  KD targets: {len(self.ecfactory_targets['KD'])}")
        print(f"  Essential genes: {len(self.essential_genes)}")

        return self.ecfactory_targets

    # ==================== iBridge Methods (Step 2) ====================

    def run_ibridge_analysis(self):
        """Run iBridge analysis if enabled."""
        if not self.enable_ibridge:
            return pd.DataFrame()

        print("\n" + "=" * 60)
        print("Step 2: Running iBridge Analysis")
        print("=" * 60)

        try:
            from baebra.ibridge_minpool import predict_ibridge_targets_optimized

            # Apply optimal constraint
            original_bound = None
            if self.optimal_constraint_bound and self.constraint_rxn_id in self.model.reactions:
                constraint_rxn = self.model.reactions.get_by_id(self.constraint_rxn_id)
                original_bound = constraint_rxn.upper_bound
                constraint_rxn.upper_bound = self.optimal_constraint_bound

            # Run iBridge
            ibridge_result = predict_ibridge_targets_optimized(
                self.model,
                self.exclude_met_dir,
                num_step=10,
                target_compound=self.product_id,
                bio_fraction=0.1,
                threshold=0.01,
                top_score_percentage=0.4,
            )

            # Restore constraint
            if original_bound is not None:
                constraint_rxn.upper_bound = original_bound

            # Process results
            self.ibridge_targets = ibridge_result[0] if isinstance(ibridge_result, tuple) else ibridge_result

            if not isinstance(self.ibridge_targets, pd.DataFrame):
                self.ibridge_targets = pd.DataFrame()

            print(f"iBridge found {len(self.ibridge_targets)} target reactions")

            # Filter biomass reactions
            self.ibridge_targets = self._filter_biomass_reactions(self.ibridge_targets)

            # Map reactions to genes
            self._map_ibridge_reactions_to_genes()

            return self.ibridge_targets

        except ImportError:
            print("iBridge module not available, skipping analysis")
            self.ibridge_targets = pd.DataFrame()
            return self.ibridge_targets
        except Exception as e:
            print(f"iBridge analysis error: {e}")
            self.ibridge_targets = pd.DataFrame()
            return self.ibridge_targets

    def _filter_biomass_reactions(self, targets):
        """Filter out biomass-related reactions."""
        if targets.empty:
            return targets

        initial_count = len(targets)
        biomass_patterns = ['Bio_', 'BIOMASS', 'biomass', 'ATPM', 'maintenance']

        mask = ~targets['Target_reactions'].str.contains('|'.join(biomass_patterns),
                                                         case=False, na=False)
        filtered = targets[mask].reset_index(drop=True)

        if len(filtered) < initial_count:
            print(f"Filtered {initial_count - len(filtered)} biomass reactions")

        return filtered

    def _map_ibridge_reactions_to_genes(self):
        """Map iBridge reactions to genes and detect conflicts."""
        print("\nMapping iBridge reactions to genes...")

        if self.ibridge_targets.empty:
            return {}

        gene_regulations_temp = {}
        gene_reactions_map = {}

        for idx, row in self.ibridge_targets.iterrows():
            rxn_id = row['Target_reactions']
            regulation_type = self._normalize_regulation_type(row.get('Regulation_type', 'KD'))

            try:
                rxn = self.model.reactions.get_by_id(rxn_id)
                genes = [gene.id for gene in rxn.genes]

                if genes:
                    for gene_id in genes:
                        if gene_id not in gene_regulations_temp:
                            gene_regulations_temp[gene_id] = []
                            gene_reactions_map[gene_id] = []

                        gene_regulations_temp[gene_id].append(regulation_type)
                        gene_reactions_map[gene_id].append((rxn_id, regulation_type))

            except KeyError:
                print(f"Reaction {rxn_id} not found")

        # Identify conflicts
        for gene_id, reg_types in gene_regulations_temp.items():
            unique_types = list(set(reg_types))

            if len(unique_types) > 1:
                self.ibridge_gene_regulations[gene_id] = 'conflict'
                print(f"  Conflict in iBridge: {gene_id} has {unique_types}")
            else:
                self.ibridge_gene_regulations[gene_id] = unique_types[0]

    def _normalize_regulation_type(self, reg_type):
        """Normalize regulation type to OE or KD."""
        if pd.isna(reg_type):
            return 'KD'

        type_str = str(reg_type).strip().lower()

        oe_keywords = ['up', 'upregulation', 'overexpression', 'oe', 'increase', 'enhance']
        kd_keywords = ['down', 'downregulation', 'knockdown', 'kd', 'decrease', 'reduce']

        if any(keyword in type_str for keyword in oe_keywords):
            return 'OE'
        elif any(keyword in type_str for keyword in kd_keywords):
            return 'KD'
        else:
            return 'KD'

    # ==================== Target Combination (Step 3) ====================

    def combine_all_targets(self):
        """Combine targets from all sources and identify conflicts."""
        print("\n" + "=" * 60)
        print("Step 3: Combining Targets and Identifying Conflicts")
        print("=" * 60)

        combined = {'OE': [], 'KD': [], 'Test': []}

        # Collect ecFactory genes
        ecfactory_genes = set()
        ecfactory_oe = set(self.ecfactory_targets.get('OE', []))
        ecfactory_kd = set(self.ecfactory_targets.get('KD', []))
        ecfactory_genes = ecfactory_oe | ecfactory_kd

        print(f"ecFactory contributed: {len(ecfactory_genes)} genes")

        # Process iBridge targets if available
        if self.enable_ibridge and not self.ibridge_targets.empty:
            new_oe = 0
            new_kd = 0
            new_test = 0

            for gene_id, reg_type in self.ibridge_gene_regulations.items():
                if gene_id not in self.gene_source_mapping:
                    self.gene_source_mapping[gene_id] = {'sources': set(), 'predictions': set()}
                self.gene_source_mapping[gene_id]['sources'].add('iBridge')

                if reg_type == 'conflict':
                    if gene_id not in ecfactory_genes:
                        combined['Test'].append(gene_id)
                        new_test += 1
                        self.conflicted_genes.append(gene_id)
                else:
                    in_oe = gene_id in ecfactory_oe
                    in_kd = gene_id in ecfactory_kd

                    if (in_oe and reg_type == 'KD') or (in_kd and reg_type == 'OE'):
                        self.conflicted_genes.append(gene_id)
                        combined['Test'].append(gene_id)
                        print(f"  Conflict detected: {gene_id}")
                    elif not in_oe and not in_kd:
                        if reg_type == 'OE':
                            combined['OE'].append(gene_id)
                            new_oe += 1
                        else:
                            combined['KD'].append(gene_id)
                            new_kd += 1

            print(f"iBridge added: OE={new_oe}, KD={new_kd}, Conflicts={new_test}")

        # Add non-conflicted ecFactory genes
        conflicted_set = set(self.conflicted_genes)

        for gene_id in ecfactory_oe:
            if gene_id not in conflicted_set:
                combined['OE'].append(gene_id)

        for gene_id in ecfactory_kd:
            if gene_id not in conflicted_set:
                combined['KD'].append(gene_id)

        # Remove duplicates
        for target_type in combined:
            combined[target_type] = list(set(combined[target_type]))

        self.combined_targets = combined
        self.conflicted_genes = list(set(self.conflicted_genes))

        print(f"\nCombined targets:")
        print(f"  OE: {len(combined['OE'])}")
        print(f"  KD: {len(combined['KD'])}")
        print(f"  Conflicts: {len(combined['Test'])}")

        return combined

    # ==================== Unified Evaluation (Step 4) - Using Original ecFbridge Logic ====================

    def unified_target_evaluation(self):
        """
        Evaluate all targets with conflict resolution and filtering.
        Modified to include special handling for Succinate (cpd00036) similar to minpool.
        """
        print("\n" + "=" * 60)
        print("Step 4: Unified Target Evaluation (With Succinate Logic)")
        print("=" * 60)

        if not self.ecfactory_instance:
            print("Error: ecFactory instance not available")
            return {}

        # --- 1. 琥珀酸特殊逻辑检测 ---
        is_succinate = False
        succinate_ids = ['cpd00036', 'succ', 'succinate']
        if self.product_id.lower() in [s.lower() for s in succinate_ids]:
            is_succinate = True
            print("\n*** 检测到琥珀酸产品 (Succinate detected) ***")

        # 需要优先评估的基因（与待关闭反应相关）
        priority_gene = 'CEY17_RS14060'
        priority_gene_evaluated = False
        priority_gene_results = {}

        # --- 2. 计算初始 Baseline ---
        print("\nCalculating baseline performance...")
        baseline_production, baseline_yield = self._calculate_baseline_performance()
        print(f"Baseline: production={baseline_production:.6f}, yield={baseline_yield:.6f}")

        # Get original kcat values
        original_kcat_values = self._get_original_kcat_values()

        # Detect additional conflicts
        self._detect_all_conflicts()

        # --- 3. 琥珀酸特殊处理逻辑 (参照 minpool) ---
        if is_succinate:
            # A. 优先评估基因 (Before closing reactions)
            all_genes_to_evaluate = (
                    self.combined_targets.get('Test', []) +
                    self.combined_targets.get('OE', []) +
                    self.combined_targets.get('KD', [])
            )

            if priority_gene in all_genes_to_evaluate:
                print(f"\n*** 优先评估基因 {priority_gene} (关联反应 rxn00173_c) ***")
                print(f"  [注意: 此基因使用反应关闭前的 baseline 进行评估]")

                gene_in_test = priority_gene in self.combined_targets.get('Test', [])
                gene_in_oe = priority_gene in self.combined_targets.get('OE', [])

                # 确定基因来源
                ecfactory_genes = set()
                for t_type in ['OE', 'KD']:
                    ecfactory_genes.update(self.ecfactory_targets.get(t_type, []))
                gene_source = 'ecFactory' if priority_gene in ecfactory_genes else 'iBridge'

                if gene_in_test:
                    # 作为冲突基因评估（测试两个方向）
                    oe_prod, oe_yield = self._test_gene_modification(priority_gene, 'OE', original_kcat_values)
                    kd_prod, kd_yield = self._test_gene_modification(priority_gene, 'KD', original_kcat_values)

                    oe_prod_imp = (
                        (oe_prod - baseline_production) / baseline_production * 100 if baseline_production > 0 else 0)
                    oe_yield_imp = ((oe_yield - baseline_yield) / baseline_yield * 100 if baseline_yield > 0 else 0)
                    kd_prod_imp = (
                        (kd_prod - baseline_production) / baseline_production * 100 if baseline_production > 0 else 0)
                    kd_yield_imp = ((kd_yield - baseline_yield) / baseline_yield * 100 if baseline_yield > 0 else 0)

                    priority_gene_results = {
                        'oe_prod_imp': oe_prod_imp, 'oe_yield_imp': oe_yield_imp,
                        'kd_prod_imp': kd_prod_imp, 'kd_yield_imp': kd_yield_imp,
                        'category': 'Test', 'source': gene_source
                    }
                else:
                    # 作为单向基因评估
                    target_type = 'OE' if gene_in_oe else 'KD'
                    production, yield_value = self._test_gene_modification(priority_gene, target_type,
                                                                           original_kcat_values)

                    prod_imp = ((
                                            production - baseline_production) / baseline_production * 100 if baseline_production > 0 else 0)
                    yield_imp = ((yield_value - baseline_yield) / baseline_yield * 100 if baseline_yield > 0 else 0)

                    priority_gene_results = {
                        'prod_imp': prod_imp, 'yield_imp': yield_imp,
                        'target_type': target_type, 'category': target_type, 'source': gene_source
                    }

                priority_gene_evaluated = True
                print(f"*** 基因 {priority_gene} 评估完成 ***")

            # B. 关闭特定反应 (Close reactions)
            print("\n*** 应用琥珀酸特殊约束：关闭反应 ***")
            reactions_to_close = ['rxn00173_c', 'rxn00288_c_reverse']
            for rxn_id in reactions_to_close:
                try:
                    rxn = self.model.reactions.get_by_id(rxn_id)
                    original_bounds = rxn.bounds
                    rxn.bounds = (0, 0)
                    print(f"  已关闭反应: {rxn_id} (原边界: {original_bounds})")
                except KeyError:
                    print(f"  警告: 反应 {rxn_id} 在模型中不存在")

            # C. 重新计算 Baseline (Re-calculate baseline)
            print("*** 重新计算 baseline（关闭反应后）***")
            baseline_production, baseline_yield = self._calculate_baseline_performance()
            print(f"New Baseline: production={baseline_production:.6f}, yield={baseline_yield:.6f}")

        # --- 4. 解决冲突 (传递优先评估结果) ---
        if self.combined_targets.get('Test'):
            print(f"\nResolving {len(self.combined_targets['Test'])} conflicted genes...")
            self._resolve_conflicts(
                baseline_production, baseline_yield, original_kcat_values,
                priority_gene if priority_gene_evaluated else None,
                priority_gene_results if priority_gene_evaluated else None
            )

        # --- 5. 评估 OE 和 KD 靶点 ---
        final_targets = {'OE': [], 'KD': []}

        for target_type in ['OE', 'KD']:
            print(f"\nEvaluating {target_type} targets ({len(self.combined_targets[target_type])})...")

            for gene_id in self.combined_targets[target_type]:
                # 检查是否是已评估的优先基因
                if priority_gene_evaluated and gene_id == priority_gene:
                    if priority_gene_results.get('category') in ['OE', 'KD']:
                        prod_imp = priority_gene_results['prod_imp']
                        yield_imp = priority_gene_results['yield_imp']
                        gene_source = priority_gene_results['source']
                        print(f"  [使用优先评估结果] {gene_id}")
                    else:
                        # 如果该基因本来在Test中但已被解决，这里跳过
                        continue
                else:
                    # 常规测试
                    production, yield_value = self._test_gene_modification(
                        gene_id, target_type, original_kcat_values
                    )
                    prod_imp = ((
                                            production - baseline_production) / baseline_production * 100 if baseline_production > 0 else 0)
                    yield_imp = ((yield_value - baseline_yield) / baseline_yield * 100 if baseline_yield > 0 else 0)

                    ecfactory_genes = set()
                    for t_type in ['OE', 'KD']:
                        ecfactory_genes.update(self.ecfactory_targets.get(t_type, []))
                    gene_source = 'ecFactory' if gene_id in ecfactory_genes else 'iBridge'

                # Store results
                self.test_results[gene_id] = {
                    'prod_improvement': prod_imp,
                    'yield_improvement': yield_imp,
                    'type': target_type,
                    'source': 'evaluation'
                }

                # Apply filtering
                if self._passes_thresholds(gene_id, target_type, prod_imp, yield_imp):
                    final_targets[target_type].append(gene_id)
                    print(
                        f"  ✔ Kept {gene_source} gene {gene_id} ({target_type}): prod +{prod_imp:.3f}%, yield +{yield_imp:.3f}%")
                else:
                    if target_type == 'KD' and (
                            prod_imp <= self.exclusion_threshold or yield_imp <= self.exclusion_threshold):
                        print(f"  ✗ Excluded {gene_id} ({target_type}): below exclusion threshold")
                    else:
                        print(f"  ✗ Filtered {gene_source} gene {gene_id} ({target_type}): below improvement threshold")

        self.combined_targets = final_targets

        print(f"\nEvaluation complete:")
        print(f"  Final OE: {len(final_targets['OE'])}")
        print(f"  Final KD: {len(final_targets['KD'])}")
        print(f"  Excluded: {len(self.excluded_genes)}")
        print(f"  Filtered: {len(self.filtered_genes)}")

        return final_targets

    def _detect_all_conflicts(self):
        """Detect all conflicting genes across categories."""
        print("\nDetecting conflicts across all categories...")

        oe_genes = set(self.combined_targets.get('OE', []))
        kd_genes = set(self.combined_targets.get('KD', []))
        test_genes = set(self.combined_targets.get('Test', []))

        new_conflicts = oe_genes & kd_genes

        if new_conflicts:
            print(f"Found {len(new_conflicts)} OE-KD conflicts")
            for gene_id in new_conflicts:
                if gene_id in self.combined_targets['OE']:
                    self.combined_targets['OE'].remove(gene_id)
                if gene_id in self.combined_targets['KD']:
                    self.combined_targets['KD'].remove(gene_id)
                if gene_id not in test_genes:
                    self.combined_targets['Test'].append(gene_id)
                if gene_id not in self.conflicted_genes:
                    self.conflicted_genes.append(gene_id)

        all_conflicts = test_genes | new_conflicts
        self.combined_targets['Test'] = list(all_conflicts)

        print(f"Total conflicts to resolve: {len(self.combined_targets['Test'])}")

    def _resolve_conflicts(self, baseline_production, baseline_yield, original_kcat_values,
                           priority_gene_id=None, priority_results=None):
        """
        Resolve conflicted genes by testing both directions.
        Modified to accept pre-calculated results for priority genes.
        """
        ecfactory_genes = set()
        for target_type in ['OE', 'KD']:
            ecfactory_genes.update(self.ecfactory_targets.get(target_type, []))

        for gene_id in self.combined_targets.get('Test', []):

            # --- 检查是否为优先基因 ---
            if priority_gene_id and gene_id == priority_gene_id and priority_results:
                print(f"\nResolving conflict for: {gene_id} (Using Pre-calculated Results)")
                oe_prod_imp = priority_results['oe_prod_imp']
                oe_yield_imp = priority_results['oe_yield_imp']
                kd_prod_imp = priority_results['kd_prod_imp']
                kd_yield_imp = priority_results['kd_yield_imp']
                gene_source = priority_results['source']

            else:
                # 常规冲突解决逻辑
                print(f"\nResolving conflict for: {gene_id}")

                # Test both modifications
                oe_prod, oe_yield = self._test_gene_modification(gene_id, 'OE', original_kcat_values)
                kd_prod, kd_yield = self._test_gene_modification(gene_id, 'KD', original_kcat_values)

                # Calculate improvements
                oe_prod_imp = (
                    (oe_prod - baseline_production) / baseline_production * 100 if baseline_production > 0 else 0)
                oe_yield_imp = ((oe_yield - baseline_yield) / baseline_yield * 100 if baseline_yield > 0 else 0)

                kd_prod_imp = (
                    (kd_prod - baseline_production) / baseline_production * 100 if baseline_production > 0 else 0)
                kd_yield_imp = ((kd_yield - baseline_yield) / baseline_yield * 100 if baseline_yield > 0 else 0)

                gene_source = 'ecFactory' if gene_id in ecfactory_genes else 'iBridge'

            print(f"  OE test: prod +{oe_prod_imp:.3f}%, yield +{oe_yield_imp:.3f}%")
            print(f"  KD test: prod +{kd_prod_imp:.3f}%, yield +{kd_yield_imp:.3f}%")

            # Choose best direction
            optimal_type, optimal_prod_imp, optimal_yield_imp = self._choose_best_direction(
                oe_prod_imp, oe_yield_imp, kd_prod_imp, kd_yield_imp
            )

            # Apply threshold filtering
            if self._passes_thresholds(gene_id, optimal_type, optimal_prod_imp, optimal_yield_imp):
                self.combined_targets[optimal_type].append(gene_id)
                self.test_results[gene_id] = {
                    'prod_improvement': optimal_prod_imp,
                    'yield_improvement': optimal_yield_imp,
                    'type': optimal_type,
                    'source': 'conflict_resolution'
                }
                print(f"  ✔ Resolved: {gene_id} -> {optimal_type} ({gene_source} gene)")
            else:
                print(f"  ✗ Filtered: {gene_id} (below thresholds)")

            # Record resolution results
            self.conflict_resolution_results[gene_id] = {
                'optimal_type': optimal_type,
                'oe_prod_improvement': oe_prod_imp,
                'oe_yield_improvement': oe_yield_imp,
                'kd_prod_improvement': kd_prod_imp,
                'kd_yield_improvement': kd_yield_imp,
                'optimal_prod_improvement': optimal_prod_imp,
                'optimal_yield_improvement': optimal_yield_imp,
                'source': gene_source,
                'saved': gene_id in self.combined_targets.get(optimal_type, [])
            }

        # Clear Test category
        self.combined_targets['Test'] = []

    def _choose_best_direction(self, oe_prod_imp, oe_yield_imp, kd_prod_imp, kd_yield_imp):
        """Choose best modification direction based on improvements."""
        # Check exclusion threshold
        oe_excluded = (oe_prod_imp <= self.exclusion_threshold or
                       oe_yield_imp <= self.exclusion_threshold)
        kd_excluded = (kd_prod_imp <= self.exclusion_threshold or
                       kd_yield_imp <= self.exclusion_threshold)

        if oe_excluded and kd_excluded:
            # Both excluded, choose less bad
            oe_score = oe_prod_imp + oe_yield_imp
            kd_score = kd_prod_imp + kd_yield_imp
            if oe_score > kd_score:
                return 'OE', oe_prod_imp, oe_yield_imp
            else:
                return 'KD', kd_prod_imp, kd_yield_imp
        elif oe_excluded:
            return 'KD', kd_prod_imp, kd_yield_imp
        elif kd_excluded:
            return 'OE', oe_prod_imp, oe_yield_imp
        else:
            # Choose based on total improvement
            oe_score = oe_prod_imp + oe_yield_imp
            kd_score = kd_prod_imp + kd_yield_imp
            if oe_score > kd_score:
                return 'OE', oe_prod_imp, oe_yield_imp
            else:
                return 'KD', kd_prod_imp, kd_yield_imp

    def _passes_thresholds(self, gene_id, target_type, prod_imp, yield_imp):
        """Check if gene passes filtering thresholds."""
        if target_type == 'KD' and (prod_imp <= self.exclusion_threshold or
                                    yield_imp <= self.exclusion_threshold):
            self.excluded_genes.append(gene_id)
            return False

        if prod_imp >= self.improvement_threshold or yield_imp >= self.improvement_threshold:
            return True
        else:
            self.filtered_genes.append(gene_id)
            return False

    def _calculate_baseline_performance(self):
        """
        Calculate baseline production and yield at 25% growth.
        使用原始ecFbridge.py的方法
        """
        try:
            with self.model as model:
                model.objective = self.biomass_rxn
                sol_growth = model.optimize()
                max_growth = sol_growth.objective_value

            with self.model as model:
                growth_rate = 0.25 * max_growth
                model.reactions.get_by_id(self.biomass_rxn).lower_bound = growth_rate
                model.objective = self.product_rxn
                sol_baseline = pfba(model)

                baseline_production = sol_baseline.fluxes[self.product_rxn]

                cs_rxn = self._get_carbon_source_reaction()
                baseline_cs_uptake = abs(sol_baseline.fluxes.get(cs_rxn, 1.0))

                baseline_yield = baseline_production / baseline_cs_uptake if baseline_cs_uptake > 0 else 0

            self.baseline_calculated = True
            return baseline_production, baseline_yield

        except Exception as e:
            print(f"Error calculating baseline: {e}")
            return 0.0, 0.0

    def _test_gene_modification(self, gene_id, mod_type, original_kcat_values):
        """
        Test the effect of modifying a single gene.
        使用原始ecFbridge.py的方法
        """
        try:
            modifications = {'OE': [], 'KD': []}
            modifications[mod_type] = [gene_id]

            with self.model as model:
                modified_model, solution, production = self._apply_modifications_and_reload_single(
                    model, modifications, original_kcat_values
                )

                if solution is None:
                    return 0.0, 0.0

                cs_rxn = self._get_carbon_source_reaction()
                cs_uptake = abs(solution.fluxes.get(cs_rxn, 1.0))
                yield_value = production / cs_uptake if cs_uptake > 0 else 0

                return production, yield_value

        except Exception as e:
            print(f"    Error testing {gene_id} {mod_type}: {e}")
            return 0.0, 0.0

    def _get_original_kcat_values(self):
        """Get original kcat_MW values from current model."""
        original_kcat_values = {}
        for reaction in self.model.reactions:
            if hasattr(reaction, 'kcat_MW') and reaction.kcat_MW:
                try:
                    original_kcat_values[reaction.id] = float(reaction.kcat_MW)
                except:
                    pass
        return original_kcat_values

    def _apply_modifications_and_reload_single(self, model, modifications, original_values=None):
        """
        Apply gene modifications and reload model for single gene testing.
        使用原始ecFbridge.py的逻辑
        """
        OEF = 5.0
        KDF = 0.2

        modified_kcat_mw_data = {}
        temp_filename = None

        try:
            orig_values = original_values or {}

            # Apply modifications
            for mod_type, genes in modifications.items():
                for gene in genes:
                    if mod_type in ['OE', 'KD']:
                        gene_reactions = model.genes.get_by_id(gene).reactions
                        for reaction in gene_reactions:
                            if hasattr(reaction, 'kcat_MW') and reaction.kcat_MW:
                                try:
                                    original_kcat = orig_values.get(reaction.id, float(reaction.kcat_MW))
                                    new_kcat = original_kcat * (OEF if mod_type == 'OE' else KDF)
                                    reaction.kcat_MW = new_kcat
                                    modified_kcat_mw_data[reaction.id] = new_kcat
                                except:
                                    pass

            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.json', mode='w', delete=False) as temp_file:
                temp_filename = temp_file.name
                cobra.io.save_json_model(model, temp_filename)

            # Modify JSON
            with open(temp_filename, 'r') as file:
                model_dict = json.load(file)

            if 'enzyme_constraint' not in model_dict:
                model_dict['enzyme_constraint'] = {
                    'enzyme_mass_fraction': 0.46,
                    'total_protein_fraction': 0.56,
                    'average_saturation': 0.5,
                    'lowerbound': 0,
                    'upperbound': 0.129
                }

            # Update kcat_MW values
            for reaction in model_dict['reactions']:
                if reaction['id'] in modified_kcat_mw_data:
                    reaction['kcat_MW'] = modified_kcat_mw_data[reaction['id']]
                    reaction['kcat'] = ''
                else:
                    reaction['kcat_MW'] = orig_values.get(reaction['id'], reaction.get('kcat_MW', ''))
                    reaction['kcat'] = ''

            with open(temp_filename, 'w') as file:
                json.dump(model_dict, file)

            # Reload model
            try:
                from ECMpy_function import get_enzyme_constraint_model
                updated_model = get_enzyme_constraint_model(temp_filename)
            except:
                updated_model = cobra.io.load_json_model(temp_filename)

            # Set growth constraint
            try:
                updated_model.objective = self.biomass_rxn
                growth_solution = updated_model.optimize()
                if growth_solution.status == 'optimal':
                    max_growth = growth_solution.objective_value
                    updated_model.reactions.get_by_id(self.biomass_rxn).lower_bound = 0.25 * max_growth
            except:
                pass

            # Optimize for product
            updated_model.objective = self.product_rxn

            try:
                solution = pfba(updated_model)
                production = solution.fluxes[self.product_rxn]
                return updated_model, solution, production
            except:
                return None, None, 0.0

        except Exception as e:
            print(f"Error applying modifications: {e}")
            return None, None, 0.0

        finally:
            if temp_filename and os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                except:
                    pass

    # ==================== Enhanced Combination Optimization (Step 5) ====================
    # 保留原有的combination优化方法，但使用保存的original_model和original_kcat_values

    def find_minimal_gene_combination(self):
        """Find minimal gene modification set with enhanced flux tracking and enzyme complex handling."""
        print("\n" + "=" * 70)
        print("Step 5: Finding Minimal Gene Combination with Enzyme Complex Handling")
        print("=" * 70)

        # Get all candidate genes
        all_candidates = {
            'OE': self.combined_targets.get('OE', []).copy(),
            'KD': self.combined_targets.get('KD', []).copy()
        }

        total_genes = len(all_candidates['OE']) + len(all_candidates['KD'])
        print(f"\nStarting with {total_genes} candidate genes:")
        print(f"  OE candidates: {len(all_candidates['OE'])}")
        print(f"  KD candidates: {len(all_candidates['KD'])}")

        if total_genes == 0:
            print("No candidate genes to optimize")
            self.minimal_gene_set = {'OE': [], 'KD': []}
            return self.minimal_gene_set

        # Identify which candidates are part of complexes
        self._map_candidates_to_complexes(all_candidates)

        # Calculate baseline with enhanced tracking
        print("\n" + "-" * 70)
        print("BASELINE PERFORMANCE (No Modifications)")
        print("-" * 70)
        baseline_data = self._calculate_detailed_baseline()
        self._print_flux_details(baseline_data, "Baseline (25% max growth)")

        # Calculate optimal performance with ALL modifications
        print("\n" + "-" * 70)
        print("OPTIMAL PERFORMANCE (All Genes Modified)")
        print("-" * 70)
        optimal_data = self._test_combination_with_details(all_candidates)
        self._print_flux_details(optimal_data, "All modifications applied")

        # Store initial flux state
        self.flux_history.append({
            'stage': 'initial_optimal',
            'genes_oe': len(all_candidates['OE']),
            'genes_kd': len(all_candidates['KD']),
            **optimal_data
        })

        # Display improvement
        print("\n" + "-" * 50)
        print("Improvement Analysis:")
        prod_imp = ((optimal_data['production'] - baseline_data['production']) /
                    baseline_data['production'] * 100 if baseline_data['production'] > 0 else 0)
        yield_imp = ((optimal_data['yield'] - baseline_data['yield']) /
                     baseline_data['yield'] * 100 if baseline_data['yield'] > 0 else 0)
        glc_change = ((optimal_data['glucose_uptake'] - baseline_data['glucose_uptake']) /
                      baseline_data['glucose_uptake'] * 100 if baseline_data['glucose_uptake'] > 0 else 0)

        print(f"  Production improvement: {prod_imp:.2f}%")
        print(f"  Yield improvement: {yield_imp:.2f}%")
        print(f"  Glucose uptake change: {glc_change:+.2f}%")

        if optimal_data['production'] <= 0:
            print("Warning: Optimal production is zero or negative")
            self.minimal_gene_set = all_candidates
            return self.minimal_gene_set

        # Store initial optimal values
        initial_optimal = optimal_data.copy()

        # Initialize minimal set
        minimal_set = {
            'OE': all_candidates['OE'].copy(),
            'KD': all_candidates['KD'].copy()
        }

        # Prepare test units (individual genes and complexes)
        test_units = self._prepare_test_units(all_candidates)

        # Sort test units by their potential impact
        test_units = self._sort_test_units(test_units)

        # Subtractive optimization
        print("\n" + "=" * 70)
        print("SUBTRACTIVE OPTIMIZATION PROCESS")
        print(f"Testing removal of {len(test_units)} units (genes/complexes)")
        print(f"Eta threshold: {self.eta_threshold}")
        print("=" * 70)

        removed_count = 0
        iteration = 0
        current_data = initial_optimal.copy()

        for unit_info in test_units:
            iteration += 1
            unit_type = unit_info['type']  # 'gene', 'simple_complex', or 'shared_complex_group'

            print(f"\n" + "-" * 60)
            print(f"[{iteration}/{len(test_units)}] Testing removal of {unit_type}: {unit_info['id']}")
            print("-" * 60)

            # Handle different unit types
            if unit_type == 'gene':
                # Simple gene removal
                removed_genes, kept_genes = self._test_gene_removal(
                    unit_info, minimal_set, current_data, initial_optimal, baseline_data
                )

            elif unit_type == 'simple_complex':
                # Simple complex without shared subunits
                removed_genes, kept_genes = self._test_simple_complex_removal(
                    unit_info, minimal_set, current_data, initial_optimal, baseline_data
                )

            elif unit_type == 'shared_complex_group':
                # Complex group with shared subunits
                removed_genes, kept_genes = self._test_shared_complex_removal(
                    unit_info, minimal_set, current_data, initial_optimal, baseline_data
                )

            # Update counts and current state
            if removed_genes:
                removed_count += len(removed_genes)
                self.removed_genes.extend(removed_genes)

                # Update current_data if genes were removed
                test_set = {
                    'OE': [g for g in minimal_set['OE'] if g not in removed_genes],
                    'KD': [g for g in minimal_set['KD'] if g not in removed_genes]
                }
                current_data = self._test_combination_with_details(test_set)

                # Update minimal_set
                minimal_set['OE'] = [g for g in minimal_set['OE'] if g not in removed_genes]
                minimal_set['KD'] = [g for g in minimal_set['KD'] if g not in removed_genes]

                # Show cumulative effect
                cumulative_production_retention = (current_data['production'] /
                                                   initial_optimal['production'] * 100)
                print(f"  Cumulative production retention: {cumulative_production_retention:.1f}%")
                print(f"  Total genes removed so far: {removed_count}")

        # Final verification
        print("\n" + "=" * 70)
        print("FINAL VERIFICATION")
        print("=" * 70)

        final_data = self._test_combination_with_details(minimal_set)

        print("\nFinal minimal set performance:")
        self._print_flux_details(final_data, "Final combination")

        # Store final flux state
        self.flux_history.append({
            'stage': 'final_minimal',
            'genes_oe': len(minimal_set['OE']),
            'genes_kd': len(minimal_set['KD']),
            **final_data
        })

        # Calculate final improvements
        print("\n" + "-" * 50)
        print("Final Improvements over Baseline:")
        prod_improvement = ((final_data['production'] - baseline_data['production']) /
                            baseline_data['production'] * 100 if baseline_data['production'] > 0 else 0)
        yield_improvement = ((final_data['yield'] - baseline_data['yield']) /
                             baseline_data['yield'] * 100 if baseline_data['yield'] > 0 else 0)
        glc_final_change = ((final_data['glucose_uptake'] - baseline_data['glucose_uptake']) /
                            baseline_data['glucose_uptake'] * 100 if baseline_data['glucose_uptake'] > 0 else 0)

        print(f"  Production improvement: {prod_improvement:.2f}%")
        print(f"  Yield improvement: {yield_improvement:.2f}%")
        print(f"  Glucose uptake change: {glc_final_change:+.2f}%")

        # Calculate retention
        retention_rate = (final_data['production'] / initial_optimal['production'] * 100
                          if initial_optimal['production'] > 0 else 0)

        print("\n" + "-" * 50)
        print("Performance Retention Analysis:")
        print(f"  {retention_rate:.1f}% of initial optimal production maintained")
        print(f"  Glucose efficiency: {(initial_optimal['glucose_uptake'] / final_data['glucose_uptake']):.2f}x")

        # Summary
        print("\n" + "=" * 70)
        print("OPTIMIZATION SUMMARY")
        print("=" * 70)
        print(f"  Initial genes: {total_genes}")
        print(f"  Removed genes: {removed_count}")
        print(f"  Final OE genes: {len(minimal_set['OE'])}")
        print(f"  Final KD genes: {len(minimal_set['KD'])}")
        print(f"  Reduction: {removed_count / total_genes * 100:.1f}%")
        print(f"  Performance retention: {retention_rate:.1f}%")

        # Store results
        self.minimal_gene_set = minimal_set
        self.combination_results = {
            'initial_genes': all_candidates,
            'minimal_genes': minimal_set,
            'removed_genes': self.removed_genes,
            'eta_values': self.eta_values,
            'complex_eta_values': self.complex_eta_values,
            'final_data': final_data,
            'baseline_data': baseline_data,
            'initial_optimal_data': initial_optimal,
            'performance_retention': retention_rate,
            'optimization_steps': self.optimization_steps,
            'flux_history': self.flux_history,
            'gene_impact_summary': self.gene_impact_summary
        }

        # Warning if significant performance loss
        if retention_rate < 80:
            print("\n⚠ WARNING: Significant performance loss detected!")
            print(f"  Only {retention_rate:.1f}% of optimal production retained")
            print("  Consider adjusting eta threshold or reviewing gene selections")

        return minimal_set

    def _map_candidates_to_complexes(self, candidates):
        """Map candidate genes to their complexes and identify relationships."""
        self.candidate_complexes = {'OE': {}, 'KD': {}}
        self.candidate_complex_groups = {'OE': {}, 'KD': {}}

        for mod_type in ['OE', 'KD']:
            candidate_genes = set(candidates[mod_type])

            # Find complexes that have all members in candidates
            for complex_id, complex_genes in self.enzyme_complexes.items():
                if all(gene in candidate_genes for gene in complex_genes):
                    self.candidate_complexes[mod_type][complex_id] = complex_genes

            # Find complex groups that have members in candidates
            for group_id, group_info in self.complex_groups.items():
                relevant_complexes = {}
                for complex_id, complex_info in group_info['complexes'].items():
                    if all(gene in candidate_genes for gene in complex_info['all_genes']):
                        relevant_complexes[complex_id] = complex_info

                if relevant_complexes:
                    self.candidate_complex_groups[mod_type][group_id] = {
                        'shared': group_info['shared'],
                        'complexes': relevant_complexes
                    }

    def _prepare_test_units(self, candidates):
        """Prepare test units (individual genes and complexes) for optimization."""
        test_units = []
        processed_genes = set()

        for mod_type in ['OE', 'KD']:
            # Add complex groups with shared subunits
            for group_id, group_info in self.candidate_complex_groups[mod_type].items():
                all_genes = set(group_info['shared'])
                for complex_info in group_info['complexes'].values():
                    all_genes.update(complex_info['all_genes'])

                test_units.append({
                    'id': group_id,
                    'type': 'shared_complex_group',
                    'mod_type': mod_type,
                    'genes': list(all_genes),
                    'group_info': group_info
                })
                processed_genes.update(all_genes)

            # Add simple complexes (no shared subunits)
            for complex_id, complex_genes in self.candidate_complexes[mod_type].items():
                # Skip if already part of a complex group
                if any(gene in processed_genes for gene in complex_genes):
                    continue

                test_units.append({
                    'id': complex_id,
                    'type': 'simple_complex',
                    'mod_type': mod_type,
                    'genes': complex_genes
                })
                processed_genes.update(complex_genes)

            # Add individual genes not in any complex
            for gene in candidates[mod_type]:
                if gene not in processed_genes:
                    test_units.append({
                        'id': gene,
                        'type': 'gene',
                        'mod_type': mod_type,
                        'genes': [gene]
                    })
                    processed_genes.add(gene)

        return test_units

    def _sort_test_units(self, test_units):
        """Sort test units by their potential impact."""
        # Calculate average impact for each unit
        for unit in test_units:
            total_impact = 0
            for gene in unit['genes']:
                if gene in self.test_results:
                    total_impact += abs(self.test_results[gene].get('prod_improvement', 0))
            unit['avg_impact'] = total_impact / len(unit['genes']) if unit['genes'] else 0

        # Sort by average impact (ascending, so less impactful units are tested first)
        return sorted(test_units, key=lambda x: x['avg_impact'])

    def _test_gene_removal(self, unit_info, minimal_set, current_data, initial_optimal, baseline_data):
        """Test removal of a single gene."""
        gene_id = unit_info['id']
        mod_type = unit_info['mod_type']

        if gene_id not in minimal_set[mod_type]:
            return [], []

        # Create test set without this gene
        test_set = {
            'OE': minimal_set['OE'].copy(),
            'KD': minimal_set['KD'].copy()
        }
        test_set[mod_type].remove(gene_id)

        # Test performance
        test_data = self._test_combination_with_details(test_set)

        # 修正：使用current_data而不是initial_optimal来计算eta
        eta = self._calculate_eta(
            current_data['production'],  # 改为使用当前状态
            current_data['glucose_uptake'],  # 改为使用当前状态
            test_data['production'],
            test_data['glucose_uptake']
        )

        self.eta_values[gene_id] = eta

        # Display results
        self._display_test_results(test_data, current_data, eta)

        # Decide whether to remove
        if eta >= self.eta_threshold:
            print(f"\n✗ REMOVED {gene_id}: Non-essential (η={eta:.4f} ≥ {self.eta_threshold})")
            return [gene_id], []
        else:
            print(f"\n✔ KEPT {gene_id}: Essential (η={eta:.4f} < {self.eta_threshold})")
            return [], [gene_id]

    def _test_simple_complex_removal(self, unit_info, minimal_set, current_data, initial_optimal, baseline_data):
        """Test removal of a simple enzyme complex (all or nothing)."""
        complex_id = unit_info['id']
        complex_genes = unit_info['genes']
        mod_type = unit_info['mod_type']

        print(f"  Complex members: {', '.join(complex_genes)}")

        # Check if all genes are still in minimal set
        if not all(gene in minimal_set[mod_type] for gene in complex_genes):
            print(f"  Some members already removed, skipping")
            return [], []

        # Create test set without the entire complex
        test_set = {
            'OE': minimal_set['OE'].copy(),
            'KD': minimal_set['KD'].copy()
        }
        for gene in complex_genes:
            if gene in test_set[mod_type]:
                test_set[mod_type].remove(gene)

        # Test performance
        test_data = self._test_combination_with_details(test_set)

        # 修正：使用current_data而不是initial_optimal来计算eta
        eta = self._calculate_eta(
            current_data['production'],  # 改为使用当前状态
            current_data['glucose_uptake'],  # 改为使用当前状态
            test_data['production'],
            test_data['glucose_uptake']
        )

        self.complex_eta_values[complex_id] = eta

        # Display results
        self._display_test_results(test_data, current_data, eta)

        # Decide whether to remove the entire complex
        if eta >= self.eta_threshold:
            print(f"\n✗ REMOVED COMPLEX {complex_id}: Non-essential (η={eta:.8f} ≥ {self.eta_threshold})")
            print(f"  Removed genes: {', '.join(complex_genes)}")
            return complex_genes, []
        else:
            print(f"\n✔ KEPT COMPLEX {complex_id}: Essential (η={eta:.8f} < {self.eta_threshold})")
            return [], complex_genes

    def _test_shared_complex_removal(self, unit_info, minimal_set, current_data, initial_optimal, baseline_data):
        """Test removal of complex groups with shared subunits."""
        group_id = unit_info['id']
        group_info = unit_info['group_info']
        mod_type = unit_info['mod_type']

        print(f"  Testing complex group with shared subunits")
        print(f"  Shared genes: {', '.join(group_info['shared'])}")

        # Test each complex in the group
        complex_effectiveness = {}

        for complex_id, complex_info in group_info['complexes'].items():
            # Test this complex alone
            test_set = {
                'OE': minimal_set['OE'].copy(),
                'KD': minimal_set['KD'].copy()
            }

            # Remove only this complex's genes
            for gene in complex_info['all_genes']:
                if gene in test_set[mod_type]:
                    test_set[mod_type].remove(gene)

            # Test performance
            test_data = self._test_combination_with_details(test_set)

            # 修正：使用current_data而不是initial_optimal来计算eta
            eta = self._calculate_eta(
                current_data['production'],  # 改为使用当前状态
                current_data['glucose_uptake'],  # 改为使用当前状态
                test_data['production'],
                test_data['glucose_uptake']
            )

            complex_effectiveness[complex_id] = {
                'eta': eta,
                'effective': eta < self.eta_threshold  # Effective if removing it hurts performance
            }

            print(f"    {complex_id}: η={eta:.4f} ({'effective' if eta < self.eta_threshold else 'non-effective'})")

        # Apply decision logic based on effectiveness
        genes_to_remove = []
        genes_to_keep = []

        effective_complexes = [cid for cid, info in complex_effectiveness.items() if info['effective']]

        if len(effective_complexes) == len(group_info['complexes']):
            # All complexes are effective: keep everything
            print(f"\n✔ All complexes in group are effective, keeping all genes")
            genes_to_keep = list(set(group_info['shared']) |
                                 set(sum([c['all_genes'] for c in group_info['complexes'].values()], [])))

        elif len(effective_complexes) == 0:
            # No complexes are effective: remove everything
            print(f"\n✗ No complexes in group are effective, removing all genes")
            genes_to_remove = list(set(group_info['shared']) |
                                   set(sum([c['all_genes'] for c in group_info['complexes'].values()], [])))

        else:
            # Some complexes are effective: keep only unique genes of effective complexes
            print(f"\n⚠ Mixed effectiveness in group:")
            print(f"  Keeping unique genes of effective complexes: {', '.join(effective_complexes)}")

            # Remove shared genes and unique genes of non-effective complexes
            genes_to_remove = list(group_info['shared'])

            for complex_id, complex_info in group_info['complexes'].items():
                if complex_id not in effective_complexes:
                    genes_to_remove.extend(complex_info['unique'])
                else:
                    genes_to_keep.extend(complex_info['unique'])

            print(f"  Removing shared genes: {', '.join(group_info['shared'])}")
            print(f"  Removing non-effective unique genes")

        # Filter to only remove genes that are actually in minimal_set
        genes_to_remove = [g for g in genes_to_remove if g in minimal_set[mod_type]]

        if genes_to_remove:
            print(f"  Total genes to remove: {', '.join(genes_to_remove)}")
        if genes_to_keep:
            print(f"  Total genes to keep: {', '.join(genes_to_keep)}")

        return genes_to_remove, genes_to_keep

    def _display_test_results(self, test_data, current_data, eta):
        """Display test results for a removal test."""
        if not self.verbose_flux:
            return

        prod_change = ((test_data['production'] - current_data['production']) /
                       current_data['production'] * 100 if current_data['production'] > 0 else 0)
        glc_change = ((test_data['glucose_uptake'] - current_data['glucose_uptake']) /
                      current_data['glucose_uptake'] * 100 if current_data['glucose_uptake'] > 0 else 0)
        bio_change = ((test_data['biomass_growth'] - current_data['biomass_growth']) /
                      current_data['biomass_growth'] * 100 if current_data['biomass_growth'] > 0 else 0)

        print("\nPerformance without this unit:")
        print(f"  Production:      {test_data['production']:.6f} ({prod_change:+.2f}% from current)")
        print(f"  Yield:           {test_data['yield']:.6f}")
        print(f"  Glucose uptake:  {test_data['glucose_uptake']:.6f} ({glc_change:+.2f}% from current)")
        print(f"  Biomass growth:  {test_data['biomass_growth']:.6f} ({bio_change:+.2f}% from current)")
        print(f"  Eta value:       {eta:.4f} (threshold: {self.eta_threshold})")

    def _calculate_detailed_baseline(self):
        """Calculate baseline performance with all flux details."""
        try:
            with self.original_model as model:
                model.objective = self.biomass_rxn
                sol_growth = model.optimize()
                max_growth = sol_growth.objective_value

            with self.original_model as model:
                growth_rate = 0.25 * max_growth
                model.reactions.get_by_id(self.biomass_rxn).lower_bound = growth_rate
                model.objective = self.product_rxn
                sol_baseline = pfba(model)

                production = sol_baseline.fluxes[self.product_rxn]
                biomass_growth = sol_baseline.fluxes[self.biomass_rxn]

                cs_rxn = self._get_carbon_source_reaction()
                glucose_uptake = abs(sol_baseline.fluxes.get(cs_rxn, 1.0))

                yield_value = production / glucose_uptake if glucose_uptake > 0 else 0

            return {
                'production': production,
                'yield': yield_value,
                'glucose_uptake': glucose_uptake,
                'biomass_growth': biomass_growth
            }

        except Exception as e:
            print(f"Error calculating baseline: {e}")
            return {
                'production': 0.0,
                'yield': 0.0,
                'glucose_uptake': 1.0,
                'biomass_growth': 0.0
            }

    def _test_combination_with_details(self, gene_modifications):
        """Test gene combination performance with detailed flux tracking."""
        try:
            model_copy = self.original_model.copy()

            total_mods = sum(len(genes) for genes in gene_modifications.values())
            if self.verbose_flux:
                print(f"  Testing combination with {total_mods} modifications "
                      f"(OE: {len(gene_modifications.get('OE', []))}, "
                      f"KD: {len(gene_modifications.get('KD', []))})")

            modified_model, solution, production = self._apply_modifications_and_reload(
                model_copy, gene_modifications, self.original_kcat_values
            )

            if solution is None or solution.status != 'optimal':
                print(f"  Warning: Solution not optimal (status: {solution.status if solution else 'None'})")
                return {
                    'production': 0.0,
                    'yield': 0.0,
                    'glucose_uptake': 0.0,
                    'biomass_growth': 0.0
                }

            # Extract flux values
            cs_rxn = self._get_carbon_source_reaction()
            glucose_uptake = abs(solution.fluxes.get(cs_rxn, 0.0))
            biomass_growth = solution.fluxes.get(self.biomass_rxn, 0.0)

            if glucose_uptake < 1e-9:
                glucose_uptake = 1e-9

            yield_value = production / glucose_uptake if glucose_uptake > 0 else 0

            return {
                'production': max(0, production),
                'yield': max(0, yield_value),
                'glucose_uptake': max(0, glucose_uptake),
                'biomass_growth': max(0, biomass_growth)
            }

        except Exception as e:
            print(f"Error testing combination: {e}")
            return {
                'production': 0.0,
                'yield': 0.0,
                'glucose_uptake': 0.0,
                'biomass_growth': 0.0
            }

    def _apply_modifications_and_reload(self, model, modifications, original_values=None):
        """Apply modifications for combination optimization."""
        OEF = 5.0
        KDF = 0.2

        modified_kcat_mw_data = {}
        temp_filename = None

        try:
            orig_values = original_values or self._get_original_kcat_values()

            # Calculate modified kcat values
            for mod_type, genes in modifications.items():
                for gene in genes:
                    if mod_type in ['OE', 'KD']:
                        try:
                            gene_reactions = model.genes.get_by_id(gene).reactions
                            for reaction in gene_reactions:
                                if hasattr(reaction, 'kcat_MW') and reaction.id in orig_values:
                                    original_kcat = orig_values[reaction.id]
                                    new_kcat = original_kcat * (OEF if mod_type == 'OE' else KDF)
                                    modified_kcat_mw_data[reaction.id] = new_kcat
                        except KeyError:
                            print(f"Warning: Gene {gene} not found in model")
                            continue

            # Save model to temporary file
            with tempfile.NamedTemporaryFile(suffix='.json', mode='w', delete=False) as temp_file:
                temp_filename = temp_file.name
                cobra.io.save_json_model(model, temp_filename)

            # Modify JSON
            with open(temp_filename, 'r') as file:
                model_dict = json.load(file)

            if 'enzyme_constraint' not in model_dict:
                model_dict['enzyme_constraint'] = {
                    'enzyme_mass_fraction': 0.46,
                    'total_protein_fraction': 0.56,
                    'average_saturation': 0.5,
                    'lowerbound': 0,
                    'upperbound': 0.129
                }

            # Update kcat values
            for reaction in model_dict['reactions']:
                reaction_id = reaction['id']

                if reaction_id in modified_kcat_mw_data:
                    reaction['kcat_MW'] = modified_kcat_mw_data[reaction_id]
                else:
                    if reaction_id in orig_values:
                        reaction['kcat_MW'] = orig_values[reaction_id]
                    else:
                        reaction['kcat_MW'] = reaction.get('kcat_MW', '')

                reaction['kcat'] = ''

            with open(temp_filename, 'w') as file:
                json.dump(model_dict, file)

            # Reload model
            try:
                from ECMpy_function import get_enzyme_constraint_model
                updated_model = get_enzyme_constraint_model(temp_filename)
            except:
                updated_model = cobra.io.load_json_model(temp_filename)

            # Set growth constraint
            try:
                updated_model.objective = self.biomass_rxn
                biomass_rxn_obj = updated_model.reactions.get_by_id(self.biomass_rxn)

                biomass_rxn_obj.lower_bound = 0
                biomass_rxn_obj.upper_bound = 1000

                max_growth_sol = updated_model.optimize()

                if max_growth_sol.status == 'optimal' and max_growth_sol.objective_value > 1e-6:
                    max_growth = max_growth_sol.objective_value
                    target_growth = 0.25 * max_growth

                    biomass_rxn_obj.lower_bound = target_growth
                    biomass_rxn_obj.upper_bound = target_growth

                    if self.verbose_flux:
                        print(f"  Max growth: {max_growth:.6f}, Fixed at: {target_growth:.6f}")
                else:
                    biomass_rxn_obj.lower_bound = 0
                    biomass_rxn_obj.upper_bound = 0

                    if self.verbose_flux:
                        print(f"  Warning: No growth possible with current modifications")

            except Exception as e:
                print(f"  Error setting growth constraint: {e}")
                biomass_rxn_obj.lower_bound = 0
                biomass_rxn_obj.upper_bound = 1000

            # Optimize for product
            updated_model.objective = self.product_rxn

            try:
                solution = updated_model.optimize()

                if solution.status == 'optimal':
                    production = solution.fluxes[self.product_rxn]
                    if abs(production) < 1e-10:
                        production = 0.0
                    return updated_model, solution, production
                else:
                    return updated_model, None, 0.0

            except Exception as e:
                print(f"  Error during product optimization: {e}")
                return updated_model, None, 0.0

        finally:
            if temp_filename and os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                except:
                    pass

    def _print_flux_details(self, data, label):
        """Print detailed flux information."""
        print(f"\n{label}:")
        print(f"  Production:      {data['production']:.6f} mol/gDW/h")
        print(f"  Yield:           {data['yield']:.6f} mol/mol")
        print(f"  Glucose uptake:  {data['glucose_uptake']:.6f} mol/gDW/h")
        print(f"  Biomass growth:  {data['biomass_growth']:.6f} 1/h")

    def _calculate_eta(self, optimal_prod, optimal_glc, test_prod, test_glc):
        """Calculate eta value for gene necessity evaluation."""
        if optimal_prod <= 1e-9:
            print(f"  Warning: Optimal production too low ({optimal_prod})")
            return 1.0

        if optimal_glc <= 1e-9:
            optimal_glc = 1.0

        if test_glc <= 1e-9:
            test_glc = 1.0

        prod_ratio = test_prod / optimal_prod if optimal_prod > 0 else 0

        if abs(optimal_glc - test_glc) < 1e-6:
            glc_ratio = 2.0
        else:
            glc_ratio = 1 + (optimal_glc / test_glc if test_glc > 0 else 0)

        glc_ratio = min(glc_ratio, 10.0)
        glc_ratio = max(glc_ratio, 0.1)

        eta = 0.5 * prod_ratio * glc_ratio

        if eta > 10:
            print(f"  Warning: Unusually high eta ({eta:.2f}), capping at 10")
            eta = 10.0

        return eta

    def _get_carbon_source_reaction(self):
        """Identify carbon source uptake reaction."""
        carbon_reactions = [
            self.constraint_rxn_id,
            'EX_cpd00027_e_reverse',
            'EX_glc_e_reverse',
            'EX_glucose_e'
        ]

        for rxn_id in carbon_reactions:
            if rxn_id in self.model.reactions:
                return rxn_id

        for rxn in self.model.reactions:
            if rxn.id.startswith('EX_'):
                for met in rxn.metabolites:
                    if any(keyword in met.name.lower() for keyword in
                           ['glucose', 'carbon', 'glycerol', 'lactate'] if met.name):
                        return rxn.id

        return self.constraint_rxn_id

    # ==================== Report Generation (Step 6) ====================

    def generate_final_report(self):
        """Generate final report with target files."""
        print("\n" + "=" * 60)
        print("Step 6: Generating Final Report")
        print("=" * 60)

        os.makedirs(self.output_dir, exist_ok=True)

        if self.enable_combination_optimization and self.minimal_gene_set:
            self._generate_targets_file(self.minimal_gene_set, f"minimal_targets_{self.product_id}.csv")
            # self._generate_targets_file(self.combined_targets, f"targets_{self.product_id}.csv")
        else:
            self._generate_targets_file(self.combined_targets, f"targets_{self.product_id}.csv")

    def _generate_targets_file(self, targets_to_report, filename):
        """Helper method to generate target file."""
        output_file = os.path.join(self.output_dir, filename)
        detailed_results = []

        for target_type in ['OE', 'KD']:
            for gene_id in targets_to_report.get(target_type, []):
                try:
                    gene = self.model.genes.get_by_id(gene_id)

                    sources = []
                    if gene_id in self.gene_source_mapping:
                        sources = list(self.gene_source_mapping[gene_id]['sources'])

                    metrics = self.test_results.get(gene_id, {})
                    impact = self.gene_impact_summary.get(gene_id, {})

                    result = {
                        'gene_id': gene_id,
                        'gene_name': gene.name if hasattr(gene, 'name') else '',
                        'type': target_type,
                        'sources': ';'.join(sources),
                        'is_conflict': gene_id in self.conflicted_genes,
                        'reactions': ';'.join([r.id for r in gene.reactions]),
                        'production_improvement': metrics.get('prod_improvement', ''),
                        'yield_improvement': metrics.get('yield_improvement', ''),
                        'eta_value': self.eta_values.get(gene_id, '') if self.enable_combination_optimization else '',
                        'glucose_impact_%': impact.get('glucose_impact', ''),
                        'biomass_impact_%': impact.get('biomass_impact', ''),
                        'status': 'kept' if gene_id not in self.removed_genes else 'removed'
                    }

                    detailed_results.append(result)

                except Exception as e:
                    print(f"Error processing gene {gene_id}: {e}")

        if detailed_results:
            df = pd.DataFrame(detailed_results)
            df.to_csv(output_file, index=False)
            print(f"Results saved to: {output_file}")

            print(f"  Total targets: {len(df)}")
            print(f"    OE targets: {len(df[df['type'] == 'OE'])}")
            print(f"    KD targets: {len(df[df['type'] == 'KD'])}")

    # ==================== Main Pipeline ====================

    def run_integrated_pipeline(self):
        """Run the complete integrated pipeline."""
        start_time = time.time()

        try:
            print("\n" + "=" * 70)
            print("Starting Integrated Target Identification Pipeline")
            print("with Enhanced Flux Tracking and Original ecFbridge Evaluation Logic")
            print("=" * 70)

            # Step 1: ecFactory analysis
            self.run_ecfactory_without_evaluation()

            # Step 2: iBridge analysis (optional)
            if self.enable_ibridge:
                self.run_ibridge_analysis()

            # Step 3: Combine targets
            self.combine_all_targets()

            # Step 4: Unified evaluation (using original ecFbridge logic)
            self.unified_target_evaluation()

            # Step 5: Combination optimization (optional)
            if self.enable_combination_optimization:
                final_targets = self.find_minimal_gene_combination()
            else:
                final_targets = self.combined_targets

            # Step 6: Generate report
            self.generate_final_report()

            # Print summary
            elapsed_time = time.time() - start_time
            print("\n" + "=" * 70)
            print("Pipeline Complete")
            print("=" * 70)
            print(f"Total time: {elapsed_time:.2f} seconds")
            print(f"\nFinal targets:")
            print(f"  OE genes: {len(final_targets['OE'])}")
            print(f"  KD genes: {len(final_targets['KD'])}")

            if self.enable_combination_optimization:
                print(f"\nOptimization statistics:")
                print(f"  Genes removed: {len(self.removed_genes)}")
                if self.combination_results:
                    initial_total = (len(self.combination_results['initial_genes']['OE']) +
                                     len(self.combination_results['initial_genes']['KD']))
                    if initial_total > 0:
                        reduction = len(self.removed_genes) / initial_total * 100
                        print(f"  Reduction: {reduction:.1f}%")
                        print(f"  Performance retention: {self.combination_results['performance_retention']:.1f}%")

            return final_targets

        except Exception as e:
            print(f"Pipeline error: {e}")
            import traceback
            traceback.print_exc()
            return {'OE': [], 'KD': []}


def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Integrated Target Identification with Enhanced Flux Tracking'
    )
    parser.add_argument('--model', required=True, help='Model file path')
    parser.add_argument('--product', required=True, help='Target product ID')
    parser.add_argument('--biomass', help='Biomass reaction ID')
    parser.add_argument('--essential-genes', help='Essential genes file')
    parser.add_argument('--enable-ibridge', action='store_true', default=True,
                        help='Enable iBridge analysis (default: enabled)')
    parser.add_argument('--disable-ibridge', action='store_true',
                        help='Disable iBridge analysis')
    parser.add_argument('--carbon-source', default='cpd00027', help='Carbon source ID')
    parser.add_argument('--exclude-metabolites', default='seed_compounds.txt',
                        help='Excluded metabolites file')
    parser.add_argument('--improvement-threshold', type=float, default=0.1,
                        help='Improvement threshold (%%, default: 0.1)')
    parser.add_argument('--exclusion-threshold', type=float, default=-5.0,
                        help='Exclusion threshold (%%, default: -5.0)')
    parser.add_argument('--eta-threshold', type=float, default=0.95,
                        help='Eta threshold for combination optimization (default: 0.95)')
    parser.add_argument('--disable-combination', action='store_true',
                        help='Disable combination optimization')
    parser.add_argument('--disable-distance-filter', action='store_true',
                        help='Disable metabolic distance filter')
    parser.add_argument('--disable-verbose-flux', action='store_true',
                        help='Disable verbose flux output')
    parser.add_argument('--output-dir', default='./prediction_target',
                        help='Output directory')

    args = parser.parse_args()

    # Process arguments
    enable_ibridge = not args.disable_ibridge
    enable_combination = not args.disable_combination
    enable_distance = not args.disable_distance_filter
    verbose_flux = not args.disable_verbose_flux

    # Create identifier instance
    identifier = IntegratedTargetIdentifierWithOptimization(
        model_file=args.model,
        product_id=args.product,
        biomass_rxn=args.biomass,
        essential_genes_file=args.essential_genes,
        enable_ibridge=enable_ibridge,
        exclude_met_dir=args.exclude_metabolites,
        c_source=args.carbon_source,
        enable_distance_filter=enable_distance,
        improvement_threshold=args.improvement_threshold,
        exclusion_threshold=args.exclusion_threshold,
        eta_threshold=args.eta_threshold,
        enable_combination_optimization=enable_combination,
        output_dir=args.output_dir,
        verbose_flux=verbose_flux
    )

    # Run pipeline
    final_targets = identifier.run_integrated_pipeline()

    print("\nAnalysis complete!")
    return final_targets


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        print("ecFbridge with Enhanced Flux Tracking (Original Evaluation Logic)")
        print("=" * 70)
        print("\nUsage examples:")
        print("  python ecFbridge_combination.py --model model.json --product cpd00103")
        print("  python ecFbridge_combination.py --model model.json --product cpd00103 --disable-combination")
        print("  python ecFbridge_combination.py --model model.json --product cpd00103 --eta-threshold 0.9")
        print("\nKey parameters:")
        print("  --improvement-threshold: Individual gene improvement threshold (default: 0.1%)")
        print("  --exclusion-threshold: Individual gene exclusion threshold (default: -5.0%)")
        print("  --eta-threshold: Combination optimization threshold (default: 0.95)")
        print("  --disable-combination: Run without combination optimization")
        print("  --disable-verbose-flux: Disable detailed flux output during optimization")
        print("\nThe complete pipeline will:")
        print("  1. Run ecFactory analysis")
        print("  2. Run iBridge analysis (optional)")
        print("  3. Combine targets and identify conflicts")
        print("  4. Evaluate individual targets (using original ecFbridge logic)")
        print("  5. Find minimal gene combination with detailed flux tracking")
        print("  6. Generate comprehensive reports")