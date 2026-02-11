#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ecFbridge: Integrated Enzyme-Constrained Target Identification System

Combines ecFactory and iBridge algorithms for comprehensive metabolic engineering
target identification with intelligent conflict resolution.

Main Features:
- ecFactory analysis with skipped individual evaluation
- Optional iBridge analysis
- Intelligent conflict gene resolution
- Threshold-based filtering (improvement and exclusion)
- Comprehensive reporting with source tracking
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

warnings.filterwarnings('ignore')


class IntegratedTargetIdentifier:
    """
    Integrated system for metabolic engineering target identification.

    Combines ecFactory and iBridge algorithms with intelligent conflict resolution
    and threshold-based filtering for robust target identification.
    """

    def __init__(self,
                 model_file,
                 product_id,
                 biomass_rxn='EX_biomass_c',
                 essential_genes_file=None,
                 enable_ibridge=True,
                 exclude_met_dir="seed_compounds.txt",
                 c_source="cpd00027",
                 air="aerobic",
                 o2='EX_cpd00007_e_reverse',
                 blocked_reactions=None,
                 improvement_threshold=0.1,
                 exclusion_threshold=-3,
                 enable_distance_filter=True,
                 output_dir='./prediction_target'):
        """
        Initialize the integrated target identifier.

        Parameters
        ----------
        model_file : str
            Path to the metabolic model file
        product_id : str
            Target product metabolite ID
        biomass_rxn : str, optional
            Biomass reaction ID (auto-detected if None)
        essential_genes_file : str, optional
            Path to essential genes file
        enable_ibridge : bool, default=True
            Whether to enable iBridge analysis
        exclude_met_dir : str
            Path to excluded metabolites file
        c_source : str
            Carbon source ID
        air : str
            Oxygen condition ('aerobic' or 'anaerobic')
        o2 : str
            Oxygen exchange reaction ID
        blocked_reactions : list, optional
            List of reactions to block
        improvement_threshold : float, default=0.1
            Minimum improvement threshold (%)
        exclusion_threshold : float, default=-5.0
            Maximum negative change threshold (%)
        enable_distance_filter : bool, default=True
            Whether to apply metabolic distance filtering
        output_dir : str
            Directory for output files
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

        # Threshold settings
        self.improvement_threshold = improvement_threshold
        self.exclusion_threshold = exclusion_threshold

        # Initialize data structures
        self._initialize_data_structures()

        # Load and configure model
        self._load_and_configure_model(biomass_rxn)

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

    def _initialize_data_structures(self):
        """Initialize internal data storage structures."""
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

    def _load_and_configure_model(self, biomass_rxn):
        """Load metabolic model and apply configuration."""
        print("Loading metabolic model...")

        # Try ECMpy loading first, fallback to standard cobra
        try:
            from ECMpy_function import get_enzyme_constraint_model_with_reaction_set_exclusion
            self.model = get_enzyme_constraint_model_with_reaction_set_exclusion(
                self.model_file
            )
        except:
            self.model = load_json_model(self.model_file)

        # Block specified reactions
        self._block_reactions()

        # Determine biomass reaction
        self.biomass_rxn = self._determine_biomass_reaction(biomass_rxn)
        print(f"Using biomass reaction: {self.biomass_rxn}")

        # Set quiet mode
        self._set_quiet_mode(True)

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

        # Try to get from objective
        try:
            if hasattr(self.model.objective, 'keys'):
                return list(self.model.objective.keys())[0].id
        except:
            pass

        # Search for common biomass reaction patterns
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
        print(f"  Improvement threshold: {self.improvement_threshold}%")
        print(f"  Exclusion threshold: {self.exclusion_threshold}%")
        print(f"  Distance filter: {'Enabled' if self.enable_distance_filter else 'Disabled'}")

    # ==================== Main Pipeline Methods ====================

    def run_integrated_pipeline(self):
        """
        Run the complete integrated pipeline.

        Workflow:
        1. Run ecFactory (without evaluation)
        2. Optionally run iBridge
        3. Combine targets and identify conflicts
        4. Intelligently resolve conflicts with thresholds
        5. Generate final report

        Returns
        -------
        dict
            Final identified targets {'OE': [...], 'KD': [...]}
        """
        start_time = time.time()

        try:
            print("\n" + "=" * 70)
            print("Starting Integrated Target Identification Pipeline")
            print("=" * 70)

            # Step 1: ecFactory analysis
            self.run_ecfactory_without_evaluation()
            if self.ecfactory_instance and hasattr(self.ecfactory_instance, 'print_enzyme_usage_details'):
                  self.ecfactory_instance.print_enzyme_usage_details('CEY17_RS08445')
                  self.ecfactory_instance.print_enzyme_usage_details('CEY17_RS08450')
            #     self.ecfactory_instance.print_enzyme_usage_details('CEY17_RS08670')
            # Step 2: iBridge analysis (optional)
            if self.enable_ibridge:
                self.run_ibridge_analysis()

            # Step 3: Combine targets
            self.combine_all_targets()

            # Step 4: Unified evaluation with conflict resolution
            self.unified_target_evaluation()

            # Step 5: Generate report
            self.generate_final_report()

            # Print summary
            self._print_pipeline_summary(time.time() - start_time)

            return self.combined_targets

        except Exception as e:
            print(f"Pipeline error: {e}")
            import traceback
            traceback.print_exc()
            return {'OE': [], 'KD': []}

    def run_ecfactory_without_evaluation(self):
        """Run ecFactory analysis without individual evaluation."""
        print("\n" + "=" * 60)
        print("Step 1: Running ecFactory Analysis")
        print("=" * 60)

        from ecFactory_light import ecFactory_light

        # Initialize ecFactory
        self.ecfactory_instance = ecFactory_light(
            model=self.model,
            product_rxn=self.product_rxn,
            biomass_rxn=self.biomass_rxn,
            essential_genes_file=self.essential_genes_file,
            pfba_factor=1,
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

        # Run pipeline steps (except evaluation)
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

    def run_ibridge_analysis(self):
        """Run iBridge analysis if enabled."""
        if not self.enable_ibridge:
            return pd.DataFrame()

        print("\n" + "=" * 60)
        print("Step 2: Running iBridge Analysis")
        print("=" * 60)

        try:
            from baebra.ibridge import predict_ibridge_targets_optimized

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
                # Record source
                if gene_id not in self.gene_source_mapping:
                    self.gene_source_mapping[gene_id] = {'sources': set(), 'predictions': set()}
                self.gene_source_mapping[gene_id]['sources'].add('iBridge')

                # Handle conflicts
                if reg_type == 'conflict':
                    # iBridge internal conflict
                    if gene_id not in ecfactory_genes:
                        combined['Test'].append(gene_id)
                        new_test += 1
                        self.conflicted_genes.append(gene_id)
                else:
                    # Check for conflicts with ecFactory
                    in_oe = gene_id in ecfactory_oe
                    in_kd = gene_id in ecfactory_kd

                    if (in_oe and reg_type == 'KD') or (in_kd and reg_type == 'OE'):
                        # Conflict between sources
                        self.conflicted_genes.append(gene_id)
                        combined['Test'].append(gene_id)
                        print(f"  Conflict detected: {gene_id}")
                    elif not in_oe and not in_kd:
                        # New gene from iBridge
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

    def unified_target_evaluation(self):
        """Evaluate all targets with conflict resolution and filtering."""
        print("\n" + "=" * 60)
        print("Step 4: Unified Target Evaluation")
        print("=" * 60)

        if not self.ecfactory_instance:
            print("Error: ecFactory instance not available")
            return {}

        # Calculate baseline
        print("\nCalculating baseline performance...")
        baseline_production, baseline_yield = self._calculate_baseline_performance()
        print(f"Baseline: production={baseline_production:.6f}, yield={baseline_yield:.6f}")

        # Get original kcat values
        original_kcat_values = self._get_original_kcat_values()

        # Detect additional conflicts
        self._detect_all_conflicts()

        # Resolve conflicts in Test category
        if self.combined_targets.get('Test'):
            print(f"\nResolving {len(self.combined_targets['Test'])} conflicted genes...")
            self._resolve_conflicts(baseline_production, baseline_yield, original_kcat_values)

        # Evaluate OE and KD targets
        final_targets = {'OE': [], 'KD': []}

        for target_type in ['OE', 'KD']:
            print(f"\nEvaluating {target_type} targets ({len(self.combined_targets[target_type])})...")

            for gene_id in self.combined_targets[target_type]:
                # Test modification
                production, yield_value = self._test_gene_modification(
                    gene_id, target_type, original_kcat_values
                )

                # Calculate improvements
                prod_imp = ((production - baseline_production) / baseline_production * 100
                            if baseline_production > 0 else 0)
                yield_imp = ((yield_value - baseline_yield) / baseline_yield * 100
                             if baseline_yield > 0 else 0)

                # Store results
                self.test_results[gene_id] = {
                    'prod_improvement': prod_imp,
                    'yield_improvement': yield_imp,
                    'type': target_type,
                    'source': 'evaluation'
                }

                # Apply filtering
                # 确定基因来源
                ecfactory_genes = set()
                for t_type in ['OE', 'KD']:
                    ecfactory_genes.update(self.ecfactory_targets.get(t_type, []))
                gene_source = 'ecFactory' if gene_id in ecfactory_genes else 'iBridge'

                if self._passes_thresholds(gene_id, target_type, prod_imp, yield_imp):
                    final_targets[target_type].append(gene_id)
                    print(
                        f"  ✓ 保留 {gene_source} 基因 {gene_id} ({target_type}): 产量改进 {prod_imp:.3f}%, 产率改进 {yield_imp:.3f}%")
                else:
                    # 判断是被排除还是被过滤
                    if target_type == 'KD' and (
                            prod_imp <= self.exclusion_threshold or yield_imp <= self.exclusion_threshold):
                        print(
                            f"  ✗ 排除基因 {gene_id} ({target_type}): 产量改进 {prod_imp:.3f}%, 产率改进 {yield_imp:.3f}% (低于排除阈值 {self.exclusion_threshold}%)")
                    else:
                        print(
                            f"  ✗ 过滤 {gene_source} 基因 {gene_id} ({target_type}): 产量改进 {prod_imp:.3f}%, 产率改进 {yield_imp:.3f}% (低于改进阈值 {self.improvement_threshold}%)")

        self.combined_targets = final_targets

        print(f"\nEvaluation complete:")
        print(f"  Final OE: {len(final_targets['OE'])}")
        print(f"  Final KD: {len(final_targets['KD'])}")
        print(f"  Excluded: {len(self.excluded_genes)}")
        print(f"  Filtered: {len(self.filtered_genes)}")

        return final_targets

    def generate_final_report(self):
        """Generate comprehensive final report."""
        print("\n" + "=" * 60)
        print("Step 5: Generating Final Report")
        print("=" * 60)

        os.makedirs(self.output_dir, exist_ok=True)
        output_file = os.path.join(self.output_dir, f"targets_{self.product_id}.csv")

        # Collect detailed results
        detailed_results = []

        for target_type in ['OE', 'KD']:
            for gene_id in self.combined_targets.get(target_type, []):
                try:
                    gene = self.model.genes.get_by_id(gene_id)

                    # Get source information
                    sources = []
                    if gene_id in self.gene_source_mapping:
                        sources = list(self.gene_source_mapping[gene_id]['sources'])

                    # Get metrics
                    metrics = self.test_results.get(gene_id, {})

                    result = {
                        'gene_id': gene_id,
                        'gene_name': gene.name if hasattr(gene, 'name') else '',
                        'type': target_type,
                        'sources': ';'.join(sources),
                        'is_conflict': gene_id in self.conflicted_genes,
                        'reactions': ';'.join([r.id for r in gene.reactions]),
                        'production_improvement': metrics.get('prod_improvement', ''),
                        'yield_improvement': metrics.get('yield_improvement', '')
                    }

                    detailed_results.append(result)

                except Exception as e:
                    print(f"Error processing gene {gene_id}: {e}")

        # Save results
        if detailed_results:
            df = pd.DataFrame(detailed_results)
            df.to_csv(output_file, index=False)
            print(f"Results saved to: {output_file}")

            # Print statistics
            self._print_report_statistics(df)

        return detailed_results

    # ==================== Helper Methods ====================

    def _filter_biomass_reactions(self, targets):
        """Filter out biomass-related reactions."""
        if targets.empty:
            return targets

        initial_count = len(targets)

        # Define biomass patterns
        biomass_patterns = ['Bio_', 'BIOMASS', 'biomass', 'ATPM', 'maintenance']

        # Filter
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

        # Process each target reaction
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
                # Conflict detected
                self.ibridge_gene_regulations[gene_id] = 'conflict'
                print(f"  Conflict in iBridge: {gene_id} has {unique_types}")
            else:
                self.ibridge_gene_regulations[gene_id] = unique_types[0]

    def _normalize_regulation_type(self, reg_type):
        """Normalize regulation type to OE or KD."""
        if pd.isna(reg_type):
            return 'KD'

        type_str = str(reg_type).strip().lower()

        # Map to standard types
        oe_keywords = ['up', 'upregulation', 'overexpression', 'oe', 'increase', 'enhance']
        kd_keywords = ['down', 'downregulation', 'knockdown', 'kd', 'decrease', 'reduce']

        if any(keyword in type_str for keyword in oe_keywords):
            return 'OE'
        elif any(keyword in type_str for keyword in kd_keywords):
            return 'KD'
        else:
            return 'KD'  # Default

    def _detect_all_conflicts(self):
        """Detect all conflicting genes across categories."""
        print("\nDetecting conflicts across all categories...")

        oe_genes = set(self.combined_targets.get('OE', []))
        kd_genes = set(self.combined_targets.get('KD', []))
        test_genes = set(self.combined_targets.get('Test', []))

        # Find OE-KD conflicts
        new_conflicts = oe_genes & kd_genes

        if new_conflicts:
            print(f"Found {len(new_conflicts)} OE-KD conflicts")
            for gene_id in new_conflicts:
                # Move to Test category
                if gene_id in self.combined_targets['OE']:
                    self.combined_targets['OE'].remove(gene_id)
                if gene_id in self.combined_targets['KD']:
                    self.combined_targets['KD'].remove(gene_id)
                if gene_id not in test_genes:
                    self.combined_targets['Test'].append(gene_id)
                if gene_id not in self.conflicted_genes:
                    self.conflicted_genes.append(gene_id)

        # Update Test category
        all_conflicts = test_genes | new_conflicts
        self.combined_targets['Test'] = list(all_conflicts)

        print(f"Total conflicts to resolve: {len(self.combined_targets['Test'])}")

    def _resolve_conflicts(self, baseline_production, baseline_yield, original_kcat_values):
        """Resolve conflicted genes by testing both directions."""
        ecfactory_genes = set()
        for target_type in ['OE', 'KD']:
            ecfactory_genes.update(self.ecfactory_targets.get(target_type, []))

        for gene_id in self.combined_targets.get('Test', []):
            print(f"\n处理冲突基因: {gene_id}")

            # 预先初始化所有变量，避免未定义错误
            optimal_type = None
            optimal_prod_imp = None
            optimal_yield_imp = None

            try:
                # Test both modifications
                oe_prod, oe_yield = self._test_gene_modification(
                    gene_id, 'OE', original_kcat_values
                )
                kd_prod, kd_yield = self._test_gene_modification(
                    gene_id, 'KD', original_kcat_values
                )

                # Calculate improvements
                oe_prod_imp = ((oe_prod - baseline_production) / baseline_production * 100
                               if baseline_production > 0 else 0)
                oe_yield_imp = ((oe_yield - baseline_yield) / baseline_yield * 100
                                if baseline_yield > 0 else 0)

                kd_prod_imp = ((kd_prod - baseline_production) / baseline_production * 100
                               if baseline_production > 0 else 0)
                kd_yield_imp = ((kd_yield - baseline_yield) / baseline_yield * 100
                                if baseline_yield > 0 else 0)

                print(f"  OE测试: 产量改进 {oe_prod_imp:.3f}%, 产率改进 {oe_yield_imp:.3f}%")
                print(f"  KD测试: 产量改进 {kd_prod_imp:.3f}%, 产率改进 {kd_yield_imp:.3f}%")

                # 确定基因来源
                gene_source = 'ecFactory' if gene_id in ecfactory_genes else 'iBridge'

                # 检查排除阈值
                oe_excluded = (oe_prod_imp <= self.exclusion_threshold or
                               oe_yield_imp <= self.exclusion_threshold)
                kd_excluded = (kd_prod_imp <= self.exclusion_threshold or
                               kd_yield_imp <= self.exclusion_threshold)

                # 处理两个方向都被排除的情况
                if oe_excluded and kd_excluded:
                    self.excluded_genes.append(gene_id)
                    print(f"  ✗ 排除冲突基因 {gene_id}: 两个方向都低于排除阈值")
                    self.conflict_resolution_results[gene_id] = {
                        'status': 'excluded',
                        'reason': '两个方向都低于排除阈值',
                        'oe_prod_improvement': oe_prod_imp,
                        'oe_yield_improvement': oe_yield_imp,
                        'kd_prod_improvement': kd_prod_imp,
                        'kd_yield_improvement': kd_yield_imp,
                        'source': gene_source
                    }
                    continue

                # 选择更好的修改类型
                if oe_excluded and not kd_excluded:
                    # OE被排除，选择KD
                    optimal_type = 'KD'
                    optimal_prod_imp = kd_prod_imp
                    optimal_yield_imp = kd_yield_imp
                    print(f"  选择 KD: OE方向被排除")

                elif kd_excluded and not oe_excluded:
                    # KD被排除，选择OE
                    optimal_type = 'OE'
                    optimal_prod_imp = oe_prod_imp
                    optimal_yield_imp = oe_yield_imp
                    print(f"  选择 OE: KD方向被排除")

                else:
                    # 两个方向都未被排除，选择更好的
                    oe_score = oe_prod_imp + oe_yield_imp
                    kd_score = kd_prod_imp + kd_yield_imp

                    if oe_score > kd_score:
                        optimal_type = 'OE'
                        optimal_prod_imp = oe_prod_imp
                        optimal_yield_imp = oe_yield_imp
                    else:
                        optimal_type = 'KD'
                        optimal_prod_imp = kd_prod_imp
                        optimal_yield_imp = kd_yield_imp

                    print(f"  选择 {optimal_type}: 得分更高 (OE:{oe_score:.3f} vs KD:{kd_score:.3f})")

                # 确保变量已正确赋值
                if optimal_type is None:
                    print(f"  错误: 无法确定最优类型，跳过基因 {gene_id}")
                    self.conflict_resolution_results[gene_id] = {
                        'status': 'error',
                        'error': 'Could not determine optimal type'
                    }
                    continue

                # 应用最终的过滤规则
                pass_threshold = False

                if gene_source == 'iBridge':
                    # iBridge基因需要通过改进阈值
                    if optimal_prod_imp >= self.improvement_threshold or optimal_yield_imp >= self.improvement_threshold:
                        pass_threshold = True
                        self.combined_targets[optimal_type].append(gene_id)
                        self.test_results[gene_id] = {
                            'prod_improvement': optimal_prod_imp,
                            'yield_improvement': optimal_yield_imp,
                            'type': optimal_type,
                            'source': 'conflict_resolution'
                        }
                        print(f"  ✓ 冲突解决: {gene_id} -> {optimal_type} (iBridge基因通过改进阈值)")
                    else:
                        self.filtered_genes.append(gene_id)
                        self.test_results[gene_id] = {
                            'prod_improvement': optimal_prod_imp,
                            'yield_improvement': optimal_yield_imp,
                            'type': optimal_type,
                            'source': 'conflict_resolution_filtered'
                        }
                        print(
                            f"  ✗ 过滤冲突基因 {gene_id}: 改进不足 (产量:{optimal_prod_imp:.3f}%, 产率:{optimal_yield_imp:.3f}%)")
                else:
                    # ecFactory基因，检查是否有正向改进
                    if optimal_prod_imp >= self.improvement_threshold or optimal_yield_imp >= self.improvement_threshold:
                        pass_threshold = True
                        self.combined_targets[optimal_type].append(gene_id)
                        self.test_results[gene_id] = {
                            'prod_improvement': optimal_prod_imp,
                            'yield_improvement': optimal_yield_imp,
                            'type': optimal_type,
                            'source': 'conflict_resolution'
                        }
                        print(f"  ✓ 冲突解决: {gene_id} -> {optimal_type} (ecFactory基因)")
                    else:
                        self.filtered_genes.append(gene_id)
                        self.test_results[gene_id] = {
                            'prod_improvement': optimal_prod_imp,
                            'yield_improvement': optimal_yield_imp,
                            'type': optimal_type,
                            'source': 'conflict_resolution_filtered'
                        }
                        print(f"  ✗ 过滤冲突基因 {gene_id}: 改进值不足")

                # 记录解决结果
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

            except Exception as e:
                print(f"  ❌ 处理基因 {gene_id} 时出错: {e}")
                # 记录错误但继续处理其他基因
                self.conflict_resolution_results[gene_id] = {
                    'status': 'error',
                    'error': str(e),
                    'source': 'ecFactory' if gene_id in ecfactory_genes else 'iBridge'
                }
                continue

        # Clear Test category
        self.combined_targets['Test'] = []
        print(f"\n冲突解决完成，共处理 {len(self.conflict_resolution_results)} 个冲突基因")

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
        # Check exclusion threshold (mainly for KD)
        if target_type == 'KD' and (prod_imp <= self.exclusion_threshold or
                                    yield_imp <= self.exclusion_threshold):
            self.excluded_genes.append(gene_id)
            return False

        # Check improvement threshold
        if prod_imp >= self.improvement_threshold or yield_imp >= self.improvement_threshold:
            return True
        else:
            self.filtered_genes.append(gene_id)
            return False

    def _calculate_baseline_performance(self):
        """Calculate baseline production and yield at 25% growth."""
        try:
            # Get maximum growth
            with self.model as model:
                model.objective = self.biomass_rxn
                sol_growth = model.optimize()
                max_growth = sol_growth.objective_value

            # Set 25% growth and optimize for product
            with self.model as model:
                growth_rate = 0.25 * max_growth
                model.reactions.get_by_id(self.biomass_rxn).lower_bound = growth_rate
                model.objective = self.product_rxn
                sol_baseline = pfba(model)

                baseline_production = sol_baseline.fluxes[self.product_rxn]

                # Calculate carbon uptake
                cs_rxn = self._get_carbon_source_reaction()
                baseline_cs_uptake = abs(sol_baseline.fluxes.get(cs_rxn, 1.0))

                baseline_yield = baseline_production / baseline_cs_uptake if baseline_cs_uptake > 0 else 0

            self.baseline_calculated = True
            return baseline_production, baseline_yield

        except Exception as e:
            print(f"Error calculating baseline: {e}")
            return 0.0, 0.0

    def _test_gene_modification(self, gene_id, mod_type, original_kcat_values):
        """Test the effect of modifying a single gene."""
        try:
            modifications = {'OE': [], 'KD': []}
            modifications[mod_type] = [gene_id]

            # Apply modifications
            with self.model as model:
                modified_model, solution, production = self._apply_modifications_and_reload(
                    model, modifications, original_kcat_values
                )

                if solution is None:
                    return 0.0, 0.0

                # Calculate yield
                cs_rxn = self._get_carbon_source_reaction()
                cs_uptake = abs(solution.fluxes.get(cs_rxn, 1.0))
                yield_value = production / cs_uptake if cs_uptake > 0 else 0

                return production, yield_value

        except Exception as e:
            print(f"    Error testing {gene_id} {mod_type}: {e}")
            return 0.0, 0.0

    def _get_original_kcat_values(self):
        """Get original kcat_MW values from model."""
        original_kcat_values = {}
        for reaction in self.model.reactions:
            if hasattr(reaction, 'kcat_MW') and reaction.kcat_MW:
                try:
                    original_kcat_values[reaction.id] = float(reaction.kcat_MW)
                except:
                    pass
        return original_kcat_values

    def _get_carbon_source_reaction(self):
        """Identify carbon source uptake reaction."""
        # Common carbon source reactions
        carbon_reactions = [
            self.constraint_rxn_id,
            'EX_cpd00027_e_reverse',
            'EX_glc_e_reverse',
            'EX_glucose_e'
        ]

        for rxn_id in carbon_reactions:
            if rxn_id in self.model.reactions:
                return rxn_id

        # Search for exchange reactions with carbon sources
        for rxn in self.model.reactions:
            if rxn.id.startswith('EX_'):
                for met in rxn.metabolites:
                    if any(keyword in met.name.lower() for keyword in
                           ['glucose', 'carbon', 'glycerol', 'lactate'] if met.name):
                        return rxn.id

        return self.constraint_rxn_id

    def _apply_modifications_and_reload(self, model, modifications, original_values=None):
        """Apply gene modifications and reload model with enzyme constraints."""
        OEF = 5.0  # Overexpression factor
        KDF = 0.2  # Knockdown factor

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

            # Modify JSON to ensure enzyme constraints
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

    def _print_pipeline_summary(self, elapsed_time):
        """Print pipeline execution summary."""
        print("\n" + "=" * 70)
        print("Pipeline Complete")
        print("=" * 70)
        print(f"Total time: {elapsed_time:.2f} seconds")

        final_targets = self.combined_targets
        print(f"Final targets: OE={len(final_targets['OE'])}, KD={len(final_targets['KD'])}")

        if self.conflicted_genes:
            print(f"Conflicts resolved: {len(self.conflicted_genes)}")

        if self.excluded_genes:
            print(f"Genes excluded (below {self.exclusion_threshold}%): {len(self.excluded_genes)}")

        if self.filtered_genes:
            print(f"Genes filtered (below {self.improvement_threshold}%): {len(self.filtered_genes)}")

    def _print_report_statistics(self, df):
        """Print report statistics."""
        print("\nFinal Statistics:")
        print(f"  Total targets: {len(df)}")
        print(f"    OE targets: {len(df[df['type'] == 'OE'])}")
        print(f"    KD targets: {len(df[df['type'] == 'KD'])}")

        if 'sources' in df.columns:
            print("\nSource distribution:")
            ecfactory_count = len(df[df['sources'].str.contains('ecFactory', na=False)])
            ibridge_count = len(df[df['sources'].str.contains('iBridge', na=False)])
            print(f"  ecFactory: {ecfactory_count}")
            print(f"  iBridge: {ibridge_count}")


def main():
    """Command-line interface for the integrated target identifier."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Integrated Target Identification System (ecFbridge)'
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
    parser.add_argument('--output-dir', default='./prediction_target',
                        help='Output directory')
    parser.add_argument('--disable-distance-filter', action='store_true',
                        help='Disable metabolic distance filter')

    args = parser.parse_args()

    # Process arguments
    enable_ibridge = not args.disable_ibridge
    enable_distance_filter = not args.disable_distance_filter

    # Create identifier instance
    identifier = IntegratedTargetIdentifier(
        model_file=args.model,
        product_id=args.product,
        biomass_rxn=args.biomass,
        essential_genes_file=args.essential_genes,
        enable_ibridge=enable_ibridge,
        exclude_met_dir=args.exclude_metabolites,
        c_source=args.carbon_source,
        enable_distance_filter=enable_distance_filter,
        improvement_threshold=args.improvement_threshold,
        exclusion_threshold=args.exclusion_threshold,
        output_dir=args.output_dir
    )

    # Run pipeline
    final_targets = identifier.run_integrated_pipeline()
    print("\nAnalysis complete!")
    return final_targets


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        print("Usage examples:")
        print("  python ecFbridge.py --model model.json --product cpd00103 --enable-ibridge")
        print("  python ecFbridge.py --model model.json --product cpd00103 --disable-ibridge")
        print("\nOptional parameters:")
        print("  --improvement-threshold 0.1  # Set improvement threshold (default 0.1%)")
        print("  --exclusion-threshold -5.0   # Set exclusion threshold (default -5.0%)")
        print("  --disable-distance-filter     # Disable metabolic distance filtering")