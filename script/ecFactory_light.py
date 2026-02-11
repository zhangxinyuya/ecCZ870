"""
ecFactory: Enzyme-Constrained Flux Analysis for Metabolic Engineering Target Identification

A comprehensive framework for identifying metabolic engineering targets using enzyme-constrained
flux analysis, including EC-FSEOF analysis, leak reaction identification, and enzyme usage
variability analysis.
"""

import cobra
import pandas as pd
import numpy as np
from cobra.flux_analysis import flux_variability_analysis, pfba
import os
import warnings
import hashlib
import json
from datetime import datetime
from contextlib import contextmanager
import tempfile

warnings.filterwarnings('ignore')


class ecFactory_light:
    """
    Enzyme-Constrained Factory for metabolic engineering target identification.

    Main features:
    - EC-FSEOF (Enzyme-Constrained FSEOF) analysis with optimized three-stage method
    - Target identification: Overexpression (OE) and Knockdown (KD) targets
    - Leak reaction analysis
    - Enzyme Usage Variability Analysis (EUVA)
    - Metabolic network topology distance filtering
    """

    def __init__(self, model, product_rxn, biomass_rxn=None,
                 essential_genes_file=None, enzyme_data_file=None,
                 pfba_factor=1, fseof_steps=10,
                 auto_identify_essential=True, additional_essential_genes=[],
                 essential_genes_cache_dir="essential_genes_cache",
                 force_reidentify_essential=False,
                 top_percentage=0.4, virtual_kcat_mw=1e-10,
                 separate_oe_kd_filtering=True,
                 filtering_strategy="include_ties"):
        """
        Initialize ecFactory framework.

        Parameters
        ----------
        model : cobra.Model
            Enzyme-constrained metabolic model
        product_rxn : str
            Target product reaction ID
        biomass_rxn : str, optional
            Biomass reaction ID (auto-detected if None)
        essential_genes_file : str, optional
            Path to essential genes file
        enzyme_data_file : str, optional
            Path to enzyme kinetics data file
        pfba_factor : float, default=1
            pFBA optimization factor
        fseof_steps : int, default=10
            Number of FSEOF scanning steps
        auto_identify_essential : bool, default=True
            Whether to auto-identify essential genes
        additional_essential_genes : list, optional
            Additional essential genes to include
        essential_genes_cache_dir : str, default="essential_genes_cache"
            Directory for essential genes cache
        force_reidentify_essential : bool, default=False
            Force re-identification of essential genes
        top_percentage : float, default=0.4
            Target selection percentage (0-1)
        virtual_kcat_mw : float, default=1e-10
            kcat_MW value for virtual reactions
        separate_oe_kd_filtering : bool, default=True
            Whether to filter OE and KD targets separately
        filtering_strategy : str, default="include_ties"
            Filtering strategy ("include_ties" or "strict_rank")
        """
        # Model initialization
        self.model = model.copy()
        self.product_rxn = product_rxn
        self.original_product_rxn = product_rxn
        self.essential_genes_cache_dir = essential_genes_cache_dir

        # Filtering parameters
        self.top_percentage = top_percentage
        self.separate_oe_kd_filtering = separate_oe_kd_filtering
        self.oe_percentage = top_percentage
        self.kd_percentage = top_percentage
        self.filtering_strategy = filtering_strategy

        # Virtual reaction parameters
        self.virtual_kcat_mw = virtual_kcat_mw
        self.virtual_reactions_added = []

        # Constraint parameters
        self.constraint_rxn_id = 'EX_cpd00027_e_reverse'
        self.optimal_constraint_bound = None

        # Auto-detect biomass reaction
        if biomass_rxn is None:
            self.biomass_rxn = list(self.model.objective.keys())[0].id
        else:
            self.biomass_rxn = biomass_rxn

        # Optimization parameters
        self.pfba_factor = pfba_factor
        self.fseof_steps = fseof_steps
        self.tol = 1e-13

        # Create cache directory
        os.makedirs(self.essential_genes_cache_dir, exist_ok=True)

        # Load essential genes
        if auto_identify_essential:
            print("Loading essential genes with caching mechanism...")
            self.essential_genes = self._load_or_identify_essential_genes(
                essential_genes_file, force_reidentify_essential
            )
        else:
            self.essential_genes = self._load_essential_genes(essential_genes_file)

        # Add additional essential genes
        if additional_essential_genes:
            original_count = len(self.essential_genes)
            self.essential_genes = list(set(self.essential_genes + additional_essential_genes))
            added_count = len(self.essential_genes) - original_count
            print(f"Added {added_count} user-specified essential genes. Total: {len(self.essential_genes)}")

        # Load enzyme data
        self.enzyme_data = self._load_enzyme_data(enzyme_data_file)

        # Check enzyme constraints
        self.has_enzyme_constraints = self._check_enzyme_constraints()

        # Initialize result storage
        self.targets = {'OE': [], 'KD': []}
        self.flux_matrix = None
        self.reaction_scores = None
        self.gene_scores = None
        self.bidirectional_genes = []

        # Leak reaction tracking
        self.leak_genes = {'transport_OE': [], 'consumption_KD': []}
        self.leak_reactions_info = []

        # Enzyme usage analysis
        self.enzyme_usage_prod = None
        self.enzyme_usage_bio = None

        self.calculate_optimal_constraint_bound()
        try:
            rxn = self.model.reactions.get_by_id('EX_cpd00197_e_reverse')
            rxn.bounds = (0, 0)
            print("Closed reaction: EX_cpd00197_e_reverse")
        except KeyError:
            print("Warning: EX_cpd00197_e_reverse not found in model")
        self.set_quiet_mode(True)
        self._print_initialization_info()

    # ==================== Configuration Methods ====================

    def set_quiet_mode(self, quiet=True):
        """Set solver quiet mode."""
        import logging
        level = logging.ERROR if quiet else logging.INFO
        logging.getLogger('cobra').setLevel(level)
        try:
            self.model.solver.configuration.verbosity = 0 if quiet else 1
        except:
            pass

    def set_filtering_strategy(self, strategy):
        """Set filtering strategy."""
        if strategy not in ["include_ties", "strict_rank"]:
            raise ValueError("Strategy must be 'include_ties' or 'strict_rank'")
        self.filtering_strategy = strategy
        print(f"Filtering strategy set to: {strategy}")

    def set_filtering_percentages(self, separate=True, oe_percentage=None, kd_percentage=None):
        """Set filtering percentages for OE and KD targets."""
        self.separate_oe_kd_filtering = separate

        if oe_percentage is not None:
            if not 0 < oe_percentage <= 1:
                raise ValueError("oe_percentage must be between 0 and 1")
            self.oe_percentage = oe_percentage
        else:
            self.oe_percentage = self.top_percentage

        if kd_percentage is not None:
            if not 0 < kd_percentage <= 1:
                raise ValueError("kd_percentage must be between 0 and 1")
            self.kd_percentage = kd_percentage
        else:
            self.kd_percentage = self.top_percentage

        print(f"Separate filtering: {'Enabled' if separate else 'Disabled'}")
        if separate:
            print(f"OE percentage: {self.oe_percentage * 100:.1f}%")
            print(f"KD percentage: {self.kd_percentage * 100:.1f}%")

    def set_top_percentage(self, percentage):
        """Set overall top percentage for target selection."""
        if not 0 < percentage <= 1:
            raise ValueError("Percentage must be between 0 and 1")
        self.top_percentage = percentage
        print(f"Top percentage set to: {percentage * 100:.1f}%")

    def _has_valid_kcat_mw_old_style(self, reaction):
        """
        使用old算法的kcat_MW检测逻辑

        Parameters:
        -----------
        reaction : cobra.Reaction
            要检查的反应

        Returns:
        --------
        bool
            是否有有效的kcat_MW值
        """
        return (hasattr(reaction, 'kcat_MW') and
                reaction.kcat_MW is not None and
                reaction.kcat_MW != '' and
                reaction.kcat_MW != 0)

    # ==================== Core Analysis Methods ====================

    def run_complete_pipeline(self):
        """
        Run complete EUVA pipeline.

        Steps:
        1. EC-FSEOF analysis
        2. Leak reaction identification
        3. Enzyme group consistency check
        4. Enzyme usage variability filtering
        5. Metabolic network topology filtering
        6. Target effectiveness evaluation

        Returns
        -------
        dict
            Final identified target genes
        """
        print("\n=== Step 1: EC-FSEOF Analysis ===")
        self.ec_fseof()

        print("\n=== Step 2: Leak Reaction Identification ===")
        self.identify_leak_reactions()

        print("\n=== Step 3: Enzyme Group Consistency Check ===")
        self.ensure_enzyme_group_consistency()

        print("\n=== Step 4: Enzyme Usage Variability Filtering ===")
        self.apply_enzyme_usage_filter()

        print("\n=== Step 5: Metabolic Network Topology Filtering ===")
        self.apply_topology_filter(max_depth=20)

        print("\n=== Step 6: Target Effectiveness Evaluation ===")
        self.evaluate_targets()

        print("\n=== Pipeline Complete ===")
        print(f"Final targets: OE={len(self.targets['OE'])}, KD={len(self.targets['KD'])}")

        # Statistics
        leak_genes_set = set(self.leak_genes['transport_OE'] + self.leak_genes['consumption_KD'])
        leak_oe = len([g for g in self.targets['OE'] if g in leak_genes_set])
        leak_kd = len([g for g in self.targets['KD'] if g in leak_genes_set])
        print(f"Leak reaction genes: OE={leak_oe}, KD={leak_kd}")

        bidirectional_oe = len([g for g in self.targets['OE'] if g in self.bidirectional_genes])
        bidirectional_kd = len([g for g in self.targets['KD'] if g in self.bidirectional_genes])
        print(f"Bidirectional genes: OE={bidirectional_oe}, KD={bidirectional_kd}")

        return self.targets

    def ec_fseof(self):
        """
        Execute optimized EC-FSEOF analysis.

        Three-stage optimization:
        1. Calculate all Pareto optimal points
        2. Create virtual reactions once
        3. Batch optimization with virtual model

        Returns
        -------
        dict
            Identified target genes
        """
        print("Starting optimized EC-FSEOF analysis...")
        print("Strategy: Three-stage optimization with virtual reactions")

        original_bounds = self._apply_optimal_constraint()
        constraint_tolerance = 0.01
        min_absolute_tolerance = 1e-6

        try:
            # Stage 1: Calculate Pareto points
            print("Stage 1: Calculating Pareto optimal points...")
            pareto_points = self._calculate_pareto_points()

            # Stage 2: Create virtual reactions
            print("\nStage 2: Creating virtual reactions...")
            with self._virtual_reaction_context() as virtual_objective:
                print(f"Virtual objective: {virtual_objective}")

                # Stage 3: Batch optimization
                print(f"\nStage 3: Batch optimization (tolerance: {constraint_tolerance * 100:.1f}%)...")
                flux_matrix = self._batch_optimize_with_virtual_model(
                    pareto_points, virtual_objective, constraint_tolerance, min_absolute_tolerance
                )

            # Stage 4: Analyze flux trends
            print("\nStage 4: Analyzing flux trends...")
            self._process_flux_matrix(flux_matrix)
            self._analyze_flux_trends()
            self._apply_filtering()

            self.save_results()
            print("\nEC-FSEOF analysis complete!")
            return self.targets

        finally:
            self._restore_constraint_bounds(original_bounds)

    def identify_leak_reactions(self):
        """
        Identify flux leak reactions.

        Returns
        -------
        dict
            Leak reaction analysis results
        """
        print("Identifying flux leak reactions...")

        self.leak_genes = {'transport_OE': [], 'consumption_KD': []}
        self.leak_reactions_info = []

        kd_genes_added = 0
        oe_genes_added = 0

        product_rxn_id = self.original_product_rxn
        print(f"Analyzing leaks based on: {product_rxn_id}")

        try:
            product_rxn = self.model.reactions.get_by_id(product_rxn_id)
            substrate_metabolites = [met for met, coeff in product_rxn.metabolites.items() if coeff < 0]

            for metabolite in substrate_metabolites:
                product_id = metabolite.id
                cytosolic_id = product_id

                # Handle extracellular products
                if product_id.endswith('_e'):
                    cytosolic_id = product_id.replace('_e', '_c')
                    try:
                        cytosolic_metabolite = self.model.metabolites.get_by_id(cytosolic_id)
                        print(f"Found cytosolic product: {cytosolic_id}")
                    except KeyError:
                        print(f"Warning: Cannot find cytosolic version {cytosolic_id}")
                        cytosolic_metabolite = metabolite
                else:
                    cytosolic_metabolite = metabolite

                # Find consuming reactions
                potential_leaks = [rxn for rxn in cytosolic_metabolite.reactions
                                   if rxn != product_rxn and rxn.metabolites[cytosolic_metabolite] < 0]

                print(f"Found {len(potential_leaks)} potential leak reactions")

                # Analyze each leak
                for rxn in potential_leaks:
                    reaction_info = self._analyze_single_leak_reaction(rxn, cytosolic_id, product_id)

                    if reaction_info['category'] == 'skip':
                        continue

                    self.leak_reactions_info.append(reaction_info)

                    # Add genes to targets
                    if reaction_info['category'] == 'transport_OE':
                        for gene in rxn.genes:
                            if gene.id not in self.leak_genes['transport_OE']:
                                self.leak_genes['transport_OE'].append(gene.id)
                                if gene.id not in self.targets['OE']:
                                    self.targets['OE'].append(gene.id)
                                    oe_genes_added += 1
                                    print(f"    -> Transport gene {gene.id} added to OE")

                    elif reaction_info['category'] == 'consumption_KD':
                        for gene in rxn.genes:
                            if gene.id not in self.leak_genes['consumption_KD']:
                                self.leak_genes['consumption_KD'].append(gene.id)
                                if gene.id not in self.targets['KD']:
                                    self.targets['KD'].append(gene.id)
                                    kd_genes_added += 1
                                    print(f"    -> Consumption gene {gene.id} added to KD")

            print(f"\nLeak analysis complete:")
            print(f"Transport OE genes: {len(self.leak_genes['transport_OE'])}")
            print(f"Consumption KD genes: {len(self.leak_genes['consumption_KD'])}")
            print(f"New OE targets: {oe_genes_added}")
            print(f"New KD targets: {kd_genes_added}")

            self._print_leak_summary()

        except Exception as e:
            print(f"Error identifying leak reactions: {e}")
            import traceback
            traceback.print_exc()

        return {
            'targets': self.targets,
            'leak_reactions': self.leak_reactions_info,
            'leak_genes': self.leak_genes,
            'oe_genes_added': oe_genes_added,
            'kd_genes_added': kd_genes_added
        }

    def apply_enzyme_usage_filter(self):
        """Apply Enzyme Usage Variability Analysis filter."""
        print("Applying Enzyme Usage Variability Analysis filter...")

        if not self.has_enzyme_constraints:
            print("Model has no enzyme constraints, skipping EUVA filter")
            return self.targets

        leak_genes_set = set(self.leak_genes['transport_OE'] + self.leak_genes['consumption_KD'])
        bidirectional_genes_set = set(self.bidirectional_genes) if hasattr(self, 'bidirectional_genes') else set()

        print(f"Leak genes (skip FVA): {len(leak_genes_set)}")
        print(f"Bidirectional genes (require FVA): {len(bidirectional_genes_set)}")

        original_bounds = self._apply_optimal_constraint()

        try:
            # Get maximum growth rate
            print("Getting maximum growth rate...")
            with self.model as m:
                m.objective = self.product_rxn
                max_product = m.optimize().objective_value
                print(f"Maximum growth rate: {max_product}")

            # Get enzyme reactions
            target_genes = set(self.targets.get('OE', []) + self.targets.get('KD', []))
            enzyme_reactions = self._get_enzyme_reactions(target_genes)
            print(f"Enzyme reactions to analyze: {len(enzyme_reactions)}")

            # Production: 0.9 * max_product (高产量)
            self.enzyme_usage_prod = self._run_scenario_fva(
                0.9 * max_product, enzyme_reactions, "production"
            )

            # Growth: 0.1 * max_product (低产量)
            self.enzyme_usage_bio = self._run_scenario_fva(
                0.1 * max_product, enzyme_reactions, "growth"
            )

            # Filter targets by enzyme usage
            print("Filtering targets by enzyme usage...")
            filtered_targets = self._filter_by_enzyme_usage(
                target_genes, leak_genes_set, bidirectional_genes_set
            )

            self._report_euva_results(filtered_targets, leak_genes_set, bidirectional_genes_set)

            self.targets = filtered_targets
            return self.targets

        except Exception as e:
            print(f"Error in EUVA filtering: {e}")
            return self.targets

        finally:
            self._restore_constraint_bounds(original_bounds)

    def apply_topology_filter(self, max_depth=2):
        """
        Apply metabolic network topology filter.

        Parameters
        ----------
        max_depth : int, default=2
            Maximum traceback depth

        Returns
        -------
        dict
            Filtered target genes
        """
        print(f"Applying metabolic network topology filter (max_depth={max_depth})...")
        print(f"Based on: {self.original_product_rxn}")
        print("Note: Only filtering KD targets")

        # Define exclusions
        excluded_reactions = {
            'rxn05150_c', 'rxn05174_c', 'rxn05195_c', 'rxn05555_c',
            'rxn05618_c', 'rxn10481_c', 'rxn10571_c', 'rxn40890_c',
            'rxn40890_c_reverse', 'rxn08766_c_reverse_num1',
            'rxn08766_c_num1', 'rxnbccaa3_c',
        }

        metabolite_reaction_exclusions = {
            'cpd00020_c': {'rxn08043_c', 'rxn15021_c','rxn01644_c',},
            # 'cpd00061_c': {'rxn00461_c','rxn00461_c_reverse'},
            'cpd00022_c': {'rxn00902_c','rxn03638_c'},
            'cpd00023_c': {'rxn06937_c'},
            'cpd00041_c': {'rxn00838_c'},
        }

        common_cofactors = {
            'cpd00002_c', 'cpd00008_c', 'cpd00018_c', 'cpd00038_c', 'cpd00031_c',
            'cpd00114_c', 'cpd00002_e', 'cpd00008_e', 'cpd00018_e', 'cpd00038_e',
            'cpd00031_e', 'cpd00114_e', 'cpd00052_c', 'cpd00096_c', 'cpd00046_c',
            'cpd00062_c', 'cpd00014_c', 'cpd00091_c', 'cpd00053_c', 'cpd00013_c',
            'cpd00106_c', 'cpd00010_c', 'cpd00644_c', 'cpd00003_c', 'cpd00004_c',
            'cpd00005_c', 'cpd00006_c', 'cpd00015_c', 'cpd00982_c', 'cpd00001_c',
            'cpd00067_c', 'cpd00009_c', 'cpd00012_c', 'cpd00011_c', 'cpd00242_c',
            'cpd00001_e', 'cpd00067_e', 'cpd00009_e', 'cpd00012_e', 'cpd00011_e',
            'cpd00242_e',
        }

        special_reaction_whitelist = {
            'rxn00789_c': ['cpd00002_c'],  # ATP as structural precursor
        }

        try:
            # Get flux-carrying reactions
            flux_carrying_reactions = self._get_flux_carrying_reactions()

            # Extract target metabolites
            target_products = self._extract_target_metabolites()

            if not target_products:
                print("Warning: No target metabolites found")
                return self.targets

            # Trace precursors
            all_precursors, network_reactions = self._trace_precursors(
                target_products, flux_carrying_reactions, excluded_reactions,
                common_cofactors, special_reaction_whitelist, max_depth
            )

            # Collect relevant reactions
            all_relevant_reactions = self._collect_relevant_reactions(
                all_precursors, network_reactions, excluded_reactions,
                metabolite_reaction_exclusions
            )

            # Filter KD targets
            filtered_targets = self._filter_kd_by_relevance(all_relevant_reactions)

            self.targets = filtered_targets
            return filtered_targets

        except Exception as e:
            print(f"Error in topology filtering: {e}")
            return self.targets

    def evaluate_targets(self):
        """Evaluate individual target effectiveness."""
        print("Evaluating individual target effectiveness...")

        # Calculate baseline
        wt_baseline = self._calculate_wildtype_baseline()

        # Save original kcat values
        original_kcat_values = self._save_original_kcat_values()

        # Evaluate targets
        beneficial_targets, target_metrics = self._evaluate_all_targets(
            wt_baseline, original_kcat_values
        )

        # Process results
        if target_metrics:
            results_df = pd.DataFrame(target_metrics)
            results_df = results_df.sort_values('prod_improvement', ascending=False)

            # Resolve conflicts
            results_df, beneficial_targets = self._resolve_dual_classifications(
                results_df, beneficial_targets
            )

            self._print_evaluation_summary(results_df, beneficial_targets)
        else:
            print("No beneficial targets found")
            results_df = pd.DataFrame()

        self.targets = beneficial_targets
        self.target_metrics = results_df

        return beneficial_targets

    def ensure_enzyme_group_consistency(self):
        """Ensure consistency within enzyme groups."""
        print("Identifying enzyme groups...")

        self.enzyme_groups = {}

        for gene in self.model.genes:
            if len(gene.reactions) > 0:
                rxn_key = tuple(sorted([r.id for r in gene.reactions]))

                if rxn_key in self.enzyme_groups:
                    self.enzyme_groups[rxn_key].append(gene.id)
                else:
                    self.enzyme_groups[rxn_key] = [gene.id]

        # Resolve inconsistencies
        modified = False

        for enzyme_group in self.enzyme_groups.values():
            if len(enzyme_group) <= 1:
                continue

            gene_targets = {}
            for gene in enzyme_group:
                for target_type in self.targets:
                    if gene in self.targets[target_type]:
                        gene_targets[gene] = target_type

            # Unify to higher priority type
            if len(set(gene_targets.values())) > 1:
                modified = True
                target_priority = {"KD": 2, "OE": 1}
                highest_priority = max(gene_targets.values(), key=lambda x: target_priority[x])

                for gene, target_type in gene_targets.items():
                    if target_type != highest_priority:
                        self.targets[target_type].remove(gene)
                        if gene not in self.targets[highest_priority]:
                            self.targets[highest_priority].append(gene)

        if modified:
            print("Resolved enzyme group inconsistencies")

        return self.targets

    # ==================== Helper Methods ====================

    def calculate_optimal_constraint_bound(self):
        """Calculate optimal carbon source constraint bound."""
        print("Calculating optimal carbon source constraint bound...")

        if self.constraint_rxn_id not in self.model.reactions:
            print(f"Warning: {self.constraint_rxn_id} not found")
            self.optimal_constraint_bound = None
            return

        constraint_rxn = self.model.reactions.get_by_id(self.constraint_rxn_id)
        original_bounds = (constraint_rxn.lower_bound, constraint_rxn.upper_bound)

        constraint_rxn.upper_bound = 1000

        try:
            print("Calculating carbon uptake for product maximization...")
            with self.model as prod_model:
                prod_model.objective = self.original_product_rxn
                prod_solution = prod_model.optimize()

                if prod_solution.status == 'optimal':
                    self.optimal_constraint_bound = abs(prod_solution.fluxes.get(self.constraint_rxn_id, 0))
                    print(f"Optimal constraint bound: {self.optimal_constraint_bound:.6f}")
                else:
                    self.optimal_constraint_bound = 2.0
                    print(f"Using default bound: {self.optimal_constraint_bound}")

        except Exception as e:
            print(f"Error calculating constraint: {e}")
            self.optimal_constraint_bound = 2.0
        finally:
            constraint_rxn.lower_bound = original_bounds[0]
            constraint_rxn.upper_bound = original_bounds[1]

    def compare_enzyme_usage(self, gene_id):
        """Compare enzyme usage for a gene between scenarios."""
        if not hasattr(self, 'enzyme_usage_prod') or not hasattr(self, 'enzyme_usage_bio'):
            return "none"

        try:
            gene_obj = self.model.genes.get_by_id(gene_id)
            reactions = [r.id for r in gene_obj.reactions]

            if not reactions:
                return "none"

            valid_reactions = []
            for rxn_id in reactions:
                rxn = self.model.reactions.get_by_id(rxn_id)

                if (self._has_valid_kcat_mw_old_style(rxn) and
                        rxn_id in self.enzyme_usage_bio.index and
                        rxn_id in self.enzyme_usage_prod.index):
                    valid_reactions.append(rxn_id)

            if not valid_reactions:
                return "noKcat_MW"

            # Analyze usage changes
            reaction_classifications = {}

            for rxn_id in valid_reactions:
                min_prod = self.enzyme_usage_prod.loc[rxn_id, 'minimum'] or 0
                max_prod = self.enzyme_usage_prod.loc[rxn_id, 'maximum'] or 0
                min_bio = self.enzyme_usage_bio.loc[rxn_id, 'minimum'] or 0
                max_bio = self.enzyme_usage_bio.loc[rxn_id, 'maximum'] or 0

                # Classify change (handle NaN)
                avg_prod = np.nanmean([min_prod, max_prod])
                avg_bio = np.nanmean([min_bio, max_bio])

                # Skip if both are NaN or zero
                if np.isnan(avg_prod) and np.isnan(avg_bio):
                    simple_class = "neutral"
                elif np.isnan(avg_bio) or avg_bio == 0:
                    simple_class = "up" if avg_prod > 0 else "neutral"
                elif np.isnan(avg_prod):
                    simple_class = "down"
                elif avg_prod > avg_bio * 1.05:
                    simple_class = "up"
                elif avg_prod < avg_bio * 0.95:
                    simple_class = "down"
                else:
                    simple_class = "neutral"

                reaction_classifications[rxn_id] = simple_class

            # Aggregate classifications
            unique_classes = set(reaction_classifications.values())

            if "up" in unique_classes and "down" in unique_classes:
                return "mixed"
            elif "up" in unique_classes:
                return "up_consistent"
            elif "down" in unique_classes:
                return "down_consistent"
            else:
                return "neutral"

        except Exception as e:
            print(f"Error comparing enzyme usage for {gene_id}: {e}")
            return "none"

    def is_enzyme_unused(self, gene_id):
        """Check if enzyme is unused in production scenario."""
        if not hasattr(self, 'enzyme_usage_prod'):
            return False

        try:
            gene_obj = self.model.genes.get_by_id(gene_id)
            for reaction in gene_obj.reactions:
                if self._has_valid_kcat_mw_old_style(reaction):
                    if reaction.id in self.enzyme_usage_prod.index:
                        max_usage = self.enzyme_usage_prod.loc[reaction.id, 'maximum']
                        if abs(max_usage) < self.tol:
                            return True
            return False
        except:
            return False

    def get_carbon_source_reaction(self):
        """Get carbon source uptake reaction ID."""
        carbon_source_ids = ['EX_cpd00027_e_reverse']

        for rxn_id in carbon_source_ids:
            if rxn_id in self.model.reactions:
                return rxn_id

        # Search for common carbon sources
        for rxn in self.model.reactions:
            if rxn.id.startswith('EX_') and len(rxn.metabolites) == 1:
                for met in rxn.metabolites:
                    met_name = met.name.lower() if met.name else ''
                    if any(keyword in met_name for keyword in
                           ['glucose', 'carbon', 'glycerol', 'lactate', 'acetate']):
                        return rxn.id

        # Return any exchange reaction as default
        for rxn in self.model.reactions:
            if rxn.id.startswith('EX_'):
                return rxn.id

        return None

    def save_results(self, output_dir="fseof_results"):
        """Save analysis results."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"Saving results to {output_dir}...")

        # Save various dataframes
        if self.flux_matrix is not None:
            self.flux_matrix.to_csv(os.path.join(output_dir, "flux_matrix.csv"))

        if self.reaction_scores is not None:
            self.reaction_scores.to_csv(os.path.join(output_dir, "reaction_scores.csv"))

        if self.gene_scores is not None:
            self.gene_scores.to_csv(os.path.join(output_dir, "gene_scores.csv"))

        # Save targets
        if hasattr(self, 'targets'):
            for target_type, genes in self.targets.items():
                with open(os.path.join(output_dir, f"targets_{target_type}.txt"), 'w') as f:
                    f.write('\n'.join(genes))

        # Save metadata
        metadata = {
            "biomass_reaction": self.biomass_rxn,
            "product_reaction": self.original_product_rxn,
            "filtering_strategy": self.filtering_strategy,
            "fseof_steps": self.fseof_steps,
            "top_percentage": self.top_percentage,
            "optimal_constraint_bound": self.optimal_constraint_bound,
        }

        with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=4)

        print("Results saved successfully")
        return output_dir

    def save_targets_to_csv(self, output_file):
        """Save targets to CSV file."""
        rows = []
        leak_genes_set = set(self.leak_genes['transport_OE'] + self.leak_genes['consumption_KD'])

        for target_type in self.targets:
            for gene_id in self.targets[target_type]:
                try:
                    gene = self.model.genes.get_by_id(gene_id)
                    row = {
                        'gene_id': gene_id,
                        'gene_name': gene.name if hasattr(gene, 'name') else '',
                        'modification': target_type,
                        'reactions': '; '.join([r.id for r in gene.reactions]),
                        'is_leak_gene': gene_id in leak_genes_set,
                        'is_bidirectional': gene_id in self.bidirectional_genes if hasattr(self,
                                                                                           'bidirectional_genes') else False,
                    }
                    rows.append(row)
                except:
                    rows.append({'gene_id': gene_id, 'modification': target_type})

        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        print(f"Targets saved to {output_file}")
        return output_file

    def print_summary(self):
        """Print analysis summary."""
        print("\n===== ecFactory Target Summary =====")
        print(f"Model: {self.model.id}")
        print(f"Product: {self.original_product_rxn}")
        print(f"Biomass: {self.biomass_rxn}")
        print(f"Strategy: {self.filtering_strategy}")
        print(f"Essential genes: {len(self.essential_genes)}")

        print("\nTargets identified:")
        print(f"  OE: {len(self.targets['OE'])} genes")
        print(f"  KD: {len(self.targets['KD'])} genes")

        print("\nLeak genes:")
        print(f"  Transport: {len(self.leak_genes['transport_OE'])}")
        print(f"  Consumption: {len(self.leak_genes['consumption_KD'])}")
        print("====================================")

    # ==================== Private Helper Methods ====================

    def _print_initialization_info(self):
        """Print initialization information."""
        print("\n" + "=" * 60)
        print("ecFactory Initialization Complete")
        print("=" * 60)
        print(f"Model: {self.model.id}")
        print(f"Product: {self.original_product_rxn}")
        print(f"Biomass: {self.biomass_rxn}")
        print(f"Strategy: {self.filtering_strategy}")
        print(f"Top %: {self.top_percentage * 100:.1f}%")
        print(f"Essential genes: {len(self.essential_genes)}")
        print(f"Enzyme constraints: {'Yes' if self.has_enzyme_constraints else 'No'}")
        print("=" * 60)

    def _apply_optimal_constraint(self):
        """Apply optimal constraint."""
        if self.optimal_constraint_bound is None:
            return None

        if self.constraint_rxn_id not in self.model.reactions:
            return None

        constraint_rxn = self.model.reactions.get_by_id(self.constraint_rxn_id)
        original_bounds = (constraint_rxn.lower_bound, constraint_rxn.upper_bound)
        constraint_rxn.upper_bound = self.optimal_constraint_bound
        return original_bounds

    def _restore_constraint_bounds(self, original_bounds):
        """Restore constraint bounds."""
        if original_bounds is None:
            return

        if self.constraint_rxn_id in self.model.reactions:
            constraint_rxn = self.model.reactions.get_by_id(self.constraint_rxn_id)
            constraint_rxn.lower_bound = original_bounds[0]
            constraint_rxn.upper_bound = original_bounds[1]

    @contextmanager
    def _virtual_reaction_context(self):
        """Context manager for virtual reactions."""
        print("Creating virtual reactions...")

        original_product = self.product_rxn
        original_virtual = self.virtual_reactions_added.copy()

        virtual_objective = self._apply_virtual_reactions()

        try:
            yield virtual_objective
        finally:
            print("Cleaning up virtual reactions...")
            self._cleanup_virtual_reactions()
            self.product_rxn = original_product
            self.virtual_reactions_added = original_virtual
            print("Model state restored")

    def _apply_virtual_reactions(self):
        """Apply virtual reaction strategy."""
        print("Creating temporary virtual reactions...")

        try:
            base_reaction = self.model.reactions.get_by_id(self.original_product_rxn)
        except KeyError:
            print(f"Error: Cannot find {self.original_product_rxn}")
            return self.original_product_rxn

        # Check if export reaction
        if not self._is_export_reaction(base_reaction):
            print("Not an export reaction, skipping virtual strategy")
            return self.original_product_rxn

        # Identify target metabolite
        target_met = None
        for met, coeff in base_reaction.metabolites.items():
            if coeff < 0:  # Consumed in export
                target_met = met
                break

        if not target_met:
            return self.original_product_rxn

        print(f"Target metabolite: {target_met.id}")

        # Create virtual metabolite
        fake_met = cobra.Metabolite(
            id="fake_e_temp",
            name="Temporary virtual metabolite",
            compartment="e"
        )
        self.model.add_metabolites([fake_met])

        virtual_reactions_kcat_mw = {}

        # Reaction 1: Product -> Virtual (efficient)
        rxn1_id = "VIRTUAL_CONVERSION_TEMP"
        rxn1 = cobra.Reaction(id=rxn1_id, lower_bound=0, upper_bound=1000)
        rxn1.add_metabolites({target_met: -1, fake_met: 1})
        virtual_reactions_kcat_mw[rxn1_id] = {"kcat": "", "kcat_MW": 1e+10}
        self.model.add_reactions([rxn1])
        self.virtual_reactions_added.append(rxn1_id)

        # Reaction 2: Nothing -> Virtual (inefficient backup)
        rxn2_id = "VIRTUAL_PRODUCTION_TEMP"
        rxn2 = cobra.Reaction(id=rxn2_id, lower_bound=0, upper_bound=1000)
        rxn2.add_metabolites({fake_met: 1})
        virtual_reactions_kcat_mw[rxn2_id] = {"kcat": "", "kcat_MW": 1e-5}
        self.model.add_reactions([rxn2])
        self.virtual_reactions_added.append(rxn2_id)

        # Reaction 3: Virtual -> Nothing (objective)
        rxn3_id = "VIRTUAL_OBJECTIVE_TEMP"
        rxn3 = cobra.Reaction(id=rxn3_id, lower_bound=0, upper_bound=1000)
        rxn3.add_metabolites({fake_met: -1})
        virtual_reactions_kcat_mw[rxn3_id] = {"kcat": "", "kcat_MW": 1e+10}
        self.model.add_reactions([rxn3])
        self.virtual_reactions_added.append(rxn3_id)

        # Reload model with virtual reactions
        self._reload_model_with_virtual_reactions(virtual_reactions_kcat_mw)

        print(f"Virtual objective: {rxn3_id}")
        return rxn3_id

    def _cleanup_virtual_reactions(self):
        """Clean up virtual reactions."""
        # Remove virtual reactions
        reactions_to_remove = []
        for rxn_id in self.virtual_reactions_added:
            if rxn_id in self.model.reactions:
                reactions_to_remove.append(self.model.reactions.get_by_id(rxn_id))

        if reactions_to_remove:
            self.model.remove_reactions(reactions_to_remove)
            print(f"Removed {len(reactions_to_remove)} virtual reactions")

        # Remove virtual metabolites
        metabolites_to_remove = []
        for met in self.model.metabolites:
            if met.id.startswith("fake_") and "temp" in met.id.lower():
                metabolites_to_remove.append(met)

        if metabolites_to_remove:
            self.model.remove_metabolites(metabolites_to_remove)
            print(f"Removed {len(metabolites_to_remove)} virtual metabolites")

        self.virtual_reactions_added = []

    def _reload_model_with_virtual_reactions(self, virtual_reactions_kcat_mw):
        """Reload model with virtual reactions."""
        temp_filename = None
        try:
            print("Reloading model with kcat_MW constraints...")

            # Extract existing kcat_MW info
            existing_kcat_mw = {}
            for reaction in self.model.reactions:
                if hasattr(reaction, 'kcat_MW') and reaction.kcat_MW:
                    try:
                        existing_kcat_mw[reaction.id] = float(reaction.kcat_MW)
                    except:
                        pass

            # Save to temporary JSON
            with tempfile.NamedTemporaryFile(suffix='.json', mode='w', delete=False) as temp_file:
                temp_filename = temp_file.name
                cobra.io.save_json_model(self.model, temp_filename)

            # Load and modify JSON
            with open(temp_filename, 'r') as file:
                model_dict = json.load(file)

            # Ensure enzyme constraint exists
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
                rxn_id = reaction['id']

                if rxn_id in virtual_reactions_kcat_mw:
                    kcat_mw_info = virtual_reactions_kcat_mw[rxn_id]
                    reaction['kcat'] = ""
                    reaction['kcat_MW'] = kcat_mw_info['kcat_MW']
                elif rxn_id in existing_kcat_mw:
                    reaction['kcat'] = ""
                    reaction['kcat_MW'] = existing_kcat_mw[rxn_id]
                else:
                    reaction['kcat'] = ""
                    reaction['kcat_MW'] = ""

            # Save modified JSON
            with open(temp_filename, 'w') as file:
                json.dump(model_dict, file, indent=2)

            # Reload model
            try:
                from ECMpy_function import get_enzyme_constraint_model_with_reaction_set_exclusion
                self.model = get_enzyme_constraint_model_with_reaction_set_exclusion(temp_filename)
                print("Model reloaded with ECMpy")
            except:
                self.model = cobra.io.load_json_model(temp_filename)
                print("Model reloaded with cobra")

        except Exception as e:
            print(f"Error reloading model: {e}")

        finally:
            if temp_filename and os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                except:
                    pass

    def _calculate_pareto_points(self):
        """Calculate Pareto optimal points."""
        print("Calculating original product maximum theoretical yield...")
        with self.model as temp_model:
            temp_model.objective = self.original_product_rxn
            sol = temp_model.optimize()
            max_product = sol.objective_value * 0.9
            print(f"Maximum product flux: {sol.objective_value:.6f}")
            print(f"FSEOF upper limit (90%): {max_product:.6f}")

        if max_product <= 0:
            raise ValueError(f"Maximum product flux is not positive: {max_product}")

        # Calculate wild-type baseline
        print("Calculating wild-type baseline...")
        with self.model as temp_model:
            temp_model.objective = self.biomass_rxn
            sol = temp_model.optimize()
            wt_biomass = sol.objective_value
            iniTarget = sol.fluxes[self.original_product_rxn]
            print(f"Wild-type max growth: {wt_biomass:.6f}")
            print(f"Wild-type product flux: {iniTarget:.6f}")

        # Create product constraint sequence
        product_constraints = np.linspace(max_product/9, max_product, self.fseof_steps)
        print(f"FSEOF constraints: {max_product/9:.3f} → {max_product:.3f} ({self.fseof_steps} steps)")

        # Calculate Pareto points
        print("Calculating all Pareto optimal points...")
        pareto_points = []

        for i, product_value in enumerate(product_constraints):
            try:
                with self.model.copy() as pareto_model:
                    if self.original_product_rxn not in pareto_model.reactions:
                        pareto_points.append((product_value, 0))
                        continue

                    # Fix product flux
                    product_rxn = pareto_model.reactions.get_by_id(self.original_product_rxn)
                    product_rxn.bounds = (product_value, product_value)

                    # Maximize biomass
                    pareto_model.objective = self.biomass_rxn
                    solution = pareto_model.optimize()

                    if solution.status != 'optimal':
                        print(f"  Point {i + 1} failed: {solution.status}")
                        pareto_points.append((product_value, 0))
                        continue

                    optimal_biomass = solution.objective_value
                    actual_product = solution.fluxes[self.original_product_rxn]
                    pareto_points.append((actual_product, optimal_biomass))

                    print(f"  Point {i + 1}: product={actual_product:.4f}, biomass={optimal_biomass:.4f}")

            except Exception as e:
                print(f"  Error at point {i + 1}: {e}")
                pareto_points.append((product_value, 0))

        print(f"Calculated {len(pareto_points)} Pareto optimal points")
        return pareto_points

    def _batch_optimize_with_virtual_model(self, pareto_points, virtual_objective,
                                           constraint_tolerance, min_absolute_tolerance):
        """Batch optimize using virtual reaction model."""
        original_reaction_ids = [r.id for r in self.model.reactions
                                 if not r.id.startswith('VIRTUAL_')]
        original_reaction_count = len(original_reaction_ids)

        flux_matrix = np.zeros((original_reaction_count, len(pareto_points)))

        print("Starting batch optimization...")
        for i, (product_flux, biomass_flux) in enumerate(pareto_points):
            try:
                print(f"  Optimizing point {i + 1}/{len(pareto_points)}: "
                      f"product={product_flux:.4f}, biomass={biomass_flux:.4f}")

                if biomass_flux <= 1e-6:
                    print(f"    Skipping: biomass too low")
                    flux_matrix[:, i] = 0
                    continue

                with self.model.copy() as opt_model:
                    # Check reactions exist
                    if (self.original_product_rxn not in opt_model.reactions or
                            self.biomass_rxn not in opt_model.reactions):
                        flux_matrix[:, i] = 0
                        continue

                    # Set product constraint with tolerance
                    if abs(product_flux) > min_absolute_tolerance:
                        product_tolerance = abs(product_flux) * constraint_tolerance
                    else:
                        product_tolerance = min_absolute_tolerance

                    product_lower = max(0, product_flux - product_tolerance)
                    product_upper = product_flux + product_tolerance

                    product_rxn = opt_model.reactions.get_by_id(self.original_product_rxn)
                    product_rxn.bounds = (product_lower, product_upper)

                    # Set biomass constraint with tolerance
                    if abs(biomass_flux) > min_absolute_tolerance:
                        biomass_tolerance = abs(biomass_flux) * constraint_tolerance
                    else:
                        biomass_tolerance = min_absolute_tolerance

                    biomass_lower = max(0, biomass_flux - biomass_tolerance)
                    biomass_upper = biomass_flux + biomass_tolerance

                    biomass_rxn = opt_model.reactions.get_by_id(self.biomass_rxn)
                    biomass_rxn.bounds = (biomass_lower, biomass_upper)

                    # Set objective
                    if virtual_objective in opt_model.reactions:
                        opt_model.objective = virtual_objective
                    else:
                        opt_model.objective = self.biomass_rxn

                    # Optimize
                    solution = opt_model.optimize()

                    if solution.status != 'optimal':
                        print(f"    Optimization failed: {solution.status}")
                        flux_matrix[:, i] = 0
                        continue

                    # Record fluxes (excluding virtual reactions)
                    for j, rxn_id in enumerate(original_reaction_ids):
                        if rxn_id in solution.fluxes:
                            flux_matrix[j, i] = solution.fluxes[rxn_id]

                    final_product = solution.fluxes.get(self.original_product_rxn, 0)
                    final_biomass = solution.fluxes.get(self.biomass_rxn, 0)

                    product_deviation = abs(final_product - product_flux) / abs(product_flux) * 100 if abs(
                        product_flux) > 1e-6 else 0
                    biomass_deviation = abs(final_biomass - biomass_flux) / abs(biomass_flux) * 100 if abs(
                        biomass_flux) > 1e-6 else 0

                    print(f"    Result: product={final_product:.4f} (dev {product_deviation:.2f}%), "
                          f"biomass={final_biomass:.4f} (dev {biomass_deviation:.2f}%)")

            except Exception as e:
                print(f"    Error at point {i + 1}: {e}")
                flux_matrix[:, i] = 0

        print("Batch optimization complete")
        return flux_matrix

    def _process_flux_matrix(self, flux_matrix):
        """Process flux matrix from optimization."""
        original_reaction_ids = [r.id for r in self.model.reactions
                                 if not r.id.startswith('VIRTUAL_')]

        # Create flux matrix DataFrame
        self.flux_matrix = pd.DataFrame(
            flux_matrix,
            index=original_reaction_ids,
            columns=[f"alpha_{i}" for i in range(flux_matrix.shape[1])]
        )

        print(f"Flux matrix dimensions: {self.flux_matrix.shape}")

        # Filter valid reactions
        with_gene_rules = [rxn_id for rxn_id in self.flux_matrix.index
                           if rxn_id in self.model.reactions and
                           self.model.reactions.get_by_id(rxn_id).gene_reaction_rule and
                           'standard' not in self.model.reactions.get_by_id(rxn_id).gene_reaction_rule]

        non_zero_flux = [rxn_id for rxn_id in with_gene_rules
                         if not np.all(np.abs(self.flux_matrix.loc[rxn_id].values) < 1e-6)]

        self.flux_matrix = self.flux_matrix.loc[non_zero_flux]
        print(f"Filtered reactions: {len(self.flux_matrix)}")

    def _analyze_flux_trends(self):
        """分析通量趋势和计算V-scores"""
        print("计算标准V_Score通量评分...")

        if self.flux_matrix is None or self.flux_matrix.empty:
            raise ValueError("Flux matrix not available")

        biomax_fluxes = self.flux_matrix.iloc[:, 0]

        target_rxns = []
        v_scores = []
        target_types = []
        has_kcat_mw_list = []

        for rxn_id in self.flux_matrix.index:
            try:
                rxn = self.model.reactions.get_by_id(rxn_id)

                # 使用old算法的kcat_MW检测方法
                has_kcat_mw = self._has_valid_kcat_mw_old_style(rxn)

                # 计算V-score的其余逻辑保持不变...
                v_biomax = abs(biomax_fluxes[rxn_id])
                flux_values = np.abs(self.flux_matrix.loc[rxn_id].values)
                avg_flux = np.mean(flux_values)

                if v_biomax < 1e-12:
                    if avg_flux > 1e-12:
                        v_score = 1000
                    else:
                        continue
                else:
                    v_score = avg_flux / v_biomax
                    if v_score > 1000:
                        v_score = 1000

                if v_score > 1.0:
                    target_type = 'OE'
                elif v_score < 1.0 and v_score > 0.05:
                    target_type = 'KD'
                else:
                    continue

                target_rxns.append(rxn_id)
                v_scores.append(v_score)
                target_types.append(target_type)
                has_kcat_mw_list.append(has_kcat_mw)

            except Exception as e:
                print(f"Error processing reaction {rxn_id}: {e}")
                continue

        # 创建reaction_scores DataFrame
        self.reaction_scores = pd.DataFrame({
            'v_score': v_scores,
            'type': target_types,
            'has_kcat_mw': has_kcat_mw_list
        }, index=target_rxns)

        if len(self.reaction_scores) > 0:
            self.reaction_scores = self.reaction_scores.sort_values('v_score', ascending=False)

        print(f"V-Score calculation complete, found {len(self.reaction_scores)} candidate reactions")

        # 安全地检查has_kcat_mw列
        if 'has_kcat_mw' in self.reaction_scores.columns:
            print(f"  With kcat_MW: {len(self.reaction_scores[self.reaction_scores['has_kcat_mw']])}")
            print(f"  Without kcat_MW: {len(self.reaction_scores[~self.reaction_scores['has_kcat_mw']])}")
        else:
            print("  Warning: has_kcat_mw column not found")

        self._calculate_gene_scores()
        self._classify_genes()

    def _calculate_gene_scores(self):
        """Calculate K-scores for genes."""
        if self.reaction_scores is None or self.reaction_scores.empty:
            raise ValueError("No V-scores available")

        print("Calculating K-scores for genes...")

        gene_scores = {}
        gene_reaction_counts = {}
        gene_reaction_types = {}
        gene_has_kcat_mw = {}

        for rxn_id, score_data in self.reaction_scores.iterrows():
            try:
                rxn = self.model.reactions.get_by_id(rxn_id)

                if len(rxn.genes) == 0:
                    continue

                v_score = score_data['v_score']
                reaction_type = score_data['type']
                has_kcat_mw = score_data.get('has_kcat_mw', False)

                for gene in rxn.genes:
                    gene_id = gene.id

                    if gene_id in gene_scores:
                        gene_scores[gene_id] += v_score
                        gene_reaction_counts[gene_id] += 1
                        if has_kcat_mw:
                            gene_has_kcat_mw[gene_id] = True
                    else:
                        gene_scores[gene_id] = v_score
                        gene_reaction_counts[gene_id] = 1
                        gene_reaction_types[gene_id] = {}
                        gene_has_kcat_mw[gene_id] = has_kcat_mw

                    if reaction_type not in gene_reaction_types[gene_id]:
                        gene_reaction_types[gene_id][reaction_type] = 0
                    gene_reaction_types[gene_id][reaction_type] += 1

            except Exception as e:
                print(f"Error processing gene for reaction {rxn_id}: {e}")
                continue

        # Calculate K-scores and determine types
        k_scores_data = {}
        self.bidirectional_genes = []

        for gene_id in gene_scores:
            k_score = gene_scores[gene_id] / gene_reaction_counts[gene_id]

            reaction_types = gene_reaction_types[gene_id]
            has_oe = 'OE' in reaction_types
            has_kd = 'KD' in reaction_types

            if has_oe and has_kd:
                self.bidirectional_genes.append(gene_id)
                gene_type = 'BOTH'
            elif k_score > 1.0:
                gene_type = 'OE'
            elif k_score < 1.0:
                gene_type = 'KD'
            else:
                gene_type = 'NEUTRAL'

            k_scores_data[gene_id] = {
                'k_score': k_score,
                'v_score_sum': gene_scores[gene_id],
                'reaction_count': gene_reaction_counts[gene_id],
                'type': gene_type,
                'reaction_types': reaction_types,
                'is_bidirectional': gene_id in self.bidirectional_genes,
                'has_kcat_mw': gene_has_kcat_mw.get(gene_id, False)
            }

        self.gene_scores = pd.DataFrame(k_scores_data).T

        if len(self.gene_scores) > 0:
            self.gene_scores = self.gene_scores.sort_values('k_score', ascending=False)

        print(f"K-scores calculated for {len(self.gene_scores)} genes")
        print(f"  Bidirectional genes: {len(self.bidirectional_genes)}")

    def _classify_genes(self):
        """Classify genes based on K-scores."""
        if self.gene_scores is None or self.gene_scores.empty:
            raise ValueError("No K-scores available")

        print("Classifying genes based on K-scores...")

        self.targets = {'OE': [], 'KD': []}

        for gene_id, gene_data in self.gene_scores.iterrows():
            gene_type = gene_data['type']
            is_bidirectional = gene_data.get('is_bidirectional', False)

            if gene_type == 'BOTH' or is_bidirectional:
                self.targets['OE'].append(gene_id)
                self.targets['KD'].append(gene_id)
            elif gene_type == 'OE':
                self.targets['OE'].append(gene_id)
            elif gene_type == 'KD':
                self.targets['KD'].append(gene_id)

        for target_type in self.targets:
            self.targets[target_type] = list(set(self.targets[target_type]))

        print(f"Gene classification complete:")
        print(f"  OE: {len(self.targets['OE'])} genes")
        print(f"  KD: {len(self.targets['KD'])} genes")

        return self.targets

    def _apply_filtering(self):
        """Apply filtering to select top targets."""
        if self.gene_scores is None or len(self.gene_scores) == 0:
            print("No candidate genes found")
            if not hasattr(self, 'targets'):
                self.targets = {'OE': [], 'KD': []}
            return

        # Filter out genes without kcat_MW
        genes_with_kcat = self.gene_scores[self.gene_scores['has_kcat_mw'] == True]
        genes_without_kcat = self.gene_scores[self.gene_scores['has_kcat_mw'] == False]

        if len(genes_without_kcat) > 0:
            print(f"Removing {len(genes_without_kcat)} genes without kcat_MW support")

        self.gene_scores = genes_with_kcat

        if len(self.gene_scores) == 0:
            print("No genes with kcat_MW support found")
            self.targets = {'OE': [], 'KD': []}
            return

        print(f"Using filtering strategy: {self.filtering_strategy}")

        if not self.separate_oe_kd_filtering:
            self._apply_overall_filtering()
        else:
            self._apply_separate_filtering()

    def _apply_overall_filtering(self):
        """Apply overall filtering."""
        genes_with_kcat = self.gene_scores[self.gene_scores['has_kcat_mw'] == True]

        both_genes = genes_with_kcat[genes_with_kcat['type'] == 'BOTH']
        normal_genes = genes_with_kcat[genes_with_kcat['type'] != 'BOTH']

        total_to_keep = max(1, int(len(genes_with_kcat) * self.top_percentage))
        normal_to_keep = max(0, total_to_keep - len(both_genes))

        print(f"Overall filtering: keeping {total_to_keep} genes")
        print(f"  Bidirectional: {len(both_genes)} (all kept)")
        print(f"  Normal genes: {normal_to_keep} from {len(normal_genes)}")

        # Filter normal genes
        if normal_to_keep > 0 and len(normal_genes) > 0:
            if self.filtering_strategy == "include_ties":
                sorted_normal = normal_genes.sort_values('k_score', ascending=False)
                if normal_to_keep <= len(sorted_normal):
                    cutoff_score = sorted_normal.iloc[normal_to_keep - 1]['k_score']
                    filtered_normal = sorted_normal[sorted_normal['k_score'] >= cutoff_score]
                else:
                    filtered_normal = sorted_normal
            else:
                filtered_normal = normal_genes.sort_values('k_score', ascending=False).head(normal_to_keep)
        else:
            filtered_normal = pd.DataFrame()

        # Combine results
        if not both_genes.empty and not filtered_normal.empty:
            filtered_genes = pd.concat([both_genes, filtered_normal])
        elif not both_genes.empty:
            filtered_genes = both_genes
        elif not filtered_normal.empty:
            filtered_genes = filtered_normal
        else:
            filtered_genes = pd.DataFrame()

        if not filtered_genes.empty:
            filtered_genes = filtered_genes.sort_values('k_score', ascending=False)

            self.targets = {'OE': [], 'KD': []}
            for gene_id, gene_data in filtered_genes.iterrows():
                gene_type = gene_data['type']
                if gene_type == 'BOTH':
                    self.targets['OE'].append(gene_id)
                    self.targets['KD'].append(gene_id)
                elif gene_type == 'OE':
                    self.targets['OE'].append(gene_id)
                elif gene_type == 'KD':
                    self.targets['KD'].append(gene_id)

            print(f"Filtering results:")
            print(f"  OE: {len(self.targets['OE'])} genes")
            print(f"  KD: {len(self.targets['KD'])} genes")
        else:
            self.targets = {'OE': [], 'KD': []}
            print("No genes passed filtering")

    def _apply_separate_filtering(self):
        """Apply separate filtering for OE and KD."""
        print(f"Separate filtering: OE top {self.oe_percentage * 100:.1f}%, KD top {self.kd_percentage * 100:.1f}%")

        genes_with_kcat = self.gene_scores[self.gene_scores['has_kcat_mw'] == True]

        oe_only = genes_with_kcat[genes_with_kcat['type'] == 'OE']
        kd_only = genes_with_kcat[genes_with_kcat['type'] == 'KD']
        both_genes = genes_with_kcat[genes_with_kcat['type'] == 'BOTH']

        selected_genes = []

        # Add all bidirectional genes
        if not both_genes.empty:
            selected_genes.append(both_genes)
            print(f"  Bidirectional: {len(both_genes)} genes (all kept)")

        # Filter OE genes
        if len(oe_only) > 0:
            selected_oe = self._filter_by_percentage(oe_only, self.oe_percentage, "OE")
            if not selected_oe.empty:
                selected_genes.append(selected_oe)

        # Filter KD genes
        if len(kd_only) > 0:
            selected_kd = self._filter_by_percentage(kd_only, self.kd_percentage, "KD")
            if not selected_kd.empty:
                selected_genes.append(selected_kd)

        # Combine results
        if selected_genes:
            filtered_gene_scores = pd.concat(selected_genes, axis=0)

            self.targets = {'OE': [], 'KD': []}
            for gene_id, gene_data in filtered_gene_scores.iterrows():
                gene_type = gene_data['type']
                if gene_type == 'BOTH':
                    self.targets['OE'].append(gene_id)
                    self.targets['KD'].append(gene_id)
                elif gene_type == 'OE':
                    self.targets['OE'].append(gene_id)
                elif gene_type == 'KD':
                    self.targets['KD'].append(gene_id)

            self.gene_scores = filtered_gene_scores

            print(f"Final results:")
            print(f"  OE: {len(self.targets['OE'])} genes")
            print(f"  KD: {len(self.targets['KD'])} genes")
        else:
            self.targets = {'OE': [], 'KD': []}
            self.gene_scores = pd.DataFrame()
            print("No genes passed filtering")

    def _filter_by_percentage(self, genes_df, percentage, category_name):
        """Filter genes by percentage."""
        n_to_keep = max(1, int(len(genes_df) * percentage))

        # Sort direction based on type
        if 'KD' in category_name or (len(genes_df) > 0 and genes_df['type'].iloc[0] == 'KD'):
            sorted_genes = genes_df.sort_values('k_score', ascending=True)
            ascending = True
        else:
            sorted_genes = genes_df.sort_values('k_score', ascending=False)
            ascending = False

        if self.filtering_strategy == "include_ties":
            cutoff_score = sorted_genes.iloc[n_to_keep - 1]['k_score']
            if ascending:
                selected = sorted_genes[sorted_genes['k_score'] <= cutoff_score]
            else:
                selected = sorted_genes[sorted_genes['k_score'] >= cutoff_score]

            same_score_count = len(sorted_genes[sorted_genes['k_score'] == cutoff_score])
            print(f"  {category_name}: {len(genes_df)} → {len(selected)} (cutoff: {cutoff_score:.3f})")
            if same_score_count > 1:
                print(f"    Including {same_score_count} genes with same score")
        else:
            selected = sorted_genes.head(n_to_keep)
            print(f"  {category_name}: {len(genes_df)} → {len(selected)} (strict top {percentage * 100:.1f}%)")

        return selected

    # Additional private helper methods for other functionality

    def _is_export_reaction(self, reaction):
        """Check if reaction is an export reaction."""
        reactants = []
        products = []

        for met, coeff in reaction.metabolites.items():
            if coeff < 0:
                reactants.append((met, abs(coeff)))
            elif coeff > 0:
                products.append((met, coeff))

        # Check biomass reaction
        biomass_products = [met for met, coeff in reactants if 'biomass' in met.id.lower() and met.id.endswith('_c')]
        if len(biomass_products) >= 1:
            return True

        # Check pure export
        if len(reactants) >= 1 and len(products) == 0:
            extracellular = [met for met, coeff in reactants if met.id.endswith('_e')]
            if len(extracellular) >= 1:
                return True

        # Check carrier-based export
        if len(reactants) >= 1 and len(products) <= 2:
            extracellular = [met for met, coeff in reactants if met.id.endswith('_e')]
            if len(extracellular) >= 1:
                carrier_molecules = {'cpd00067_c', 'cpd00001_c', 'cpd00009_c', 'cpd00010_c'}
                product_mets = [met for met, coeff in products]
                carrier_products = [met for met in product_mets if met.id in carrier_molecules]

                if len(carrier_products) >= len(product_mets) * 0.5:
                    return True

        return False

    def _analyze_single_leak_reaction(self, rxn, cytosolic_id, extracellular_id):
        """Analyze a single leak reaction."""
        reaction_info = {
            'id': rxn.id,
            'name': rxn.name if hasattr(rxn, 'name') and rxn.name else 'No name',
            'equation': str(rxn.reaction),
            'subsystem': rxn.subsystem if hasattr(rxn, 'subsystem') else 'Unknown',
            'genes': [gene.id for gene in rxn.genes],
            'has_kcat_MW': self._has_valid_kcat_mw_old_style(rxn),
            'category': 'unknown'
        }

        # Skip exchange and biomass reactions
        if rxn.id.startswith("EX_") or rxn.id.startswith("Bio_") or "biomass" in rxn.id.lower():
            reaction_info['category'] = 'skip'
            return reaction_info

        # Check if transport reaction
        is_transport = self._is_transport_reaction(rxn, cytosolic_id, extracellular_id)

        if is_transport:
            if reaction_info['has_kcat_MW']:
                print(f"🚛 Transport reaction (with kcat_MW): {rxn.id} - add to OE")
                reaction_info['category'] = 'transport_OE'
            else:
                print(f"🚛 Transport reaction (no kcat_MW): {rxn.id} - skip")
                reaction_info['category'] = 'skip'
        else:
            print(f"💧 Consumption reaction: {rxn.id} - add to KD")
            reaction_info['category'] = 'consumption_KD'

        return reaction_info

    def _is_transport_reaction(self, rxn, cytosolic_id, extracellular_id):
        """Check if reaction is a transport reaction."""
        base_product_id = cytosolic_id.replace('_c', '').replace('_e', '')
        met_ids = {met.id for met in rxn.metabolites.keys()}

        return f"{base_product_id}_c" in met_ids and f"{base_product_id}_e" in met_ids

    def _print_leak_summary(self):
        """Print leak reaction summary."""
        if not self.leak_reactions_info:
            print("No leak reactions found")
            return

        print("\n" + "=" * 80)
        print("Leak Reaction Details:")
        print("=" * 80)

        transport = [r for r in self.leak_reactions_info if r['category'] == 'transport_OE']
        consumption = [r for r in self.leak_reactions_info if r['category'] == 'consumption_KD']

        if transport:
            print(f"\n🚛 Transport reactions (OE targets, {len(transport)}):")
            for i, reaction in enumerate(transport[:5], 1):
                print(f"{i}. {reaction['id']}")
                print(f"   Equation: {reaction['equation']}")
                print(f"   Genes: {', '.join(reaction['genes']) if reaction['genes'] else 'None'}")

        if consumption:
            print(f"\n💧 Consumption reactions (KD targets, {len(consumption)}):")
            for i, reaction in enumerate(consumption[:5], 1):
                print(f"{i}. {reaction['id']}")
                print(f"   Equation: {reaction['equation']}")
                print(f"   Genes: {', '.join(reaction['genes']) if reaction['genes'] else 'None'}")

    # Continue with remaining helper methods...
    # (Due to space, I'll include key methods)

    def _get_enzyme_reactions(self, target_genes):
        """Get enzyme reactions for target genes."""
        enzyme_reactions = []
        for gene_id in target_genes:
            try:
                gene_obj = self.model.genes.get_by_id(gene_id)
                for reaction in gene_obj.reactions:
                    if hasattr(reaction, 'enzyme_reaction'):
                        enzyme_reactions.append(reaction.enzyme_reaction)
                    else:
                        enzyme_reactions.append(reaction.id)
            except KeyError:
                print(f"Warning: Gene {gene_id} not found in model")
                continue

        return list(set(enzyme_reactions))

    def _run_scenario_fva(self, product_rate, enzyme_reactions, scenario_name):
        """Run FVA with selective tolerance relaxation for problematic reactions."""
        tolerance_levels = [0.01, 0.03, 0.05, 0.08]

        # Initialize result dataframe
        final_result = pd.DataFrame()
        remaining_reactions = enzyme_reactions.copy()
        tolerance_used = {}  # Track which tolerance was used for each reaction

        for tol_factor in tolerance_levels:
            if not remaining_reactions:
                break

            tolerance = product_rate * tol_factor
            lower = product_rate - tolerance
            upper = product_rate

            with self.model as scenario_model:
                scenario_model.reactions.get_by_id(self.product_rxn).bounds = (lower, upper)
                scenario_model.objective = self.biomass_rxn
                solution = scenario_model.optimize()

                if solution.status != 'optimal':
                    if tol_factor < tolerance_levels[-1]:
                        print(
                            f"  {scenario_name}: Optimization failed with {tol_factor * 100:.1f}% tolerance, retrying all...")
                        continue
                    else:
                        print(f"  {scenario_name}: Optimization failed even with max tolerance")
                        return pd.DataFrame()

                # Run FVA on remaining reactions
                fva_result = flux_variability_analysis(
                    scenario_model, reaction_list=remaining_reactions,
                    fraction_of_optimum=1, processes=1
                )

                # Separate successful and NaN reactions
                valid_mask = ~(fva_result['minimum'].isna() | fva_result['maximum'].isna())
                valid_results = fva_result[valid_mask]
                nan_reactions = fva_result[~valid_mask].index.tolist()

                # Store valid results
                if not valid_results.empty:
                    if final_result.empty:
                        final_result = valid_results
                    else:
                        final_result = pd.concat([final_result, valid_results])

                    # Track tolerance used
                    for rxn in valid_results.index:
                        tolerance_used[rxn] = tol_factor

                # Report progress
                if nan_reactions and tol_factor < tolerance_levels[-1]:
                    print(
                        f"  {scenario_name}: Found {len(nan_reactions)} NaN reactions with {tol_factor * 100:.1f}% tolerance, will retry them...")
                    remaining_reactions = nan_reactions
                elif nan_reactions:
                    print(f"  {scenario_name}: {len(nan_reactions)} reactions still NaN with max tolerance")
                    # Add NaN reactions to final result to maintain complete index
                    for rxn in nan_reactions:
                        final_result.loc[rxn] = {'minimum': np.nan, 'maximum': np.nan}
                        tolerance_used[rxn] = tol_factor
                else:
                    if tol_factor > 0.01 and len(tolerance_used) > 0:
                        relaxed_count = sum(1 for t in tolerance_used.values() if t > 0.01)
                        print(
                            f"  {scenario_name}: Success! {relaxed_count}/{len(tolerance_used)} reactions needed relaxed tolerance")
                    remaining_reactions = []

        return final_result

    def _filter_by_enzyme_usage(self, target_genes, leak_genes_set, bidirectional_genes_set):
        """Filter targets by enzyme usage patterns."""
        enzyme_comparison = {}

        for gene_id in target_genes:
            if gene_id in leak_genes_set:
                enzyme_comparison[gene_id] = "leak_gene_skip"
                continue

            comparison = self.compare_enzyme_usage(gene_id)
            enzyme_comparison[gene_id] = comparison

            if gene_id in bidirectional_genes_set:
                print(f"  Bidirectional gene {gene_id}: enzyme usage={comparison}")

        # Apply filtering rules
        filtered_targets = {'OE': [], 'KD': []}

        # Filter OE targets
        for gene in self.targets.get('OE', []):
            comparison = enzyme_comparison.get(gene, "none")
            if comparison in ["up_consistent", "mixed", "leak_gene_skip"]:
                filtered_targets['OE'].append(gene)
                if comparison == "leak_gene_skip":
                    print(f"Keep OE target {gene}: leak gene")
                else:
                    print(f"Keep OE target {gene}: {comparison}")
            else:
                print(f"Remove OE target {gene}: {comparison}")

        # Filter KD targets
        for gene in self.targets.get('KD', []):
            comparison = enzyme_comparison.get(gene, "none")
            if comparison in ["down_consistent", "mixed", "leak_gene_skip"] or self.is_enzyme_unused(gene):
                filtered_targets['KD'].append(gene)
                if comparison == "leak_gene_skip":
                    print(f"Keep KD target {gene}: leak gene")
                elif self.is_enzyme_unused(gene):
                    print(f"Keep KD target {gene}: enzyme unused")
                else:
                    print(f"Keep KD target {gene}: {comparison}")
            else:
                print(f"Remove KD target {gene}: {comparison}")

        return filtered_targets

    def _report_euva_results(self, filtered_targets, leak_genes_set, bidirectional_genes_set):
        """Report EUVA filtering results."""
        print("\nEUVA filtering results:")
        for target_type in ['OE', 'KD']:
            if target_type in self.targets:
                original = len(self.targets[target_type])
                filtered = len(filtered_targets[target_type])
                leak_count = len([g for g in filtered_targets[target_type] if g in leak_genes_set])

                bidirectional_original = len([g for g in self.targets[target_type] if g in bidirectional_genes_set])
                bidirectional_retained = len([g for g in filtered_targets[target_type] if g in bidirectional_genes_set])
                bidirectional_filtered = bidirectional_original - bidirectional_retained

                print(
                    f"{target_type}: {original} → {filtered} (retention: {filtered / original * 100:.1f}% if original > 0 else 0)")
                print(f"  - Leak genes: {leak_count} (all retained)")
                print(
                    f"  - Bidirectional: {bidirectional_original} → {bidirectional_retained} (filtered: {bidirectional_filtered})")

    # Essential genes methods

    def _load_or_identify_essential_genes(self, essential_genes_file=None, force_reidentify=False):
        """Load or identify essential genes with caching."""
        essential_genes = []

        if not force_reidentify:
            cached_genes = self._load_essential_genes_cache()
            if cached_genes is not None:
                essential_genes.extend(cached_genes)

                if essential_genes_file:
                    file_genes = self._load_essential_genes(essential_genes_file)
                    if file_genes:
                        original_count = len(essential_genes)
                        essential_genes = list(set(essential_genes + file_genes))
                        added_count = len(essential_genes) - original_count
                        print(f"Merged {added_count} additional essential genes from file")
                        self._save_essential_genes_cache(essential_genes)

                return essential_genes

        print("Identifying essential genes...")
        identified_genes = self._identify_essential_genes()

        if essential_genes_file:
            file_genes = self._load_essential_genes(essential_genes_file)
            if file_genes:
                identified_genes = list(set(identified_genes + file_genes))

        self._save_essential_genes_cache(identified_genes)
        return identified_genes

    def _identify_essential_genes(self, growth_fraction=0.001):
        """Identify essential genes through knockout simulation."""
        print("Running in silico gene knockout...")
        essential_genes = []

        original_objective = self.model.objective
        self.model.objective = self.biomass_rxn

        try:
            solution = self.model.optimize()
            wild_type_growth = solution.objective_value

            if wild_type_growth <= 0:
                print("Warning: Wild-type model shows no growth")
                return []

            essential_threshold = wild_type_growth * growth_fraction
            print(f"Wild-type growth: {wild_type_growth}")
            print(f"Essential threshold: {essential_threshold}")

            total_genes = len(self.model.genes)
            print(f"Testing {total_genes} genes...")

            for i, gene in enumerate(self.model.genes):
                if i % 100 == 0 and i > 0:
                    print(f"Progress: {i}/{total_genes}, found {len(essential_genes)} essential")

                with self.model as knockout_model:
                    gene.knock_out()

                    try:
                        ko_solution = knockout_model.optimize()
                        ko_growth = ko_solution.objective_value

                        if ko_growth < essential_threshold:
                            essential_genes.append(gene.id)
                    except:
                        essential_genes.append(gene.id)

        except Exception as e:
            print(f"Error during essential gene identification: {e}")
        finally:
            self.model.objective = original_objective

        print(f"Identified {len(essential_genes)} essential genes")
        return essential_genes

    def _load_essential_genes(self, file_path=None):
        """Load essential genes from file."""
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    return [line.strip() for line in f.readlines()]
            except Exception as e:
                print(f"Error loading essential genes: {e}")
        return []

    def _load_enzyme_data(self, file_path=None):
        """Load enzyme kinetics data."""
        if file_path and os.path.exists(file_path):
            try:
                return pd.read_csv(file_path)
            except Exception as e:
                print(f"Error loading enzyme data: {e}")
        return pd.DataFrame()

    def _check_enzyme_constraints(self):
        """Check if model has enzyme constraints."""
        if hasattr(self.model, 'constraints'):
            return True

        if hasattr(self.model, 'json') and isinstance(self.model.json, dict):
            if 'enzyme_constraint' in self.model.json:
                return True

        for attr_name in ['_enzyme_constraint', 'ecm_constraints', 'protein_constraint']:
            if hasattr(self.model, attr_name):
                return True

        try:
            if 'enzyme_constraint' in self.model.constraints:
                return True
        except AttributeError:
            pass

        return False

    def _generate_model_signature(self):
        """Generate unique model signature."""
        model_info = {
            'reactions': sorted([r.id for r in self.model.reactions]),
            'metabolites': sorted([m.id for m in self.model.metabolites]),
            'genes': sorted([g.id for g in self.model.genes]),
            'model_id': self.model.id,
            'biomass_rxn': self.biomass_rxn
        }

        model_str = json.dumps(model_info, sort_keys=True)
        signature = hashlib.md5(model_str.encode()).hexdigest()
        return signature

    def _get_essential_genes_cache_path(self):
        """Get essential genes cache file path."""
        model_signature = self._generate_model_signature()
        cache_filename = f"essential_genes_{model_signature}.json"
        return os.path.join(self.essential_genes_cache_dir, cache_filename)

    def _save_essential_genes_cache(self, essential_genes):
        """Save essential genes to cache."""
        cache_path = self._get_essential_genes_cache_path()

        cache_data = {
            'essential_genes': essential_genes,
            'model_id': self.model.id,
            'biomass_rxn': self.biomass_rxn,
            'total_genes': len(self.model.genes),
            'total_reactions': len(self.model.reactions),
            'identification_date': datetime.now().isoformat(),
            'model_signature': self._generate_model_signature()
        }

        try:
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
            print(f"Essential genes cached: {cache_path}")
        except Exception as e:
            print(f"Warning: Failed to cache essential genes: {e}")

    def _load_essential_genes_cache(self):
        """Load essential genes from cache."""
        cache_path = self._get_essential_genes_cache_path()

        if not os.path.exists(cache_path):
            return None

        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)

            if cache_data.get('model_signature') != self._generate_model_signature():
                print("Warning: Model signature mismatch")
                return None

            essential_genes = cache_data.get('essential_genes', [])
            print(f"Loaded {len(essential_genes)} essential genes from cache")
            return essential_genes

        except Exception as e:
            print(f"Warning: Failed to load cache: {e}")
            return None

    def clear_essential_genes_cache(self):
        """Clear essential genes cache."""
        cache_path = self._get_essential_genes_cache_path()
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
                print(f"Cache cleared: {cache_path}")
            except Exception as e:
                print(f"Error clearing cache: {e}")
        else:
            print("No cache file found")

    # Additional helper methods for other analyses

    def _get_flux_carrying_reactions(self):
        """Get reactions with non-zero flux."""
        print("Optimizing for product to get flux-carrying reactions...")

        with self.model as m:
            m.objective = self.original_product_rxn
            solution = m.optimize()

            if solution.status != 'optimal':
                print(f"Warning: Optimization failed with status: {solution.status}")
                return set()

            flux_carrying = set()
            for rxn_id, flux_value in solution.fluxes.items():
                if abs(flux_value) > 1e-6:
                    flux_carrying.add(rxn_id)

            print(f"Found {len(flux_carrying)} flux-carrying reactions")
            return flux_carrying

    def _extract_target_metabolites(self):
        """Extract target product metabolites."""
        print("Extracting target product metabolites...")

        try:
            product_reaction = self.model.reactions.get_by_id(self.original_product_rxn)
        except KeyError:
            print(f"Error: Cannot find {self.original_product_rxn}")
            return set()

        target_products = set()

        # Check for products (positive coefficients)
        for met, coeff in product_reaction.metabolites.items():
            if coeff > 0:
                target_products.add(met.id)
                print(f"Target product (coeff>0): {met.id}")

        # If none found, check export reactions
        if not target_products:
            print("No products found, checking export reaction...")
            for met, coeff in product_reaction.metabolites.items():
                if coeff < 0:
                    target_products.add(met.id)
                    print(f"Target product (export, coeff<0): {met.id}")

        # Check for cytosolic versions
        if not target_products:
            print("Checking for cytosolic versions...")
            for met, coeff in product_reaction.metabolites.items():
                if coeff < 0 and met.id.endswith('_e'):
                    cytosolic_id = met.id.replace('_e', '_c')
                    try:
                        self.model.metabolites.get_by_id(cytosolic_id)
                        target_products.add(cytosolic_id)
                        print(f"Found cytosolic version: {cytosolic_id}")
                    except KeyError:
                        pass

        return target_products

    def _trace_precursors(self, target_products, flux_carrying_reactions,
                          excluded_reactions, common_cofactors,
                          special_reaction_whitelist, max_depth):
        """Trace metabolic precursors."""
        print(f"Tracing precursors from target products (max_depth={max_depth})...")

        all_precursors = set(target_products)
        network_reactions = {self.original_product_rxn}
        current_level_metabolites = target_products.copy()

        for depth in range(1, max_depth + 1):
            next_level_metabolites = set()
            level_reactions = set()

            for metabolite_id in current_level_metabolites:
                try:
                    metabolite = self.model.metabolites.get_by_id(metabolite_id)

                    # Find producing reactions
                    producing_reactions = []
                    for rxn in metabolite.reactions:
                        if (rxn.id != self.original_product_rxn and
                                rxn.id in flux_carrying_reactions and
                                rxn.metabolites[metabolite] > 0):

                            if rxn.id in excluded_reactions:
                                continue
                            producing_reactions.append(rxn)

                    for rxn in producing_reactions:
                        level_reactions.add(rxn.id)

                        # Check if transamination reaction
                        if self._is_transamination_reaction(rxn):
                            print(f"  Found transamination in synthesis: {rxn.id}")

                        # Identify precursors
                        for met, coeff in rxn.metabolites.items():
                            if coeff < 0 and met.id not in all_precursors:

                                should_exclude = met.id in common_cofactors

                                # Check whitelist
                                if should_exclude and rxn.id in special_reaction_whitelist:
                                    allowed_mets = special_reaction_whitelist[rxn.id]
                                    if met.id in allowed_mets:
                                        should_exclude = False
                                        print(f"  Whitelist: {met.id} in {rxn.id}")

                                if not should_exclude:
                                    next_level_metabolites.add(met.id)

                except KeyError:
                    continue

            all_precursors.update(next_level_metabolites)
            network_reactions.update(level_reactions)
            current_level_metabolites = next_level_metabolites

            if not next_level_metabolites:
                print(f"  Tracing stopped at depth {depth}")
                break

        print(f"Total precursors traced: {len(all_precursors)}")
        print(f"Network reactions: {len(network_reactions)}")

        return all_precursors, network_reactions

    def _is_transamination_reaction(self, reaction):
        """Check if reaction is transamination."""
        transamination_metabolites = {
            'cpd00023_c', 'cpd00053_c', 'cpd00024_c',
            'cpd00041_c', 'cpd00032_c', 'cpd00013_c', 'cpd00013_e',
        }

        reaction_metabolites = {met.id for met in reaction.metabolites.keys()}

        # Check classic combinations
        if ('cpd00023_c' in reaction_metabolites and 'cpd00024_c' in reaction_metabolites):
            return True

        if ('cpd00023_c' in reaction_metabolites and 'cpd00053_c' in reaction_metabolites):
            return True

        if ('cpd00041_c' in reaction_metabolites and 'cpd00032_c' in reaction_metabolites):
            return True

        # Check reaction name
        reaction_name = reaction.name.lower() if hasattr(reaction, 'name') and reaction.name else ''
        if any(keyword in reaction_name for keyword in ['transaminase', 'aminotransferase', 'transamination']):
            return True

        return False

    def _collect_relevant_reactions(self, all_precursors, network_reactions,
                                    excluded_reactions, metabolite_reaction_exclusions):
        """Collect all relevant reactions."""
        all_relevant_reactions = set(network_reactions)
        transamination_excluded = set()
        transamination_kept = set()
        excluded_by_mapping = set()

        # Mark transamination reactions in synthesis path
        for rxn_id in network_reactions:
            try:
                rxn = self.model.reactions.get_by_id(rxn_id)
                if self._is_transamination_reaction(rxn):
                    transamination_kept.add(rxn_id)
            except:
                continue

        # Collect other reactions involving precursors
        for metabolite_id in all_precursors:
            try:
                metabolite = self.model.metabolites.get_by_id(metabolite_id)
                for rxn in metabolite.reactions:
                    if rxn.id not in all_relevant_reactions and not rxn.id.startswith('Bio_'):

                        if rxn.id in excluded_reactions:
                            continue

                        # Check metabolite-reaction exclusion mapping
                        if metabolite_id in metabolite_reaction_exclusions:
                            if rxn.id in metabolite_reaction_exclusions[metabolite_id]:
                                excluded_by_mapping.add(rxn.id)
                                print(f"  Excluding mapped: {metabolite_id} -> {rxn.id}")
                                continue

                        # Exclude non-synthesis transamination
                        if self._is_transamination_reaction(rxn):
                            transamination_excluded.add(rxn.id)
                            continue

                        all_relevant_reactions.add(rxn.id)

            except KeyError:
                continue

        print(f"Final relevant reaction network: {len(all_relevant_reactions)}")
        print(f"Kept synthesis transamination: {len(transamination_kept)}")
        print(f"Excluded non-synthesis transamination: {len(transamination_excluded)}")
        print(f"Excluded by mapping rules: {len(excluded_by_mapping)}")

        return all_relevant_reactions

    def _filter_kd_by_relevance(self, all_relevant_reactions):
        """Filter KD targets by relevance to product synthesis."""
        print("Filtering KD targets...")

        filtered_targets = {
            'OE': self.targets['OE'].copy(),  # Keep all OE targets
            'KD': []
        }

        for gene_id in self.targets['KD']:
            try:
                gene = self.model.genes.get_by_id(gene_id)
                keep_gene = False

                # Check if gene participates in relevant reactions
                for reaction in gene.reactions:
                    if reaction.id in all_relevant_reactions:
                        keep_gene = True
                        print(f"Keep KD target {gene_id}: relevant to synthesis")
                        break

                if keep_gene:
                    filtered_targets['KD'].append(gene_id)
                else:
                    print(f"Remove KD target {gene_id}: not relevant to synthesis")

            except KeyError:
                continue

        print(f"\nTopology filter results:")
        print(f"OE targets: {len(self.targets['OE'])} → {len(filtered_targets['OE'])} (no filter)")
        print(f"KD targets: {len(self.targets['KD'])} → {len(filtered_targets['KD'])} "
              f"(removed {len(self.targets['KD']) - len(filtered_targets['KD'])})")

        return filtered_targets

    def _calculate_wildtype_baseline(self):
        """Calculate wild-type baseline metrics."""
        print("Calculating wild-type baseline...")

        # Max product
        with self.model as model:
            model.objective = self.original_product_rxn
            sol_prod = model.optimize()
            wt_max_product = sol_prod.objective_value
            print(f"Wild-type max product: {wt_max_product:.6f}")

        # Max growth
        with self.model as model:
            model.objective = self.biomass_rxn
            sol_growth = model.optimize()
            wt_max_growth = sol_growth.objective_value
            wt_product_at_max_growth = sol_growth.fluxes[self.original_product_rxn]
            print(f"Wild-type max growth: {wt_max_growth:.6f}")
            print(f"Product at max growth: {wt_product_at_max_growth:.6f}")

        # Baseline scenario (25% growth)
        with self.model as model:
            growth_rate = 0.25 * wt_max_growth
            model.reactions.get_by_id(self.biomass_rxn).lower_bound = growth_rate
            model.objective = self.original_product_rxn
            sol_baseline = pfba(model)
            baseline_production = sol_baseline.fluxes[self.original_product_rxn]

            cs_rxn = self.get_carbon_source_reaction()
            baseline_cs_uptake = abs(sol_baseline.fluxes.get(cs_rxn, 0))

            if baseline_cs_uptake == 0:
                baseline_cs_uptake = 1.0

        baseline_yield = baseline_production / baseline_cs_uptake if baseline_cs_uptake > 0 else 0

        print(f"Baseline growth rate (25%): {growth_rate:.6f}")
        print(f"Baseline production: {baseline_production:.6f}")
        print(f"Baseline carbon uptake: {baseline_cs_uptake:.6f}")
        print(f"Baseline yield: {baseline_yield:.6f}")

        return {
            'wt_max_product': wt_max_product,
            'wt_max_growth': wt_max_growth,
            'growth_rate': growth_rate,
            'baseline_production': baseline_production,
            'baseline_cs_uptake': baseline_cs_uptake,
            'baseline_yield': baseline_yield
        }

    def _save_original_kcat_values(self):
        """Save original kcat_MW values."""
        original_kcat_values = {}
        with self.model as model:
            for reaction in model.reactions:
                if self._has_valid_kcat_mw_old_style(reaction):
                    try:
                        original_kcat_values[reaction.id] = float(reaction.kcat_MW)
                    except:
                        pass

        print(f"Found {len(original_kcat_values)} reactions with kcat_MW values")
        return original_kcat_values

    def _evaluate_all_targets(self, wt_baseline, original_kcat_values):
        """Evaluate all targets."""
        beneficial_targets = {'OE': [], 'KD': []}
        target_metrics = []

        leak_genes_set = set(self.leak_genes['transport_OE'] + self.leak_genes['consumption_KD'])
        print(f"Including {len(leak_genes_set)} leak reaction genes")
        print(f"Bidirectional genes: {len(self.bidirectional_genes)}")

        all_targets = {'OE': self.targets['OE'], 'KD': self.targets['KD']}
        total_genes = sum(len(genes) for genes in all_targets.values())
        print(f"Evaluating {total_genes} potential targets")

        for target_type, genes in all_targets.items():
            print(f"\nEvaluating {len(genes)} {target_type} targets...")

            for i, gene in enumerate(genes):
                if i % 10 == 0 and i > 0:
                    print(f"  Progress: {i}/{len(genes)}")

                try:
                    is_leak = gene in leak_genes_set
                    is_bidirectional = gene in self.bidirectional_genes

                    modifications = {
                        'OE': [gene] if target_type == 'OE' else [],
                        'KD': [gene] if target_type == 'KD' else []
                    }

                    modified_model, solution, production = self._apply_modifications(
                        self.model, modifications, original_kcat_values
                    )

                    if solution is None:
                        print(f"  Gene {gene} ({target_type}): infeasible")
                        continue

                    # Calculate metrics
                    cs_rxn = self.get_carbon_source_reaction()
                    cs_uptake = abs(solution.fluxes.get(cs_rxn, 0))

                    if cs_uptake == 0:
                        cs_uptake = 1.0

                    yield_value = production / cs_uptake if cs_uptake > 0 else 0
                    prod_improvement = (production - wt_baseline['baseline_production']) / wt_baseline[
                        'baseline_production'] * 100
                    yield_improvement = (yield_value - wt_baseline['baseline_yield']) / wt_baseline[
                        'baseline_yield'] * 100 if wt_baseline['baseline_yield'] > 0 else 0

                    # Check if beneficial
                    if prod_improvement > 0.1 or yield_improvement > 0.1:
                        beneficial_targets[target_type].append(gene)

                        metrics = {
                            'gene': gene,
                            'type': target_type,
                            'is_leak_gene': is_leak,
                            'is_bidirectional': is_bidirectional,
                            'production': production,
                            'yield': yield_value,
                            'prod_improvement': prod_improvement,
                            'yield_improvement': yield_improvement,
                            'growth': solution.fluxes[self.biomass_rxn]
                        }

                        target_metrics.append(metrics)
                        print(f"  Gene {gene}: +{prod_improvement:.2f}% production, +{yield_improvement:.2f}% yield")
                    else:
                        print(f"  Gene {gene}: no significant improvement")

                except Exception as e:
                    print(f"  Error testing {gene}: {str(e)}")

        return beneficial_targets, target_metrics

    def _apply_modifications(self, model, modifications, original_values=None):
        """Apply gene modifications and reload model."""
        import tempfile
        import cobra
        from cobra.flux_analysis import pfba

        OEF = 5.0  # Overexpression factor
        KDF = 0.2  # Knockdown factor

        modified_kcat_mw_data = {}
        temp_filename = None

        try:
            orig_values = original_values if original_values else {}

            # Apply modifications
            for mod_type, genes in modifications.items():
                for gene in genes:
                    if mod_type in ['OE', 'KD']:
                        gene_reactions = model.genes.get_by_id(gene).reactions
                        for reaction in gene_reactions:
                            if self._has_valid_kcat_mw_old_style(reaction):
                                try:
                                    original_kcat = orig_values.get(reaction.id, float(reaction.kcat_MW))
                                    if mod_type == 'OE':
                                        new_kcat = original_kcat * OEF
                                    else:
                                        new_kcat = original_kcat * KDF
                                    reaction.kcat_MW = new_kcat
                                    modified_kcat_mw_data[reaction.id] = new_kcat
                                except:
                                    pass

            # Save and reload
            with tempfile.NamedTemporaryFile(suffix='.json', mode='w', delete=False) as temp_file:
                temp_filename = temp_file.name
                cobra.io.save_json_model(model, temp_filename)

            with open(temp_filename, 'r') as file:
                model_dict = json.load(file)

            # Ensure enzyme constraint
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
            updated_model.objective = self.original_product_rxn

            try:
                solution = pfba(updated_model)
                production = solution.fluxes[self.original_product_rxn]
                return updated_model, solution, production

            except cobra.exceptions.Infeasible:
                return None, None, 0.0

            except Exception:
                return None, None, 0.0

        except Exception:
            return None, None, 0.0

        finally:
            if temp_filename and os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                except:
                    pass

    def _resolve_dual_classifications(self, results_df, beneficial_targets):
        """Resolve genes that appear in both OE and KD."""
        overlap_genes = set(beneficial_targets['OE']) & set(beneficial_targets['KD'])

        if overlap_genes:
            print(f"\nResolving {len(overlap_genes)} dual-classification conflicts...")

            for gene in overlap_genes:
                oe_result = results_df[(results_df['gene'] == gene) & (results_df['type'] == 'OE')]
                kd_result = results_df[(results_df['gene'] == gene) & (results_df['type'] == 'KD')]

                if not oe_result.empty and not kd_result.empty:
                    oe_prod = oe_result.iloc[0]['prod_improvement']
                    kd_prod = kd_result.iloc[0]['prod_improvement']
                    oe_yield = oe_result.iloc[0]['yield_improvement']
                    kd_yield = kd_result.iloc[0]['yield_improvement']

                    # Choose better modification
                    if oe_prod > kd_prod or (oe_prod == kd_prod and oe_yield >= kd_yield):
                        beneficial_targets['KD'].remove(gene)
                        results_df = results_df[~((results_df['gene'] == gene) & (results_df['type'] == 'KD'))]
                        print(f"  Gene {gene}: chose OE (prod +{oe_prod:.2f}% vs +{kd_prod:.2f}%)")
                    else:
                        beneficial_targets['OE'].remove(gene)
                        results_df = results_df[~((results_df['gene'] == gene) & (results_df['type'] == 'OE'))]
                        print(f"  Gene {gene}: chose KD (prod +{kd_prod:.2f}% vs +{oe_prod:.2f}%)")

            results_df = results_df.sort_values('prod_improvement', ascending=False)

        return results_df, beneficial_targets

    def _print_evaluation_summary(self, results_df, beneficial_targets):
        """Print evaluation summary."""
        leak_genes_set = set(self.leak_genes['transport_OE'] + self.leak_genes['consumption_KD'])

        print("\nBeneficial targets evaluation results:")
        print(f"Total targets found: {sum(len(targets) for targets in beneficial_targets.values())}")
        print(f"OE targets: {len(beneficial_targets['OE'])}")
        print(f"KD targets: {len(beneficial_targets['KD'])}")

        leak_oe = len([g for g in beneficial_targets['OE'] if g in leak_genes_set])
        leak_kd = len([g for g in beneficial_targets['KD'] if g in leak_genes_set])
        print(f"Leak genes: OE={leak_oe}, KD={leak_kd}")

        bidirectional_count = len([g for g in self.bidirectional_genes
                                   if g in beneficial_targets['OE'] or g in beneficial_targets['KD']])
        print(f"Bidirectional genes: {bidirectional_count}")

        print("\nTop 10 targets by production improvement:")
        if len(results_df) > 0:
            display_df = results_df[['gene', 'type', 'is_leak_gene', 'is_bidirectional',
                                     'prod_improvement', 'yield_improvement']].head(10)
            for _, row in display_df.iterrows():
                leak_mark = " 🔧" if row['is_leak_gene'] else ""
                bidirectional_mark = " 🔄" if row['is_bidirectional'] else ""
                print(f"  {row['gene']} ({row['type']}){leak_mark}{bidirectional_mark}: "
                      f"production +{row['prod_improvement']:.2f}%, yield +{row['yield_improvement']:.2f}%")

    def print_enzyme_usage_details(self, gene_id):
        """
        打印指定基因的详细酶用量信息

        Parameters:
        -----------
        gene_id : str
            基因ID
        """
        if not hasattr(self, 'enzyme_usage_prod') or not hasattr(self, 'enzyme_usage_bio'):
            print("错误: 未运行enzyme_fva_filter，无酶用量数据")
            return

        try:
            gene_obj = self.model.genes.get_by_id(gene_id)
            print(f"\n{'=' * 100}")
            print(f"基因 {gene_id} 的详细酶用量分析")
            print(f"基因名称: {gene_obj.name if hasattr(gene_obj, 'name') else '未知'}")
            print(f"{'=' * 100}")

            valid_reactions = 0
            for reaction in gene_obj.reactions:
                rxn_id = reaction.id

                # 检查反应是否有酶用量数据
                if (hasattr(reaction, 'kcat_MW') and reaction.kcat_MW and
                        rxn_id in self.enzyme_usage_bio.index and
                        rxn_id in self.enzyme_usage_prod.index):

                    valid_reactions += 1

                    # 获取酶用量数据
                    min_prod = self.enzyme_usage_prod.loc[rxn_id, 'minimum']
                    max_prod = self.enzyme_usage_prod.loc[rxn_id, 'maximum']
                    min_bio = self.enzyme_usage_bio.loc[rxn_id, 'minimum']
                    max_bio = self.enzyme_usage_bio.loc[rxn_id, 'maximum']

                    # 计算平均值和变化（处理NaN）
                    avg_prod = np.nanmean([min_prod, max_prod])
                    avg_bio = np.nanmean([min_bio, max_bio])

                    # 处理NaN情况的变化率计算
                    if np.isnan(avg_bio) or avg_bio == 0:
                        if np.isnan(avg_prod) or avg_prod == 0:
                            avg_change = 0
                        else:
                            avg_change = 100  # 从0或NaN到有值，视为100%增长
                    elif np.isnan(avg_prod):
                        avg_change = -100  # 从有值到NaN，视为100%下降
                    else:
                        avg_change = ((avg_prod - avg_bio) / avg_bio * 100)

                    # 打印反应信息
                    print(f"\n反应 {valid_reactions}: {rxn_id}")
                    print(f"  反应式: {reaction.reaction}")
                    print(f"  代谢途径: {reaction.subsystem if hasattr(reaction, 'subsystem') else '未知'}")
                    print(f"  kcat_MW: {reaction.kcat_MW}")

                    # 格式化输出，处理NaN
                    def format_value(val):
                        return f"{val:.6f}" if not np.isnan(val) else "NaN"

                    print(
                        f"  生产情景酶用量: 最小={format_value(min_prod)}, 最大={format_value(max_prod)}, 平均={format_value(avg_prod)}")
                    print(
                        f"  生长情景酶用量: 最小={format_value(min_bio)}, 最大={format_value(max_bio)}, 平均={format_value(avg_bio)}")
                    print(f"  平均酶用量变化: {avg_change:+.2f}%")

                    # 判断变化趋势
                    if avg_change > 5:
                        trend = "🔺 上调 (适合过表达)"
                    elif avg_change < -5:
                        trend = "🔻 下调 (适合敲低)"
                    else:
                        trend = "➡️ 中性 (变化不明显)"

                    print(f"  变化趋势: {trend}")

            if valid_reactions == 0:
                print("该基因没有有效的酶用量数据")
                return

            # 打印整体分类结果
            classification = self.compare_enzyme_usage(gene_id)
            print(f"\n{'=' * 50}")
            print(f"整体分类结果: {classification}")

            # 检查是否为双向调控基因
            if hasattr(self, 'bidirectional_genes') and gene_id in self.bidirectional_genes:
                print(f"注意: 该基因为双向调控基因 🔄")

                # 显示该基因在不同反应中的分类
                if hasattr(self, 'gene_scores') and gene_id in self.gene_scores.index:
                    gene_data = self.gene_scores.loc[gene_id]
                    if 'reaction_types' in gene_data:
                        print(f"反应类型分布: {gene_data['reaction_types']}")

            print(f"{'=' * 100}")

        except Exception as e:
            print(f"错误: {e}")
