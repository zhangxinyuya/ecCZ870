#!/usr/bin/env python
# Author: Zhang xinyu
# Date: 2026-10-15
"""Naming rules and exclusion sets for ModelSEED and BiGG metabolic models."""

import re


# ModelSEED cofactors

_MODELSEED_COFACTORS = {
    'cpd00002_c', 'cpd00008_c', 'cpd00018_c', 'cpd00038_c', 'cpd00031_c', 'cpd00126_c',
    'cpd00052_c', 'cpd00096_c', 'cpd00046_c', 'cpd00062_c', 'cpd00014_c', 'cpd00091_c',
    'cpd00002_e', 'cpd00008_e', 'cpd00018_e', 'cpd00038_e', 'cpd00031_e', 'cpd00126_e',
    'cpd00052_e', 'cpd00096_e', 'cpd00046_e', 'cpd00062_e', 'cpd00014_e', 'cpd00091_e',
    'cpd00003_c', 'cpd00004_c', 'cpd00005_c', 'cpd00006_c', 'cpd00015_c', 'cpd00982_c',
    'cpd00050_c', 'cpd01270_c', 'cpd00003_e', 'cpd00004_e', 'cpd00005_e', 'cpd00006_e',
    'cpd00015_e', 'cpd00982_e', 'cpd00050_e', 'cpd01270_e', 'cpd00010_c', 'cpd00022_c',
    'cpd00078_c', 'cpd00070_c', 'cpd15500_c', 'cpd15499_c', 'cpd00001_c', 'cpd00067_c',
    'cpd00009_c', 'cpd00012_c', 'cpd00011_c', 'cpd00242_c', 'cpd00007_c', 'cpd00013_c',
    'cpd00001_e', 'cpd00067_e', 'cpd00009_e', 'cpd00012_e', 'cpd00011_e', 'cpd00242_e',
    'cpd00007_e', 'cpd00013_e', 'cpd00087_c', 'cpd00125_c', 'cpd00017_c', 'cpd00019_c',
    'protein_c', 'dna_c', 'rna_c',
}


# BiGG cofactors


_BIGG_COFACTORS = {
    '2fe1s_c', '2fe2s_c', '35cgmp_c', '4fe4s_c', 'ACP_c', 'ade_c', 'adn_c', 'adp_c', 'ag_c', 'amp_c',
    'aps_c', 'arbtn_c', 'atp_c', 'bwco_c', 'bwco1gdp_c', 'bwcogdp_c', 'ca2_c', 'camp_c', 'cd2_c',
    'cdp_c', 'cl_c', 'cmp_c', 'co2_c', 'coa_c', 'cobalt2_c', 'ctp_c', 'cu_c', 'cu2_c', 'dad_DASH_2_c',
    'dad_DASH_5_c', 'dadp_c', 'datp_c', 'dcdp_c', 'dcmp_c', 'dctp_c', 'dgdp_c', 'dgmp_c', 'dgtp_c',
    'didp_c', 'dimp_c', 'ditp_c', 'dnad_c', 'dtdp_c', 'dtmp_c', 'dttp_c', 'dudp_c', 'dump_c', 'duri_c',
    'dutp_c', 'fad_c', 'fadh2_c', 'fe2_c', 'fe3_c', 'fmn_c', 'fmnh2_c', 'gdp_c', 'gmp_c', 'gtp_c',
    'h_c', 'h2_c', 'h2o_c', 'h2o2_c', 'h2s_c', 'hco3_c', 'hg2_c', 'idp_c', 'imp_c', 'ins_c', 'ipdp_c',
    'iscs_c', 'iscu_c', 'iscu_DASH_2fe2s_c', 'iscu_DASH_2fe2s2_c', 'iscu_DASH_4fe4s_c', 'itp_c', 'k_c',
    'mg2_c', 'mn2_c', 'na1_c', 'nad_c', 'nadh_c', 'nadp_c', 'nadph_c', 'nh4_c', 'ni2_c', 'nmn_c',
    'no_c', 'no2_c', 'no3_c', 'o2_c', 'pi_c', 'ppi_c', 'pppi_c', 'q8_c', 'q8h2_c', 'sertrna_c',
    'sertrna_LSQBKT_sec_RSQBKT__c', 'so3_c', 'so4_c', 'sufbcd_c', 'sufbcd_DASH_2fe2s_c',
    'sufbcd_DASH_2fe2s2_c', 'sufbcd_DASH_4fe4s_c', 'sufse_c', 'sufsesh_c', 'sulfac_c', 'trdrd_c',
    'trnaala_c', 'trnaarg_c', 'trnaasn_c', 'trnaasp_c', 'trnacys_c', 'trnagln_c', 'trnaglu_c',
    'trnagly_c', 'trnahis_c', 'trnaile_c', 'trnaleu_c', 'trnalys_c', 'trnamet_c', 'trnaphe_c',
    'trnapro_c', 'trnasecys_c', 'trnaser_c', 'trnathr_c', 'trnatrp_c', 'trnatyr_c', 'trnaval_c',
    'trptrna_c', 'udp_c', 'ump_c', 'ura_c', 'urate_c', 'urea_c', 'uri_c', 'utp_c', 'xdp_c', 'xmp_c',
    'xtp_c', 'zn2_c','protein_c', 'dna_c', 'rna_c',
}


# ModelSEED exclusions

_MODELSEED_EXCLUDED_METABOLITES = {'cpd15500_c', 'cpd15499_c'}
_MODELSEED_EXCLUDED_REACTIONS   = {'rxn05226_c', 'rxn05501_c', 'rxn05647_c','rxn00558_c_reverse'}


# BiGG exclusions


_BIGG_EXCLUDED_METABOLITES = {'q8_c', 'q8h2_c', 'mqn8_c', 'mql8_c'}
_BIGG_EXCLUDED_REACTIONS   = set()


# Reactions to close

_MODELSEED_CLOSE_REACTIONS = ['EX_cpd00197_e_reverse',]
_BIGG_CLOSE_REACTIONS      = []


class NamingConvention:
    """Manage naming rules, cofactors, exclusions, and reactions to close.

    Supported conventions: modelseed and bigg."""


    VALID_CONVENTIONS = ('modelseed', 'bigg')

    def __init__(self, convention='modelseed'):
        convention = convention.lower().strip()
        if convention not in self.VALID_CONVENTIONS:
            raise ValueError(
                f"Unknown naming convention '{convention}'. "
                f"Choose from: {self.VALID_CONVENTIONS}"
            )
        self.name = convention

        if convention == 'modelseed':
            self.COMMON_COFACTORS     = _MODELSEED_COFACTORS.copy()
            self.EXCLUDED_METABOLITES = _MODELSEED_EXCLUDED_METABOLITES.copy()
            self.EXCLUDED_REACTIONS   = _MODELSEED_EXCLUDED_REACTIONS.copy()
            self.CLOSE_REACTIONS      = list(_MODELSEED_CLOSE_REACTIONS)
        else:
            self.COMMON_COFACTORS     = _BIGG_COFACTORS.copy()
            self.EXCLUDED_METABOLITES = _BIGG_EXCLUDED_METABOLITES.copy()
            self.EXCLUDED_REACTIONS   = _BIGG_EXCLUDED_REACTIONS.copy()
            self.CLOSE_REACTIONS      = list(_BIGG_CLOSE_REACTIONS)


    def is_cofactor(self, metabolite_id):
        """Check whether the metabolite is listed as a cofactor."""
        return metabolite_id in self.COMMON_COFACTORS

    def is_excluded_metabolite(self, metabolite_id):
        """Check the metabolite exclusion set."""
        return metabolite_id in self.EXCLUDED_METABOLITES

    def is_excluded_reaction(self, reaction_id):
        """Check the reaction exclusion set."""
        return reaction_id in self.EXCLUDED_REACTIONS


    def is_bio_reaction(self, reaction_id):
        """Check the Bio_ prefix for ModelSEED or BIOMASS_ for BiGG."""
        if self.name == 'modelseed':
            return reaction_id.startswith('Bio_')
        else:
            return reaction_id.upper().startswith('BIOMASS_')


    def make_exchange_rxn(self, compound_id):
        """Build EX_<compound_id>_e from a compartment-free compound ID."""
        return f'EX_{compound_id}_e'


    def normalize_reaction_id(self, reaction_id):
        """Extract the ModelSEED base reaction ID; leave BiGG IDs unchanged."""
        if self.name == 'modelseed':
            pattern = r'(rxn\d{5}_[ce])'
            match = re.search(pattern, reaction_id)
            return match.group(1) if match else reaction_id
        else:
            return reaction_id


    def find_isozyme_reactions(self, model, target_reaction_id):
        """Match ModelSEED reactions without _numN suffixes; return the BiGG target ID."""
        if self.name == 'modelseed':
            base_id = re.sub(r'_num\d+$', '', target_reaction_id)
            return [rxn.id for rxn in model.reactions
                    if re.sub(r'_num\d+$', '', rxn.id) == base_id]
        else:


            return [target_reaction_id]


    def is_cytoplasmic_metabolite(self, metabolite_id):
        """Check the cytoplasmic suffix _c."""
        return metabolite_id.endswith('_c')


    def find_carbon_source_fallback(self, model, c_source_rxn):
        """Find the first reaction containing cpd00027 or glc__D; otherwise keep the input ID."""
        if self.name == 'modelseed':
            keyword = 'cpd00027'
        else:
            keyword = 'glc__D'

        fallback = [r.id for r in model.reactions if keyword in r.id]
        return fallback[0] if fallback else c_source_rxn


    def extract_product_tag(self, product_rxn):
        """Remove EX_ and _e from a product exchange ID for file naming."""
        tag = product_rxn
        if tag.startswith('EX_'):
            tag = tag[3:]
        if tag.endswith('_e'):
            tag = tag[:-2]
        return tag


    def __repr__(self):
        return (
            f"NamingConvention('{self.name}', "
            f"cofactors={len(self.COMMON_COFACTORS)}, "
            f"excl_met={len(self.EXCLUDED_METABOLITES)}, "
            f"excl_rxn={len(self.EXCLUDED_REACTIONS)})"
        )

    def summary(self):
        """Print convention settings and a biomass ID example."""
        print(f"  Naming convention : {self.name}")
        print(f"    Cofactors       : {len(self.COMMON_COFACTORS)}")
        print(f"    Excluded mets   : {len(self.EXCLUDED_METABOLITES)}")
        print(f"    Excluded rxns   : {len(self.EXCLUDED_REACTIONS)}")
        bio_pattern = 'Bio_cgATCC14067_c' if self.name == 'modelseed' else 'CG_biomass_cgl_ATCC13032'
        print(f"    Biomass pattern : {bio_pattern}")
