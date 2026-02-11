"""
iBridge 优化版本 - 使用真实酶成本最小化(minpool)获取通量分布
调整后的优先级：Neg_to_Pos > Pos_to_Neg > Pos_Optimization

修改说明：
- find_active_reactions(): 用 minpool 替代虚拟反应策略
- run_moma_analysis(): 用 minpool 替代 pFBA 获取野生型参考
- 其他所有功能保持原 ibridge.py 不变
"""

import os
import re
import json
import tempfile
from math import isnan
import numpy as np
import pandas as pd
import networkx as nx
from collections import deque
from cobra.flux_analysis import moma
import cobra

os.environ['NUMEXPR_NUM_THREADS'] = '4'

# 全局配置
COMMON_COFACTORS = {
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
    'protein_c','dna_c','rna_c',
}

EXCLUDED_METABOLITES = {'cpd15500_c', 'cpd15499_c'}
EXCLUDED_REACTIONS = {'rxn05226_c','rxn05501_c','rxn05647_c'}  # ptsG
METABOLIC_EXCLUSION_RULES = [
    {
        'source_metabolites': {'cpd00020_c', 'cpd00020_e'},
        'target_metabolites': {'cpd00498_c', 'cpd19047_c'},
        'description': '阻止cpd00020转化为特定代谢物'
    },
    {
        'source_metabolites': {'cpd00020_c'},
        'target_metabolites': {'cpd02120_c'},
        'description': '阻止cpd00020转化为特定代谢物'
    },
    {
        'source_metabolites': {'cpd00023_c', 'cpd00023_e'},
        'target_metabolites': {'cpd00066_c', 'cpd00069_c'},
        'description': '阻止cpd00023转化为特定代谢物'
    },
    {
        'source_metabolites': {'cpd00023_c'},
        'target_metabolites': {'cpd00053_c'},
        'description': '阻止rxn00555_c中L-Glutamine_c转化为L-Glutamate_c'
    },
    {
        'source_metabolites': {'cpd00022_c'},
        'target_metabolites': {'cpd01646_c'},
        'description': '阻止rxn00902_c中Acetyl-CoA转化为2-Isopropylmalate'
    },
]

def is_cofactor(metabolite_id):
    return metabolite_id in COMMON_COFACTORS

def is_excluded_metabolite(metabolite_id):
    return metabolite_id in EXCLUDED_METABOLITES

def is_bio_reaction(reaction_id):
    return reaction_id.startswith('Bio_')

def is_excluded_reaction(reaction_id):
    """检查反应是否在直接排除列表中"""
    return reaction_id in EXCLUDED_REACTIONS

def normalize_reaction_id(reaction_id):
    """标准化反应ID，提取核心模式"""
    pattern = r'(rxn\d{5}_[ce])'
    match = re.search(pattern, reaction_id)
    return match.group(1) if match else reaction_id

def get_unique_reaction_count(reaction_list):
    """获取去重后的反应数"""
    if not reaction_list:
        return 0
    return len(set(normalize_reaction_id(rxn) for rxn in reaction_list))

def check_enzyme_constraints(model):
    """检查模型是否有酶约束"""
    # 检查模型约束
    if hasattr(model, 'constraints'):
        return True

    # 检查JSON格式的酶约束信息
    if hasattr(model, 'json') and isinstance(model.json, dict):
        if 'enzyme_constraint' in model.json:
            return True

    # 检查反应是否有kcat_MW属性（增加数值验证）
    for reaction in model.reactions:
        if hasattr(reaction, 'kcat_MW') and reaction.kcat_MW:
            try:
                kcat_mw_val = float(reaction.kcat_MW)
                if kcat_mw_val > 0:
                    return True
            except (ValueError, TypeError):
                continue

    return False


# ==================== 新增：minpool 优化方法 ====================

def optimize_minimize_enzyme_pool(model, objective_rxn, fraction_of_optimum=1.0):
    """
    两阶段优化：最小化真实酶池用量

    目标函数:
    minimize: Σ(v_i / kcat_MW_i) + ε × Σ(v_j)  (对无kcat_MW的反应)

    参数:
    -----
    model : cobra.Model
        代谢模型
    objective_rxn : str
        目标反应ID
    fraction_of_optimum : float
        最优值的保留比例（默认1.0）

    返回:
    -----
    solution : cobra.Solution
        优化解
    """
    # ===== 阶段1: 找到最大目标值 =====
    model.objective = objective_rxn
    model.objective_direction = 'max'

    max_objective = model.slim_optimize()

    if max_objective is None or isnan(max_objective) or max_objective <= 1e-9:
        return model.optimize()

    # ===== 阶段2: 固定目标值，最小化酶池 =====
    target_flux = max_objective * fraction_of_optimum
    tolerance = abs(target_flux) * 0.001

    obj_rxn = model.reactions.get_by_id(objective_rxn)
    original_bounds = obj_rxn.bounds
    obj_rxn.bounds = (target_flux - tolerance, target_flux + tolerance)

    # 构建酶成本目标函数
    enzyme_cost_objective = {}
    default_penalty = 1e-9  # 对无kcat_MW反应的小惩罚

    for rxn in model.reactions:
        if hasattr(rxn, 'kcat_MW') and rxn.kcat_MW:
            try:
                kcat_mw = float(rxn.kcat_MW)
                if kcat_mw > 0:
                    # 有kcat_MW：使用真实酶成本
                    enzyme_cost_objective[rxn] = -1.0 / kcat_mw
                else:
                    # kcat_MW为0或负：使用小惩罚
                    enzyme_cost_objective[rxn] = -default_penalty
            except (ValueError, TypeError):
                enzyme_cost_objective[rxn] = -default_penalty
        else:
            # 无kcat_MW（如交换反应）：使用小惩罚，避免极端值
            enzyme_cost_objective[rxn] = -default_penalty

    model.objective = enzyme_cost_objective
    model.objective_direction = 'max'  # max(-cost) = min(cost)

    solution = model.optimize()

    # 恢复原始边界
    obj_rxn.bounds = original_bounds

    return solution


# ==================== 修改1：find_active_reactions 使用 minpool ====================

def find_active_reactions(model, target_rxn, biomass_rxn, bio_fraction=0.1):
    """
    识别活跃反应 - 使用真实酶成本最小化(minpool)

    替代原来的虚拟反应策略
    """
    print("分析活跃反应（真实酶成本最小化）...")

    # 检查是否有酶约束
    has_enzyme_constraints = check_enzyme_constraints(model)

    if not has_enzyme_constraints:
        print("  模型无酶约束，使用标准方法")
        return find_active_reactions_standard(model, target_rxn, biomass_rxn, bio_fraction)

    print("  检测到酶约束，应用 minpool 方法")

    try:
        with model as m:
            # 设置生长约束
            m.objective = biomass_rxn
            max_bio = m.slim_optimize()

            if isnan(max_bio) or max_bio < 1e-6:
                raise ValueError("无法达到生长要求")

            m.reactions.get_by_id(biomass_rxn).lower_bound = max_bio * bio_fraction

            # 使用 minpool 方法获取通量分布
            solution = optimize_minimize_enzyme_pool(m, target_rxn, fraction_of_optimum=1.0)

            if solution.status != 'optimal':
                raise ValueError("minpool 优化失败")

            # 提取活跃反应（排除生物量反应）
            active_reactions = [
                rxn_id for rxn_id, flux in solution.fluxes.items()
                if abs(flux) > 1e-6
                and not is_bio_reaction(rxn_id)
            ]

            print(f"  minpool 方法识别活跃反应数: {len(active_reactions)}")
            return active_reactions

    except Exception as e:
        print(f"  minpool 优化失败: {e}")
        print("  回退到标准方法")
        return find_active_reactions_standard(model, target_rxn, biomass_rxn, bio_fraction)


def find_active_reactions_standard(model, target_rxn, biomass_rxn, bio_fraction=0.1):
    """标准活跃反应识别方法（原版）"""
    with model as m:
        m.objective = biomass_rxn
        max_bio = m.slim_optimize()

        m.reactions.get_by_id(biomass_rxn).lower_bound = max_bio * bio_fraction
        m.objective = target_rxn
        solution = m.optimize()

        if solution.status != 'optimal':
            raise ValueError("无法找到可行解")

        active_reactions = [
            rxn_id for rxn_id, flux in solution.fluxes.items()
            if abs(flux) > 1e-6 and not is_bio_reaction(rxn_id)
        ]

    print(f"  标准方法识别活跃反应数: {len(active_reactions)}")
    return active_reactions


# ==================== 修改2：run_moma_analysis 野生型参考使用 minpool ====================

def run_moma_analysis(model, biomass_rxn, target_rxn, num_pts=10):
    """
    执行MOMA分析生成多个通量分布

    修改：野生型参考使用 minpool 方法（有酶约束时）
    """
    print(f"运行MOMA分析 (采样点: {num_pts})...")

    # 检查是否有酶约束
    has_enzyme_constraints = check_enzyme_constraints(model)

    # 获取野生型参考分布
    with model.copy() as m:
        m.objective = biomass_rxn
        if has_enzyme_constraints:
            print("  使用 minpool 获取野生型参考")
            wt_dist = optimize_minimize_enzyme_pool(m, biomass_rxn, fraction_of_optimum=1.0)
        else:
            print("  使用标准 FBA 获取野生型参考")
            wt_dist = m.optimize()

    # 确定目标反应范围
    with model as m:
        m.objective = target_rxn
        max_const = m.slim_optimize()
        m.objective_direction = 'min'
        min_const = m.slim_optimize()

    # 扩展采样范围10%
    extended_max = max_const * 1.1 if max_const > 0 else max_const * 0.9
    extended_min = min_const * 1.1 if min_const < 0 else min_const * 0.9

    results = {}
    successful_points = 0

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
                    successful_points += 1
        except:
            continue

    if successful_points < 3:
        raise ValueError("MOMA分析成功点数过少")

    df_flux = pd.DataFrame.from_dict(results)
    df_cov = df_flux.abs().T.cov()

    print(f"  成功采样: {successful_points}/{num_pts}")
    return df_cov[target_rxn], df_flux


# ===== 以下保持iBridge原有功能不变 =====

def extract_metabolite_directions(model):
    """提取代谢物在反应中的方向信息"""
    dict_reactant, dict_product = {}, {}

    for met in model.metabolites:
        dict_reactant[met.id] = []
        dict_product[met.id] = []

        for rxn in met.reactions:
            coeff = rxn.get_coefficient(met.id)
            if coeff < 0:
                dict_reactant[met.id].append(rxn.id)
            elif coeff > 0:
                dict_product[met.id].append(rxn.id)

    return dict_reactant, dict_product


def build_reaction_network(model, reactions_list=None):
    """构建反应连接网络"""
    G = nx.DiGraph()
    reactions_to_use = reactions_list or [rxn.id for rxn in model.reactions]

    # 添加反应节点
    for rxn_id in reactions_to_use:
        if not is_bio_reaction(rxn_id):
            G.add_node(rxn_id)

    # 构建连接关系
    for rxn1_id in reactions_to_use:
        if is_bio_reaction(rxn1_id):
            continue

        try:
            rxn1 = model.reactions.get_by_id(rxn1_id)
        except KeyError:
            continue

        # 获取非辅因子产物
        rxn1_products = [
            met.id for met, coeff in rxn1.metabolites.items()
            if coeff > 0 and not is_cofactor(met.id)
        ]

        # 寻找连接反应
        for rxn2_id in reactions_to_use:
            if rxn1_id == rxn2_id or is_bio_reaction(rxn2_id):
                continue

            try:
                rxn2 = model.reactions.get_by_id(rxn2_id)
            except KeyError:
                continue

            rxn2_substrates = [
                met.id for met, coeff in rxn2.metabolites.items()
                if coeff < 0 and not is_cofactor(met.id)
            ]

            shared = set(rxn1_products) & set(rxn2_substrates)
            if shared and G.has_node(rxn1_id) and G.has_node(rxn2_id):
                G.add_edge(rxn1_id, rxn2_id, weight=len(shared))

    return G


def build_metabolic_network(model, reactions_list=None):
    """构建代谢物-反应网络"""
    G = nx.DiGraph()
    reactions_to_use = reactions_list or [rxn.id for rxn in model.reactions]

    for rxn_id in reactions_to_use:
        try:
            rxn = model.reactions.get_by_id(rxn_id)
        except KeyError:
            continue

        if is_bio_reaction(rxn_id):
            continue

        non_cofactor_mets = [
            met for met in rxn.metabolites
            if not is_cofactor(met.id)
        ]

        if len(non_cofactor_mets) <= 1:
            continue

        G.add_node(rxn_id, type='reaction')

        for met, coeff in rxn.metabolites.items():
            if is_cofactor(met.id):
                continue

            G.add_node(met.id, type='metabolite')

            if coeff < 0:
                G.add_edge(met.id, rxn_id)
            elif coeff > 0:
                G.add_edge(rxn_id, met.id)

    return G


def calculate_backup_topology_weights(model, target_metabolite):
    """备用拓扑权重计算 - 基于代谢物共享反应的相似度"""
    try:
        target_met = model.metabolites.get_by_id(target_metabolite)
    except KeyError:
        return {target_metabolite: (0, 1.0)}

    target_reactions = set(rxn.id for rxn in target_met.reactions if not is_bio_reaction(rxn.id))
    weights = {target_metabolite: (0, 1.0)}

    for met in model.metabolites:
        if met.id == target_metabolite or is_excluded_metabolite(met.id):
            continue

        met_reactions = set(rxn.id for rxn in met.reactions if not is_bio_reaction(rxn.id))

        if met_reactions:
            # 计算反应相似度
            shared = len(target_reactions & met_reactions)
            total = len(target_reactions | met_reactions)

            if total > 0:
                similarity = shared / total
                # 转换为距离和权重
                distance = max(1, int(3 * (1 - similarity)))  # 距离1-3
                weight = 0.7 ** distance
                weights[met.id] = (distance, weight)
            else:
                weights[met.id] = (3, 0.343)  # 0.7^3

    return weights


def calculate_topology_weights(target, network, max_depth=3, decay_factor=0.7):
    """计算拓扑权重，基于到目标的距离"""
    if target not in network:
        print(f"    目标 {target} 不在网络中")
        return {target: (0, 1.0)}

    distances = {target: 0}
    queue = deque([(target, 0)])
    visited = {target}

    while queue:
        current, depth = queue.popleft()
        if depth >= max_depth:
            continue

        # 获取前驱和后继节点
        predecessors = list(network.predecessors(current))
        successors = list(network.successors(current))
        neighbors = predecessors + successors

        for neighbor in neighbors:
            if neighbor not in visited:
                distances[neighbor] = depth + 1
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))

    # 计算衰减权重
    weights = {}
    for node, distance in distances.items():
        weight = decay_factor ** distance
        weights[node] = (distance, weight)

    print(f"    从 {target} 计算得到 {len(weights)} 个权重")
    return weights


def calculate_metabolite_scores(model, df_cov, active_reactions, reaction_weights,
                               metabolite_weights, threshold=0.01):
    """计算代谢物评分"""
    print("计算代谢物评分...")

    # 应用反应拓扑权重
    df_cov_weighted = df_cov.copy()
    for rxn_id in df_cov.index:
        if rxn_id in reaction_weights:
            _, weight = reaction_weights[rxn_id]
            df_cov_weighted[rxn_id] = df_cov[rxn_id] * (1 + 0.5 * weight)

    # 筛选显著反应
    df_pcov = df_cov_weighted[df_cov_weighted.abs() > threshold].dropna()
    pos_rxns = list(df_pcov[df_pcov > 0].index)
    neg_rxns = list(df_pcov[df_pcov < 0].index)

    active_set = set(active_reactions)
    metabolite_scores = {}

    for met in model.metabolites:
        if is_excluded_metabolite(met.id):
            metabolite_scores[met.id] = {
                'Metabolite_ID': met.id,
                'Score_Positive': 0.0,
                'Score_Negative': 0.0,
                'Score_Normalized': 0.0,
                'Score_Weighted': 0.0,
                'Topology_Distance': float('inf'),
                'Topology_Weight': 0.0,
                'Total_Reactions': 0,
                'Active_Reactions': 0,
                'Positive_Reactions': 0,
                'Negative_Reactions': 0,
                'Is_Excluded': True
            }
            continue

        # 获取相关反应
        rxn_candidates = [rxn.id for rxn in met.reactions]
        rxn_pos = list(set(pos_rxns) & set(rxn_candidates))
        rxn_neg = list(set(neg_rxns) & set(rxn_candidates))

        # 计算基础分数
        score_pos = df_pcov[rxn_pos].abs().sum() if rxn_pos else 0
        score_neg = df_pcov[rxn_neg].abs().sum() if rxn_neg else 0

        # 归一化
        active_in_met = list(set(rxn_candidates) & active_set)
        norm_factor = np.sqrt(1 + get_unique_reaction_count(active_in_met))
        score_norm = (score_pos - score_neg) / norm_factor

        # 应用拓扑权重
        if met.id in metabolite_weights:
            distance, topo_weight = metabolite_weights[met.id]
        else:
            # 为未在权重字典中的代谢物设置默认权重
            distance, topo_weight = float('inf'), 0.0

        score_weighted = score_norm * (1 + 0.5 * topo_weight)

        metabolite_scores[met.id] = {
            'Metabolite_ID': met.id,
            'Score_Positive': float(score_pos),
            'Score_Negative': float(score_neg),
            'Score_Normalized': float(score_norm),
            'Score_Weighted': float(score_weighted),
            'Topology_Distance': distance if distance != float('inf') else 999,
            'Topology_Weight': float(topo_weight),
            'Total_Reactions': len(rxn_candidates),
            'Active_Reactions': len(active_in_met),
            'Positive_Reactions': len(rxn_pos),
            'Negative_Reactions': len(rxn_neg),
            'Is_Excluded': False
        }

    return metabolite_scores


def find_target_reactions(model, dict_reactant, dict_product, metabolite_scores, exclude_met_file):
    """识别目标反应候选 - 调整优先级版本，优化同类型内的选择"""
    print("识别目标反应...")

    # 读取排除代谢物列表
    excluded_mets = set()
    try:
        with open(exclude_met_file, 'r') as f:
            excluded_mets = {line.strip() for line in f}
    except FileNotFoundError:
        print(f"  警告: 未找到排除文件 {exclude_met_file}")

    # 分类代谢物
    met_pos = []
    met_neg = []

    for met_id, scores in metabolite_scores.items():
        if not met_id.endswith('_c') or met_id in excluded_mets or scores['Is_Excluded']:
            continue

        if scores['Score_Weighted'] > 0:
            met_pos.append(met_id)
        elif scores['Score_Weighted'] < 0:
            met_neg.append(met_id)

    target_reactions = {}

    # ==========================================
    # 优先级1: 负向到正向转化 (Neg_to_Pos) - 最高优先级
    # ==========================================
    print("  处理类型: Neg_to_Pos (优先级最高)")

    neg_to_pos_count = 0
    excluded_count_1 = 0

    for met1 in met_neg:
        score1 = abs(metabolite_scores[met1]['Score_Weighted'])
        for met2 in met_pos:
            score2 = abs(metabolite_scores[met2]['Score_Weighted'])

            # 消耗负向代谢物、产生正向代谢物的反应 -> 上调
            r_np = set(dict_product[met2]) & set(dict_reactant[met1])
            for rxn_id in r_np:
                # 检查是否被直接排除
                if is_excluded_reaction(rxn_id):
                    print(f"    跳过被排除的反应: {rxn_id}")
                    excluded_count_1 += 1
                    continue

                new_soc = score1 + score2

                # 检查是否需要更新（仅在同类型内比较SoC）
                if rxn_id not in target_reactions:
                    target_reactions[rxn_id] = {
                        'reaction_id': rxn_id,
                        'source_metabolite': met1,
                        'target_metabolite': met2,
                        'regulation_type': 'Up',
                        'conversion_type': 'Neg_to_Pos',
                        'soc_score': new_soc,
                        'source_score': -score1,
                        'target_score': score2,
                        'priority_reason': 'Primary_Bridge_Reaction'
                    }
                    neg_to_pos_count += 1
                elif target_reactions[rxn_id]['conversion_type'] == 'Neg_to_Pos':
                    # 同类型内，选择SoC更大的配对
                    if new_soc > target_reactions[rxn_id]['soc_score']:
                        print(
                            f"    Neg_to_Pos更新 {rxn_id}: SoC从{target_reactions[rxn_id]['soc_score']:.3f}提升到{new_soc:.3f}")
                        target_reactions[rxn_id] = {
                            'reaction_id': rxn_id,
                            'source_metabolite': met1,
                            'target_metabolite': met2,
                            'regulation_type': 'Up',
                            'conversion_type': 'Neg_to_Pos',
                            'soc_score': new_soc,
                            'source_score': -score1,
                            'target_score': score2,
                            'priority_reason': 'Primary_Bridge_Reaction'
                        }

    print(f"    发现 {neg_to_pos_count} 个 Neg_to_Pos 反应")
    if excluded_count_1 > 0:
        print(f"    排除 {excluded_count_1} 个被禁用的反应")

    # ==========================================
    # 优先级2: 正向到负向转化 (Pos_to_Neg) - 次高优先级
    # ==========================================
    print("  处理类型: Pos_to_Neg (优先级次高)")

    # 使用全局定义的排除规则
    active_rules = []
    for rule in METABOLIC_EXCLUSION_RULES:
        active_sources = rule['source_metabolites'] & set(met_pos)
        if active_sources:
            active_rules.append({
                'sources': active_sources,
                'targets': rule['target_metabolites'],
                'description': rule['description']
            })
            print(f"    检测到正向代谢物: {', '.join(active_sources)}")
            print(f"    将阻止这些代谢物转化为 {', '.join(rule['target_metabolites'])} 的反应被识别为KD酶点")

    pos_to_neg_count = 0
    skipped_due_to_priority = 0
    skipped_due_to_special_rule = 0
    skipped_due_to_better_products = 0
    excluded_count_2 = 0

    for met1 in met_neg:
        score1 = abs(metabolite_scores[met1]['Score_Weighted'])

        for met2 in met_pos:
            score2 = abs(metabolite_scores[met2]['Score_Weighted'])

            # 检查是否违反任何排除规则
            skip_due_to_rule = False
            for rule in active_rules:
                if met2 in rule['sources'] and met1 in rule['targets']:
                    print(f"    跳过特殊反应: {met2} -> {met1} ({rule['description']})")
                    skipped_due_to_special_rule += 1
                    skip_due_to_rule = True
                    break

            if skip_due_to_rule:
                continue

            # 消耗正向代谢物、产生负向代谢物的反应 -> 下调
            r_pn = set(dict_reactant[met2]) & set(dict_product[met1])
            for rxn_id in r_pn:
                # 检查是否被直接排除
                if is_excluded_reaction(rxn_id):
                    print(f"    跳过被排除的反应: {rxn_id}")
                    excluded_count_2 += 1
                    continue

                # 新增检查: 如果反应同时产生评分更高的正代谢物，则跳过Pos_to_Neg分类
                should_skip_pos_to_neg = False
                try:
                    rxn = model.reactions.get_by_id(rxn_id)
                    met2_score = metabolite_scores[met2]['Score_Weighted']

                    # 检查所有产物中是否有评分更高的正代谢物
                    for product_met, coeff in rxn.metabolites.items():
                        if (coeff > 0 and
                                product_met.id in metabolite_scores and
                                not is_cofactor(product_met.id)):
                            product_score = metabolite_scores[product_met.id]['Score_Weighted']
                            if product_score > 0 and product_score > met2_score:
                                should_skip_pos_to_neg = True
                                print(
                                    f"    跳过Pos_to_Neg分类 {rxn_id}: 产物{product_met.id}({product_score:.3f}) > 底物{met2}({met2_score:.3f})")
                                skipped_due_to_better_products += 1
                                break

                except KeyError:
                    pass

                if should_skip_pos_to_neg:
                    continue

                new_soc = score1 + score2

                # 检查是否已被更高优先级分类
                if rxn_id not in target_reactions:
                    target_reactions[rxn_id] = {
                        'reaction_id': rxn_id,
                        'source_metabolite': met2,
                        'target_metabolite': met1,
                        'regulation_type': 'Down',
                        'conversion_type': 'Pos_to_Neg',
                        'soc_score': new_soc,
                        'source_score': score2,
                        'target_score': -score1,
                        'priority_reason': 'Secondary_Prevention'
                    }
                    pos_to_neg_count += 1
                elif target_reactions[rxn_id]['conversion_type'] == 'Pos_to_Neg':
                    # 同类型内，选择SoC更大的配对
                    if new_soc > target_reactions[rxn_id]['soc_score']:
                        print(
                            f"    Pos_to_Neg更新 {rxn_id}: SoC从{target_reactions[rxn_id]['soc_score']:.3f}提升到{new_soc:.3f}")
                        target_reactions[rxn_id] = {
                            'reaction_id': rxn_id,
                            'source_metabolite': met2,
                            'target_metabolite': met1,
                            'regulation_type': 'Down',
                            'conversion_type': 'Pos_to_Neg',
                            'soc_score': new_soc,
                            'source_score': score2,
                            'target_score': -score1,
                            'priority_reason': 'Secondary_Prevention'
                        }
                else:
                    # 被更高优先级占用
                    skipped_due_to_priority += 1

    print(f"    发现 {pos_to_neg_count} 个 Pos_to_Neg 反应")
    if skipped_due_to_priority > 0:
        print(f"    由于优先级跳过 {skipped_due_to_priority} 个冲突反应")
    if skipped_due_to_special_rule > 0:
        print(f"    由于特殊规则跳过 {skipped_due_to_special_rule} 个反应")
    if skipped_due_to_better_products > 0:
        print(f"    由于产生更好产物跳过 {skipped_due_to_better_products} 个反应")
    if excluded_count_2 > 0:
        print(f"    排除 {excluded_count_2} 个被禁用的反应")

    # ==========================================
    # 优先级3: 正向代谢物间优化 (Pos_Optimization) - 最低优先级
    # ==========================================
    print("  处理类型: Pos_Optimization (优先级最低)")

    sorted_pos = sorted([(met, abs(metabolite_scores[met]['Score_Weighted']))
                         for met in met_pos], key=lambda x: x[1], reverse=True)

    pos_optimization_count = 0
    pos_optimization_updates = 0
    skipped_due_to_higher_priority = 0
    excluded_count_3 = 0

    for i in range(len(sorted_pos)):
        for j in range(i):
            met_lower, score_lower = sorted_pos[i]
            met_higher, score_higher = sorted_pos[j]

            # 找到消耗低分正向代谢物、产生高分正向代谢物的反应
            r_lh = set(dict_reactant[met_lower]) & set(dict_product[met_higher])

            if r_lh and score_higher > score_lower:
                for rxn_id in r_lh:
                    # 检查是否被直接排除
                    if is_excluded_reaction(rxn_id):
                        print(f"    跳过被排除的反应: {rxn_id}")
                        excluded_count_3 += 1
                        continue

                    new_soc = score_higher - score_lower

                    # 调试输出
                    print(
                        f"    检查反应 {rxn_id}: {met_lower}({score_lower:.3f}) -> {met_higher}({score_higher:.3f}), SoC={new_soc:.3f}")

                    # 检查是否需要更新
                    if rxn_id not in target_reactions:
                        # 新反应，直接添加
                        target_reactions[rxn_id] = {
                            'reaction_id': rxn_id,
                            'source_metabolite': met_lower,
                            'target_metabolite': met_higher,
                            'regulation_type': 'Up',
                            'conversion_type': 'Pos_Optimization',
                            'soc_score': new_soc,
                            'source_score': score_lower,
                            'target_score': score_higher,
                            'priority_reason': 'Tertiary_Optimization'
                        }
                        pos_optimization_count += 1
                        print(f"      ✓ 新增反应")

                    elif target_reactions[rxn_id]['conversion_type'] == 'Pos_Optimization':
                        # 同类型，比较SoC选择最优
                        old_soc = target_reactions[rxn_id]['soc_score']
                        if new_soc > old_soc:
                            old_source = target_reactions[rxn_id]['source_metabolite']
                            old_target = target_reactions[rxn_id]['target_metabolite']

                            target_reactions[rxn_id] = {
                                'reaction_id': rxn_id,
                                'source_metabolite': met_lower,
                                'target_metabolite': met_higher,
                                'regulation_type': 'Up',
                                'conversion_type': 'Pos_Optimization',
                                'soc_score': new_soc,
                                'source_score': score_lower,
                                'target_score': score_higher,
                                'priority_reason': 'Tertiary_Optimization'
                            }
                            pos_optimization_updates += 1
                            print(
                                f"      ✓ 更新: {old_source}->{old_target}(SoC={old_soc:.3f}) 替换为 {met_lower}->{met_higher}(SoC={new_soc:.3f})")
                        else:
                            print(f"      × 保持现有配对(SoC={old_soc:.3f} >= {new_soc:.3f})")

                    else:
                        # 被更高优先级占用
                        skipped_due_to_higher_priority += 1
                        print(f"      × 被更高优先级({target_reactions[rxn_id]['conversion_type']})占用")

    print(f"    发现 {pos_optimization_count} 个新 Pos_Optimization 反应")
    if pos_optimization_updates > 0:
        print(f"    更新 {pos_optimization_updates} 个 Pos_Optimization 反应配对")
    if skipped_due_to_higher_priority > 0:
        print(f"    由于更高优先级跳过 {skipped_due_to_higher_priority} 个冲突反应")
    if excluded_count_3 > 0:
        print(f"    排除 {excluded_count_3} 个被禁用的反应")

    total_reactions = len(target_reactions)
    print(f"\n  总计: {total_reactions} 个目标反应")

    # 输出各类型统计
    type_counts = {}
    for rxn_data in target_reactions.values():
        conv_type = rxn_data['conversion_type']
        type_counts[conv_type] = type_counts.get(conv_type, 0) + 1

    print("  类型分布:")
    for conv_type in ['Neg_to_Pos', 'Pos_to_Neg', 'Pos_Optimization']:
        if conv_type in type_counts:
            print(f"    {conv_type}: {type_counts[conv_type]} 个反应")

    return target_reactions


def find_isozyme_reactions(model, target_reaction_id):
    """查找同工酶反应"""
    base_id = re.sub(r'_num\d+$', '', target_reaction_id)
    return [rxn.id for rxn in model.reactions
            if re.sub(r'_num\d+$', '', rxn.id) == base_id]


def process_target_reactions(target_reactions, metabolite_scores, active_reactions,
                             model, top_score_percentage=0.4, return_all=False):
    """处理目标反应 - 使用调整后的优先级"""
    print("处理目标反应...")

    if not target_reactions:
        if return_all:
            return pd.DataFrame(), pd.DataFrame()
        return pd.DataFrame()

    active_set = set(active_reactions)
    results = []

    # 构建结果数据
    for rxn_data in target_reactions.values():
        rxn_id = rxn_data['reaction_id']
        is_active = rxn_id in active_set

        # 获取反应信息
        try:
            rxn = model.reactions.get_by_id(rxn_id)
            reaction_name = rxn.name or rxn_id
            reaction_equation = str(rxn.reaction)
        except:
            reaction_name = rxn_id
            reaction_equation = "Unknown"

        # 获取同工酶信息
        isozymes = find_isozyme_reactions(model, rxn_id)
        isozyme_count = len(isozymes)
        active_isozymes = sum(1 for iso in isozymes if iso in active_set)

        source_met = rxn_data['source_metabolite']
        target_met = rxn_data['target_metabolite']

        result = {
            'Target_reactions': rxn_id,
            'Reaction_name': reaction_name,
            'Reaction_equation': reaction_equation,
            'Regulation_type': rxn_data['regulation_type'],
            'Conversion_type': rxn_data['conversion_type'],
            'Priority_reason': rxn_data.get('priority_reason', 'Unknown'),
            'SoC': rxn_data['soc_score'],
            'Source_metabolite': source_met,
            'Destination_metabolite': target_met,
            'Source_score': rxn_data['source_score'],
            'Target_score': rxn_data['target_score'],
            'Source_Distance': metabolite_scores[source_met]['Topology_Distance'],
            'Dest_Distance': metabolite_scores[target_met]['Topology_Distance'],
            'Source_Topo_Weight': metabolite_scores[source_met]['Topology_Weight'],
            'Dest_Topo_Weight': metabolite_scores[target_met]['Topology_Weight'],
            'Is_Active': is_active,
            'Isozyme_count': isozyme_count,
            'Active_isozymes': active_isozymes,
            'All_isozymes': ';'.join(isozymes) if len(isozymes) > 1 else rxn_id,
            'Selection_reason': '',
            'Type_rank': 0,
            'Global_SoC_rank': 0,
        }
        results.append(result)

    df = pd.DataFrame(results)

    # 调整后的优先级排序顺序
    priority_order = {
        'Neg_to_Pos': 1,      # 最高优先级
        'Pos_to_Neg': 2,      # 次高优先级
        'Pos_Optimization': 3  # 最低优先级
    }

    conversion_types = df['Conversion_type'].unique()
    selected_reactions = []

    # 按类型分别处理
    print(f"  按SoC取前{top_score_percentage:.0%}（同分全选），再补充活跃反应:")

    for conv_type in conversion_types:
        type_mask = df['Conversion_type'] == conv_type
        type_df = df[type_mask].copy()

        # 按SoC分数降序排序
        type_df_sorted = type_df.sort_values('SoC', ascending=False)

        # 设置类型内SoC排名
        for rank, idx in enumerate(type_df_sorted.index):
            df.loc[idx, 'Type_rank'] = rank + 1

        # 按SoC取前N%，处理同分问题
        ideal_count = max(1, int(len(type_df_sorted) * top_score_percentage))

        if ideal_count >= len(type_df_sorted):
            top_soc_reactions = type_df_sorted
            cutoff_soc = type_df_sorted['SoC'].min()
        else:
            cutoff_soc = type_df_sorted.iloc[ideal_count - 1]['SoC']
            top_soc_reactions = type_df_sorted[type_df_sorted['SoC'] >= cutoff_soc]

        # 标记前N%的选择原因
        for idx in top_soc_reactions.index:
            rank = df.loc[idx, 'Type_rank']
            is_active = df.loc[idx, 'Is_Active']
            soc_score = df.loc[idx, 'SoC']

            if is_active:
                df.loc[idx, 'Selection_reason'] = f'Top_{top_score_percentage:.0%}_SoC_{conv_type}_Active_Rank_{rank}_Score_{soc_score:.3f}'
            else:
                df.loc[idx, 'Selection_reason'] = f'Top_{top_score_percentage:.0%}_SoC_{conv_type}_Inactive_Rank_{rank}_Score_{soc_score:.3f}'

        # 在剩余反应中找活跃反应作为补充
        remaining_reactions = type_df_sorted[~type_df_sorted.index.isin(top_soc_reactions.index)]
        active_supplements = remaining_reactions[remaining_reactions['Is_Active'] == True]

        # 标记补充的活跃反应
        for idx in active_supplements.index:
            rank = df.loc[idx, 'Type_rank']
            soc_score = df.loc[idx, 'SoC']
            df.loc[idx, 'Selection_reason'] = f'Active_supplement_{conv_type}_SoC_Rank_{rank}_Score_{soc_score:.3f}'

        # 合并当前类型的选中反应
        type_selected = pd.concat([top_soc_reactions, active_supplements])
        selected_reactions.append(type_selected)

        print(f"    {conv_type}: 选中 {len(type_selected)} 个反应")

    # 合并所有选中的反应
    if selected_reactions:
        final_df = pd.concat(selected_reactions, ignore_index=True)

        # 最终排序（使用调整后的优先级）
        final_df['Priority_Order'] = final_df['Conversion_type'].map(priority_order).fillna(999)
        final_df = final_df.sort_values(['Priority_Order', 'SoC'], ascending=[True, False])
        final_df = final_df.reset_index(drop=True)
        final_df['Global_Rank'] = range(1, len(final_df) + 1)

        # 设置全局SoC排名
        final_df_soc_sorted = final_df.sort_values('SoC', ascending=False)
        for global_rank, idx in enumerate(final_df_soc_sorted.index):
            final_df.loc[idx, 'Global_SoC_rank'] = global_rank + 1

        final_df = final_df.drop('Priority_Order', axis=1)
    else:
        final_df = pd.DataFrame()

    # 为完整数据集也添加排序信息
    df_all = df.copy()
    df_all['Priority_Order'] = df_all['Conversion_type'].map(priority_order).fillna(999)
    df_all = df_all.sort_values(['Priority_Order', 'SoC'], ascending=[True, False])
    df_all = df_all.reset_index(drop=True)
    df_all['Global_Rank'] = range(1, len(df_all) + 1)

    # 设置全局SoC排名
    df_all_soc_sorted = df_all.sort_values('SoC', ascending=False)
    for global_rank, idx in enumerate(df_all_soc_sorted.index):
        df_all.loc[idx, 'Global_SoC_rank'] = global_rank + 1

    df_all = df_all.drop('Priority_Order', axis=1)

    print(f"  最终筛选结果: {len(final_df)} 个反应")

    if return_all:
        return df_all, final_df
    return final_df


def predict_ibridge_targets_optimized(model, exclude_met_file,
                                      target_compound=None,
                                      biomass_rxn='Bio_cgATCC14067_c',
                                      num_step=20,
                                      bio_fraction=0.1,
                                      threshold=0.01,
                                      top_score_percentage=0.4,
                                      max_depth=5,
                                      decay_factor=0.7,
                                      save_results=True,
                                      verbose=True):
    """
    iBridge优化版本 - 使用minpool方法获取通量分布
    使用调整后的优先级: Neg_to_Pos > Pos_to_Neg > Pos_Optimization
    """

    if verbose:
        print("=" * 60)
        print("iBridge 优化版本 - 使用 minpool 方法")
        print("优先级: Neg_to_Pos > Pos_to_Neg > Pos_Optimization")
        print("=" * 60)
        print(f"MOMA采样: {num_step}, 筛选比例: {top_score_percentage:.0%}")
        print(f"拓扑深度: {max_depth}, 衰减因子: {decay_factor}")

    # 确定目标反应
    if target_compound:
        target_chem_id = target_compound
    else:
        target_chem_id = '_'.join(model.id.split('_')[1:])
        target_chem_id = target_chem_id.split('_path_')[0]

    target_rxn = f'EX_{target_chem_id}_e'
    if verbose:
        print(f"目标反应: {target_rxn}")

    # 可行性检查
    with model as m:
        m.objective = biomass_rxn
        max_bio = m.slim_optimize()
        m.objective = target_rxn
        max_prod = m.slim_optimize()

        if isnan(max_bio) or max_bio < 1e-6:
            raise ValueError("模型无法生长")
        if isnan(max_prod) or max_prod < 1e-6:
            raise ValueError("无法产生目标产物")

    # 核心分析流程
    dict_reactant, dict_product = extract_metabolite_directions(model)

    # 使用 minpool 方法的活跃反应识别
    active_reactions = find_active_reactions(model, target_rxn, biomass_rxn, bio_fraction)

    # 使用 minpool 方法获取野生型参考的 MOMA 分析
    df_cov, df_flux = run_moma_analysis(model, biomass_rxn, target_rxn, num_step)

    # 拓扑网络分析
    print("构建拓扑网络...")
    reaction_network = build_reaction_network(model, active_reactions)
    reaction_weights = calculate_topology_weights(target_rxn, reaction_network,
                                                  max_depth, decay_factor)

    metabolic_network = build_metabolic_network(model, active_reactions)
    # 修复目标代谢物ID
    if target_rxn.startswith('EX_'):
        target_metabolite = target_rxn[3:-2] + '_c'  # EX_cpd00001_e -> cpd00001_c
    else:
        target_metabolite = target_rxn

    metabolite_weights = calculate_topology_weights(target_metabolite, metabolic_network,
                                                    max_depth, decay_factor)

    print(f"  目标代谢物: {target_metabolite}")
    print(f"  反应拓扑权重: {len(reaction_weights)}")
    print(f"  代谢物拓扑权重: {len(metabolite_weights)}")

    # 如果拓扑权重计算失败，使用备用方案
    if len(metabolite_weights) <= 1:
        print("  使用备用拓扑权重方案...")
        metabolite_weights = calculate_backup_topology_weights(model, target_metabolite)

    # 代谢物评分和目标反应识别
    metabolite_scores = calculate_metabolite_scores(
        model, df_cov, active_reactions, reaction_weights, metabolite_weights, threshold
    )

    # 新的调用方式（添加model参数）
    target_reactions = find_target_reactions(model, dict_reactant, dict_product, metabolite_scores, exclude_met_file)

    # 处理和筛选结果
    all_results, final_results = process_target_reactions(target_reactions, metabolite_scores,
                                                          active_reactions, model,
                                                          top_score_percentage, return_all=True)

    # 保存标准化输出文件
    if save_results:
        print("\n保存结果文件...")

        # 1. 通量矩阵
        # df_flux.to_csv('flux_matrix.csv', index=True)
        # print("  ✓ flux_matrix.csv")

        # 2. 代谢物评分
        metabolite_df = pd.DataFrame.from_dict(metabolite_scores, orient='index')
        metabolite_df = metabolite_df.sort_values('Score_Weighted',
                                                  key=lambda x: x.abs(), ascending=False)
        # metabolite_df.to_csv('metabolite_scores.csv', index=False)
        # print("  ✓ metabolite_scores.csv")

        # 3. 反应详细信息
        if not all_results.empty:
            output_columns = [
                'Target_reactions', 'Selection_reason', 'Global_Rank', 'Type_rank',
                'Regulation_type', 'Conversion_type', 'Priority_reason', 'SoC',
                'Is_Active', 'Source_metabolite', 'Destination_metabolite',
                'Source_score', 'Target_score', 'Source_Distance', 'Dest_Distance',
                'Source_Topo_Weight', 'Dest_Topo_Weight', 'Isozyme_count',
                'Active_isozymes', 'All_isozymes', 'Reaction_name', 'Reaction_equation'
            ]

            all_results_output = all_results[output_columns]
            # all_results_output.to_csv('reaction_details_all.csv', index=False)
            # print(f"  ✓ reaction_details_all.csv ({len(all_results)} 个反应)")

            if not final_results.empty:
                final_results_output = final_results[output_columns]
                # final_results_output.to_csv('reaction_details.csv', index=False)
                # print(f"  ✓ reaction_details.csv ({len(final_results)} 个筛选反应)")

    # 结果总结
    if verbose:
        print("\n" + "=" * 60)
        print("iBridge 分析完成（minpool 方法）")
        print("=" * 60)
        print(f"候选反应总数: {len(all_results)}")
        print(f"筛选后反应数: {len(final_results)}")

        if not final_results.empty:
            print(f"活跃反应: {sum(final_results['Is_Active'])}")
            print(f"上调反应: {sum(final_results['Regulation_type'] == 'Up')}")
            print(f"下调反应: {sum(final_results['Regulation_type'] == 'Down')}")

            print(f"\n前5个最佳候选反应（按新优先级）:")
            for i, (_, row) in enumerate(final_results.head().iterrows(), 1):
                active_mark = "✓" if row['Is_Active'] else "✗"
                print(f"  {i}. {row['Target_reactions']} ({row['Regulation_type']}) "
                      f"类型: {row['Conversion_type']} "
                      f"SoC: {row['SoC']:.3f} 活跃: {active_mark}")

    return final_results