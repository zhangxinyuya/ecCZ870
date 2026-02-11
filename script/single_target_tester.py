#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单基因靶点效果测试工具

功能：测试单个基因的上调(OE)或下调(KD)对目标产物产量的影响

使用方法：
    python single_target_tester.py model.json cpd00227 CEY17_RS03645 OE
    
或在Python中：
    from single_target_tester import test_single_target
    result = test_single_target('model.json', 'cpd00227', 'CEY17_RS03645', 'OE')
"""

import json
import os
import tempfile
import warnings
from copy import deepcopy

warnings.filterwarnings('ignore')


class SingleTargetTester:
    """单基因靶点效果测试器"""
    
    def __init__(self, 
                 model_file,
                 product_id,
                 biomass_rxn=None,
                 optimization_method='enzyme_pool',
                 growth_fraction=0.25,
                 oe_factor=5.0,
                 kd_factor=0.2):
        """
        初始化测试器
        
        Parameters
        ----------
        model_file : str
            模型JSON文件路径
        product_id : str
            目标产物ID (如 'cpd00227')
        biomass_rxn : str, optional
            生物量反应ID，None则自动检测
        optimization_method : str
            优化方法: 'enzyme_pool' 或 'pfba'
        growth_fraction : float
            生长率约束比例 (默认0.25，即25%最大生长)
        oe_factor : float
            过表达因子 (默认5.0)
        kd_factor : float
            敲低因子 (默认0.2)
        """
        self.model_file = model_file
        self.product_id = product_id
        self.product_rxn = f'EX_{product_id}_e'
        self.optimization_method = optimization_method
        self.growth_fraction = growth_fraction
        self.oe_factor = oe_factor
        self.kd_factor = kd_factor
        
        # 加载模型
        print(f"加载模型: {model_file}")
        self.model = self._load_model(model_file)
        
        # 确定生物量反应
        self.biomass_rxn = biomass_rxn or self._detect_biomass_rxn()
        print(f"生物量反应: {self.biomass_rxn}")
        print(f"产物反应: {self.product_rxn}")
        print(f"优化方法: {optimization_method}")
        
        # 保存原始kcat_MW值
        self.original_kcat_values = self._save_original_kcat()
        
        # 设置静默模式
        self._set_quiet_mode()
    
    def _load_model(self, model_file):
        """加载模型"""
        try:
            from ECMpy_function import get_enzyme_constraint_model
            return get_enzyme_constraint_model(model_file)
        except:
            import cobra
            return cobra.io.load_json_model(model_file)
    
    def _detect_biomass_rxn(self):
        """自动检测生物量反应"""
        # 从目标函数获取
        try:
            return list(self.model.objective.keys())[0].id
        except:
            pass
        
        # 搜索常见模式
        for rxn in self.model.reactions:
            if any(pattern in rxn.id for pattern in ['EX_biomass_c']):
                return rxn.id
        
        raise ValueError("无法自动检测生物量反应，请手动指定 biomass_rxn")
    
    def _save_original_kcat(self):
        """保存原始kcat_MW值"""
        kcat_values = {}
        for rxn in self.model.reactions:
            if hasattr(rxn, 'kcat_MW') and rxn.kcat_MW:
                try:
                    kcat_values[rxn.id] = float(rxn.kcat_MW)
                except:
                    pass
        print(f"找到 {len(kcat_values)} 个反应具有kcat_MW值")
        return kcat_values
    
    def _set_quiet_mode(self):
        """设置静默模式"""
        import logging
        logging.getLogger('cobra').setLevel(logging.ERROR)
        try:
            self.model.solver.configuration.verbosity = 0
        except:
            pass
    
    def _get_carbon_source_rxn(self):
        """获取碳源反应"""
        candidates = ['EX_cpd00027_e_reverse', 'EX_glc__D_e_reverse', 'EX_glc_e_reverse']
        for rxn_id in candidates:
            if rxn_id in self.model.reactions:
                return rxn_id
        
        # 搜索葡萄糖相关反应
        for rxn in self.model.reactions:
            if 'EX_' in rxn.id and ('glc' in rxn.id.lower() or 'cpd00027' in rxn.id):
                return rxn.id
        
        return None
    
    def optimize_minimize_enzyme_pool(self, model, objective_rxn, fraction_of_optimum=1.0):
        """
        两阶段优化：最小化酶池使用
        
        Stage 1: 找到最大目标值
        Stage 2: 固定目标，最小化酶成本
        """
        # Stage 1: 最大化目标
        model.objective = objective_rxn
        model.objective_direction = 'max'
        max_objective = model.slim_optimize()
        
        if max_objective is None or max_objective <= 1e-9:
            return model.optimize()
        
        # Stage 2: 固定目标，最小化酶成本
        target_flux = max_objective * fraction_of_optimum
        tolerance = abs(target_flux) * 0.001
        
        obj_rxn = model.reactions.get_by_id(objective_rxn)
        original_bounds = obj_rxn.bounds
        obj_rxn.bounds = (target_flux - tolerance, target_flux + tolerance)
        
        # 构建酶成本目标函数
        enzyme_cost_objective = {}
        default_penalty = 1e-9
        
        for rxn in model.reactions:
            if hasattr(rxn, 'kcat_MW') and rxn.kcat_MW:
                try:
                    kcat_mw = float(rxn.kcat_MW)
                    if kcat_mw > 0:
                        enzyme_cost_objective[rxn] = -1.0 / kcat_mw
                    else:
                        enzyme_cost_objective[rxn] = -default_penalty
                except:
                    enzyme_cost_objective[rxn] = -default_penalty
            else:
                enzyme_cost_objective[rxn] = -default_penalty
        
        model.objective = enzyme_cost_objective
        model.objective_direction = 'max'
        
        solution = model.optimize()
        obj_rxn.bounds = original_bounds
        
        return solution
    
    def _optimize(self, model, objective_rxn):
        """根据设置选择优化方法"""
        if self.optimization_method == 'enzyme_pool':
            return self.optimize_minimize_enzyme_pool(model, objective_rxn)
        else:
            from cobra.flux_analysis import pfba
            model.objective = objective_rxn
            return pfba(model)
    
    def get_baseline(self):
        """
        获取野生型基线
        
        Returns
        -------
        dict
            包含 max_growth, baseline_production, baseline_yield 等
        """
        print("\n" + "="*60)
        print("计算野生型基线")
        print("="*60)
        
        results = {}
        
        # 1. 最大生长率
        with self.model as m:
            m.objective = self.biomass_rxn
            sol = m.optimize()
            results['max_growth'] = sol.objective_value
            print(f"最大生长率: {results['max_growth']:.6f}")
        
        # 2. 最大产物产量
        with self.model as m:
            m.objective = self.product_rxn
            sol = m.optimize()
            results['max_product'] = sol.objective_value
            print(f"最大产物产量: {results['max_product']:.6f}")
        
        # 3. 约束生长率下的产物产量
        with self.model as m:
            growth_constraint = self.growth_fraction * results['max_growth']
            m.reactions.get_by_id(self.biomass_rxn).lower_bound = growth_constraint
            
            sol = self._optimize(m, self.product_rxn)
            
            results['constrained_growth'] = growth_constraint
            results['baseline_production'] = sol.fluxes[self.product_rxn]
            
            # 计算产率
            cs_rxn = self._get_carbon_source_rxn()
            if cs_rxn:
                cs_uptake = abs(sol.fluxes.get(cs_rxn, 0))
                results['carbon_uptake'] = cs_uptake
                results['baseline_yield'] = results['baseline_production'] / cs_uptake if cs_uptake > 0 else 0
            else:
                results['carbon_uptake'] = 0
                results['baseline_yield'] = 0
        
        print(f"约束生长率 ({self.growth_fraction*100:.0f}%): {results['constrained_growth']:.6f}")
        print(f"基线产量: {results['baseline_production']:.6f}")
        print(f"基线产率: {results['baseline_yield']:.6f}")
        
        return results
    
    def get_gene_info(self, gene_id):
        """获取基因信息"""
        try:
            gene = self.model.genes.get_by_id(gene_id)
            reactions = list(gene.reactions)
            
            info = {
                'gene_id': gene_id,
                'gene_name': gene.name if hasattr(gene, 'name') else '',
                'reaction_count': len(reactions),
                'reactions': []
            }
            
            for rxn in reactions:
                rxn_info = {
                    'id': rxn.id,
                    'name': rxn.name if hasattr(rxn, 'name') else '',
                    'subsystem': rxn.subsystem if hasattr(rxn, 'subsystem') else '',
                    'kcat_MW': self.original_kcat_values.get(rxn.id),
                    'equation': rxn.reaction
                }
                info['reactions'].append(rxn_info)
            
            return info
        except KeyError:
            return None
    
    def print_gene_info(self, gene_id):
        """打印基因信息"""
        info = self.get_gene_info(gene_id)
        
        if info is None:
            print(f"错误: 基因 {gene_id} 不存在于模型中")
            return
        
        print(f"\n基因: {info['gene_id']}")
        if info['gene_name']:
            print(f"名称: {info['gene_name']}")
        print(f"关联反应数: {info['reaction_count']}")
        
        print("\n关联反应:")
        print("-" * 80)
        for rxn in info['reactions']:
            kcat_str = f"{rxn['kcat_MW']:.4f}" if rxn['kcat_MW'] else "无"
            print(f"  {rxn['id']}")
            print(f"    kcat_MW: {kcat_str}")
            print(f"    反应式: {rxn['equation'][:60]}...")
    
    def test_modification(self, gene_id, modification_type='OE'):
        """
        测试单个基因修改的效果
        
        Parameters
        ----------
        gene_id : str
            基因ID
        modification_type : str
            修改类型: 'OE' (过表达) 或 'KD' (敲低)
        
        Returns
        -------
        dict
            测试结果
        """
        print(f"\n" + "="*60)
        print(f"测试基因 {gene_id} 的 {modification_type} 效果")
        print("="*60)
        
        # 验证基因存在
        gene_info = self.get_gene_info(gene_id)
        if gene_info is None:
            return {'status': 'error', 'message': f'基因 {gene_id} 不存在'}
        
        self.print_gene_info(gene_id)
        
        # 获取基线
        baseline = self.get_baseline()
        
        # 应用修改并测试
        print(f"\n应用 {modification_type} 修改...")
        factor = self.oe_factor if modification_type == 'OE' else self.kd_factor
        print(f"修改因子: {factor}x")
        
        modified_results = self._test_with_modification(gene_id, modification_type, baseline)
        
        # 汇总结果
        results = {
            'gene_id': gene_id,
            'modification': modification_type,
            'factor': factor,
            'status': 'success',
            
            # 基线
            'baseline_production': baseline['baseline_production'],
            'baseline_yield': baseline['baseline_yield'],
            
            # 修改后
            'modified_production': modified_results['production'],
            'modified_yield': modified_results['yield'],
            
            # 改进
            'prod_improvement': modified_results['prod_improvement'],
            'yield_improvement': modified_results['yield_improvement'],
            
            # 判断
            'is_effective': modified_results['prod_improvement'] > 0.1 or modified_results['yield_improvement'] > 0.1
        }
        
        # 打印结果
        self._print_results(results)
        
        return results
    
    def _test_with_modification(self, gene_id, modification_type, baseline):
        """应用修改并测试"""
        import cobra
        
        factor = self.oe_factor if modification_type == 'OE' else self.kd_factor
        
        # 加载模型JSON
        with open(self.model_file, 'r') as f:
            model_dict = json.load(f)
        
        # 修改kcat_MW
        modified_rxns = []
        for rxn in model_dict['reactions']:
            gene_rule = rxn.get('gene_reaction_rule', '')
            if gene_id in gene_rule:
                original_kcat = self.original_kcat_values.get(rxn['id'])
                if original_kcat:
                    new_kcat = original_kcat * factor
                    rxn['kcat_MW'] = new_kcat
                    modified_rxns.append({
                        'id': rxn['id'],
                        'original': original_kcat,
                        'new': new_kcat
                    })
        
        print(f"\n修改了 {len(modified_rxns)} 个反应的kcat_MW:")
        for r in modified_rxns:
            print(f"  {r['id']}: {r['original']:.4f} -> {r['new']:.4f}")
        
        # 确保酶约束存在
        if 'enzyme_constraint' not in model_dict:
            model_dict['enzyme_constraint'] = {
                'enzyme_mass_fraction': 0.46,
                'total_protein_fraction': 0.56,
                'average_saturation': 0.5,
                'lowerbound': 0,
                'upperbound': 0.129
            }
        
        # 保存并重新加载
        with tempfile.NamedTemporaryFile(suffix='.json', mode='w', delete=False) as f:
            json.dump(model_dict, f)
            temp_file = f.name
        
        try:
            modified_model = self._load_model(temp_file)
            self._set_quiet_mode()
            
            # 测试
            with modified_model as m:
                # 获取新的最大生长率
                m.objective = self.biomass_rxn
                sol_growth = m.optimize()
                max_growth = sol_growth.objective_value
                
                # 约束生长率
                m.reactions.get_by_id(self.biomass_rxn).lower_bound = self.growth_fraction * max_growth
                
                # 优化产物
                sol = self._optimize(m, self.product_rxn)
                
                production = sol.fluxes[self.product_rxn]
                
                cs_rxn = self._get_carbon_source_rxn()
                cs_uptake = abs(sol.fluxes.get(cs_rxn, 1.0)) if cs_rxn else 1.0
                yield_value = production / cs_uptake if cs_uptake > 0 else 0
            
            # 计算改进
            prod_imp = ((production - baseline['baseline_production']) / 
                       baseline['baseline_production'] * 100) if baseline['baseline_production'] > 0 else 0
            yield_imp = ((yield_value - baseline['baseline_yield']) / 
                        baseline['baseline_yield'] * 100) if baseline['baseline_yield'] > 0 else 0
            
            return {
                'production': production,
                'yield': yield_value,
                'prod_improvement': prod_imp,
                'yield_improvement': yield_imp
            }
            
        finally:
            os.remove(temp_file)
    
    def _print_results(self, results):
        """打印测试结果"""
        print("\n" + "="*60)
        print("测试结果")
        print("="*60)
        
        print(f"\n基因: {results['gene_id']}")
        print(f"修改类型: {results['modification']} ({results['factor']}x)")
        
        print(f"\n{'指标':<20} {'基线':<15} {'修改后':<15} {'变化':<15}")
        print("-" * 65)
        
        prod_change = f"{results['prod_improvement']:+.4f}%"
        yield_change = f"{results['yield_improvement']:+.4f}%"
        
        print(f"{'产量':<20} {results['baseline_production']:<15.6f} {results['modified_production']:<15.6f} {prod_change:<15}")
        print(f"{'产率':<20} {results['baseline_yield']:<15.6f} {results['modified_yield']:<15.6f} {yield_change:<15}")
        
        print("\n" + "-" * 65)
        if results['is_effective']:
            print(f"✓ 该修改有效！建议将 {results['gene_id']} 作为 {results['modification']} 靶点")
        else:
            print(f"✗ 该修改效果不显著 (改进 < 0.1%)")
    
    def test_both_directions(self, gene_id):
        """测试OE和KD两个方向"""
        print(f"\n" + "="*70)
        print(f"测试基因 {gene_id} 的双向效果")
        print("="*70)
        
        oe_result = self.test_modification(gene_id, 'OE')
        kd_result = self.test_modification(gene_id, 'KD')
        
        # 比较结果
        print("\n" + "="*70)
        print("双向比较")
        print("="*70)
        
        print(f"\n{'方向':<10} {'产量改进':<15} {'产率改进':<15} {'是否有效':<10}")
        print("-" * 50)
        print(f"{'OE':<10} {oe_result['prod_improvement']:+.4f}%{'':<6} {oe_result['yield_improvement']:+.4f}%{'':<6} {'✓' if oe_result['is_effective'] else '✗':<10}")
        print(f"{'KD':<10} {kd_result['prod_improvement']:+.4f}%{'':<6} {kd_result['yield_improvement']:+.4f}%{'':<6} {'✓' if kd_result['is_effective'] else '✗':<10}")
        
        # 推荐
        print("\n推荐:")
        if oe_result['is_effective'] and kd_result['is_effective']:
            if oe_result['prod_improvement'] > kd_result['prod_improvement']:
                print(f"  两个方向都有效，OE效果更好 (+{oe_result['prod_improvement']:.4f}%)")
            else:
                print(f"  两个方向都有效，KD效果更好 (+{kd_result['prod_improvement']:.4f}%)")
        elif oe_result['is_effective']:
            print(f"  推荐 OE (过表达)")
        elif kd_result['is_effective']:
            print(f"  推荐 KD (敲低)")
        else:
            print("  两个方向都不显著，该基因可能不是有效靶点")
        
        return {'OE': oe_result, 'KD': kd_result}


# ==================== 便捷函数 ====================

def test_single_target(model_file, product_id, gene_id, modification='OE', 
                       biomass_rxn=None, method='enzyme_pool'):
    """
    一键测试单个靶点
    
    Parameters
    ----------
    model_file : str
        模型文件路径
    product_id : str
        产物ID (如 'cpd00227')
    gene_id : str
        基因ID
    modification : str
        'OE', 'KD', 或 'both'
    biomass_rxn : str, optional
        生物量反应ID
    method : str
        优化方法: 'enzyme_pool' 或 'pfba'
    
    Returns
    -------
    dict
        测试结果
    """
    tester = SingleTargetTester(
        model_file=model_file,
        product_id=product_id,
        biomass_rxn=biomass_rxn,
        optimization_method=method
    )
    
    if modification.lower() == 'both':
        return tester.test_both_directions(gene_id)
    else:
        return tester.test_modification(gene_id, modification.upper())


def batch_test_targets(model_file, product_id, gene_list, modification='OE',
                       biomass_rxn=None, method='enzyme_pool'):
    """
    批量测试多个靶点
    
    Parameters
    ----------
    model_file : str
        模型文件路径
    product_id : str
        产物ID
    gene_list : list
        基因ID列表
    modification : str
        'OE' 或 'KD'
    
    Returns
    -------
    pd.DataFrame
        所有测试结果
    """
    import pandas as pd
    
    tester = SingleTargetTester(
        model_file=model_file,
        product_id=product_id,
        biomass_rxn=biomass_rxn,
        optimization_method=method
    )
    
    results = []
    for gene_id in gene_list:
        print(f"\n{'='*60}")
        print(f"测试 {gene_id}")
        result = tester.test_modification(gene_id, modification)
        results.append(result)
    
    df = pd.DataFrame(results)
    df = df.sort_values('prod_improvement', ascending=False)
    
    print("\n" + "="*70)
    print("批量测试结果汇总")
    print("="*70)
    print(df[['gene_id', 'modification', 'prod_improvement', 'yield_improvement', 'is_effective']])
    
    return df


# ==================== 主函数 ====================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) >= 4:
        model_file = sys.argv[1]
        product_id = sys.argv[2]
        gene_id = sys.argv[3]
        modification = sys.argv[4] if len(sys.argv) > 4 else 'OE'
        method = sys.argv[5] if len(sys.argv) > 5 else 'enzyme_pool'
        
        test_single_target(model_file, product_id, gene_id, modification, method=method)
    
    else:
        print("""
单基因靶点效果测试工具
======================

命令行使用:
    python single_target_tester.py <模型.json> <产物ID> <基因ID> [OE/KD/both] [enzyme_pool/pfba]

示例:
    python single_target_tester.py model.json cpd00227 CEY17_RS03645 OE
    python single_target_tester.py model.json cpd00227 CEY17_RS03645 both
    python single_target_tester.py model.json cpd00227 CEY17_RS03645 KD pfba

Python中使用:
    from single_target_tester import test_single_target, batch_test_targets
    
    # 测试单个靶点
    result = test_single_target(
        model_file='model.json',
        product_id='cpd00227',
        gene_id='CEY17_RS03645',
        modification='OE'  # 或 'KD' 或 'both'
    )
    
    # 批量测试
    results = batch_test_targets(
        model_file='model.json',
        product_id='cpd00227',
        gene_list=['CEY17_RS03645', 'CEY17_RS08445', 'CEY17_RS08450'],
        modification='OE'
    )
        """)
