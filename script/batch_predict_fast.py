#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ecFactory 批量预测脚本 - 修复版
解决了多进程环境下 sys.path 传递和模块导入的问题

模块依赖:
- ecFbridge_minpool.py
- ecFactory_light_minpool.py
"""

import pandas as pd
import sys
import os
import time
import warnings
from datetime import datetime
from multiprocessing import Pool, cpu_count
import traceback

warnings.filterwarnings('ignore')

# ==================== 路径配置 ====================
# 获取当前脚本所在目录的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 配置依赖模块的路径（请根据实际情况修改）
PATHS_TO_ADD = [
    os.path.join(SCRIPT_DIR, 'MEResource'),
    os.path.join(SCRIPT_DIR, 'script'),
    SCRIPT_DIR,  # 当前目录
]

# 在主进程中添加路径
for path in PATHS_TO_ADD:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

# 尝试导入 tqdm
try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("提示: 安装 tqdm 可获得更好的进度显示 (pip install tqdm)")


# ==================== 子进程初始化 ====================
def _init_worker(paths_to_add):
    """
    子进程初始化函数
    在每个子进程启动时调用，确保模块路径正确设置
    """
    import sys
    import warnings
    warnings.filterwarnings('ignore')

    # 在子进程中添加路径
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)


# ==================== 单产物处理函数 ====================
def process_single_product(args):
    """
    处理单个产物的函数（在子进程中运行）
    """
    product_id, model_file, batch_output_dir, paths_to_add, kwargs = args
    start_time = time.time()

    # 确保子进程中路径正确
    import sys
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)

    try:

        from ecFbridge_minpool import IntegratedTargetIdentifier

        # 创建标识器并运行
        identifier = IntegratedTargetIdentifier(
            model_file=model_file,
            product_id=product_id,
            output_dir=batch_output_dir,
            **kwargs
        )
        final_targets = identifier.run_integrated_pipeline()

        return {
            'product_id': product_id,
            'OE_targets': len(final_targets.get('OE', [])),
            'KD_targets': len(final_targets.get('KD', [])),
            'status': 'success',
            'time': time.time() - start_time,
            'output_file': f"targets_{product_id}.csv",
            'targets': final_targets,
            'error': None
        }

    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        return {
            'product_id': product_id,
            'OE_targets': 0,
            'KD_targets': 0,
            'status': 'failed',
            'time': time.time() - start_time,
            'output_file': 'N/A',
            'targets': {'OE': [], 'KD': []},
            'error': error_msg
        }


# ==================== 批量预测主函数 ====================
def batch_predict(
        model_file,
        product_ids,
        output_base_dir='./results',
        n_workers=None,
        resume_from=None,
        show_progress=True,
        save_interval=3,
        **kwargs
):
    """
    并行批量预测函数

    Parameters
    ----------
    model_file : str
        模型文件路径
    product_ids : list
        产物ID列表
    output_base_dir : str
        输出基础目录
    n_workers : int, optional
        并行进程数，默认为 CPU核心数-1
    resume_from : str, optional
        从指定目录继续处理（断点续传）
    show_progress : bool
        是否显示进度条
    save_interval : int
        每处理几个产物保存一次中间结果
    **kwargs : dict
        传递给 IntegratedTargetIdentifier 的其他参数

    Returns
    -------
    tuple
        (results_dict, output_dir)
    """

    # 确定进程数
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    else:
        n_workers = max(1, min(n_workers, cpu_count()))

    # 转换为绝对路径
    model_file = os.path.abspath(model_file)

    # 创建或恢复输出目录
    if resume_from and os.path.exists(resume_from):
        batch_output_dir = os.path.abspath(resume_from)
        print(f"📂 从已有批次继续: {batch_output_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_output_dir = os.path.abspath(
            os.path.join(output_base_dir, f"batch_{timestamp}")
        )
        os.makedirs(batch_output_dir, exist_ok=True)

    # 检查已处理的产物（断点续传支持）
    processed_products = set()
    summary_file = os.path.join(batch_output_dir, 'batch_summary.csv')
    existing_summary = []

    if os.path.exists(summary_file):
        existing_df = pd.read_csv(summary_file)
        existing_summary = existing_df.to_dict('records')
        processed_products = set(
            existing_df[existing_df['status'] == 'success']['product_id']
        )
        if processed_products:
            print(f"✓ 发现 {len(processed_products)} 个已完成的产物")

    # 过滤待处理产物
    remaining_products = [pid for pid in product_ids if pid not in processed_products]

    if not remaining_products:
        print("✓ 所有产物都已处理完成！")
        return {}, batch_output_dir

    # 打印配置信息
    print(f"\n{'=' * 70}")
    print(f"🚀 ecFactory 批量预测")
    print(f"{'=' * 70}")
    print(f"📊 总产物数: {len(product_ids)}")
    print(f"✅ 已完成: {len(processed_products)}")
    print(f"⏳ 待处理: {len(remaining_products)}")
    print(f"🔧 并行进程数: {n_workers}")
    print(f"📁 模型文件: {model_file}")
    print(f"📁 输出目录: {batch_output_dir}")
    print(f"{'=' * 70}\n")

    # 准备进程参数
    process_args = [
        (pid, model_file, batch_output_dir, PATHS_TO_ADD, kwargs)
        for pid in remaining_products
    ]

    # 开始并行处理
    start_time = time.time()
    all_results = []

    # 创建进程池配置
    pool_config = {
        'processes': n_workers,
        'initializer': _init_worker,
        'initargs': (PATHS_TO_ADD,)
    }

    if HAS_TQDM and show_progress:
        # 带进度条的处理
        with Pool(**pool_config) as pool:
            iterator = pool.imap_unordered(process_single_product, process_args)

            with tqdm(
                    total=len(remaining_products),
                    desc="处理进度",
                    unit="个",
                    ncols=100
            ) as pbar:

                for i, result in enumerate(iterator, 1):
                    all_results.append(result)

                    # 更新进度条
                    if result['status'] == 'success':
                        status = f"✓ {result['product_id']}"
                    else:
                        status = f"✗ {result['product_id']}"
                    pbar.set_postfix_str(status)
                    pbar.update(1)

                    # 定期保存
                    if i % save_interval == 0:
                        _save_results(all_results, existing_summary,
                                      batch_output_dir, summary_file, final=False)
    else:
        # 无进度条的处理
        with Pool(**pool_config) as pool:
            print("⏳ 处理中...")
            for i, result in enumerate(
                    pool.imap_unordered(process_single_product, process_args), 1
            ):
                all_results.append(result)

                # 打印进度
                icon = "✅" if result['status'] == 'success' else "❌"
                print(
                    f"{icon} [{i}/{len(remaining_products)}] {result['product_id']} "
                    f"(OE:{result['OE_targets']}, KD:{result['KD_targets']}, "
                    f"{result['time']:.1f}s)"
                )

                # 定期保存
                if i % save_interval == 0:
                    _save_results(all_results, existing_summary,
                                  batch_output_dir, summary_file, final=False)

    # 最终保存
    _save_results(all_results, existing_summary, batch_output_dir, summary_file, final=True)

    # 统计信息
    total_time = time.time() - start_time
    success_count = sum(1 for r in all_results if r['status'] == 'success')
    failed_count = len(all_results) - success_count
    avg_time = sum(r['time'] for r in all_results) / len(all_results) if all_results else 0
    speedup = (len(all_results) * avg_time) / total_time if total_time > 0 else 1

    # 打印统计
    print(f"\n{'=' * 70}")
    print(f"✅ 批量预测完成!")
    print(f"{'=' * 70}")
    print(f"⏱️  总耗时: {total_time:.1f}秒 ({total_time / 60:.1f}分钟)")
    print(f"✅ 成功: {success_count}/{len(all_results)}")
    print(f"❌ 失败: {failed_count}/{len(all_results)}")
    print(f"⚡ 平均: {avg_time:.1f}秒/个")
    print(f"🚀 加速比: {speedup:.1f}x")

    # 处理失败情况
    if failed_count > 0:
        print(f"\n⚠️  失败的产物:")
        error_log_file = os.path.join(batch_output_dir, 'error_log.txt')

        with open(error_log_file, 'w', encoding='utf-8') as f:
            for r in all_results:
                if r['status'] == 'failed':
                    short_error = str(r['error']).split('\n')[0][:80]
                    print(f"   - {r['product_id']}: {short_error}")

                    f.write(f"{'=' * 60}\n")
                    f.write(f"Product: {r['product_id']}\n")
                    f.write(f"Time: {r['time']:.1f}s\n")
                    f.write(f"Error:\n{r['error']}\n\n")

        print(f"\n   📄 详细错误日志: {error_log_file}")

    # 输出文件信息
    print(f"\n📁 输出文件:")
    print(f"   - 摘要: {summary_file}")
    combined_file = os.path.join(batch_output_dir, 'all_targets_combined.csv')
    if os.path.exists(combined_file):
        print(f"   - 合并结果: {combined_file}")
    print(f"{'=' * 70}\n")

    # 返回结果
    results_dict = {r['product_id']: r['targets'] for r in all_results}
    return results_dict, batch_output_dir


# ==================== 辅助函数 ====================
def _save_results(new_results, existing_summary, batch_output_dir, summary_file, final=False):
    """保存结果到文件"""

    # 保存摘要
    summary_data = existing_summary + [
        {
            'product_id': r['product_id'],
            'OE_targets': r['OE_targets'],
            'KD_targets': r['KD_targets'],
            'status': r['status'],
            'time': round(r['time'], 2),
            'output_file': r['output_file'],
            'error': str(r.get('error', ''))[:200] if r.get('error') else ''
        }
        for r in new_results
    ]
    pd.DataFrame(summary_data).to_csv(summary_file, index=False)

    # 最终保存时合并所有靶点
    if final:
        all_targets = []
        for result in new_results:
            product_id = result['product_id']
            targets = result['targets']
            for target_type in ['OE', 'KD']:
                for gene_id in targets.get(target_type, []):
                    all_targets.append({
                        'product_id': product_id,
                        'gene_id': gene_id,
                        'type': target_type
                    })

        if all_targets:
            combined_file = os.path.join(batch_output_dir, 'all_targets_combined.csv')
            new_df = pd.DataFrame(all_targets)

            if os.path.exists(combined_file):
                existing_df = pd.read_csv(combined_file)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df.drop_duplicates(inplace=True)
            else:
                combined_df = new_df

            combined_df.to_csv(combined_file, index=False)


def batch_predict_serial(model_file, product_ids, output_base_dir='./results', **kwargs):
    """
    串行批量预测（用于调试，不使用多进程）
    """
    from ecFbridge_minpool import IntegratedTargetIdentifier

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_output_dir = os.path.join(output_base_dir, f"batch_{timestamp}")
    os.makedirs(batch_output_dir, exist_ok=True)

    results = {}
    failed = []

    print(f"\n{'=' * 70}")
    print(f"🔧 串行模式（调试用）")
    print(f"{'=' * 70}\n")

    for i, product_id in enumerate(product_ids, 1):
        print(f"\n[{i}/{len(product_ids)}] 处理 {product_id}...")
        try:
            identifier = IntegratedTargetIdentifier(
                model_file=model_file,
                product_id=product_id,
                output_dir=batch_output_dir,
                **kwargs
            )
            final_targets = identifier.run_integrated_pipeline()
            results[product_id] = final_targets
            print(f"✅ {product_id}: OE={len(final_targets.get('OE', []))}, "
                  f"KD={len(final_targets.get('KD', []))}")
        except Exception as e:
            print(f"❌ {product_id}: {e}")
            traceback.print_exc()
            results[product_id] = {'OE': [], 'KD': []}
            failed.append(product_id)

    print(f"\n{'=' * 70}")
    print(f"完成: 成功 {len(product_ids) - len(failed)}, 失败 {len(failed)}")
    print(f"{'=' * 70}\n")

    return results, batch_output_dir


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    # ========== 配置参数 ==========

    # 产物列表
    PRODUCT_IDS = [
        "cpd00039", "cpd00322", "cpd00060", "cpd00107",
        "cpd00156", "cpd00161", "cpd00227", "cpd00129",
        "cpd00051", "cpd00064", "cpd00274", "cpd00069",
        "cpd00065", "cpd00054", "cpd00084", "cpd00119",
        "cpd00036", "cpd00066",
    ]

    # 模型文件路径
    MODEL_FILE = "./model_Calibration/DLKcat_Calibration/ecCG827_DLKcat_Calibration_modify.json"

    # 输出目录
    OUTPUT_DIR = "prediction_target_ecFbridge"

    # ========== 运行预测 ==========

    # 方式1: 并行预测（推荐）
    results, output_dir = batch_predict(
        model_file=MODEL_FILE,
        product_ids=PRODUCT_IDS,
        output_base_dir=OUTPUT_DIR,
        n_workers=None,  # 自动选择进程数
        show_progress=True,  # 显示进度条
        save_interval=3,  # 每3个产物保存一次
        # 以下为 IntegratedTargetIdentifier 的参数
        enable_ibridge=True,
        improvement_threshold=0.1,
        exclusion_threshold=-5.0,
        enable_distance_filter=True,
    )

    # 方式2: 串行预测（调试用）
    # results, output_dir = batch_predict_serial(
    #     model_file=MODEL_FILE,
    #     product_ids=PRODUCT_IDS[:2],  # 先测试前2个
    #     output_base_dir=OUTPUT_DIR,
    #     enable_ibridge=True,
    #     improvement_threshold=0.1,
    #     exclusion_threshold=-5.0,
    #     enable_distance_filter=True,
    # )

    print(f"✅ 所有结果保存在: {output_dir}")