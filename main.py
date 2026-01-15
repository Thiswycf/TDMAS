"""
主程序入口
实现完整的训练pipeline：数据收集、优化、验证
"""

import asyncio
import argparse
import os
import json
import yaml
import shutil
import importlib
import subprocess
import sys
import datetime
from typing import Optional
from metagpt.logs import logger
from termcolor import cprint

from utils.llm_manager import get_global_llm_manager
from utils.path_utils import (
    ensure_dir,
    workspace_path,
    build_model_name,
    evaluation_output_dir,
    solve_rate_file,
)
import ScoreFlow.params
from tdmas.data_collector import DataCollector
from tdmas.preference_data import generate_preference_pairs, save_preference_data
from tdmas.evaluator import Evaluator


def load_config(config_path: str = "config/running_config.yaml") -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def copy_config_to_workspace(config_path: str, dataset: str, zcp: str = "accuracy"):
    """复制配置文件到workspace文件夹"""
    workspace_config_dir = workspace_path("config", dataset, zcp)
    ensure_dir(workspace_config_dir)

    config_filename = os.path.basename(config_path)
    dest_path = os.path.join(workspace_config_dir, config_filename)

    shutil.copy2(config_path, dest_path)
    logger.info(f"配置文件已复制到: {dest_path}")


def load_dataset(dataset_name: str, data_type: str = "validate"):
    """加载数据集"""
    bench_dic = ScoreFlow.params.bench_dic
    if dataset_name not in bench_dic:
        raise ValueError(f"未知的数据集: {dataset_name}")

    benchmark_info = bench_dic[dataset_name]
    benchmark_name = benchmark_info["benchmark_name"]

    # 加载benchmark模块
    benchmark_module_path = f"ScoreFlow.benchmark.{benchmark_name}"
    module = importlib.import_module(benchmark_module_path)
    benchmark_class = getattr(module, benchmark_info["module_name"])
    benchmark = benchmark_class(
        name=dataset_name, file_path="data", log_path="")

    # 加载数据
    data_file = f"datasets/{benchmark_name}/{data_type}.jsonl"
    data = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    return benchmark, data


async def collect_training_data(
    model_name: str,
    dataset: str,
    benchmark,
    training_data: list,
    epoch: int,
    limit: Optional[int] = None,
    max_depth: int = 5,
    max_concurrent_request: int = 10,
    max_concurrent_execute_code: int = 128,
    train_ask_num: int = 8,
    max_loop: int = 5,
    zcp: str = "accuracy"
):
    """收集训练数据"""
    cprint(
        f"[TDMAS] 开始收集训练数据 | Epoch {epoch} | Dataset {dataset} | Train Ask Num {train_ask_num}", color="blue")

    # 限制数据量
    if limit is not None:
        training_data = training_data[:limit]

    # 创建数据收集器
    collector = DataCollector(model_name, dataset, benchmark)

    # 收集数据（对每个问题重复询问 train_ask_num 次）
    collected_data = await collector.collect_batch(
        training_data,
        max_depth=max_depth,
        max_concurrent_request=max_concurrent_request,
        max_concurrent_execute_code=max_concurrent_execute_code,
        train_ask_num=train_ask_num,
        max_loop=max_loop
    )

    cprint(
        f"[TDMAS] 训练数据收集完成 | Epoch {epoch} | 收集了 {len(collected_data)} 条数据", color="green")
    logger.info(f"训练数据收集完成，共收集 {len(collected_data)} 条数据")

    # 计算训练集准确率
    correct_count = 0
    total_count = 0

    # 按问题分组统计（每个问题有ask_num个回答，只统计第一个回答的准确率）
    for data in collected_data:
        correct_count += data.get('correct', 0)
        total_count += 1

    train_accuracy = (correct_count / total_count * 100) if total_count > 0 else 0.0

    cprint(
        f"[TDMAS] 训练集准确率 | Epoch {epoch} | Accuracy: {train_accuracy:.4f}% ({correct_count}/{total_count})",
        color="green"
    )
    logger.info(
        f"训练集准确率: {train_accuracy:.4f}% ({correct_count}/{total_count})")

    # 保存训练集准确率到文件
    result_dir = evaluation_output_dir(dataset, zcp)
    ensure_dir(result_dir)
    result_file = solve_rate_file(dataset, zcp, "optimize")
    with open(result_file, "a") as f:
        f.write(
            f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] | Limit {limit} | Epoch {epoch} | Train Solve Rate: {train_accuracy:.4f}%\n"
        )

    return collected_data


async def generate_preference_data_from_collected(
    collected_data: list,
    dataset: str,
    zcp: str,
    epoch: int,
    similarity_threshold: Optional[float] = None,
    preference_pairs_limit: Optional[int] = None
):
    """从收集的数据生成preference data"""
    cprint(f"[TDMAS] 开始生成preference data | Epoch {epoch}", color="blue")

    preference_pairs = generate_preference_pairs(
        collected_data, similarity_threshold=similarity_threshold, preference_pairs_limit=preference_pairs_limit)
    save_preference_data(preference_pairs, dataset, zcp, epoch)

    cprint(
        f"[TDMAS] preference data生成完成 | Epoch {epoch} | 生成了 {len(preference_pairs)} 个偏好对", color="green")
    logger.info(f"preference data生成完成，共 {len(preference_pairs)} 个偏好对")


async def optimize_model(
    epoch: int,
    dataset: str,
    zcp: str = "accuracy",
    train_only: bool = False
):
    """优化模型（DPO训练）"""
    cprint(f"[TDMAS] 开始优化模型 | Epoch {epoch}", color="blue")

    # 在独立子进程中运行优化，避免主进程持有CUDA上下文
    trainer_script = os.path.join(
        os.path.dirname(__file__), "tdmas/optimize.py")
    cmd = [
        sys.executable,
        trainer_script,
        "--dataset",
        dataset,
        "--epoch",
        str(epoch),
        "--zcp",
        zcp,
    ]
    if train_only:
        cmd.append("--train_only")

    subprocess.run(cmd, check=True)

    cprint(f"[TDMAS] 模型优化完成 | Epoch {epoch}", color="green")
    logger.info(f"模型优化完成")


async def evaluate_model(
    model_name: str,
    dataset: str,
    benchmark,
    test_data: list,
    epoch: int,
    config: dict,
    zcp: str = "accuracy"
):
    """评估模型性能"""
    cprint(f"[TDMAS] 开始评估模型 | Epoch {epoch} | Dataset {dataset}", color="blue")
    
    limit=config.get('limit')
    max_depth=config.get('max_depth', 5)
    max_loop=config.get('max_loop', 5)
    max_concurrent_request=config.get('max_concurrent_request', 10)
    max_concurrent_execute_code=config.get('max_concurrent_execute_code', 128)
    test_ask_num=config.get('test_ask_num', 8)

    # 限制数据量
    if limit is not None:
        test_data = test_data[:limit]

    # 创建评估器
    evaluator = Evaluator(model_name, dataset, benchmark)

    # 评估
    results = await evaluator.evaluate_batch(
        test_data,
        max_depth=max_depth,
        max_concurrent_request=max_concurrent_request,
        max_concurrent_execute_code=max_concurrent_execute_code,
        max_loop=max_loop,
        test_ask_num=test_ask_num
    )

    # 计算准确率
    accuracy = evaluator.calculate_accuracy(results)
    accuracy_percentage = accuracy * 100

    cprint(
        f"[TDMAS] 模型评估完成 | Epoch {epoch} | 测试集准确率: {accuracy_percentage:.4f}%", color="green")
    logger.info(f"模型评估完成，测试集准确率: {accuracy_percentage:.4f}%")

    # 保存测试集准确率到文件
    result_dir = evaluation_output_dir(dataset, zcp)
    ensure_dir(result_dir)
    result_file = solve_rate_file(dataset, zcp, "inference")
    with open(result_file, "a") as f:
        f.write(
            f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] | Limit {limit} | Epoch {epoch} | Test Solve Rate: {accuracy_percentage:.4f}%\n"
        )

    return accuracy, results


async def run_training_epoch(
    config: dict,
    epoch: int,
    benchmark,
    training_data: list,
    model_name: str,
    train_only: bool = False
):
    """运行一个训练epoch"""
    dataset = config['dataset']
    zcp = config.get('zcp', 'accuracy')
    limit = config.get('limit')
    max_depth = config.get('max_depth', 5)
    max_concurrent_request = config.get('max_concurrent_request', 10)
    max_concurrent_execute_code = config.get('max_concurrent_execute_code', 128)
    train_ask_num = config.get('train_ask_num', 8)
    max_loop = config.get('max_loop', 5)

    cprint(f"[TDMAS] {'='*10} Epoch {epoch} {'='*10}", color="blue")

    # 1. 收集训练数据
    collected_data = await collect_training_data(
        model_name,
        dataset,
        benchmark,
        training_data,
        epoch,
        limit=limit,
        max_depth=max_depth,
        max_concurrent_request=max_concurrent_request,
        max_concurrent_execute_code=max_concurrent_execute_code,
        train_ask_num=train_ask_num,
        max_loop=max_loop,
        zcp=zcp
    )
    get_global_llm_manager().clear_all()
    cprint("All LLM models have been cleared!", color="green")
    logger.info("All LLM models have been cleared!")

    # 2. 生成preference data
    similarity_threshold = config.get('similarity_threshold')
    preference_pairs_limit = config.get('preference_pairs_limit')
    await generate_preference_data_from_collected(
        collected_data,
        dataset,
        zcp,
        epoch,
        similarity_threshold=similarity_threshold,
        preference_pairs_limit=preference_pairs_limit
    )

    # 3. 优化模型
    await optimize_model(epoch, dataset, zcp, train_only=train_only)

    return


async def run_pipeline(config_path: str = "config/running_config.yaml"):
    """运行训练pipeline"""
    # 加载配置
    config = load_config(config_path)

    dataset = config['dataset']
    start_epoch = config.get('start_epoch', 0)
    end_epoch = config.get('end_epoch', 3)
    task_type = config.get('task_type', 'full_pipeline')
    zcp = config.get('zcp', 'accuracy')
    max_output_tokens = config.get('max_output_tokens', 1024)

    get_global_llm_manager(agent_max_output_tokens=max_output_tokens)

    # 复制配置文件到workspace
    copy_config_to_workspace(config_path, dataset, zcp)

    # 加载数据集
    benchmark, training_data = load_dataset(dataset, "validate")
    _, test_data = load_dataset(dataset, "test")
    
    # 为每个entry分配唯一问题id（递增）
    next_auto_id = 0
    for entry in training_data:
        entry['id'] = f"auto_{next_auto_id}"
        next_auto_id += 1
    for entry in test_data:
        entry['id'] = f"auto_{next_auto_id}"
        next_auto_id += 1

    # 加载模型配置
    with open("config/local_llm.yaml", "r") as f:
        llm_config = yaml.safe_load(f)

    base_model_name = config.get(
        'model_name', llm_config.get('default_llm', 'Qwen3-8B'))

    # 确定运行模式
    train_only = (task_type == 'train_only')
    infer_only = (task_type == 'infer_only')

    cprint(
        f"[TDMAS] 开始运行 | Mode={task_type} | Dataset={dataset} | Epoch={start_epoch}→{end_epoch} | ZCP={zcp}",
        color="green"
    )

    if infer_only:
        # 只执行评估
        for epoch in range(start_epoch, end_epoch + 1):
            eval_model_name = build_model_name(base_model_name, dataset, zcp, epoch)
            accuracy, results = await evaluate_model(
                eval_model_name,
                dataset,
                benchmark,
                test_data,
                epoch,
                config,
                zcp=zcp
            )
    elif train_only:
        # 只执行训练
        for epoch in range(start_epoch, end_epoch):
            _model_name = build_model_name(base_model_name, dataset, zcp, epoch)
            await run_training_epoch(
                config,
                epoch,
                benchmark,
                training_data,
                _model_name,
                train_only=True
            )
    else:
        # 完整的pipeline：训练 + 评估
        for epoch in range(start_epoch, end_epoch):
            _model_name = build_model_name(base_model_name, dataset, zcp, epoch)
            accuracy, results = await evaluate_model(
                _model_name,
                dataset,
                benchmark,
                test_data,
                epoch,
                config,
                zcp=zcp
            )
            await run_training_epoch(
                config,
                epoch,
                benchmark,
                training_data,
                _model_name,
                train_only=False
            )
        _model_name = build_model_name(base_model_name, dataset, zcp, end_epoch)
        accuracy, results = await evaluate_model(
            _model_name,
            dataset,
            benchmark,
            test_data,
            end_epoch,
            config,
            zcp=zcp
        )

    get_global_llm_manager().clear_all()
    cprint("[TDMAS] Pipeline完成!", color="green")
    logger.info("Pipeline完成!")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="TDMAS主程序")
    parser.add_argument(
        "--config",
        type=str,
        default="config/running_config.yaml",
        help="配置文件路径"
    )

    args = parser.parse_args()

    asyncio.run(run_pipeline(args.config))


if __name__ == "__main__":
    main()
