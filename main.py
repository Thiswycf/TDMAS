"""
主程序入口
实现完整的训练pipeline：数据收集、优化、验证
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["disable_custom_all_reduce"] = "True"
os.environ["NCCL_IB_DISABLE"] = "True"
os.environ["NCCL_P2P_DISABLE"] = "True"
os.environ["NCCL_SOCKET_IFNAME"] = "lo"

import sys
import yaml
import json
import pickle
import shutil
import asyncio
import datetime
import argparse
import importlib
import subprocess
from typing import Optional
from termcolor import cprint

from tdmas.evaluator import Evaluator
from tdmas.preference_data import generate_preference_pairs, save_preference_data
from tdmas.data_collector import DataCollector
import ScoreFlow.params
from utils.path_utils import (
    ensure_dir,
    workspace_path,
    build_model_name,
    evaluation_output_dir,
    solve_rate_file,
    wsft_data_dir,
    wsft_data_file,
    ppo_data_dir,
    ppo_data_file,
)
from utils.llm_manager import get_global_llm_manager
from metagpt.logs import logger
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("VLLM_CONFIGURE_LOGGING", "0")


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
    max_debug_attempts: int = 2,
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
        max_loop=max_loop,
        max_debug_attempts=max_debug_attempts
    )

    cprint(
        f"[TDMAS] 训练数据收集完成 | Epoch {epoch} | 收集了 {len(collected_data)} 条数据", color="green")
    logger.info(f"训练数据收集完成，共收集 {len(collected_data)} 条数据")

    # 计算训练集准确率
    correct_count = 0
    total_count = len(training_data) * train_ask_num

    # 按问题分组统计（每个问题有ask_num个回答，只统计第一个回答的准确率）
    for data in collected_data:
        assert data["correctness"] == 1.0 or data[
            "correctness"] == 0.0, f'Got invalid correctness: {data["correctness"]}'
        correct_count += data["correctness"]

    train_accuracy = (correct_count / total_count *
                      100) if total_count > 0 else 0.0

    cprint(
        f"[TDMAS] 训练集准确率 | Epoch {epoch} | Accuracy: {train_accuracy:.4f}% ({int(correct_count)}/{total_count})",
        color="green"
    )
    logger.info(
        f"训练集准确率: {train_accuracy:.4f}% ({int(correct_count)}/{total_count})")

    # 保存训练集准确率到文件
    result_dir = evaluation_output_dir(dataset, zcp)
    ensure_dir(result_dir)
    result_file = solve_rate_file(dataset, zcp, "optimize")
    with open(result_file, "a") as f:
        f.write(
            f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] | Limit {limit} | Epoch {epoch} | Train Solve Rate: {train_accuracy:.4f}%\n"
        )

    training_data = []
    for data in collected_data:
        training_data.extend(data["single_problem_train_data"])

    return training_data


def generate_preference_data_from_collected(
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
    """优化模型（PPO训练）"""
    cprint(f"[TDMAS] 开始优化模型 | Epoch {epoch}", color="blue")

    # 在独立子进程中运行优化，避免主进程持有CUDA上下文
    trainer_script = os.path.join(
        # os.path.dirname(__file__), "tdmas/wsft_optimize.py")
        os.path.dirname(__file__), "tdmas/ppo_optimize.py")
    cmd = [
        sys.executable,
        trainer_script,
        "--dataset",
        dataset,
        "--epoch",
        str(epoch),
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
    limit = config.get('limit')
    
    # 检查是否已经评估过
    result_file = solve_rate_file(dataset, zcp, "inference")
    if os.path.exists(result_file):
        try:
            with open(result_file, "r") as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    # 检查最后一行是否包含当前的limit和epoch
                    if f"Limit {limit} | Epoch {epoch}" in last_line:
                        cprint(f"[TDMAS] 模型已评估 | Epoch {epoch} | Limit {limit} | 跳过评估", color="green")
                        logger.info(f"模型已评估，跳过评估：Limit {limit}, Epoch {epoch}")
                        # 从最后一行提取准确率
                        import re
                        match = re.search(r'Test Solve Rate: ([\d.]+)%', last_line)
                        if match:
                            test_accuracy = float(match.group(1))
                            cprint(f"[TDMAS] 使用已有评估结果 | Accuracy: {test_accuracy:.4f}%", color="green")
                            return test_accuracy, []
                        else:
                            return 0.0, []
        except Exception as e:
            logger.warning(f"检查评估记录时出错: {e}")

    cprint(f"[TDMAS] 开始评估模型 | Epoch {epoch} | Dataset {dataset}", color="blue")

    max_depth = config.get('max_depth', 5)
    max_loop = config.get('max_loop', 5)
    max_debug_attempts = config.get('max_debug_attempts', 2)
    max_concurrent_request = config.get('max_concurrent_request', 10)
    max_concurrent_execute_code = config.get(
        'max_concurrent_execute_code', 128)
    test_ask_num = config.get('test_ask_num', 8)

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
        test_ask_num=test_ask_num,
        max_debug_attempts=max_debug_attempts
    )

    # 计算训练集准确率
    correct_count = 0
    total_count = len(test_data) * test_ask_num

    # 按问题分组统计（每个问题有ask_num个回答，只统计第一个回答的准确率）
    for data in results:
        assert data["correctness"] == 1.0 or data[
            "correctness"] == 0.0, f'Got invalid correctness: {data["correctness"]}'
        correct_count += data["correctness"]

    test_accuracy = (correct_count / total_count *
                     100) if total_count > 0 else 0.0

    cprint(
        f"[TDMAS] 测试集准确率 | Epoch {epoch} | Accuracy: {test_accuracy:.4f}% ({int(correct_count)}/{total_count})",
        color="green"
    )
    logger.info(
        f"测试集准确率: {test_accuracy:.4f}% ({int(correct_count)}/{total_count})")

    # 保存测试集准确率到文件
    result_dir = evaluation_output_dir(dataset, zcp)
    ensure_dir(result_dir)
    result_file = solve_rate_file(dataset, zcp, "inference")
    with open(result_file, "a") as f:
        f.write(
            f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] | Limit {limit} | Epoch {epoch} | Test Solve Rate: {test_accuracy:.4f}%\n"
        )

    return test_accuracy, results


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
    max_concurrent_execute_code = config.get(
        'max_concurrent_execute_code', 128)
    train_ask_num = config.get('train_ask_num', 8)
    max_loop = config.get('max_loop', 5)
    max_debug_attempts = config.get('max_debug_attempts', 2)

    cprint(f"[TDMAS] {'='*10} Epoch {epoch} {'='*10}", color="blue")


    # ensure_dir(wsft_data_dir(dataset, zcp))
    # data_path = wsft_data_file(dataset, zcp, epoch)
    ensure_dir(ppo_data_dir(dataset, zcp))
    data_dir = ppo_data_file(dataset, zcp, epoch)

    # if True:
    if not os.path.exists(data_dir):
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
            max_debug_attempts=max_debug_attempts,
            zcp=zcp
        )
        get_global_llm_manager().clear_all()
        cprint("All LLM models have been cleared!", color="green")
        logger.info("All LLM models have been cleared!")

        # # 2. 生成preference data
        # similarity_threshold = config.get('similarity_threshold')
        # preference_pairs_limit = config.get('preference_pairs_limit')
        # generate_preference_data_from_collected(
        #     collected_data,
        #     dataset,
        #     zcp,
        #     epoch,
        #     similarity_threshold=similarity_threshold,
        #     preference_pairs_limit=preference_pairs_limit
        # )

    #     # 2. 生成wsft training data
    #     with open(data_dir, 'wb') as f:
    #         pickle.dump(collected_data, f)

    #     cprint(
    #         f"[TDMAS] weighted-SFT data 生成完成 | Epoch {epoch} | 生成了 {len(collected_data)} 条数据，保存到 {data_dir}", color="green")
    #     logger.info(f"weighted-SFT data 生成完成，共 {len(collected_data)} 条数据，保存到 {data_dir}")
    # else:
    #     cprint(
    #         f"[TDMAS] weighted-SFT data 已存在 | Epoch {epoch} | 从 {data_dir} 加载", color="green")
    #     logger.info(f"weighted-SFT data 已存在，从 {data_dir} 加载")

        # 2. 生成ppo training data
        with open(data_dir, 'wb') as f:
            pickle.dump(collected_data, f)

        cprint(
            f"[TDMAS] PPO data 生成完成 | Epoch {epoch} | 生成了 {len(collected_data)} 条数据，保存到 {data_dir}", color="green")
        logger.info(f"PPO data 生成完成，共 {len(collected_data)} 条数据，保存到 {data_dir}")
    else:
        cprint(
            f"[TDMAS] PPO data 已存在 | Epoch {epoch} | 从 {data_dir} 加载", color="green")
        logger.info(f"PPO data 已存在，从 {data_dir} 加载")


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
            if epoch == 0 and config.get('ignore_first_evolution', False):
                continue
            eval_model_name = build_model_name(
                base_model_name, dataset, zcp, epoch)
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
            _model_name = build_model_name(
                base_model_name, dataset, zcp, epoch)
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
            if not (epoch == 0 and config.get('ignore_first_evolution', False)):
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
        _model_name = build_model_name(
            base_model_name, dataset, zcp, end_epoch)
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
