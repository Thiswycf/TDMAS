#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM实例化管理器，根据llm_name从local_llm.yaml中获取配置
实现多进程LLM池管理系统，每个进程管理一个LLM实例
当显存不够时，根据使用次数排序，终止较少使用的LLM进程
"""

from utils.path_utils import (
    finetuned_model_path,
    legacy_finetuned_model_path,
    parse_finetuned_model_name,
    resolve_existing_path,
)
from utils.const import *
from maas.logs import logger
from utils.VLLMAdapter import VLLMAdapter
import asyncio
import yaml
import os
import sys
import time
import psutil
import queue
import traceback
import multiprocessing
from typing import Optional, Dict, Tuple, Any, List
import itertools
from vllm import SamplingParams
from transformers import AutoTokenizer

# 在导入可能初始化CUDA的库之前设置多进程启动方式
multiprocessing.set_start_method('spawn', force=True)

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath("../../.."))

os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_SOCKET_IFNAME"] = "lo"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"


class LLMProcess(multiprocessing.Process):
    """LLM实例进程类，每个进程管理一个LLM实例"""

    def __init__(self, llm_name: str, model_config: dict, config: dict, request_queue: multiprocessing.Queue, response_dict: Dict[str, Any], response_lock: multiprocessing.Lock, expected_requests: multiprocessing.Value, max_expected_requests: int):
        """初始化LLM进程

        Args:
            llm_name: LLM名称
            model_config: 模型配置
            config: 全局配置
            request_queue: 请求队列
            response_dict: 响应字典，用于存储响应
            response_lock: 响应字典的锁
            expected_requests: 共享的预期请求数（用于控制等待时长）
            max_expected_requests: 最大预期请求数
        """
        super().__init__()
        self.llm_name = llm_name
        self.model_config = model_config
        self.config = config
        self.request_queue = request_queue
        self.response_dict = response_dict  # 使用字典存储响应
        self.response_lock = response_lock  # 响应字典的锁
        self.expected_requests = expected_requests  # 共享的预期请求数
        self.max_expected_requests = max_expected_requests  # 最大预期请求数
        self.llm_instance = None
        self.running = True

    def _increment_expected_requests(self):
        """增加预期请求数"""
        with self.expected_requests.get_lock():
            self.expected_requests.value = min(
                self.expected_requests.value + 1, self.max_expected_requests)

    def _decrement_expected_requests(self):
        """减少预期请求数"""
        with self.expected_requests.get_lock():
            self.expected_requests.value = max(
                self.expected_requests.value - 1, 0)

    def run(self):
        """进程运行函数，初始化LLM实例并处理请求"""
        try:
            # 设置CUDA_VISIBLE_DEVICES环境变量
            cuda_visible_devices = self.config["CUDA_VISIBLE_DEVICES"]
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
            logger.info(
                f"进程 {self.pid} 设置CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")

            # 初始化LLM实例
            model_path = self.model_config.get("model_path")
            max_model_len = self.model_config.get("max_model_len", 16384)

            dtype = "bfloat16"
            tensor_parallel_size = self.config["CUDA_VISIBLE_DEVICES"].count(
                ',') + 1
            gpu_memory_utilization = self.model_config.get(
                "gpu_memory_utilization", 0.9)
            temperature = self.config.get("temperature", 0.2)

            logger.info(f"进程 {self.pid} 初始化LLM实例: {model_path}")

            # 初始化LLM实例
            try:
                self.llm_instance = VLLMAdapter(
                    model_path=model_path,
                    dtype=dtype,
                    tensor_parallel_size=tensor_parallel_size,
                    gpu_memory_utilization=gpu_memory_utilization,
                    max_model_len=max_model_len
                )
            except Exception as init_error:
                # 检查是否是OutOfMemory错误
                error_str = str(init_error).lower()
                error_traceback = traceback.format_exc().lower()
                is_oom = ("out of memory" in error_str or
                          "outofmemory" in error_str or
                          "out of memory" in error_traceback or
                          "outofmemory" in error_traceback)

                if is_oom:
                    # 通过response_dict传递OOM错误信息
                    with self.response_lock:
                        self.response_dict["init_error"] = {
                            "type": "init_error",
                            "error_type": "oom",
                            "error": str(init_error),
                            "traceback": traceback.format_exc()
                        }
                    logger.error(
                        f"进程 {self.pid} 初始化LLM实例时发生OutOfMemory错误: {init_error}")
                # 重新抛出异常，让外层异常处理逻辑处理
                raise

            logger.info(f"进程 {self.pid} LLM实例初始化成功: {model_path}")

            # 发送就绪信号
            with self.response_lock:
                self.response_dict["ready"] = {
                    "type": "ready",
                    "llm_name": self.llm_name
                }

            # 处理请求
            while self.running:
                try:
                    requests = []
                    # 先尝试获取第一个请求
                    try:
                        request = self.request_queue.get(timeout=1)
                        if request["type"] == "stop":
                            # 停止进程
                            self.running = False
                            break
                        requests.append(request)
                    except queue.Empty:
                        continue

                    import time
                    start_time = time.time()
                    queue_size = self.request_queue.qsize()
                    # 使用进程内的共享expected_requests值，避免访问全局管理器（spawn模式下会创建新实例）
                    with self.expected_requests.get_lock():
                        expected_requests_val = max(
                            self.expected_requests.value, 1)  # 防止除以0
                    time_to_wait_more_request = TIME_TO_WAIT_MORE_REQUEST * \
                        (1 - queue_size / expected_requests_val)
                    time_to_wait_more_request = max(
                        time_to_wait_more_request, 2)
                    time_to_wait_more_request = min(
                        time_to_wait_more_request, TIME_TO_WAIT_MORE_REQUEST)
                    while time.time() - start_time < time_to_wait_more_request:
                        try:
                            request = self.request_queue.get(timeout=0.01)
                            if request["type"] == "stop":
                                # 停止进程
                                self.running = False
                                break
                            requests.append(request)
                        except queue.Empty:
                            continue

                    if not requests:
                        continue

                    # 存储any_stop信号
                    any_stop = any(req["type"] == "stop" for req in requests)
                    # 按采样参数分类
                    groups: Dict[Tuple, List[Tuple[int, dict]]] = {}
                    for idx, req in enumerate(requests):
                        if req["type"] == "generate":
                            sampling_params = req["kwargs"]["sampling_params"]
                            key = str(sampling_params)
                            groups.setdefault(key, []).append((idx, req))

                    # 对每一类并行输入llm_instance
                    for key, group in groups.items():
                        prompts = [req["kwargs"]["prompt"] for _, req in group]
                        sampling_params = group[0][1]["kwargs"]["sampling_params"]
                        responses = self.llm_instance.generate(
                            prompts=prompts, sampling_params=sampling_params)
                        # 分发结果
                        for (idx, request), resp in zip(group, responses):
                            try:
                                if isinstance(resp, list) and len(resp) > 0:
                                    resp = resp[0]
                            except Exception as e:
                                logger.error(f"处理生成结果失败: {e}")
                                generated_text = f"处理生成结果失败: {e}"
                            with self.response_lock:
                                self.response_dict[request["request_id"]] = {
                                    "type": "generate_response",
                                    "request_id": request["request_id"],
                                    "response": resp
                                }

                    # 处理stop请求
                    if any_stop:
                        self.running = False
                except Exception as e:
                    # 处理请求错误
                    for request in requests:
                        request_id = request[
                            "request_id"] if "request_id" in request else f"error_{time.time()}"
                        with self.response_lock:
                            self.response_dict[request_id] = {
                                "type": "error",
                                "request_id": request_id,
                                "error": str(e),
                                "traceback": traceback.format_exc()
                            }
                    requests.clear()
        except KeyboardInterrupt:
            # 捕获KeyboardInterrupt异常，确保进程能够正常退出
            logger.info(f"进程 {self.pid} 接收到中断信号，正在退出")
            self.running = False
        except Exception as e:
            logger.warning(f"进程 {self.pid} 初始化或运行失败: {e}")
        finally:
            # 清理资源
            if hasattr(self, 'llm_instance') and self.llm_instance is not None:
                try:
                    # 释放LLM实例资源
                    del self.llm_instance
                except:
                    pass
            logger.info(f"进程 {self.pid} 已退出")

    def stop(self):
        """停止进程"""
        self.running = False
        if self.is_alive():
            try:
                # 发送停止请求
                self.request_queue.put({"type": "stop"})
                # 等待进程终止
                self.join(timeout=5)

                # 如果进程仍在运行，强制终止
                if self.is_alive():
                    logger.warning(f"进程 {self.pid} 未能正常终止，正在强制终止")
                    self.terminate()
                    self.join(timeout=5)
            except Exception as e:
                logger.error(f"停止进程 {self.pid} 失败: {e}")
                # 强制终止
                try:
                    self.terminate()
                    self.join(timeout=5)
                except:
                    pass


class LLMManager:
    """多进程LLM实例化管理器"""

    def __init__(self, config_path: str = "config/local_llm.yaml", agent_max_output_tokens: int = 256, max_expected_requests: int = 512):
        """初始化LLM管理器

        Args:
            config_path: local_llm.yaml配置文件路径
        """
        self.config_path = config_path
        self.instantiated_llms: Dict[str, Dict[str, Any]] = {}  # 存储LLM进程信息
        self.llm_usage_count: Dict[str, int] = {}  # 记录每个LLM的使用次数
        self._load_config()
        self.request_queues: Dict[str, multiprocessing.Queue] = {}
        # 使用字典存储响应，key为request_id
        self.response_dicts: Dict[str, Dict[str, Any]] = {}
        self.response_locks: Dict[str, multiprocessing.Lock] = {}  # 每个响应字典的锁
        self.manager = multiprocessing.Manager()  # 用于创建进程共享的字典
        self.request_counter = 0

        self.AGENT_MAX_OUTPUT_TOKENS: int = agent_max_output_tokens
        self.MAX_EXPECTED_REQUESTS: int = max_expected_requests

        # Tokenizer缓存：同一个model_name共享同一个tokenizer实例
        self.tokenizers: Dict[str, AutoTokenizer] = {}

    def _load_config(self):
        """加载local_llm.yaml配置"""
        try:
            with open(self.config_path, "r") as file:
                self.config = yaml.safe_load(file)
            logger.info(f"成功加载配置文件: {self.config_path}")
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            self.config = {}

    @staticmethod
    def _get_available_gpus() -> list:
        """获取可用GPU列表及其显存使用情况

        Returns:
            可用GPU列表，每个元素包含id、memory_free、memory_used、memory_total等信息
        """
        try:
            # 延迟导入GPUtil，避免在主进程中初始化CUDA
            import GPUtil
            gpus = GPUtil.getGPUs()
            available_gpus = []

            for gpu in gpus:
                # 降低GPU可用的判断标准，只要有空闲内存就认为可用
                if gpu.memoryFree > 0:  # 只要有空闲内存就认为可用
                    available_gpus.append({
                        "id": gpu.id,
                        "memory_free": gpu.memoryFree / 1024,  # 转换为GB
                        "memory_used": gpu.memoryUsed / 1024,  # 转换为GB
                        "memory_total": gpu.memoryTotal / 1024  # 转换为GB
                    })

            # 按空闲内存降序排序
            available_gpus.sort(key=lambda x: (
                x["memory_free"], int(x["id"])), reverse=True)
            return available_gpus
        except Exception as e:
            logger.error(f"获取GPU信息失败: {e}")
            # 如果获取GPU信息失败，返回所有GPU
            return [{"id": i, "memory_free": 20.0, "memory_used": 0.0, "memory_total": 24.0} for i in range(8)]

    @staticmethod
    def _get_available_gpu_memory() -> float:
        """获取可用GPU显存总量

        Returns:
            可用GPU显存总量(GB)
        """
        try:
            # 延迟导入GPUtil，避免在主进程中初始化CUDA
            import GPUtil
            gpus = GPUtil.getGPUs()
            total_free_memory = 0.0
            for gpu in gpus:
                if gpu.memoryFree > 0:  # 只要有空闲内存就计算
                    total_free_memory += gpu.memoryFree / 1024  # 转换为GB
            return total_free_memory
        except Exception as e:
            logger.error(f"获取GPU显存信息失败: {e}")
            # 如果获取GPU信息失败，返回默认值160GB（8个GPU*20GB）
            return 160.0

    @staticmethod
    def _get_llm_memory_requirement(model_name: str) -> float:
        """估算LLM模型所需的显存

        Args:
            model_name: 模型名称

        Returns:
            模型所需显存(GB)
        """
        if "30b" in model_name.lower():
            return 21.6 * 4
        elif "14b" in model_name.lower():
            return 21.6 * 2
            # return 21.6 * 4 # NOTE: modified
        elif "7b" in model_name.lower() or "8b" in model_name.lower():
            # return 21.6
            # return 21.6 * 1.2  # NOTE: modified
            return 21.6 * 1.8  # NOTE: modified
        elif "4B" in model_name or "4b" in model_name:
            return 21.6
        elif "tiny" in model_name.lower() or "1.1b" in model_name.lower():
            return 21.6
        else:
            return 21.6  # 默认

    @staticmethod
    def _calculate_gpu_memory_utilization(allocated_gpus: List[Dict], llm_memory: float,
                                          default_utilization: float = 0.9) -> float:
        """计算GPU内存利用率

        Args:
            allocated_gpus: 分配的GPU列表，每个GPU包含id、memory_free、memory_total等信息
            llm_memory: 模型所需的显存(GB)
            default_utilization: 默认内存利用率

        Returns:
            计算出的GPU内存利用率（0.0-1.0之间）
        """
        if not allocated_gpus:
            return default_utilization

        # 计算总可用显存和总显存
        total_free_memory = sum(gpu["memory_free"] for gpu in allocated_gpus)
        total_memory = sum(gpu["memory_total"] for gpu in allocated_gpus)

        if total_free_memory >= llm_memory:
            # 计算“已占内存 + 所需内存”占总显存的比例
            occupied_ratio = (
                total_memory - total_free_memory + llm_memory) / total_memory
            # 确保利用率在合理范围内（0.92）
            if occupied_ratio > 0.92:
                logger.warning(
                    f"计算出的GPU内存利用率({occupied_ratio:.2f})超过0.92，可能会导致运行时内存不足")
            return occupied_ratio
        else:
            # 如果可用显存不足，使用默认值（这种情况应该不会发生，因为分配时已经检查过）
            logger.error(
                f"分配的GPU总显存({total_free_memory:.2f}GB)不足以满足需求({llm_memory:.2f}GB)，使用默认利用率{default_utilization}")
            return default_utilization

    @staticmethod
    def _allocate_gpus(llm_memory: float, llm_name: str = None,
                       default_gpu_memory_utilization: float = 0.9,
                       chosen_strategy: str = "max") -> Dict[str, Any]:
        """为LLM进程分配可用的GPU（使用满足条件的最小剩余内存GPU或组合）

        Args:
            llm_memory: 模型所需的显存(GB)
            llm_name: LLM实例名称
            default_gpu_memory_utilization: 默认GPU内存利用率
            chosen_strategy: 选择策略，"min"表示选择最小剩余内存GPU，"max"表示选择最大剩余内存GPU

        Returns:
            包含以下键的字典:
                - cuda_visible_devices: CUDA_VISIBLE_DEVICES环境变量值（字符串）
                - gpu_memory_utilization: 计算出的GPU内存利用率（0.0-1.0）
                - allocated_gpus: 分配的GPU列表（用于调试和日志）
        """
        available_gpus = LLMManager._get_available_gpus()
        if not available_gpus:
            raise RuntimeError("没有可用的GPU设备")

        # 计算模型所需的显存（考虑安全余量）
        required_memory = llm_memory * (1 + SAFETY_MARGIN_RATIO)

        chosen_strategy = chosen_strategy.lower()
        if chosen_strategy not in ["min", "max"]:
            raise ValueError(
                f"chosen_strategy 必须是 'min' 或 'max'，当前值为 {chosen_strategy}")
        chosen_func = min if chosen_strategy == "min" else max

        # 跳过已被占用的GPU（空闲内存小于1GB）
        gpus_filtered = [
            gpu for gpu in available_gpus if gpu["memory_free"] > 1.0]
        if not gpus_filtered:
            logger.warning("没有空闲内存大于1GB的GPU，将使用所有可用GPU")
            gpus_filtered = available_gpus
            if not gpus_filtered:
                raise RuntimeError("没有可用的GPU设备")
        available_gpus = gpus_filtered

        allocated_gpus = None
        cuda_visible_devices = None

        gpu_count = 0
        assert MAX_GPU_COUNT <= 64, f"MAX_GPU_COUNT 不能超过 64, 当前值为 {MAX_GPU_COUNT}"
        while cuda_visible_devices is None and gpu_count < MAX_GPU_COUNT:
            gpu_count += 1
            # llm_name is not None 代表是为 LLM 的多注意力头分配 GPU （而不是为 LLM 微调分配 GPU ）
            # 一般情况下 LLM 的多注意力头数都是 2 的幂次个（默认 GPU 总数不会超过 64 ）
            if llm_name is not None and (64 % gpu_count != 0):
                continue

            candidates = []
            for quad in itertools.combinations(available_gpus, gpu_count):
                if all(gpu["memory_free"] >= required_memory / gpu_count for gpu in quad):
                    if sum(gpu["memory_free"] for gpu in quad) >= required_memory:
                        candidates.append(quad)
            if candidates:
                # 选用剩余内存总量最小/最大的组合
                best_quad = chosen_func(
                    candidates, key=lambda gs: sum(g["memory_free"] for g in gs))
                allocated_gpus = list(best_quad)
                cuda_visible_devices = ",".join(
                    str(g["id"]) for g in best_quad)
                if llm_name is not None:
                    logger.info(
                        f"分配GPU [{cuda_visible_devices}] 给LLM {llm_name} 进程，空闲内存: {[round(g['memory_free'],2) for g in best_quad]}GB")
                else:
                    logger.info(
                        f"分配GPU [{cuda_visible_devices}] 给 LLM 微调，空闲内存: {[round(g['memory_free'],2) for g in best_quad]}GB")
            else:
                logger.info(
                    f"不存在 {gpu_count} 块 GPU 满足显存需求，将尝试 {gpu_count+1} 块 GPU 组合")

        if allocated_gpus is None or cuda_visible_devices is None:
            logger.error(f"所有GPU组合都不可用，llm_name：{llm_name}")
            exit(1)

        # 计算GPU内存利用率
        gpu_memory_utilization = LLMManager._calculate_gpu_memory_utilization(
            allocated_gpus, llm_memory, default_gpu_memory_utilization
        )

        logger.info(
            f"计算出的GPU内存利用率: {gpu_memory_utilization:.3f} (模型所需显存: {llm_memory:.2f}GB, 分配GPU数: {len(allocated_gpus)})")

        return {
            "cuda_visible_devices": cuda_visible_devices,
            "gpu_memory_utilization": gpu_memory_utilization,
            "allocated_gpus": allocated_gpus
        }

    def _remove_least_used_llm(self) -> Tuple[str, bool]:
        """移除使用次数最少的LLM实例

        Returns:
            (被移除的LLM名称, 是否成功移除)
        """
        if not self.instantiated_llms or not self.llm_usage_count:
            return "", False

        # 按使用次数排序，选择使用次数最少的LLM
        sorted_llms = sorted(self.llm_usage_count.items(), key=lambda x: x[1])
        least_used_llm = sorted_llms[0][0]

        # 终止进程并移除实例
        if least_used_llm in self.instantiated_llms:
            process_info = self.instantiated_llms[least_used_llm]
            llm_process = process_info["process"]
            try:
                # 发送停止请求
                if least_used_llm in self.request_queues:
                    try:
                        self.request_queues[least_used_llm].put(
                            {"type": "stop"})
                    except:
                        pass

                # 等待进程终止
                llm_process.join(timeout=10)

                # 如果进程仍在运行，强制终止
                if llm_process.is_alive():
                    llm_process.terminate()
                    llm_process.join(timeout=5)

                logger.info(
                    f"成功终止LLM进程: {least_used_llm} (PID: {llm_process.pid})")
            except Exception as e:
                logger.error(
                    f"终止LLM进程失败: {least_used_llm} (PID: {llm_process.pid}), 错误: {e}")
                # 强制终止
                try:
                    llm_process.terminate()
                    llm_process.join(timeout=5)
                except:
                    pass

        # 移除实例和记录
        if least_used_llm in self.instantiated_llms:
            del self.instantiated_llms[least_used_llm]
        if least_used_llm in self.llm_usage_count:
            del self.llm_usage_count[least_used_llm]
        if least_used_llm in self.request_queues:
            try:
                del self.request_queues[least_used_llm]
            except:
                pass

        logger.info(f"成功移除使用次数最少的LLM: {least_used_llm}")
        return least_used_llm, True

    def _get_model_path(self, llm_name: str) -> str:
        """获取模型路径（公共方法，供get_llm_process和get_tokenizer复用）

        Args:
            llm_name: LLM名称

        Returns:
            模型路径
        """
        if 'finetuned' in llm_name:  # finetuned generator model
            assert 'EPOCH' in llm_name, "Finetuned generator model name must contain epoch number, model name: " + llm_name
            assert 'ZCP' in llm_name, "Finetuned generator model name must contain zcp, model name: " + llm_name
            data_set, zcp, epoch = parse_finetuned_model_name(llm_name)
            if zcp is None or epoch is None:
                epoch = llm_name.split("EPOCH")[-1]
                zcp = llm_name[llm_name.index(
                    'ZCP')+3:llm_name.index('EPOCH')-1]
            with open("config/generator_config.yaml", "r") as file:
                generator_config = yaml.safe_load(file)
            candidate_paths = []
            if data_set:
                candidate_paths.append(
                    finetuned_model_path(data_set, zcp, epoch))
            candidate_paths.append(legacy_finetuned_model_path(zcp, epoch))
            model_path = resolve_existing_path(
                candidate_paths[0], candidate_paths[1:])
        else:
            model_config: dict = self.config.get("models").get(llm_name)
            if not model_config:
                raise ValueError(f"配置中未找到模型配置: {llm_name}")
            model_path = model_config.get("model_path")

        return model_path

    def get_tokenizer(self, llm_name: Optional[str] = None) -> Optional[AutoTokenizer]:
        """获取tokenizer实例（统一管理，同一个model_name共享同一个tokenizer）

        Args:
            llm_name: LLM名称，None则使用配置中的默认模型

        Returns:
            tokenizer实例，如果加载失败则返回None
        """
        # 确定要使用的LLM名称
        if llm_name is None:
            llm_name = self.config.get("default_llm")

        if llm_name == 'random':
            if not self.instantiated_llms:
                logger.warning("当前没有可用的LLM实例（llm池为空），无法获取tokenizer")
                return None
            # 任意选择一个已有llm_name
            llm_name = next(iter(self.instantiated_llms))

        # 如果已经缓存了tokenizer，直接返回
        if llm_name in self.tokenizers:
            return self.tokenizers[llm_name]

        # 首次加载tokenizer
        try:
            model_path = self._get_model_path(llm_name)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.tokenizers[llm_name] = tokenizer
            logger.info(f"成功加载并缓存tokenizer: {llm_name} (路径: {model_path})")
            return tokenizer
        except Exception as e:
            logger.warning(f"加载tokenizer失败: {llm_name}, 错误: {e}，将返回None")
            return None

    def get_llm_process(self, llm_name: Optional[str] = None, oom_retry_start_time: Optional[float] = None) -> Tuple[str, Dict[str, Any]]:
        """获取或初始化LLM进程

        Args:
            llm_name: LLM名称，None则使用配置中的默认模型

        Returns:
            (LLM名称, 进程信息字典)

        Raises:
            RuntimeError: 当没有实例化LLM可返回时
        """
        # 确定要使用的LLM名称
        if llm_name is None:
            # logger.info(f"NO LLM name is specified, the default model will be used: {self.config.get('default_llm')}")
            llm_name = self.config.get("default_llm")

        if llm_name == 'random':
            if not self.instantiated_llms:
                raise RuntimeError("当前没有可用的LLM实例（llm池为空），无法分配随机模型。")
            # 任意选择一个已有llm_name
            llm_name = next(iter(self.instantiated_llms))

        # 使用公共方法获取模型路径
        model_path = self._get_model_path(llm_name)

        # 获取模型配置（用于后续初始化）
        if 'finetuned' in llm_name:  # finetuned generator model
            with open("config/generator_config.yaml", "r") as file:
                generator_config = yaml.safe_load(file)
            model_config = {
                "model_path": model_path,
                "max_model_len": generator_config["max_model_len"],
            }
        else:
            model_config: dict = self.config.get("models").get(llm_name)
            if not model_config:
                raise ValueError(f"配置中未找到模型配置: {llm_name}")

        # 如果llm_name已实例化，直接返回并更新使用次数
        if llm_name and llm_name in self.instantiated_llms:
            # logger.info(f"Return the instantiated LLM: {llm_name}")
            self.llm_usage_count[llm_name] = self.llm_usage_count.get(
                llm_name, 0) + 1
            return llm_name, self.instantiated_llms[llm_name]

        # 尝试初始化新的LLM进程
        retry_count = 0
        max_retries = len(self.instantiated_llms)  # 最多重试次数为当前实例化的LLM数量

        while retry_count <= max_retries:
            try:
                # 检查可用显存是否足够
                available_memory = LLMManager._get_available_gpu_memory()

                llm_memory = LLMManager._get_llm_memory_requirement(llm_name)

                # 预留安全余量
                required_memory = llm_memory * (1 + SAFETY_MARGIN_RATIO)

                logger.info(
                    f"可用GPU显存: {available_memory:.2f}GB, 模型所需显存: {required_memory:.2f}GB")

                # 如果可用显存不足，尝试移除使用次数最少的LLM
                if available_memory < required_memory and self.instantiated_llms:
                    logger.warning(f"显存不足，尝试移除使用次数最少的LLM实例")
                    self._remove_least_used_llm()
                    # 等待释放显存
                    time.sleep(3)
                    retry_count += 1
                    continue

                # 分配GPU并获取内存利用率（接口化算法）
                default_gpu_memory_utilization = model_config.get(
                    "gpu_memory_utilization", 0.9)
                allocation_result = LLMManager._allocate_gpus(
                    llm_memory,
                    llm_name,
                    default_gpu_memory_utilization
                )

                # 从分配结果中提取信息
                cuda_visible_devices = allocation_result["cuda_visible_devices"]
                gpu_memory_utilization = allocation_result["gpu_memory_utilization"]
                allocated_gpus = allocation_result["allocated_gpus"]

                # 更新模型配置中的内存利用率
                model_config["gpu_memory_utilization"] = gpu_memory_utilization

                # 创建进程配置
                process_config = self.config.copy()
                process_config["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

                # 创建请求队列和共享响应字典
                request_queue = multiprocessing.Queue()
                response_dict = self.manager.dict()
                response_lock = multiprocessing.Lock()
                # 创建共享的expected_requests值，用于子进程控制等待时长
                expected_requests = multiprocessing.Value(
                    'i', 1)  # 初始值为1，防止除以0

                # 初始化LLM进程
                logger.info(f"初始化LLM进程: {llm_name} 中...")
                start_time = time.time()

                llm_process = LLMProcess(
                    llm_name=llm_name,
                    model_config=model_config,
                    config=process_config,
                    request_queue=request_queue,
                    response_dict=response_dict,
                    response_lock=response_lock,
                    expected_requests=expected_requests,
                    max_expected_requests=self.MAX_EXPECTED_REQUESTS
                )

                # 启动进程
                llm_process.start()

                # 等待进程启动并检查是否成功
                time.sleep(2)  # 给进程一些时间启动

                # 等待模型初始化完成（等待ready信号）
                logger.info(f"等待LLM进程 {llm_process.pid} 初始化完成...")
                ready_start_time = time.time()
                ready_received = False

                while time.time() - ready_start_time < TIME_FOR_PROCESS_TO_READY:
                    try:
                        with response_lock:
                            if "ready" in response_dict:
                                signal = response_dict["ready"]
                                if signal.get("type") == "ready" and signal.get("llm_name") == llm_name:
                                    ready_received = True
                                    # 移除ready信号
                                    del response_dict["ready"]
                                    break
                    except Exception as e:
                        logger.error(f"检查ready信号时出错: {e}")

                    # 检查进程是否仍然存活
                    if not llm_process.is_alive():
                        time.sleep(5)  # 等待可能存在的竞争进程
                        # 检查是否是OutOfMemory错误（当两次返回的分配信息不一致时，判定 OOM）
                        is_oom = LLMManager._allocate_gpus(
                            llm_memory,
                            llm_name,
                            default_gpu_memory_utilization
                        ) != allocation_result

                        if is_oom:
                            logger.warning("指定分配的 GPU 被占，尝试重新分配")
                            # 记录OOM重试开始时间
                            if oom_retry_start_time is None:
                                oom_retry_start_time = time.time()

                            # 检查总等待时间是否超过限制
                            total_wait_time = time.time() - oom_retry_start_time
                            if total_wait_time >= TIME_FOR_OOM_RETRY:
                                logger.error(
                                    f"OOM重试总等待时间超过{TIME_FOR_OOM_RETRY}秒，退出")
                                exit(1)

                            # 等待30秒后重试
                            wait_time = 30
                            logger.info(
                                f"等待{wait_time}秒后重试初始化LLM进程（已等待{total_wait_time:.1f}秒，剩余{TIME_FOR_OOM_RETRY - total_wait_time:.1f}秒）")
                            time.sleep(wait_time)

                            # 清理当前失败的进程资源
                            try:
                                if llm_name in self.instantiated_llms:
                                    del self.instantiated_llms[llm_name]
                                if llm_name in self.llm_usage_count:
                                    del self.llm_usage_count[llm_name]
                                if llm_name in self.request_queues:
                                    try:
                                        del self.request_queues[llm_name]
                                    except:
                                        pass
                                if llm_name in self.response_dicts:
                                    try:
                                        del self.response_dicts[llm_name]
                                    except:
                                        pass
                                if llm_name in self.response_locks:
                                    try:
                                        del self.response_locks[llm_name]
                                    except:
                                        pass
                            except Exception as e:
                                logger.error(f"清理失败进程资源时出错: {e}")

                            # 重新调用get_llm_process
                            logger.info(f"重新尝试初始化LLM进程: {llm_name}")
                            return self.get_llm_process(llm_name, oom_retry_start_time)
                        else:
                            logger.error(f"LLM进程在初始化过程中终止（非OOM错误）")
                            exit(1)

                    time.sleep(0.5)  # 短暂休眠，减少CPU占用

                if not ready_received:
                    raise RuntimeError(
                        f"LLM进程初始化超时: {TIME_FOR_PROCESS_TO_READY}秒")

                # 记录初始化时间
                initialization_time = time.time() - start_time
                logger.info(
                    f"LLM进程 {llm_process.pid} 启动成功并已就绪，初始化时间: {initialization_time:.2f}秒")

                # 记录进程信息
                process_info = {
                    "pid": llm_process.pid,
                    "process": llm_process,
                    "cuda_visible_devices": cuda_visible_devices,
                    "gpu_memory_utilization": gpu_memory_utilization,
                    "allocated_gpus": allocated_gpus,
                    "initialized_time": time.time()
                }

                self.instantiated_llms[llm_name] = process_info

                # 记录使用次数
                self.llm_usage_count[llm_name] = 1

                # 记录队列和响应字典
                self.request_queues[llm_name] = request_queue
                self.response_dicts[llm_name] = response_dict
                self.response_locks[llm_name] = response_lock

                return llm_name, process_info

            except Exception as e:
                logger.error(f"初始化LLM进程失败: {e}")
                logger.error(traceback.format_exc())

                # 清理资源
                if llm_name in self.instantiated_llms:
                    if "process" in self.instantiated_llms[llm_name]:
                        try:
                            self.instantiated_llms[llm_name]['process'].terminate(
                            )
                            self.instantiated_llms[llm_name]['process'].join(
                                timeout=5)
                        except:
                            pass
                    del self.instantiated_llms[llm_name]
                if llm_name in self.llm_usage_count:
                    del self.llm_usage_count[llm_name]
                if llm_name in self.request_queues:
                    try:
                        del self.request_queues[llm_name]
                    except:
                        pass
                if llm_name in self.response_dicts:
                    try:
                        del self.response_dicts[llm_name]
                    except:
                        pass
                if llm_name in self.response_locks:
                    try:
                        del self.response_locks[llm_name]
                    except:
                        pass

                # 尝试移除一个LLM后重试
                if self.instantiated_llms:
                    logger.warning(f"尝试移除一个LLM实例后重试")
                    self._remove_least_used_llm()
                    time.sleep(3)
                    retry_count += 1
                else:
                    raise RuntimeError(f"无法初始化LLM进程: {e}")

        raise RuntimeError("尝试所有可能的组合后仍无法初始化LLM进程")

    async def generate(self, llm_name: Optional[str] = None, prompt: str = "", **kwargs) -> str:
        """生成文本

        Args:
            llm_name: LLM名称，None则使用默认模型
            prompt: 提示词
            **kwargs: 其他生成参数

        Returns:
            生成的文本
        """
        # llm_name = 'Llama-3.1-8B-Instruct' # NOTE: modified
        # 获取或初始化LLM进程
        llm_name, process_info = self.get_llm_process(llm_name)

        # 检查进程是否仍在运行
        if not process_info["process"].is_alive():
            logger.warning(f"LLM进程 {llm_name} 已终止，重新初始化")
            # 清理资源
            del self.instantiated_llms[llm_name]
            del self.llm_usage_count[llm_name]
            del self.request_queues[llm_name]
            # 重新获取进程
            llm_name, process_info = self.get_llm_process(llm_name)

        # 生成请求ID
        self.request_counter += 1
        request_id = f"req_{self.request_counter}"

        # 准备请求
        # 处理参数，确保prompt在kwargs中
        if "prompt" not in kwargs and prompt:
            kwargs["prompt"] = prompt
        if "sampling_params" not in kwargs:
            # 如果没有提供sampling_params，创建一个默认的
            kwargs["sampling_params"] = VLLMAdapterWrapper.get_sampling_params(
                max_tokens=self.AGENT_MAX_OUTPUT_TOKENS)

        request = {
            "type": "generate",
            "request_id": request_id,
            "kwargs": kwargs
        }

        # 通过进程对象增加预期请求数
        process_info["process"]._increment_expected_requests()

        # 异步发送请求
        await asyncio.to_thread(self.request_queues[llm_name].put, request)
        start_time = time.time()
        response_dict = self.response_dicts[llm_name]
        response_lock = self.response_locks[llm_name]

        while time.time() - start_time < TIME_FOR_AGENT_TO_RESPONSE:
            try:
                # 定义一个同步函数来检查响应字典
                def check_response():
                    with response_lock:
                        if request_id in response_dict:
                            response = response_dict[request_id]
                            # 移除已处理的响应
                            del response_dict[request_id]
                            return response
                    return None

                # 使用to_thread运行同步函数
                response = await asyncio.to_thread(check_response)
                if response:
                    process_info["process"]._decrement_expected_requests()
                    if response["type"] == "generate_response":
                        # 通过进程对象减少预期请求数
                        return response["response"]
                    elif response["type"] == "error":
                        logger.error(f"生成请求错误: {response['error']}")
                        logger.error(response['traceback'])
                        raise RuntimeError(f"生成请求失败: {response['error']}")

                # 短暂休眠，减少CPU占用
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"检查响应时出错: {e}")
                continue

        raise TimeoutError(
            f"生成请求超时: {TIME_FOR_AGENT_TO_RESPONSE}秒；请求ID: {request_id}；llm_name: {llm_name}；prompt: {prompt}；kwargs: {kwargs}")

    def get_all_instantiated_llms(self) -> Dict[str, VLLMAdapter]:
        """获取所有已实例化的LLM

        Returns:
            已实例化的LLM字典
        """
        return self.instantiated_llms

    def remove_llm(self, llm_name: str) -> bool:
        """移除指定的LLM实例

        Args:
            llm_name: LLM名称

        Returns:
            是否成功移除
        """
        if llm_name not in self.instantiated_llms:
            return False

        # 停止进程
        process_info = self.instantiated_llms[llm_name]
        llm_process = process_info["process"]

        try:
            llm_process.stop()
            logger.info(f"成功停止LLM进程: {llm_name} (PID: {llm_process.pid})")
        except Exception as e:
            logger.error(
                f"停止LLM进程失败: {llm_name} (PID: {llm_process.pid}), 错误: {e}")

        # 移除实例和记录
        del self.instantiated_llms[llm_name]
        if llm_name in self.llm_usage_count:
            del self.llm_usage_count[llm_name]
        if llm_name in self.request_queues:
            del self.request_queues[llm_name]
        if llm_name in self.response_dicts:
            del self.response_dicts[llm_name]
        if llm_name in self.response_locks:
            del self.response_locks[llm_name]

        logger.info(f"移除LLM实例: {llm_name}")
        return True

    def clear_all(self):
        """清除所有LLM实例"""
        # 终止所有进程
        for llm_name, process_info in self.instantiated_llms.items():
            try:
                llm_process = process_info["process"]
                llm_process.stop()
                logger.info(f"成功终止LLM进程: {llm_name} (PID: {llm_process.pid})")
            except Exception as e:
                logger.error(f"终止LLM进程失败: {llm_name}, 错误: {e}")

        # 清除所有记录
        self.instantiated_llms.clear()
        self.llm_usage_count.clear()
        self.request_queues.clear()
        self.response_dicts.clear()
        self.response_locks.clear()

        logger.info("清除所有LLM实例")

    def clear_all_finetuned_llms(self):
        """清除所有LLM实例"""
        for llm_name, process_info in self.instantiated_llms.items():
            if 'finetuned' not in llm_name:
                continue
            try:
                llm_process = process_info["process"]
                llm_process.stop()
                logger.info(f"成功终止LLM进程: {llm_name} (PID: {llm_process.pid})")
            except Exception as e:
                logger.error(f"终止LLM进程失败: {llm_name}, 错误: {e}")

        self.instantiated_llms = {
            k: v for k, v in self.instantiated_llms.items() if 'finetuned' not in k}
        self.llm_usage_count = {
            k: v for k, v in self.llm_usage_count.items() if 'finetuned' not in k}
        self.request_queues = {
            k: v for k, v in self.request_queues.items() if 'finetuned' not in k}
        self.response_dicts = {
            k: v for k, v in self.response_dicts.items() if 'finetuned' not in k}
        self.response_locks = {
            k: v for k, v in self.response_locks.items() if 'finetuned' not in k}

        logger.info("清除所有finetuned LLM实例")

    def get_llm_stats(self) -> Dict[str, Dict]:
        """获取LLM实例的统计信息

        Returns:
            包含所有LLM统计信息的字典
        """
        stats = {}
        for llm_name, process_info in self.instantiated_llms.items():
            stats[llm_name] = {
                "usage_count": self.llm_usage_count.get(llm_name, 0),
                "pid": process_info["pid"],
                "model": self.config.get("models").get(llm_name).get("model_path"),
                "cuda_visible_devices": process_info.get("cuda_visible_devices", "N/A")
            }
        return stats


class VLLMAdapterWrapper:
    """VLLMAdapter包装器类，封装与LLMManager的通信
    对外提供与BaseLLM相同的接口，确保与ActionNode兼容
    """

    def __init__(self, llm_manager: LLMManager, llm_name: Optional[str] = None):
        """初始化VLLMAdapter包装器

        Args:
            llm_manager: LLM管理器实例
            llm_name: LLM名称
        """
        self.llm_manager = llm_manager
        self.llm_name = llm_name
        # BaseLLM所需的属性
        self.auto_max_tokens = False
        self.cost_manager = None
        self.model = llm_name
        self.system_prompt = ""
        self.auto_retry = False

    def generate(self, prompts: List[str], sampling_params: Any):
        """生成文本，与VLLMAdapter的generate方法签名一致

        Args:
            prompts: 提示词列表
            sampling_params: 采样参数

        Returns:
            生成的结果
        """
        results = []
        for prompt in prompts:
            result = self.llm_manager.generate(
                self.llm_name,
                prompt=prompt,
                sampling_params=sampling_params,
            )
            # 创建与vllm.generate返回格式兼容的结果对象

            class GeneratedText:
                def __init__(self, text):
                    self.text = text
                    self.token_ids = []
                    self.generated_tokens = len(text.split())
                    self.prompt_token_ids = []
                    self.finished = True
                    self.finish_reason = "stop"

            generated_text = GeneratedText(result)
            results.append(generated_text)
        return results

    async def aask(self, msg: str, system_msgs=None, format_msgs=None, images=None, timeout=None, stream=None):
        """异步生成文本，与BaseLLM的aask方法签名一致

        Args:
            msg: 提示词
            system_msgs: 系统消息
            format_msgs: 格式消息
            images: 图像（用于多模态模型）
            timeout: 超时时间
            stream: 是否流式输出

        Returns:
            生成的文本
        """
        # 处理系统消息
        full_prompt = ""
        if system_msgs:
            if isinstance(system_msgs, list):
                for system_msg in system_msgs:
                    full_prompt += f"<system>{system_msg}</system>\n"
            else:
                full_prompt += f"<system>{system_msgs}</system>\n"

        # 添加用户消息
        full_prompt += msg

        # 调用LLMManager的generate方法生成文本
        return await self.llm_manager.generate(
            self.llm_name,
            prompt=full_prompt,
            sampling_params=VLLMAdapterWrapper.get_sampling_params(
                temperature=0.2, max_tokens=get_global_llm_manager().AGENT_MAX_OUTPUT_TOKENS, logprobs=20)
        )

    @staticmethod
    def get_sampling_params(**kwargs):
        """获取采样参数"""
        return SamplingParams(
            temperature=kwargs.get("temperature", 0.2),
            top_p=kwargs.get("top_p", 0.95),
            top_k=kwargs.get("top_k", 40),
            max_tokens=kwargs.get("max_tokens", 1024),
            logprobs=kwargs.get("logprobs", None),
        )


# 创建全局LLM管理器实例
global_llm_manager = None


def get_global_llm_manager(**kwargs) -> LLMManager:
    """获取全局LLM管理器实例

    Returns:
        全局LLM管理器实例
    """
    global global_llm_manager
    if global_llm_manager is None:
        global_llm_manager = LLMManager(**kwargs)
    return global_llm_manager


def get_llm_instance(llm_name: Optional[str] = None, **kwargs) -> VLLMAdapter:
    """便捷函数：获取或初始化LLM实例

    Args:
        llm_name: LLM模型名称

    Returns:
        VLLMAdapter实例
    """
    manager = get_global_llm_manager(**kwargs)
    return VLLMAdapterWrapper(manager, llm_name)


def generate(llm_name: Optional[str] = None, **kwargs) -> Any:
    """便捷函数：生成文本

    Args:
        llm_name: LLM名称
        **kwargs: 生成参数

    Returns:
        生成结果
    """
    manager = get_global_llm_manager()
    return manager.generate(llm_name, **kwargs)
