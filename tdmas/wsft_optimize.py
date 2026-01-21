#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["fix_mistral_regex"] = "True"
os.environ["disable_custom_all_reduce"] = "True"
os.environ["NCCL_IB_DISABLE"] = "True"
os.environ["NCCL_P2P_DISABLE"] = "True"
os.environ["NCCL_SOCKET_IFNAME"] = "lo"

import re
import sys
import pickle
import json
import random
import time
import yaml
import torch
import warnings
import glob
import shutil
from torch import nn
from torch.utils.data import Dataset
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field

# 过滤不会影响训练效果的警告
warnings.filterwarnings("ignore", message=".*Detected different devices in the system.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*is deprecated.*")
warnings.filterwarnings("ignore", message=".*using the __call__ method is faster.*")
warnings.filterwarnings("ignore", message=".*max_length.*is ignored.*")
warnings.filterwarnings("ignore", message=".*use_cache.*is incompatible.*")
warnings.filterwarnings("ignore", message=".*torch.utils.checkpoint.*use_reentrant.*")
warnings.filterwarnings("ignore", message=".*The tokenizer has new PAD/BOS/EOS tokens.*")
warnings.filterwarnings("ignore", message=".*Setting `save_embedding_layers` to `True`.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="peft")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from metagpt.logs import logger
from utils.llm_manager import LLMManager
from utils.path_utils import wsft_data_file, finetuned_epoch_dir, ensure_dir, finetuned_model_path, build_model_name


# -------------------------
# Config
# -------------------------
@dataclass
class Args:
    # Models
    policy_model_name: str = field(default="Qwen3-8B")
    policy_model_path: str = field(default="../models/Qwen3-8B")

    # Data
    dataset: str = field(default="")
    epoch: int = field(default=0)
    zcp: str = field(default="accuracy")
    seed: int = field(default=42)

    # Tokenization / generation
    max_prompt_len: int = field(default=2048)
    max_new_tokens: int = field(default=1024)
    temperature: float = field(default=0.7)
    top_p: float = field(default=0.9)

    # weighted-SFT
    learning_rate: float = field(default=1e-6)
    num_train_epochs: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=8)
    save_steps: int = field(default=200)
    max_train_samples: int = field(default=4096)

    # Reward processing
    reward_clip_min: float = field(default=0.1)  # 避免权重为0
    reward_clip_max: float = field(default=10.0)  # 防止权重过大

    # KL / rewards (保留用于兼容性)
    kl_coef: float = field(default=0.02)
    missing_eos_penalty: float = field(default=1.0)

    # Memory savers
    use_lora: bool = field(default=True)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)

    load_in_4bit: bool = field(default=True)
    bnb_4bit_compute_dtype: str = field(default="bfloat16")

    # Qwen3 thinking mode
    enable_thinking: bool = field(default=False)

    # Logging / saving
    save_steps: int = field(default=200)
    log_steps: int = field(default=10)


def find_latest_checkpoint(output_dir):
    """查找最近的checkpoint目录"""
    checkpoint_pattern = os.path.join(output_dir, "checkpoint-*")
    checkpoint_dirs = glob.glob(checkpoint_pattern)
    
    if not checkpoint_dirs:
        return None
    
    # 按数字序号排序（例如 checkpoint-100 > checkpoint-50）
    checkpoint_dirs.sort(key=lambda x: int(x.split("-")[-1]), reverse=True)
    return checkpoint_dirs[0]


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Reward Processing
# -------------------------
def process_rewards(rewards: List[float], clip_min: float = 0.1, clip_max: float = 10.0) -> np.ndarray:
    """处理奖励值：normalize → clip → weight

    Args:
        rewards: 原始奖励列表
        clip_min: 裁剪最小值
        clip_max: 裁剪最大值

    Returns:
        处理后的权重数组
    """
    rewards = np.array(rewards, dtype=np.float32)

    # 1. Normalize: 减去均值，除以标准差
    if len(rewards) > 1 and np.std(rewards) > 0:
        rewards = (rewards - np.mean(rewards)) / np.std(rewards)
    else:
        # 如果只有一个样本或标准差为0，保持原值
        pass

    # 2. Clip: 限制范围
    rewards = np.clip(rewards, clip_min, clip_max)

    # 3. Weight: 确保为正值（通过指数变换）
    weights = np.exp(rewards)

    return weights


# -------------------------
# Data Loading
# -------------------------
def load_training_data(dataset: str, zcp: str, epoch: int, max_train_samples: int = 1000) -> List[Dict]:
    """从pickle文件加载训练数据"""
    data_file = wsft_data_file(dataset, zcp, epoch)
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"训练数据文件不存在: {data_file}")

    logger.info(f"正在加载训练数据: {data_file}")
    with open(data_file, 'rb') as f:
        training_data = pickle.load(f)

    logger.info(f"加载 {len(training_data)} 条原始训练样本")

    # 随机采样
    random.shuffle(training_data)
    training_data = training_data[:max_train_samples]

    logger.info(f"加载 {len(training_data)} 条随机采样训练样本")

    # 提取并处理reward
    raw_rewards = []
    for sample in training_data:
        reward_dict = sample.get('reward', {})
        if isinstance(reward_dict, dict):
            total_reward = reward_dict.get('total_reward', 0.0)
        else:
            total_reward = float(reward_dict) if reward_dict else 0.0
        raw_rewards.append(total_reward)

    # 处理reward得到权重
    weights = process_rewards(raw_rewards)

    # 更新training_data中的reward为处理后的权重
    for i, sample in enumerate(training_data):
        sample['weight'] = weights[i]
        sample['original_reward'] = raw_rewards[i]

    logger.info(f"奖励处理完成 | 原始奖励范围: [{min(raw_rewards):.4f}, {max(raw_rewards):.4f}] | 权重范围: [{min(weights):.4f}, {max(weights):.4f}]")

    return training_data


def format_prompt_for_tokenizer(tokenizer: AutoTokenizer, prompt: List) -> str:
    """将prompt格式化为tokenizer可用的字符串

    Args:
        tokenizer: tokenizer对象
        prompt: prompt列表，可能是字符串列表或字典列表（chat格式）

    Returns:
        格式化后的prompt字符串
    """
    if not prompt:
        return ""

    # 如果是chat格式（字典列表）
    if isinstance(prompt[0], dict):
        # 使用chat template
        return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    # 如果是字符串列表
    elif isinstance(prompt[0], str):
        # 简单拼接
        return "\n".join(prompt)
    else:
        # 其他情况，尝试转换为字符串
        return str(prompt)


# -------------------------
# Weighted SFT Dataset
# -------------------------
class WeightedSFTDataset(Dataset):
    """支持权重的SFT数据集"""

    def __init__(self, data: List[Dict], tokenizer: AutoTokenizer, max_length: int = 2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # 格式化prompt和response
        prompt_str = format_prompt_for_tokenizer(self.tokenizer, sample['prompt'])
        response_str = sample['response']

        # 分别tokenize prompt和response
        prompt_tokenized = self.tokenizer(
            prompt_str,
            add_special_tokens=False,  # 不添加特殊token，避免重复
            return_tensors=None
        )
        response_tokenized = self.tokenizer(
            response_str,
            add_special_tokens=False,  # 不添加特殊token
            return_tensors=None
        )

        # 拼接input_ids和attention_mask
        input_ids = prompt_tokenized['input_ids'] + response_tokenized['input_ids']
        attention_mask = prompt_tokenized['attention_mask'] + response_tokenized['attention_mask']

        # 截断到最大长度
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]

        # 创建labels：复制input_ids，但将prompt部分mask为-100
        labels = input_ids.copy()
        prompt_len = len(prompt_tokenized['input_ids'])
        labels[:prompt_len] = [-100] * prompt_len  # mask prompt部分

        # 获取权重
        weight = sample.get('weight', 1.0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'weight': weight,
            'original_reward': sample.get('original_reward', 0.0),
        }


# -------------------------
# Weighted Loss
# -------------------------
class WeightedCrossEntropyLoss(nn.Module):
    """支持样本级权重的交叉熵损失"""

    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, labels, weights=None):
        """
        Args:
            logits: [batch_size, seq_len, vocab_size]
            labels: [batch_size, seq_len]
            weights: [batch_size] 或 None
        """
        # 计算每个token的损失
        loss = self.ce_loss(logits.view(-1, logits.size(-1)), labels.view(-1))

        # 重塑为 [batch_size, seq_len]
        loss = loss.view(labels.shape)

        # 只计算response部分的损失（非padding且非prompt部分）
        # 这里假设labels中-100表示忽略的token
        valid_mask = (labels != -100)

        if weights is not None:
            # 扩展权重到序列维度
            weights = weights.unsqueeze(-1).expand_as(loss)
            # 只对有效的token应用权重
            weighted_loss = loss * weights * valid_mask
        else:
            weighted_loss = loss * valid_mask

        # 计算平均损失
        total_valid_tokens = valid_mask.sum()
        if total_valid_tokens > 0:
            return weighted_loss.sum() / total_valid_tokens
        else:
            # 返回一个与logits相关的零张量，保持梯度连接
            return logits.sum() * 0.0


# -------------------------
# Weighted SFT Trainer
# -------------------------
class WeightedSFTTrainer(Trainer):
    """支持权重的SFT训练器"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weighted_loss_fn = WeightedCrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """计算加权损失"""
        labels = inputs.get('labels')
        weights = inputs.get('weights')

        outputs = model(**{k: v for k, v in inputs.items() if k not in ['labels', 'weights', 'original_rewards']})
        logits = outputs.logits

        # 计算加权损失
        loss = self.weighted_loss_fn(logits, labels, weights)

        return (loss, outputs) if return_outputs else loss


# -------------------------
# Training Monitor
# -------------------------
class TrainingMonitor:
    """训练监控器，用于记录和可视化训练过程"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        ensure_dir(output_dir)

        # 记录训练指标
        self.steps = []
        self.losses = []
        self.rewards = []
        self.weights = []
        self.learning_rates = []
        self.accuracies = []  # 基于correctness的准确率

        # 时间记录
        self.start_time = None
        self.step_times = []

        # 日志文件
        self.log_file = os.path.join(output_dir, "training_log.json")
        self.metrics_file = os.path.join(output_dir, "metrics.jsonl")

    def start_training(self):
        """开始训练"""
        self.start_time = time.time()
        logger.info("训练监控器已启动")

    def log_step(
        self,
        step: int,
        loss: Optional[float] = None,
        reward: Optional[float] = None,
        weight: Optional[float] = None,
        learning_rate: Optional[float] = None,
        accuracy: Optional[float] = None,
        stats: Optional[Dict] = None,
    ):
        """记录一步训练指标"""
        step_time = time.time()
        if self.start_time:
            elapsed = step_time - self.start_time
            self.step_times.append(elapsed)

        self.steps.append(step)
        if loss is not None:
            self.losses.append(float(loss))
        if reward is not None:
            self.rewards.append(float(reward))
        if weight is not None:
            self.weights.append(float(weight))
        if learning_rate is not None:
            self.learning_rates.append(float(learning_rate))
        if accuracy is not None:
            self.accuracies.append(float(accuracy))

        # 保存到JSONL文件
        metrics = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "elapsed_time": float(elapsed) if self.start_time else 0,
        }
        if loss is not None:
            metrics["loss"] = float(loss)
        if reward is not None:
            metrics["reward"] = float(reward)
        if weight is not None:
            metrics["weight"] = float(weight)
        if learning_rate is not None:
            metrics["learning_rate"] = float(learning_rate)
        if accuracy is not None:
            metrics["accuracy"] = float(accuracy)
        if stats:
            # 处理stats中的数值类型，确保能被JSON序列化
            processed_stats = {}
            for k, v in stats.items():
                if isinstance(v, (np.floating, np.integer, torch.Tensor)):
                    processed_stats[k] = float(v)
                elif isinstance(v, dict):
                    # 递归处理嵌套字典
                    nested_processed = {}
                    for nk, nv in v.items():
                        if isinstance(nv, (np.floating, np.integer, torch.Tensor)):
                            nested_processed[nk] = float(nv)
                        else:
                            nested_processed[nk] = nv
                    processed_stats[k] = nested_processed
                else:
                    processed_stats[k] = v
            metrics.update(processed_stats)

        with open(self.metrics_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + '\n')

        # 绘制loss曲线
        self.plot_metrics()

    def estimate_remaining_time(self, current_step: int, total_steps: int) -> str:
        """估算剩余训练时间"""
        if not self.start_time or current_step == 0:
            return "计算中..."

        # 计算平均每步时间
        if len(self.step_times) >= 2:
            # 使用最近的时间差计算平均速度
            recent_times = self.step_times[-min(100, len(self.step_times)):]
            time_diffs = np.diff(recent_times)
            if len(time_diffs) > 0:
                avg_time_per_step = np.mean(time_diffs)
            else:
                avg_time_per_step = self.step_times[-1] / current_step
        else:
            # 如果只有一步，使用总时间除以步数
            avg_time_per_step = self.step_times[-1] / \
                current_step if current_step > 0 else 0

        remaining_steps = total_steps - current_step
        remaining_seconds = avg_time_per_step * remaining_steps

        if remaining_seconds < 60:
            return f"{remaining_seconds:.1f}秒"
        elif remaining_seconds < 3600:
            return f"{remaining_seconds/60:.1f}分钟"
        else:
            hours = remaining_seconds // 3600
            minutes = (remaining_seconds % 3600) // 60
            return f"{int(hours)}小时{int(minutes)}分钟"

    def plot_metrics(self):
        """绘制训练指标曲线"""
        if len(self.steps) < 2:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Weighted SFT Training Metrics', fontsize=16, fontweight='bold')

        # Loss曲线
        if self.losses:
            # 只使用与losses长度匹配的steps部分
            steps_subset = self.steps[-len(self.losses):]
            axes[0, 0].plot(steps_subset, self.losses, 'b-',
                            linewidth=1.5, alpha=0.7)
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].grid(True, alpha=0.3)
            if len(self.losses) > 10:
                # 添加移动平均
                window = min(50, len(self.losses) // 10)
                moving_avg = np.convolve(
                    self.losses, np.ones(window)/window, mode='valid')
                axes[0, 0].plot(steps_subset[window-1:], moving_avg, 'r-',
                                linewidth=2, label=f'Moving Avg (window={window})')
                axes[0, 0].legend()

        # Reward曲线
        if self.rewards:
            # 只使用与rewards长度匹配的steps部分
            steps_subset = self.steps[-len(self.rewards):]
            axes[0, 1].plot(steps_subset, self.rewards,
                            'g-', linewidth=1.5, alpha=0.7)
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Reward')
            axes[0, 1].set_title('Average Reward')
            axes[0, 1].grid(True, alpha=0.3)
            if len(self.rewards) > 10:
                window = min(50, len(self.rewards) // 10)
                moving_avg = np.convolve(
                    self.rewards, np.ones(window)/window, mode='valid')
                axes[0, 1].plot(steps_subset[window-1:], moving_avg, 'r-',
                                linewidth=2, label=f'Moving Avg (window={window})')
                axes[0, 1].legend()

        # Weight曲线
        if self.weights:
            # 只使用与weights长度匹配的steps部分
            steps_subset = self.steps[-len(self.weights):]
            axes[1, 0].plot(steps_subset, self.weights,
                            'm-', linewidth=1.5, alpha=0.7)
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Weight')
            axes[1, 0].set_title('Average Weight')
            axes[1, 0].grid(True, alpha=0.3)

        # 准确率曲线
        if self.accuracies:
            # 只使用与accuracies长度匹配的steps部分
            steps_subset = self.steps[-len(self.accuracies):]
            axes[1, 1].plot(steps_subset, self.accuracies,
                            'c-', linewidth=1.5, alpha=0.7)
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].set_title('Training Accuracy')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim([0, 1])

        plt.tight_layout()
        plot_file = os.path.join(self.output_dir, "training_metrics.png")
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"训练指标图已保存: {plot_file}")

    def save_summary(self, total_steps: int):
        """保存训练总结"""
        if not self.start_time:
            return

        total_time = time.time() - self.start_time
        summary = {
            "total_steps": total_steps,
            "total_time_seconds": total_time,
            "total_time_formatted": str(timedelta(seconds=int(total_time))),
            "average_time_per_step": total_time / total_steps if total_steps > 0 else 0,
            "final_loss": self.losses[-1] if self.losses else None,
            "final_reward": self.rewards[-1] if self.rewards else None,
            "final_weight": self.weights[-1] if self.weights else None,
            "final_accuracy": self.accuracies[-1] if self.accuracies else None,
            "max_reward": max(self.rewards) if self.rewards else None,
            "min_loss": min(self.losses) if self.losses else None,
        }

        summary_file = os.path.join(self.output_dir, "training_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"训练总结已保存: {summary_file}")
        return summary


# -------------------------
# Data Collator
# -------------------------
class WeightedDataCollator:
    """支持权重的Data Collator"""

    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features):
        # 提取权重
        weights = [f.pop('weight') for f in features]
        original_rewards = [f.pop('original_reward') for f in features]

        # 手动collate数据
        batch = {}

        # 处理input_ids, attention_mask, labels
        if 'input_ids' in features[0]:
            # 先手动截断所有序列到max_length，确保长度一致
            input_ids = []
            attention_mask = []
            labels = []
            
            for f in features:
                # 截断input_ids和attention_mask
                ids = f['input_ids'][:self.max_length]
                mask = f['attention_mask'][:self.max_length]
                lbls = f['labels'][:self.max_length] if 'labels' in f else None
                
                input_ids.append(ids)
                attention_mask.append(mask)
                if lbls is not None:
                    labels.append(lbls)
            
            # 使用tokenizer.pad进行padding
            padded = self.tokenizer.pad(
                [{'input_ids': ids, 'attention_mask': mask} for ids, mask in zip(input_ids, attention_mask)],
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )

            batch['input_ids'] = padded['input_ids']
            batch['attention_mask'] = padded['attention_mask']

            if labels:
                # 手动填充labels到max_length，使用-100作为padding值
                padded_labels = []
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else -100
                
                for lbl in labels:
                    if len(lbl) < self.max_length:
                        # 填充到max_length，使用-100（忽略的token）
                        padded_lbl = lbl + [-100] * (self.max_length - len(lbl))
                    else:
                        padded_lbl = lbl[:self.max_length]
                    padded_labels.append(padded_lbl)
                
                batch['labels'] = torch.tensor(padded_labels, dtype=torch.long)

        # 添加权重到batch
        batch['weights'] = torch.tensor(weights, dtype=torch.float32)
        batch['original_rewards'] = original_rewards

        return batch


# -------------------------
# Training Callbacks
# -------------------------
class WeightedSFTCallback(TrainerCallback):
    def __init__(self, monitor, args, training_samples):
        self.monitor: MetricMonitor = monitor
        self.args = args
        self.training_samples = training_samples
        self.step_count = 0

    def on_train_begin(self, args, state, control, **kwargs):
        pass

    def on_epoch_begin(self, args, state, control, **kwargs):
        pass

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        self.step_count += 1
        loss = logs.get("loss")
        learning_rate = logs.get("learning_rate")

        # 计算平均权重和奖励（需要访问当前batch的数据）
        # 从训练数据中采样一些来计算平均值
        sample_indices = np.random.choice(len(self.training_samples), min(100, len(self.training_samples)), replace=False)
        avg_weight = np.mean([self.training_samples[i]['weight'] for i in sample_indices])
        avg_reward = np.mean([self.training_samples[i]['original_reward'] for i in sample_indices])

        self.monitor.log_step(
            step=self.step_count,
            loss=loss,
            reward=avg_reward,
            weight=avg_weight,
            learning_rate=learning_rate,
        )

def load_config(config_path: str = "config/running_config.yaml") -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Weighted SFT训练脚本")
    parser.add_argument("--dataset", type=str, default="GSM8K", help="数据集名称")
    parser.add_argument("--epoch", type=int, default=0, help="训练epoch")
    parser.add_argument("--zcp", type=str, default="accuracy", help="ZCP")
    parser.add_argument("--train_only", action="store_true", help="仅训练模式")

    cli = parser.parse_args()

    # 加载配置文件
    config_path = "config/running_config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    config = load_config(config_path)

    # 获取optimize_config，如果不存在则使用默认值
    optimize_config = config.get('optimize_config', {})

    # 处理布尔值
    def get_bool(value, default=False):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ["1", "true", "yes", "y"]
        return default

    # 构建参数对象，从配置文件读取参数
    args = Args(
        policy_model_name=config.get('model_name'),
        dataset=cli.dataset,
        epoch=cli.epoch,
        zcp=cli.zcp,
        seed=optimize_config.get('seed', 42),
        max_prompt_len=optimize_config.get('max_prompt_len', 2048),
        max_new_tokens=optimize_config.get('max_new_tokens', 1024),
        temperature=optimize_config.get('temperature', 0.7),
        top_p=optimize_config.get('top_p', 0.9),

        learning_rate=float(optimize_config.get('learning_rate', 1e-6)),
        num_train_epochs=optimize_config.get('num_train_epochs', 1),
        per_device_train_batch_size=optimize_config.get(
            'per_device_train_batch_size', 1),
        gradient_accumulation_steps=optimize_config.get(
            'gradient_accumulation_steps', 8),
        save_steps=optimize_config.get('save_steps', 200),
        log_steps=optimize_config.get('log_steps', 10),
        max_train_samples=optimize_config.get('max_train_samples', 1000),

        reward_clip_min=optimize_config.get('reward_clip_min', 0.1),
        reward_clip_max=optimize_config.get('reward_clip_max', 10.0),
        kl_coef=optimize_config.get('kl_coef', 0.02),
        missing_eos_penalty=optimize_config.get('missing_eos_penalty', 1.0),
        use_lora=get_bool(optimize_config.get('use_lora', True)),
        lora_r=optimize_config.get('lora_r', 16),
        lora_alpha=optimize_config.get('lora_alpha', 32),
        lora_dropout=optimize_config.get('lora_dropout', 0.05),
        load_in_4bit=get_bool(optimize_config.get('load_in_4bit', True)),
        bnb_4bit_compute_dtype=optimize_config.get(
            'bnb_4bit_compute_dtype', 'bfloat16'),
        enable_thinking=get_bool(
            optimize_config.get('enable_thinking', False)),
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = LLMManager._allocate_gpus(llm_memory=48.0)["cuda_visible_devices"]

    # 加载 local_llm 配置文件
    local_llm_config_path = "config/local_llm.yaml"
    if not os.path.exists(local_llm_config_path):
        raise FileNotFoundError(f"配置文件不存在: {local_llm_config_path}")

    args.policy_model_name = build_model_name(args.policy_model_name, args.dataset, args.zcp, args.epoch)
    if 'finetuned' not in args.policy_model_name:
        local_llm_config = load_config(local_llm_config_path)
        args.policy_model_path = local_llm_config["models"][args.policy_model_name]["model_path"]
    else:
        args.policy_model_path = finetuned_model_path(args.dataset, args.zcp, args.epoch)

    logger.info(f"已从配置文件加载参数: {config_path}")

    # 设置输出目录
    output_dir = finetuned_epoch_dir(args.dataset, args.zcp, args.epoch + 1)
    ensure_dir(output_dir)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Weighted SFT训练需要CUDA支持")

    logger.info(
        f"开始Weighted SFT训练 | Dataset: {args.dataset} | Epoch: {args.epoch} | Output: {output_dir}")

    # -------------------------
    # 加载训练数据
    # -------------------------
    training_samples = load_training_data(args.dataset, args.zcp, args.epoch, args.max_train_samples)
    if len(training_samples) == 0:
        raise ValueError("训练数据为空")

    logger.info(f"加载了 {len(training_samples)} 个训练样本")

    # -------------------------
    # Tokenizer
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        args.policy_model_path, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # -------------------------
    # Model + LoRA
    # -------------------------
    quant_config = None
    if args.load_in_4bit:
        compute_dtype = torch.bfloat16 if args.bnb_4bit_compute_dtype == "bfloat16" else torch.float16
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.policy_model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quant_config,
    )

    try:
        model.generation_config.enable_thinking = args.enable_thinking
    except Exception:
        pass

    model.resize_token_embeddings(len(tokenizer))

    if args.use_lora:
        if args.load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj",
                            "o_proj", "up_proj", "down_proj", "gate_proj"],
        )
        model = get_peft_model(model, lora_cfg)

    # -------------------------
    # Dataset
    # -------------------------
    dataset = WeightedSFTDataset(training_samples, tokenizer, max_length=args.max_prompt_len)
    warmup_steps = int(len(training_samples) // args.gradient_accumulation_steps * 0.1)

    data_collator = WeightedDataCollator(tokenizer, max_length=args.max_prompt_len)

    # -------------------------
    # Training Arguments
    # -------------------------
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=args.log_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="no",
        load_best_model_at_end=False,
        metric_for_best_model=None,
        greater_is_better=False,
        dataloader_num_workers=0,  # 避免序列化问题
        remove_unused_columns=False,  # 保留所有列
        label_names=["labels"],  # 指定标签列名
    )

    # -------------------------
    # Trainer
    # -------------------------
    trainer = WeightedSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    # -------------------------
    # Check Final Model
    # -------------------------
    final_dir = finetuned_model_path(args.dataset, args.zcp, args.epoch + 1)
    final_model_exists = os.path.exists(final_dir)
    
    if final_model_exists:
        logger.info(f"检测到已存在final模型: {final_dir}")
        
        # 检查是否为合并后的完整模型
        # 完整模型会包含pytorch_model.bin或model.safetensors等文件
        # 检测是否存在合并后的完整模型文件
        # 1) 单文件权重
        single_files = ['pytorch_model.bin', 'model.safetensors', 'consolidated.00.pth']
        has_single = any(os.path.exists(os.path.join(final_dir, f)) for f in single_files)

        # 2) 多文件权重（至少包含一个 model-00001-of-*.safetensors 或 .bin）
        shard_files = os.listdir(final_dir)
        has_shards = any(
            re.fullmatch(r'model-\d{5}-of-\d{5}\.(safetensors|bin)', f)
            for f in shard_files
        )

        is_merged = has_single or has_shards
        if is_merged:
            logger.info("已存在合并后的完整模型，跳过本次训练")
            logger.info("=" * 60)
            logger.info("训练已跳过!")
            logger.info("使用已存在的合并后模型")
            logger.info("=" * 60)
            return
        else:
            logger.info("检测到未合并的模型，开始合并权重...")
            # 如果使用了LoRA，合并权重
            if args.use_lora and hasattr(trainer.model, 'merge_and_unload'):
                try:
                    trainer.model = trainer.model.merge_and_unload()
                    trainer.save_model(final_dir)
                    tokenizer.save_pretrained(final_dir)
                    logger.info("权重合并完成并已保存")
                    logger.info("=" * 60)
                    logger.info("权重合并已完成!")
                    logger.info("=" * 60)
                    return
                except Exception as e:
                    logger.warning(f"合并权重失败: {str(e)}")
                    logger.info("将继续进行完整训练")
    
    # -------------------------
    # Training Monitor
    # -------------------------
    monitor = TrainingMonitor(output_dir)
    monitor.start_training()

    callback = WeightedSFTCallback(monitor, args, training_samples)
    trainer.add_callback(callback)

    # -------------------------
    # Training
    # -------------------------
    logger.info("开始训练...")
    
    # 检查是否有最近的checkpoint
    latest_checkpoint = find_latest_checkpoint(output_dir)
    if latest_checkpoint:
        logger.info(f"找到最近的checkpoint: {latest_checkpoint}")
        logger.info("从checkpoint继续训练...")
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        logger.info("没有找到checkpoint，从头开始训练...")
        trainer.train()

    # -------------------------
    # Save Final Model
    # -------------------------
    final_dir = finetuned_model_path(args.dataset, args.zcp, args.epoch + 1)
    
    # 如果使用了LoRA，先合并权重到基础模型
    if args.use_lora and hasattr(trainer.model, 'merge_and_unload'):
        logger.info("正在合并LoRA权重到基础模型...")
        try:
            trainer.model = trainer.model.merge_and_unload()
            logger.info("LoRA权重合并完成")
        except Exception as e:
            logger.warning(f"合并LoRA权重失败: {str(e)}")
            logger.warning("将保存未合并的LoRA模型")
    
    trainer.save_model(final_dir)
    logger.info(f"最终模型已保存: {final_dir}")
    
    # 保存tokenizer
    tokenizer.save_pretrained(final_dir)
    logger.info(f"Tokenizer已保存: {final_dir}")
    
    # -------------------------
    # Delete All Checkpoints
    # -------------------------
    logger.info("开始清理checkpoint...")
    checkpoint_pattern = os.path.join(output_dir, "checkpoint-*")
    checkpoint_dirs = glob.glob(checkpoint_pattern)
    
    for checkpoint_dir in checkpoint_dirs:
        try:
            shutil.rmtree(checkpoint_dir)
            logger.info(f"已删除checkpoint: {checkpoint_dir}")
        except Exception as e:
            logger.warning(f"删除checkpoint失败: {checkpoint_dir}, 错误: {str(e)}")
    
    logger.info("checkpoint清理完成")

    # 保存训练总结
    summary = monitor.save_summary(int(trainer.state.global_step))
    monitor.plot_metrics()  # 最终绘制一次

    if trainer.accelerator.is_main_process:
        logger.info("=" * 60)
        logger.info("训练完成!")
        logger.info(f"总步数: {trainer.state.global_step}")
        logger.info(f"总时间: {summary.get('total_time_formatted', 'N/A')}")
        logger.info(f"最终Loss: {summary.get('final_loss', 'N/A')}")
        logger.info(f"最终Reward: {summary.get('final_reward', 'N/A')}")
        logger.info(f"最终Weight: {summary.get('final_weight', 'N/A')}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()