#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pickle
import json
import random
import time
import yaml
import torch
from torch import nn
import numpy as np
import glob
import shutil
import re
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import warnings
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore", category=UserWarning, module="peft")

from metagpt.logs import logger
from utils.llm_manager import LLMManager
from utils.path_utils import (
    ppo_data_file,
    finetuned_epoch_dir,
    ensure_dir,
    finetuned_model_path,
    build_model_name
)


# TRL PPO (experimental API used by TRL official example script)


# -------------------------
# Config
# -------------------------
@dataclass
class Args:
    # Models
    policy_model_name: str = field(default="Qwen3-8B")
    policy_model_path: str = field(default="../models/Qwen3-8B")
    ref_policy_model_path: str = field(default="../models/Qwen3-8B")

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

    # PPO
    learning_rate: float = field(default=1e-6)
    total_episodes: int = field(default=1000)
    per_device_train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=8)
    num_ppo_epochs: int = field(default=1)
    num_mini_batches: int = field(default=1)

    # KL / rewards
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

    # Visualization
    plot_steps: int = field(default=50)  # 每多少步绘制一次loss曲线


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_latest_checkpoint(output_dir):
    """查找最近的checkpoint目录"""
    checkpoint_pattern = os.path.join(output_dir, "checkpoint-*")
    checkpoint_dirs = glob.glob(checkpoint_pattern)
    
    if not checkpoint_dirs:
        return None
    
    # 按数字序号排序（例如 checkpoint-100 > checkpoint-50）
    checkpoint_dirs.sort(key=lambda x: int(x.split("-")[-1]), reverse=True)
    return checkpoint_dirs[0]


# -------------------------
# Data Loading
# -------------------------
def load_training_data(dataset: str, zcp: str, epoch: int) -> List[Dict]:
    """从pickle文件加载训练数据"""
    data_file = ppo_data_file(dataset, zcp, epoch)
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"训练数据文件不存在: {data_file}")

    logger.info(f"正在加载训练数据: {data_file}")
    with open(data_file, 'rb') as f:
        training_samples = pickle.load(f)

    logger.info(f"成功加载 {len(training_samples)} 条训练样本")

    # 数据已经是turn级别，直接返回
    return training_samples


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
# Training Monitor
# -------------------------
class TrainingMonitor:
    """训练监控器，用于记录和可视化训练过程"""

    def __init__(self, output_dir: str, plot_steps: int = 50):
        self.output_dir = output_dir
        self.plot_steps = plot_steps
        ensure_dir(output_dir)

        # 记录训练指标
        self.steps = []
        self.losses = []
        self.rewards = []
        self.kl_divergences = []
        self.learning_rates = []
        self.accuracies = []  # 基于correctness的准确率
        self.value_losses = []  # Value Head的损失

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
        kl_divergence: Optional[float] = None,
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
            self.losses.append(loss)
        if reward is not None:
            self.rewards.append(reward)
        if kl_divergence is not None:
            self.kl_divergences.append(kl_divergence)
        if learning_rate is not None:
            self.learning_rates.append(learning_rate)
        if accuracy is not None:
            self.accuracies.append(accuracy)
        if stats and 'value_loss' in stats:
            self.value_losses.append(stats['value_loss'])

        # 保存到JSONL文件
        metrics = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "elapsed_time": elapsed if self.start_time else 0,
        }
        if loss is not None:
            metrics["loss"] = loss
        if reward is not None:
            metrics["reward"] = reward
        if kl_divergence is not None:
            metrics["kl_divergence"] = kl_divergence
        if learning_rate is not None:
            metrics["learning_rate"] = learning_rate
        if accuracy is not None:
            metrics["accuracy"] = accuracy
        if stats:
            metrics.update(stats)

        with open(self.metrics_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + '\n')

        # 定期绘制loss曲线
        if step % self.plot_steps == 0:
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
        fig.suptitle('PPO Training Metrics', fontsize=16, fontweight='bold')

        # Loss曲线
        if self.losses:
            axes[0, 0].plot(self.steps, self.losses, 'b-',
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
                axes[0, 0].plot(self.steps[window-1:], moving_avg, 'r-',
                                linewidth=2, label=f'Moving Avg (window={window})')
                axes[0, 0].legend()

        # Reward曲线
        if self.rewards:
            axes[0, 1].plot(self.steps, self.rewards,
                            'g-', linewidth=1.5, alpha=0.7)
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Reward')
            axes[0, 1].set_title('Average Reward')
            axes[0, 1].grid(True, alpha=0.3)
            if len(self.rewards) > 10:
                window = min(50, len(self.rewards) // 10)
                moving_avg = np.convolve(
                    self.rewards, np.ones(window)/window, mode='valid')
                axes[0, 1].plot(self.steps[window-1:], moving_avg, 'r-',
                                linewidth=2, label=f'Moving Avg (window={window})')
                axes[0, 1].legend()

        # KL散度曲线
        if self.kl_divergences:
            axes[1, 0].plot(self.steps, self.kl_divergences,
                            'm-', linewidth=1.5, alpha=0.7)
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('KL Divergence')
            axes[1, 0].set_title('KL Divergence')
            axes[1, 0].grid(True, alpha=0.3)

        # Value Head损失曲线
        if self.value_losses:
            axes[1, 1].plot(self.steps, self.value_losses,
                            'y-', linewidth=1.5, alpha=0.7)
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Value Loss')
            axes[1, 1].set_title('Value Head Loss')
            axes[1, 1].grid(True, alpha=0.3)
        # 准确率曲线
        elif self.accuracies:
            axes[1, 1].plot(self.steps, self.accuracies,
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
            "final_accuracy": self.accuracies[-1] if self.accuracies else None,
            "max_reward": max(self.rewards) if self.rewards else None,
            "min_loss": min(self.losses) if self.losses else None,
        }

        summary_file = os.path.join(self.output_dir, "training_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"训练总结已保存: {summary_file}")
        return summary


class PPOLoss(nn.Module):
    """PPO损失函数"""

    def __init__(self, clip_range: float = 0.2, kl_coef: float = 0.02):
        super().__init__()
        self.clip_range = clip_range
        self.kl_coef = kl_coef

    def forward(
        self,
        new_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        new_log_probs_ref: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            new_log_probs: [batch_size, seq_len] 当前策略的log概率
            old_log_probs: [batch_size, seq_len] 旧策略的log概率
            advantages: [batch_size, seq_len] 优势函数值
            new_log_probs_ref: [batch_size, seq_len] 参考策略的log概率（可选，用于KL惩罚）

        Returns:
            包含各种损失的字典
        """
        # 计算概率比率
        ratio = torch.exp(new_log_probs - old_log_probs)

        # 计算未裁剪的PPO损失
        surr1 = ratio * advantages
        # 计算裁剪后的PPO损失
        surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages

        # PPO损失（取最小值）
        policy_loss = -torch.min(surr1, surr2).mean()

        # KL散度惩罚（使用当前策略和参考策略的KL散度）
        kl_divergence = None
        if new_log_probs_ref is not None:
            # KL(new || ref) = E[log(new) - log(ref)] = E[new_log_probs - new_log_probs_ref]
            kl_divergence = (new_log_probs - new_log_probs_ref).mean()
            kl_loss = self.kl_coef * kl_divergence
        else:
            kl_loss = torch.tensor(0.0, device=new_log_probs.device)

        # 总损失
        total_loss = policy_loss + kl_loss

        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "kl_loss": kl_loss,
            "kl_divergence": kl_divergence,
            "ratio": ratio.mean(),
        }


class ValueHead(nn.Module):
    """Value Head用于估计状态价值baseline"""

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size] 隐藏状态

        Returns:
            [batch_size, seq_len, 1] 价值估计
        """
        # 将hidden_states转换为float32类型，与ValueHead的权重保持一致
        hidden_states = hidden_states.to(torch.float32)
        values = self.value_head(hidden_states)
        return values.squeeze(-1)  # [batch_size, seq_len]


def collate_batch(
    samples: List[Dict],
    tokenizer: AutoTokenizer,
    max_prompt_len: int,
    max_new_tokens: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    将样本批次整理为模型输入格式

    Args:
        samples: 样本列表
        tokenizer: tokenizer对象
        max_prompt_len: 最大prompt长度
        max_new_tokens: 最大生成token数
        device: 设备

    Returns:
        包含模型输入和标签的字典
    """
    batch_prompts = []
    batch_responses = []
    batch_old_logprobs = []
    batch_rewards = []

    for sample in samples:
        # 格式化prompt
        prompt_str = format_prompt_for_tokenizer(tokenizer, sample['prompt'])
        batch_prompts.append(prompt_str)
        batch_responses.append(sample['response'])

        # 处理old_logprobs
        old_logprobs = sample['logprobs']
        # 转换为tensor并截断/填充
        old_logprobs_tensor = torch.tensor(old_logprobs, dtype=torch.float32)
        if len(old_logprobs_tensor) > max_new_tokens:
            old_logprobs_tensor = old_logprobs_tensor[:max_new_tokens]
        # 如果长度不足，填充到max_new_tokens
        elif len(old_logprobs_tensor) < max_new_tokens:
            pad_len = max_new_tokens - len(old_logprobs_tensor)
            old_logprobs_tensor = torch.cat([old_logprobs_tensor, torch.zeros(pad_len, dtype=torch.float32)])
        batch_old_logprobs.append(old_logprobs_tensor)

        # 处理reward（实际数据中是字典）
        reward_dict = sample['reward']
        total_reward = reward_dict['total_reward']
        batch_rewards.append(total_reward)

    # Tokenize prompts
    q = tokenizer(
        batch_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_len,
    )
    q_input_ids = q["input_ids"].to(device)
    q_attention_mask = q["attention_mask"].to(device)

    # Tokenize responses
    r = tokenizer(
        batch_responses,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_new_tokens,
    )
    r_input_ids = r["input_ids"].to(device)
    r_attention_mask = r["attention_mask"].to(device)

    # 填充old_logprobs到相同长度
    max_logprob_len = max(len(lp) for lp in batch_old_logprobs)
    padded_old_logprobs = torch.zeros(len(batch_old_logprobs), max_logprob_len, dtype=torch.float32, device=device)
    for i, lp in enumerate(batch_old_logprobs):
        padded_old_logprobs[i, :len(lp)] = lp

    # 转换rewards为tensor
    rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32, device=device)

    return {
        "q_input_ids": q_input_ids,
        "q_attention_mask": q_attention_mask,
        "r_input_ids": r_input_ids,
        "r_attention_mask": r_attention_mask,
        "old_logprobs": padded_old_logprobs,
        "rewards": rewards_tensor,
    }


def compute_advantages(
    rewards: torch.Tensor,
    hidden_states: torch.Tensor,
    value_head: nn.Module,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> torch.Tensor:
    """
    使用Value Head估计baseline，并计算advantage

    Args:
        rewards: [batch_size] 奖励值
        hidden_states: [batch_size, seq_len, hidden_size] 模型的隐藏状态
        value_head: Value Head模型
        gamma: 折扣因子
        lam: GAE参数

    Returns:
        [batch_size, seq_len] 优势函数值
    """
    batch_size, seq_len, _ = hidden_states.shape

    # 使用Value Head估计baseline
    with torch.no_grad():
        baseline = value_head(hidden_states)  # [batch_size, seq_len]

    # 扩展rewards到序列维度
    rewards_expanded = rewards.unsqueeze(-1).expand(-1, seq_len)

    # 计算GAE (Generalized Advantage Estimation)
    advantages = torch.zeros_like(rewards_expanded)
    for i in range(batch_size):
        gae = 0.0
        for t in reversed(range(seq_len)):
            delta = rewards_expanded[i, t] + gamma * baseline[i, t] - baseline[i, t]
            gae = delta + gamma * lam * gae
            advantages[i, t] = gae

    # 归一化优势
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages


def load_config(config_path: str = "config/running_config.yaml") -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    import argparse

    parser = argparse.ArgumentParser(description="PPO训练脚本")
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
        policy_model_name=optimize_config.get('policy_model'),
        dataset=cli.dataset,
        epoch=cli.epoch,
        zcp=cli.zcp,
        seed=optimize_config.get('seed', 42),
        max_prompt_len=optimize_config.get('max_prompt_len', 2048),
        max_new_tokens=optimize_config.get('max_new_tokens', 1024),
        temperature=optimize_config.get('temperature', 0.7),
        top_p=optimize_config.get('top_p', 0.9),
        learning_rate=float(optimize_config.get('learning_rate', 1e-6)),
        total_episodes=optimize_config.get('total_episodes', 1000),
        per_device_train_batch_size=optimize_config.get(
            'per_device_train_batch_size', 1),
        gradient_accumulation_steps=optimize_config.get(
            'gradient_accumulation_steps', 8),
        num_ppo_epochs=optimize_config.get('num_ppo_epochs', 1),
        num_mini_batches=optimize_config.get('num_mini_batches', 1),
        kl_coef=float(optimize_config.get('kl_coef', 0.02)),
        missing_eos_penalty=float(optimize_config.get('missing_eos_penalty', 1.0)),
        use_lora=get_bool(optimize_config.get('use_lora', True)),
        lora_r=optimize_config.get('lora_r', 16),
        lora_alpha=optimize_config.get('lora_alpha', 32),
        lora_dropout=float(optimize_config.get('lora_dropout', 0.05)),
        load_in_4bit=get_bool(optimize_config.get('load_in_4bit', True)),
        bnb_4bit_compute_dtype=optimize_config.get(
            'bnb_4bit_compute_dtype', 'bfloat16'),
        enable_thinking=get_bool(
            optimize_config.get('enable_thinking', False)),
        save_steps=optimize_config.get('save_steps', 200),
        log_steps=optimize_config.get('log_steps', 10),
        plot_steps=optimize_config.get('plot_steps', 50),
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = LLMManager._allocate_gpus(llm_memory=21.6)["cuda_visible_devices"] # NOTE: modified

    # 加载 local_llm 配置文件
    local_llm_config_path = "config/local_llm.yaml"
    if not os.path.exists(local_llm_config_path):
        raise FileNotFoundError(f"配置文件不存在: {local_llm_config_path}")

    local_llm_config = load_config(local_llm_config_path)
    args.ref_policy_model_path = local_llm_config["models"][args.policy_model_name]["model_path"]
    args.policy_model_name = build_model_name(args.policy_model_name, args.dataset, args.zcp, args.epoch)
    if 'finetuned' not in args.policy_model_name:
        args.policy_model_path = args.ref_policy_model_path
    else:
        args.policy_model_path = finetuned_model_path(args.dataset, args.zcp, args.epoch)

    logger.info(f"已从配置文件加载参数: {config_path}")

    # 设置输出目录
    output_dir = finetuned_epoch_dir(args.dataset, args.zcp, args.epoch + 1)
    ensure_dir(output_dir)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("PPO训练需要CUDA支持")

    logger.info(
        f"开始PPO训练 | Dataset: {args.dataset} | Epoch: {args.epoch} | Output: {output_dir}")

    # -------------------------
    # 加载训练数据
    # -------------------------
    training_samples = load_training_data(args.dataset, args.zcp, args.epoch)
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

    policy = AutoModelForCausalLM.from_pretrained(
        args.policy_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quant_config,
    )

    try:
        policy.generation_config.enable_thinking = args.enable_thinking
    except Exception:
        pass

    policy.resize_token_embeddings(len(tokenizer))

    # 创建参考策略（用于KL散度）
    ref_policy = AutoModelForCausalLM.from_pretrained(
        args.ref_policy_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quant_config,
    )
    ref_policy.eval()

    if args.use_lora:
        if args.load_in_4bit:
            policy = prepare_model_for_kbit_training(policy)
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj",
                            "o_proj", "up_proj", "down_proj", "gate_proj"],
        )
        policy = get_peft_model(policy, lora_cfg)

    # -------------------------
    # PPO Loss & Optimizer
    # -------------------------
    ppo_loss_fn = PPOLoss(clip_range=0.2, kl_coef=args.kl_coef)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate)

    # -------------------------
    # Value Head
    # -------------------------
    # 获取模型的隐藏层大小
    hidden_size = policy.config.hidden_size
    value_head = ValueHead(hidden_size, dropout=0.1).to(device)
    value_optimizer = torch.optim.AdamW(value_head.parameters(), lr=args.learning_rate)

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
            if args.use_lora and hasattr(policy, 'merge_and_unload'):
                try:
                    policy = policy.merge_and_unload()
                    policy.save_pretrained(final_dir)
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
    monitor = TrainingMonitor(output_dir, plot_steps=args.plot_steps)
    monitor.start_training()

    # -------------------------
    # Training loop
    # -------------------------
    global_step = 0
    sample_idx = 0
    n_samples = len(training_samples)

    # 检查是否有最近的checkpoint
    latest_checkpoint = find_latest_checkpoint(output_dir)
    if latest_checkpoint:
        logger.info(f"找到最近的checkpoint: {latest_checkpoint}")
        logger.info("从checkpoint继续训练...")
        
        # 从checkpoint加载模型
        policy = AutoModelForCausalLM.from_pretrained(
            latest_checkpoint,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=quant_config,
        )
        
        # 从checkpoint加载optimizer状态
        optimizer_path = os.path.join(latest_checkpoint, "optimizer.pt")
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path))
            logger.info("已加载optimizer状态")
        
        # 从checkpoint目录名提取global_step
        checkpoint_step = int(latest_checkpoint.split("-")[-1])
        global_step = checkpoint_step
        logger.info(f"已恢复到step: {global_step}")
    else:
        logger.info("没有找到checkpoint，从头开始训练...")

    logger.info(f"开始训练循环 | 总步数: {args.total_episodes} | 总样本数: {n_samples}")

    policy.train()
    
    # 用于reward归一化的统计信息
    reward_mean = 0.0
    reward_std = 1.0
    reward_count = 0
    reward_history = []

    while global_step < args.total_episodes:
        # 准备批次数据
        batch_samples = []
        for _ in range(args.per_device_train_batch_size):
            batch_samples.append(training_samples[sample_idx % n_samples])
            sample_idx += 1

        # 整理批次
        batch = collate_batch(
            batch_samples,
            tokenizer,
            args.max_prompt_len,
            args.max_new_tokens,
            device,
        )

        # 计算当前策略的log概率
        with torch.no_grad():
            # 使用参考策略计算log概率
            ref_outputs = ref_policy(
                batch["r_input_ids"],
                attention_mask=batch["r_attention_mask"],
                labels=batch["r_input_ids"],
            )
            ref_logits = ref_outputs.logits

        # 计算当前策略的log概率
        policy_outputs = policy(
            batch["r_input_ids"],
            attention_mask=batch["r_attention_mask"],
            labels=batch["r_input_ids"],
            output_hidden_states=True,
        )
        policy_logits = policy_outputs.logits
        hidden_states = policy_outputs.hidden_states[-1]  # 使用最后一层的隐藏状态

        # 使用 Value Head 估计 baseline
        baseline = value_head(hidden_states)  # [batch_size, seq_len]

        # 计算 advantage: 使用GAE (Generalized Advantage Estimation)
        rewards = batch["rewards"]  # [batch_size]
        
        # 更新reward统计信息（用于归一化）
        reward_history.extend(rewards.cpu().tolist())
        if len(reward_history) > 1000:  # 只保留最近1000个reward
            reward_history = reward_history[-1000:]
        if len(reward_history) > 10:
            reward_mean = np.mean(reward_history)
            reward_std = np.std(reward_history) + 1e-8
        
        # 可选：归一化rewards（使用移动平均）
        if len(reward_history) > 10:
            rewards_normalized = (rewards - reward_mean) / reward_std
        else:
            rewards_normalized = rewards
        
        seq_len = baseline.shape[1]
        
        # 使用GAE计算advantage
        gamma = 0.99
        lam = 0.95
        advantages = torch.zeros_like(baseline)
        for i in range(baseline.shape[0]):
            gae = 0.0
            for t in reversed(range(seq_len)):
                # 计算TD error
                if t == seq_len - 1:
                    # 最后一步：使用归一化后的reward
                    delta = rewards_normalized[i] - baseline[i, t]
                else:
                    # 中间步：value_next - value（中间步没有reward）
                    delta = gamma * baseline[i, t+1] - baseline[i, t]
                gae = delta + gamma * lam * gae
                advantages[i, t] = gae
        
        # 归一化 advantage（按batch和sequence维度）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 计算log概率
        # shift logits和labels以对齐
        shift_logits = policy_logits[..., :-1, :].contiguous()
        shift_labels = batch["r_input_ids"][..., 1:].contiguous()
        shift_ref_logits = ref_logits[..., :-1, :].contiguous()

        # 计算log概率
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        new_log_probs = torch.gather(log_probs, -1, shift_labels.unsqueeze(-1)).squeeze(-1)

        # 计算参考策略的log概率
        ref_log_probs = torch.nn.functional.log_softmax(shift_ref_logits, dim=-1)
        new_log_probs_ref = torch.gather(ref_log_probs, -1, shift_labels.unsqueeze(-1)).squeeze(-1)

        # 调整old_logprobs的长度以匹配
        old_logprobs = batch["old_logprobs"]
        if old_logprobs.shape[1] != new_log_probs.shape[1]:
            if old_logprobs.shape[1] > new_log_probs.shape[1]:
                old_logprobs = old_logprobs[:, :new_log_probs.shape[1]]
            else:
                pad_len = new_log_probs.shape[1] - old_logprobs.shape[1]
                old_logprobs = torch.nn.functional.pad(old_logprobs, (0, pad_len))

        # 调整advantages的长度
        if advantages.shape[1] != new_log_probs.shape[1]:
            if advantages.shape[1] > new_log_probs.shape[1]:
                advantages = advantages[:, :new_log_probs.shape[1]]
            else:
                pad_len = new_log_probs.shape[1] - advantages.shape[1]
                advantages = torch.nn.functional.pad(advantages, (0, pad_len))

        # 计算PPO损失
        loss_dict = ppo_loss_fn(
            new_log_probs,
            old_logprobs,
            advantages,
            new_log_probs_ref,
        )

        loss = loss_dict["total_loss"]

        # 计算 Value Head 的损失 (MSE loss: (value - target)^2)
        # 使用累积奖励作为目标值（从当前步到序列结束的累积奖励）
        value_targets = torch.zeros_like(baseline)
        for i in range(baseline.shape[0]):
            # 从后往前计算累积奖励（使用归一化后的reward）
            cumulative_reward = rewards_normalized[i]
            for t in reversed(range(seq_len)):
                value_targets[i, t] = cumulative_reward
                # 对于中间步，累积奖励需要折扣
                if t > 0:
                    cumulative_reward = cumulative_reward * gamma  # gamma=0.99
        
        value_loss = nn.functional.mse_loss(baseline, value_targets)

        # 总损失 = PPO损失 + Value Head损失
        total_loss = loss + value_loss

        # 反向传播
        total_loss = total_loss / args.gradient_accumulation_steps
        total_loss.backward()

        # 梯度累积
        if (global_step + 1) % args.gradient_accumulation_steps == 0:
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(value_head.parameters(), max_norm=1.0)
            # 更新参数
            optimizer.step()
            value_optimizer.step()
            optimizer.zero_grad()
            value_optimizer.zero_grad()

        global_step += 1

        # 计算指标
        avg_reward = batch["rewards"].mean().item()
        loss_value = loss_dict["total_loss"].item()
        value_loss_value = value_loss.item()
        policy_loss_value = loss_dict["policy_loss"].item()
        kl_divergence_value = loss_dict["kl_divergence"].item() if loss_dict["kl_divergence"] is not None else 0.0
        ratio_value = loss_dict["ratio"].item()
        
        # 记录advantage统计信息用于调试
        avg_advantage = advantages.mean().item()
        std_advantage = advantages.std().item()

        # 记录指标
        monitor.log_step(
            step=global_step,
            loss=loss_value,
            reward=avg_reward,
            kl_divergence=kl_divergence_value,
            learning_rate=args.learning_rate,
            accuracy=None,
            stats={
                "policy_loss": policy_loss_value,
                "value_loss": value_loss_value,
                "ratio": ratio_value,
                "avg_advantage": avg_advantage,
                "std_advantage": std_advantage,
            },
        )

        # 打印日志
        if global_step % args.log_steps == 0:
            remaining_time = monitor.estimate_remaining_time(
                global_step, args.total_episodes)
            log_msg = (
                f"[Step {global_step}/{args.total_episodes}] "
                f"Loss: {loss_value:.4f} | Value Loss: {value_loss_value:.4f} | Reward: {avg_reward:.4f} | "
                f"KL: {kl_divergence_value:.4f} | "
                f"Ratio: {ratio_value:.4f} | "
                f"Adv: {avg_advantage:.4f}±{std_advantage:.4f}"
            )
            log_msg += f" | 剩余时间: {remaining_time}"
            logger.info(log_msg)

        # 保存检查点
        if global_step % args.save_steps == 0:
            checkpoint_dir = os.path.join(
                output_dir, f"checkpoint-{global_step}")
            policy.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            logger.info(f"检查点已保存: {checkpoint_dir}")

    # 如果使用了LoRA，先合并权重到基础模型
    if args.use_lora and hasattr(policy, 'merge_and_unload'):
        logger.info("正在合并LoRA权重到基础模型...")
        try:
            policy = policy.merge_and_unload()
            logger.info("LoRA权重合并完成")
        except Exception as e:
            logger.warning(f"合并LoRA权重失败: {str(e)}")
            logger.warning("将保存未合并的LoRA模型")

    policy.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"最终模型已保存: {final_dir}")

    # 保存训练总结
    summary = monitor.save_summary(args.total_episodes)
    monitor.plot_metrics()  # 最终绘制一次

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

    logger.info("=" * 60)
    logger.info("训练完成!")
    logger.info(f"总步数: {args.total_episodes}")
    logger.info(f"总时间: {summary.get('total_time_formatted', 'N/A')}")
    logger.info(f"最终Loss: {summary.get('final_loss', 'N/A')}")
    logger.info(f"最终Reward: {summary.get('final_reward', 'N/A')}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
