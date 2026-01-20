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
from torch.utils.data import Dataset
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trl.trainer import PPOConfig, PPOTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from metagpt.logs import logger
from utils.llm_manager import LLMManager
from utils.path_utils import ppo_data_file, finetuned_epoch_dir, ensure_dir, finetuned_model_path, build_model_name
# TRL PPO (experimental API used by TRL official example script)


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
        collected_data = pickle.load(f)

    logger.info(f"成功加载 {len(collected_data)} 条问题数据")
    return collected_data

    # # 展开数据：将每个问题的所有turn展开为独立的训练样本
    # training_samples = []
    # for problem_data in collected_data:
    #     problem_id = problem_data.get('problem_id', 'unknown')
    #     question = problem_data.get('question', '')
    #     correctness = problem_data.get('correctness', 0.0)
    #     single_problem_train_data = problem_data.get(
    #         'single_problem_train_data', [])

    #     for turn_idx, turn_data in enumerate(single_problem_train_data):
    #         prompt = turn_data.get('prompt', [])
    #         response = turn_data.get('response', '')
    #         reward_dict = turn_data.get('reward', {})

    #         # 提取总奖励
    #         if isinstance(reward_dict, dict):
    #             total_reward = reward_dict.get('total_reward', 0.0)
    #         else:
    #             total_reward = float(
    #                 reward_dict) if reward_dict is not None else 0.0

    #         training_samples.append({
    #             'problem_id': problem_id,
    #             'question': question,
    #             'correctness': correctness,
    #             'prompt': prompt,
    #             'response': response,
    #             'reward': total_reward,
    #             'turn_idx': turn_idx,
    #         })

    # logger.info(f"展开后共有 {len(training_samples)} 个训练样本")
    # return training_samples


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

        # 准确率曲线
        if self.accuracies:
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


class DummyRewardModel(nn.Module):
    """占位奖励模型，仅用于满足PPOTrainer对reward_model的device迁移要求"""
    def __init__(self):
        super().__init__()
        self.register_buffer("dummy_value", torch.tensor(0.0))
    def forward(self, *args, **kwargs):
        return self.dummy_value

class ConstantLengthDataset(Dataset):
    """用于PPOTrainer初始化的占位Dataset，不参与实际训练"""
    def __init__(self, length: int, pad_token_id: int):
        self.length = length
        self.pad_token_id = pad_token_id

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor([self.pad_token_id], dtype=torch.long),
            "attention_mask": torch.tensor([1], dtype=torch.long),
        }

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
        policy_model_name=config.get('model_name'),
        dataset=cli.dataset,
        epoch=cli.epoch,
        zcp=cli.zcp,
        seed=optimize_config.get('seed', 42),
        max_prompt_len=optimize_config.get('max_prompt_len', 2048),
        max_new_tokens=optimize_config.get('max_new_tokens', 1024),
        temperature=optimize_config.get('temperature', 0.7),
        top_p=optimize_config.get('top_p', 0.9),
        learning_rate=optimize_config.get('learning_rate', 1e-6),
        total_episodes=optimize_config.get('total_episodes', 1000),
        per_device_train_batch_size=optimize_config.get(
            'per_device_train_batch_size', 1),
        gradient_accumulation_steps=optimize_config.get(
            'gradient_accumulation_steps', 8),
        num_ppo_epochs=optimize_config.get('num_ppo_epochs', 1),
        num_mini_batches=optimize_config.get('num_mini_batches', 1),
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
        save_steps=optimize_config.get('save_steps', 200),
        log_steps=optimize_config.get('log_steps', 10),
        plot_steps=optimize_config.get('plot_steps', 50),
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = LLMManager._allocate_gpus(llm_memory=65.0)["cuda_visible_devices"]

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
    output_dir = finetuned_epoch_dir(args.dataset, args.zcp, args.epoch)
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
    # Policy model + LoRA
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

    ref_policy = None

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
    # PPO Trainer
    # -------------------------
    ppo_config = PPOConfig(
        output_dir=output_dir,
        learning_rate=args.learning_rate,
        total_episodes=args.total_episodes,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_ppo_epochs=args.num_ppo_epochs,
        num_mini_batches=args.num_mini_batches,
        kl_coef=args.kl_coef,
        missing_eos_penalty=args.missing_eos_penalty,
    )

    # PPOTrainer需要reward_model和Dataset实例才能完成初始化
    placeholder_reward_model = DummyRewardModel()
    placeholder_dataset = ConstantLengthDataset(
        length=len(training_samples),
        pad_token_id=tokenizer.pad_token_id
    )

    trainer = PPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model=placeholder_reward_model,
        train_dataset=placeholder_dataset,
    )

    # -------------------------
    # Training Monitor
    # -------------------------
    monitor = TrainingMonitor(output_dir, plot_steps=args.plot_steps)
    monitor.start_training()

    # -------------------------
    # Training loop
    # -------------------------
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    global_step = 0
    sample_idx = 0
    n_samples = len(training_samples)

    logger.info(f"开始训练循环 | 总步数: {args.total_episodes} | 总样本数: {n_samples}")

    while global_step < args.total_episodes:
        # 准备批次数据
        batch_prompts = []
        batch_reference_rewards = []  # 参考reward（来自收集的数据）
        batch_correctness = []

        for _ in range(args.per_device_train_batch_size):
            sample = training_samples[sample_idx % n_samples]
            sample_idx += 1

            # 格式化prompt
            prompt_str = format_prompt_for_tokenizer(
                tokenizer, sample['prompt'])
            batch_prompts.append(prompt_str)

            # 保存参考reward（用于监控）
            batch_reference_rewards.append(sample['reward'])

            # 记录correctness用于准确率计算
            batch_correctness.append(sample['correctness'])

        # Tokenize prompts
        q = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_prompt_len,
        )
        q_input_ids = q["input_ids"].to(trainer.accelerator.device)
        q_attention_mask = q["attention_mask"].to(trainer.accelerator.device)

        # 从当前策略生成responses（这才是真正的PPO）
        with torch.no_grad():
            response_ids = trainer.generate(
                q_input_ids,
                attention_mask=q_attention_mask,
                **gen_kwargs,
            )

        # Decode responses（用于监控）
        responses = []
        for i in range(response_ids.shape[0]):
            prompt_len = q_attention_mask[i].sum().item()
            gen_part = response_ids[i, int(prompt_len):]
            text = tokenizer.decode(gen_part, skip_special_tokens=True).strip()
            responses.append(text)

        # 计算rewards
        rewards = torch.tensor(
            batch_reference_rewards, dtype=torch.float32, device=trainer.accelerator.device)

        # PPO step
        stats = trainer.step(q_input_ids, response_ids, rewards)

        global_step += 1

        # 计算指标
        avg_reward = rewards.mean().item()
        avg_correctness = np.mean(batch_correctness)
        # 从stats中提取各种指标
        loss = stats.get("ppo/loss/total", stats.get("objective/kl", 0.0))
        kl = stats.get("ppo/objective/kl", stats.get("objective/kl", None))
        policy_loss = stats.get("ppo/policy/mean", None)
        value_loss = stats.get("ppo/value/mean", None)
        entropy = stats.get("ppo/entropy/mean", None)
        lr = stats.get("learning_rate", args.learning_rate)

        # 记录指标
        monitor.log_step(
            step=global_step,
            loss=loss,
            reward=avg_reward,
            kl_divergence=kl,
            learning_rate=lr,
            accuracy=avg_correctness,
            stats={
                **stats,
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
            },
        )

        # 打印日志
        if global_step % args.log_steps == 0 and trainer.accelerator.is_main_process:
            remaining_time = monitor.estimate_remaining_time(
                global_step, args.total_episodes)
            kl_display = f"{kl:.4f}" if kl is not None else "N/A"
            log_msg = (
                f"[Step {global_step}/{args.total_episodes}] "
                f"Loss: {loss:.4f} | Reward: {avg_reward:.4f} | "
                f"KL: {kl_display} | Accuracy: {avg_correctness:.4f}"
            )
            if policy_loss is not None:
                log_msg += f" | Policy Loss: {policy_loss:.4f}"
            if value_loss is not None:
                log_msg += f" | Value Loss: {value_loss:.4f}"
            log_msg += f" | 剩余时间: {remaining_time}"
            logger.info(log_msg)

        # 保存检查点
        if global_step % args.save_steps == 0:
            checkpoint_dir = os.path.join(
                output_dir, f"checkpoint-{global_step}")
            trainer.save_pretrained(checkpoint_dir)
            logger.info(f"检查点已保存: {checkpoint_dir}")

    # 最终保存
    final_dir = finetuned_model_path(args.dataset, args.zcp, args.epoch)
    trainer.save_pretrained(final_dir)
    logger.info(f"最终模型已保存: {final_dir}")

    # 保存训练总结
    summary = monitor.save_summary(args.total_episodes)
    monitor.plot_metrics()  # 最终绘制一次

    if trainer.accelerator.is_main_process:
        logger.info("=" * 60)
        logger.info("训练完成!")
        logger.info(f"总步数: {args.total_episodes}")
        logger.info(f"总时间: {summary.get('total_time_formatted', 'N/A')}")
        logger.info(f"最终Loss: {summary.get('final_loss', 'N/A')}")
        logger.info(f"最终Reward: {summary.get('final_reward', 'N/A')}")
        logger.info(f"最终准确率: {summary.get('final_accuracy', 'N/A')}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
