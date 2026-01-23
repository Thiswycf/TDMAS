#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析训练指标脚本
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def analyze_training_metrics(metrics_file):
    """分析训练指标"""
    if not os.path.exists(metrics_file):
        print(f"指标文件不存在: {metrics_file}")
        return
    
    # 读取数据
    data = []
    with open(metrics_file, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                continue
    
    if not data:
        print("没有数据")
        return
    
    # 提取指标
    steps = [d['step'] for d in data]
    losses = [d.get('loss', 0) for d in data if d.get('loss') is not None]
    rewards = [d.get('reward', 0) for d in data if d.get('reward') is not None]
    kls = [d.get('kl_divergence', 0) for d in data if d.get('kl_divergence') is not None]
    value_losses = [d.get('value_loss', 0) for d in data if d.get('value_loss') is not None]
    
    # 计算统计信息
    print("=" * 60)
    print("训练指标分析")
    print("=" * 60)
    print(f"总步数: {len(data)}")
    print(f"\nLoss统计:")
    print(f"  范围: {min(losses):.4f} - {max(losses):.4f}")
    print(f"  平均: {np.mean(losses):.4f}")
    print(f"  最后10步平均: {np.mean(losses[-10:]):.4f}")
    print(f"  趋势: {'下降' if np.mean(losses[-10:]) < np.mean(losses[:10]) else '上升或波动'}")
    
    print(f"\nReward统计:")
    print(f"  范围: {min(rewards):.4f} - {max(rewards):.4f}")
    print(f"  平均: {np.mean(rewards):.4f}")
    print(f"  最后10步平均: {np.mean(rewards[-10:]):.4f}")
    print(f"  标准差: {np.std(rewards):.4f}")
    
    print(f"\nKL散度统计:")
    if kls:
        print(f"  范围: {min(kls):.4f} - {max(kls):.4f}")
        print(f"  平均: {np.mean(kls):.4f}")
        print(f"  最后10步平均: {np.mean(kls[-10:]):.4f}")
        print(f"  趋势: {'稳定' if np.std(kls[-50:]) < 0.1 else '波动'}")
    
    print(f"\nValue Loss统计:")
    if value_losses:
        print(f"  范围: {min(value_losses):.4f} - {max(value_losses):.4f}")
        print(f"  平均: {np.mean(value_losses):.4f}")
        print(f"  最后10步平均: {np.mean(value_losses[-10:]):.4f}")
    
    # 分析问题
    print(f"\n问题诊断:")
    issues = []
    
    if len(losses) > 50:
        recent_loss = np.mean(losses[-50:])
        early_loss = np.mean(losses[:50])
        if recent_loss > early_loss * 1.1:
            issues.append("Loss在后期上升，可能存在过拟合或不稳定")
        elif recent_loss < early_loss * 0.9:
            issues.append("Loss持续下降，训练正常")
    
    if kls and np.mean(kls[-50:]) > 0.1:
        issues.append(f"KL散度偏高 ({np.mean(kls[-50:]):.4f})，可能需要降低KL系数")
    
    if np.std(rewards) > 2.0:
        issues.append(f"Reward波动较大 (std={np.std(rewards):.4f})，可能需要reward归一化")
    
    if not issues:
        issues.append("训练指标正常，无明显问题")
    
    for issue in issues:
        print(f"  - {issue}")
    
    print("=" * 60)
    
    return {
        'steps': steps,
        'losses': losses,
        'rewards': rewards,
        'kls': kls,
        'value_losses': value_losses,
        'issues': issues
    }

if __name__ == "__main__":
    import sys
    metrics_file = sys.argv[1] if len(sys.argv) > 1 else "tdmas_workspace/finetuned/AIME24/accuracy/1/metrics.jsonl"
    analyze_training_metrics(metrics_file)
