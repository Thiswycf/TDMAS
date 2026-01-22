"""
奖励计算模块
计算监督得分奖励、一致性奖励和token奖励
"""


def calculate_reward(
    supervision_score: int,
    consistency_score: int,
    input_tokens: int,
    output_tokens: int,
    supervision_weight: float = 1.0,
    consistency_weight: float = 0.1,
    token_weight: float = -1e-5
) -> dict:
    """计算总奖励
    
    奖励 = supervision_weight * 监督得分奖励 + 
          consistency_weight * 一致性奖励 + 
          token_weight * token奖励
    
    Args:
        supervision_score: 监督得分
        consistency_score: 一致性得分
        input_tokens: 输入token数
        output_tokens: 输出token数
        supervision_weight: 监督得分权重
        consistency_weight: 一致性得分权重
        token_weight: token权重
    
    Returns:
        包含各项奖励的字典
    """
    supervision_reward = supervision_weight * supervision_score
    consistency_reward = consistency_weight * consistency_score
    token_reward = token_weight * output_tokens
    
    total_reward = (
        supervision_reward +
        consistency_reward +
        token_reward
    )
    
    return {
        'total_reward': total_reward,
        'supervision_reward': supervision_reward,
        'consistency_reward': consistency_reward,
        'token_reward': token_reward,
        'supervision_weight': supervision_weight,
        'consistency_weight': consistency_weight,
        'token_weight': token_weight
    }
