"""
创建小规模数据集用于快速测试
"""

import json
import random


def create_small_dataset():
    """创建小规模的HotpotQA数据集"""
    print("📊 创建小规模测试数据集...")
    
    # 读取原始数据
    with open('data/hotpotqa_official_train.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    with open('data/hotpotqa_official_val.json', 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    
    # 随机选择小样本
    random.seed(42)
    small_train = random.sample(train_data, min(200, len(train_data)))
    small_val = random.sample(val_data, min(50, len(val_data)))
    
    # 保存小数据集
    with open('data/hotpotqa_small_train.json', 'w', encoding='utf-8') as f:
        json.dump(small_train, f, indent=2, ensure_ascii=False)
    
    with open('data/hotpotqa_small_val.json', 'w', encoding='utf-8') as f:
        json.dump(small_val, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 小数据集创建完成:")
    print(f"   训练集: {len(small_train)} 样本")
    print(f"   验证集: {len(small_val)} 样本")
    
    return len(small_train), len(small_val)


if __name__ == "__main__":
    create_small_dataset()
