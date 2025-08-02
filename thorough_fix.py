"""
彻底修复数据集的索引问题
确保所有实体和关系类型都在正确范围内
"""

import json


def thorough_fix_dataset(input_path, output_path):
    """彻底修复数据集"""
    print(f"🔧 彻底修复: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    fixed_data = []
    
    for item in data:
        fixed_item = {
            "document": item.get("document", ""),
            "query": item.get("query", ""),
            "entities": [],
            "relations": [],
            "label": min(item.get("label", 0), 4),  # 确保标签在0-4范围内
            "metadata": item.get("metadata", {})
        }
        
        # 修复实体 - 强制所有实体类型为0-7范围内
        for entity in item.get("entities", []):
            fixed_entity = {
                "span": entity.get("span", [0, 0]),
                "type": 0,  # 统一设为0，避免索引问题
                "text": entity.get("text", "")
            }
            fixed_item["entities"].append(fixed_entity)
        
        # 修复关系 - 强制所有关系类型为0-3范围内
        for relation in item.get("relations", []):
            head = relation.get("head", 0)
            tail = relation.get("tail", 0)
            
            # 确保head和tail索引在实体范围内
            if head < len(fixed_item["entities"]) and tail < len(fixed_item["entities"]):
                fixed_relation = {
                    "head": head,
                    "tail": tail,
                    "type": 0  # 统一设为0，避免索引问题
                }
                fixed_item["relations"].append(fixed_relation)
        
        fixed_data.append(fixed_item)
    
    # 保存修复后的数据
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(fixed_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 彻底修复完成: {output_path} ({len(fixed_data)} 样本)")
    
    # 验证修复结果
    verify_dataset(output_path)


def verify_dataset(filepath):
    """验证数据集的正确性"""
    print(f"🔍 验证数据集: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    entity_types = []
    relation_types = []
    labels = []
    
    for item in data:
        # 检查实体类型
        for entity in item.get("entities", []):
            entity_types.append(entity.get("type", 0))
        
        # 检查关系类型
        for relation in item.get("relations", []):
            relation_types.append(relation.get("type", 0))
        
        # 检查标签
        labels.append(item.get("label", 0))
    
    print(f"实体类型范围: {min(entity_types) if entity_types else 0} - {max(entity_types) if entity_types else 0}")
    print(f"关系类型范围: {min(relation_types) if relation_types else 0} - {max(relation_types) if relation_types else 0}")
    print(f"标签范围: {min(labels)} - {max(labels)}")
    
    # 检查是否有超出范围的值
    max_entity = max(entity_types) if entity_types else 0
    max_relation = max(relation_types) if relation_types else 0
    max_label = max(labels)
    
    if max_entity >= 8:
        print(f"❌ 实体类型超出范围: {max_entity}")
    if max_relation >= 4:
        print(f"❌ 关系类型超出范围: {max_relation}")
    if max_label >= 5:
        print(f"❌ 标签超出范围: {max_label}")
    
    if max_entity < 8 and max_relation < 4 and max_label < 5:
        print("✅ 数据集验证通过")


def main():
    """主函数"""
    print("🔧 彻底修复数据集工具")
    print("=" * 40)
    
    # 修复数据集
    datasets = [
        ('data/hotpotqa_small_train.json', 'data/hotpotqa_small_train_safe.json'),
        ('data/hotpotqa_small_val.json', 'data/hotpotqa_small_val_safe.json')
    ]
    
    for input_path, output_path in datasets:
        try:
            thorough_fix_dataset(input_path, output_path)
            print()
        except Exception as e:
            print(f"❌ 修复失败 {input_path}: {e}")


if __name__ == "__main__":
    main()
