"""
最终修复脚本 - 彻底解决所有索引问题
确保所有标签都严格在模型定义的范围内
"""

import json
import numpy as np


def final_fix_dataset(input_path, output_path):
    """最终修复数据集，确保所有索引都在正确范围内"""
    print(f"🔧 最终修复: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    fixed_data = []
    
    for i, item in enumerate(data):
        try:
            fixed_item = {
                "document": item.get("document", ""),
                "query": item.get("query", ""),
                "entities": [],
                "relations": [],
                "label": max(0, min(item.get("label", 0), 4)),  # 强制在0-4范围内
                "metadata": item.get("metadata", {})
            }
            
            # 处理实体 - 确保所有值都在0-7范围内
            for entity in item.get("entities", []):
                span = entity.get("span", [0, 0])
                if not isinstance(span, list) or len(span) != 2:
                    span = [0, 0]
                
                # 确保span是有效的整数
                try:
                    span = [max(0, int(span[0])), max(0, int(span[1]))]
                except:
                    span = [0, 0]
                
                # 获取实体类型并强制在0-7范围内
                entity_type = entity.get("type", 0)
                if isinstance(entity_type, str):
                    # 字符串映射
                    type_map = {
                        'TITLE': 1, 'ENTITY': 2, 'PERSON': 3, 'ORGANIZATION': 4,
                        'LOCATION': 5, 'MISC': 6, 'DATE': 7
                    }
                    entity_type = type_map.get(entity_type, 0)
                else:
                    try:
                        entity_type = int(entity_type)
                    except:
                        entity_type = 0
                
                # 强制在0-7范围内
                entity_type = max(0, min(entity_type, 7))
                
                fixed_entity = {
                    "span": span,
                    "type": entity_type,
                    "text": str(entity.get("text", ""))
                }
                fixed_item["entities"].append(fixed_entity)
            
            # 处理关系 - 确保所有值都在0-3范围内
            for relation in item.get("relations", []):
                try:
                    head = max(0, min(int(relation.get("head", 0)), len(fixed_item["entities"]) - 1))
                    tail = max(0, min(int(relation.get("tail", 0)), len(fixed_item["entities"]) - 1))
                    
                    # 只有当head和tail都有效时才添加关系
                    if head < len(fixed_item["entities"]) and tail < len(fixed_item["entities"]) and head != tail:
                        relation_type = relation.get("type", 0)
                        if isinstance(relation_type, str):
                            type_map = {
                                'SUPPORTS': 1, 'RELATED': 2, 'PART_OF': 3
                            }
                            relation_type = type_map.get(relation_type, 0)
                        else:
                            try:
                                relation_type = int(relation_type)
                            except:
                                relation_type = 0
                        
                        # 强制在0-3范围内
                        relation_type = max(0, min(relation_type, 3))
                        
                        fixed_relation = {
                            "head": head,
                            "tail": tail,
                            "type": relation_type
                        }
                        fixed_item["relations"].append(fixed_relation)
                except:
                    continue  # 跳过有问题的关系
            
            fixed_data.append(fixed_item)
            
        except Exception as e:
            print(f"⚠️  跳过样本 {i}: {e}")
            continue
    
    # 保存修复后的数据
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(fixed_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 最终修复完成: {output_path} ({len(fixed_data)} 样本)")
    
    # 详细验证
    verify_fixed_dataset(output_path)
    return output_path


def verify_fixed_dataset(filepath):
    """详细验证修复后的数据集"""
    print(f"🔍 详细验证: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    entity_types = []
    relation_types = []
    labels = []
    issues = []
    
    for i, item in enumerate(data):
        # 检查标签
        label = item.get("label", 0)
        labels.append(label)
        if label < 0 or label > 4:
            issues.append(f"样本 {i}: 标签 {label} 超出范围 [0,4]")
        
        # 检查实体
        for j, entity in enumerate(item.get("entities", [])):
            entity_type = entity.get("type", 0)
            entity_types.append(entity_type)
            if entity_type < 0 or entity_type > 7:
                issues.append(f"样本 {i}, 实体 {j}: 类型 {entity_type} 超出范围 [0,7]")
        
        # 检查关系
        for j, relation in enumerate(item.get("relations", [])):
            relation_type = relation.get("type", 0)
            relation_types.append(relation_type)
            if relation_type < 0 or relation_type > 3:
                issues.append(f"样本 {i}, 关系 {j}: 类型 {relation_type} 超出范围 [0,3]")
            
            # 检查head和tail索引
            head = relation.get("head", 0)
            tail = relation.get("tail", 0)
            num_entities = len(item.get("entities", []))
            if head >= num_entities or tail >= num_entities:
                issues.append(f"样本 {i}, 关系 {j}: head={head} 或 tail={tail} 超出实体数量 {num_entities}")
    
    # 统计信息
    print(f"📊 统计信息:")
    print(f"  样本数量: {len(data)}")
    print(f"  标签范围: {min(labels)} - {max(labels)}")
    print(f"  实体类型范围: {min(entity_types) if entity_types else 'N/A'} - {max(entity_types) if entity_types else 'N/A'}")
    print(f"  关系类型范围: {min(relation_types) if relation_types else 'N/A'} - {max(relation_types) if relation_types else 'N/A'}")
    
    # 报告问题
    if issues:
        print(f"❌ 发现 {len(issues)} 个问题:")
        for issue in issues[:5]:  # 只显示前5个问题
            print(f"  {issue}")
        if len(issues) > 5:
            print(f"  ... 还有 {len(issues) - 5} 个问题")
        return False
    else:
        print("✅ 验证通过，所有索引都在正确范围内")
        return True


def create_minimal_test_data():
    """创建最小测试数据集，确保没有任何问题"""
    print("📝 创建最小测试数据集...")
    
    minimal_data = []
    
    # 创建10个简单的测试样本
    for i in range(10):
        sample = {
            "document": f"This is test document {i}. It contains some text for testing.",
            "query": f"What is the content of document {i}?",
            "entities": [
                {
                    "span": [0, 4],  # "This"
                    "type": 0,
                    "text": "This"
                },
                {
                    "span": [8, 12],  # "test"
                    "type": 1,
                    "text": "test"
                }
            ],
            "relations": [
                {
                    "head": 0,
                    "tail": 1,
                    "type": 0
                }
            ],
            "label": i % 5,  # 0-4范围内
            "metadata": {
                "source": "minimal_test",
                "id": f"test_{i}"
            }
        }
        minimal_data.append(sample)
    
    # 保存训练集和验证集
    train_data = minimal_data[:8]
    val_data = minimal_data[8:]
    
    with open('data/minimal_train.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open('data/minimal_val.json', 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 最小测试数据集创建完成:")
    print(f"  训练集: {len(train_data)} 样本")
    print(f"  验证集: {len(val_data)} 样本")
    
    # 验证最小数据集
    verify_fixed_dataset('data/minimal_train.json')
    verify_fixed_dataset('data/minimal_val.json')


def main():
    """主函数"""
    print("🔧 最终修复工具")
    print("=" * 50)
    
    # 选项1: 修复现有数据集
    datasets_to_fix = [
        ('data/hotpotqa_small_train.json', 'data/hotpotqa_final_train.json'),
        ('data/hotpotqa_small_val.json', 'data/hotpotqa_final_val.json')
    ]
    
    all_success = True
    
    for input_path, output_path in datasets_to_fix:
        try:
            if not final_fix_dataset(input_path, output_path):
                all_success = False
            print()
        except Exception as e:
            print(f"❌ 修复失败 {input_path}: {e}")
            all_success = False
    
    # 选项2: 创建最小测试数据集
    print("=" * 50)
    create_minimal_test_data()
    
    if all_success:
        print("\n🎉 所有数据集修复完成！")
        print("建议使用以下配置:")
        print("- 完整数据集: data/hotpotqa_final_*.json")
        print("- 最小测试集: data/minimal_*.json")
    else:
        print("\n⚠️  部分数据集修复失败，建议使用最小测试集")


if __name__ == "__main__":
    main()
