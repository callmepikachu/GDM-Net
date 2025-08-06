"""
调试数据集问题
检查实体和关系类型的范围
"""

import json
import numpy as np


def analyze_dataset(filepath):
    """分析数据集中的实体和关系类型"""
    print(f"📊 分析数据集: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    entity_types = []
    relation_types = []
    labels = []
    
    for item in data:
        # 收集实体类型
        for entity in item.get('entities', []):
            if isinstance(entity.get('type'), int):
                entity_types.append(entity['type'])
            elif isinstance(entity.get('type'), str):
                # 如果是字符串，尝试转换
                try:
                    entity_types.append(hash(entity['type']) % 10)  # 简单哈希到0-9
                except:
                    entity_types.append(0)
        
        # 收集关系类型
        for relation in item.get('relations', []):
            if isinstance(relation.get('type'), int):
                relation_types.append(relation['type'])
            elif isinstance(relation.get('type'), str):
                try:
                    relation_types.append(hash(relation['type']) % 10)  # 简单哈希到0-9
                except:
                    relation_types.append(0)
        
        # 收集标签
        labels.append(item.get('label', 0))
    
    print(f"实体类型范围: {min(entity_types) if entity_types else 'N/A'} - {max(entity_types) if entity_types else 'N/A'}")
    print(f"关系类型范围: {min(relation_types) if relation_types else 'N/A'} - {max(relation_types) if relation_types else 'N/A'}")
    print(f"标签范围: {min(labels)} - {max(labels)}")
    print(f"唯一实体类型: {sorted(set(entity_types)) if entity_types else []}")
    print(f"唯一关系类型: {sorted(set(relation_types)) if relation_types else []}")
    print(f"唯一标签: {sorted(set(labels))}")
    
    return {
        'max_entity_type': max(entity_types) if entity_types else 0,
        'max_relation_type': max(relation_types) if relation_types else 0,
        'max_label': max(labels),
        'entity_types': sorted(set(entity_types)) if entity_types else [],
        'relation_types': sorted(set(relation_types)) if relation_types else [],
        'labels': sorted(set(labels))
    }


def fix_dataset(filepath, output_path, max_entity_type=7, max_relation_type=9):
    """修复数据集中的类型索引问题"""
    print(f"🔧 修复数据集: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    fixed_data = []
    
    for item in data:
        fixed_item = item.copy()
        
        # 修复实体类型
        fixed_entities = []
        for entity in item.get('entities', []):
            fixed_entity = entity.copy()
            entity_type = entity.get('type', 0)
            
            if isinstance(entity_type, str):
                # 字符串类型映射
                type_mapping = {
                    'TITLE': 1,
                    'ENTITY': 2,
                    'PERSON': 3,
                    'ORGANIZATION': 4,
                    'LOCATION': 5,
                    'MISC': 6
                }
                fixed_entity['type'] = type_mapping.get(entity_type, 0)
            elif isinstance(entity_type, int):
                # 确保在范围内
                fixed_entity['type'] = min(entity_type, max_entity_type)
            else:
                fixed_entity['type'] = 0
            
            fixed_entities.append(fixed_entity)
        
        fixed_item['entities'] = fixed_entities
        
        # 修复关系类型
        fixed_relations = []
        for relation in item.get('relations', []):
            fixed_relation = relation.copy()
            relation_type = relation.get('type', 0)
            
            if isinstance(relation_type, str):
                # 字符串类型映射
                type_mapping = {
                    'SUPPORTS': 1,
                    'RELATED': 2,
                    'PART_OF': 3,
                    'LOCATED_IN': 4
                }
                fixed_relation['type'] = type_mapping.get(relation_type, 0)
            elif isinstance(relation_type, int):
                # 确保在范围内
                fixed_relation['type'] = min(relation_type, max_relation_type)
            else:
                fixed_relation['type'] = 0
            
            fixed_relations.append(fixed_relation)
        
        fixed_item['relations'] = fixed_relations
        
        # 确保标签在合理范围内
        fixed_item['label'] = min(item.get('label', 0), 4)
        
        fixed_data.append(fixed_item)
    
    # 保存修复后的数据
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(fixed_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 修复完成: {output_path} ({len(fixed_data)} 样本)")
    return output_path


def main():
    """主函数"""
    print("🔍 数据集调试工具")
    print("=" * 40)
    
    # 分析数据集
    datasets = [
        'data/hotpotqa_small_train.json',
        'data/hotpotqa_small_val.json'
    ]
    
    max_entity = 0
    max_relation = 0
    
    for dataset in datasets:
        try:
            stats = analyze_dataset(dataset)
            max_entity = max(max_entity, stats['max_entity_type'])
            max_relation = max(max_relation, stats['max_relation_type'])
            print()
        except Exception as e:
            print(f"❌ 分析失败 {dataset}: {e}")
    
    print(f"📋 建议的模型配置:")
    print(f"num_entities: {max_entity + 1}")
    print(f"num_relations: {max_relation + 1}")
    
    # 修复数据集
    print(f"\n🔧 修复数据集...")
    for dataset in datasets:
        try:
            output_path = dataset.replace('.json', '_fixed.json')
            fix_dataset(dataset, output_path, max_entity_type=7, max_relation_type=9)
        except Exception as e:
            print(f"❌ 修复失败 {dataset}: {e}")


if __name__ == "__main__":
    main()
