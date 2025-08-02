"""
真实数据集转换脚本
将您的数据转换为GDM-Net格式
"""

import json
import pandas as pd
from typing import List, Dict, Any


def convert_from_csv(csv_path: str, output_path: str):
    """从CSV文件转换数据"""
    df = pd.read_csv(csv_path)
    
    converted_data = []
    for _, row in df.iterrows():
        sample = {
            "document": row['document'],
            "query": row.get('query', ''),
            "entities": json.loads(row.get('entities', '[]')),
            "relations": json.loads(row.get('relations', '[]')),
            "label": int(row.get('label', 0)),
            "metadata": {"source": "csv", "id": str(row.get('id', ''))}
        }
        converted_data.append(sample)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)
    
    print(f"转换完成: {len(converted_data)} 个样本 -> {output_path}")


def convert_from_jsonl(jsonl_path: str, output_path: str):
    """从JSONL文件转换数据"""
    converted_data = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                # 根据您的数据格式调整这里
                sample = {
                    "document": data.get('text', data.get('document', '')),
                    "query": data.get('query', ''),
                    "entities": data.get('entities', []),
                    "relations": data.get('relations', []),
                    "label": data.get('label', 0),
                    "metadata": data.get('metadata', {})
                }
                converted_data.append(sample)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)
    
    print(f"转换完成: {len(converted_data)} 个样本 -> {output_path}")


def convert_docred_format(docred_path: str, output_path: str):
    """转换DocRED格式数据"""
    with open(docred_path, 'r', encoding='utf-8') as f:
        docred_data = json.load(f)
    
    converted_data = []
    for item in docred_data:
        # 合并句子
        document = ' '.join([' '.join(sent) for sent in item['sents']])
        
        # 转换实体
        entities = []
        for entity in item['vertexSet']:
            for mention in entity:
                entities.append({
                    "span": [mention['pos'][0], mention['pos'][1]],
                    "type": mention.get('type', 'MISC'),
                    "text": mention['name']
                })
        
        # 转换关系
        relations = []
        for relation in item.get('labels', []):
            relations.append({
                "head": relation['h'],
                "tail": relation['t'],
                "type": relation['r']
            })
        
        sample = {
            "document": document,
            "query": "",  # DocRED没有查询
            "entities": entities,
            "relations": relations,
            "label": 0,  # 默认标签
            "metadata": {"source": "docred", "title": item.get('title', '')}
        }
        converted_data.append(sample)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)
    
    print(f"转换完成: {len(converted_data)} 个样本 -> {output_path}")


if __name__ == "__main__":
    print("数据转换脚本")
    print("请根据您的数据格式选择相应的转换函数")
    print("1. convert_from_csv() - CSV格式")
    print("2. convert_from_jsonl() - JSONL格式") 
    print("3. convert_docred_format() - DocRED格式")
