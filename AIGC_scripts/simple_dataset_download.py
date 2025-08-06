"""
简化的数据集下载脚本
直接下载和转换数据集，避免复杂的依赖问题
"""

import os
import json
import requests
from pathlib import Path
from tqdm import tqdm
import random


def create_sample_hotpotqa():
    """创建 HotpotQA 风格的示例数据集"""
    print("📝 创建 HotpotQA 风格的示例数据集...")
    
    # 示例数据模板
    templates = [
        {
            "question": "Which company was founded first, {company1} or {company2}?",
            "contexts": [
                ["{company1}", ["{company1} was founded in {year1} by {founder1}.", "It is a {industry1} company."]],
                ["{company2}", ["{company2} was founded in {year2} by {founder2}.", "It specializes in {industry2}."]]
            ],
            "answer": "{answer}",
            "supporting_facts": [["{company1}", 0], ["{company2}", 0]]
        },
        {
            "question": "Who is the current CEO of {company}?",
            "contexts": [
                ["{company}", ["{company} is a technology company.", "{ceo} is the current CEO of {company}."]],
                ["Leadership", ["The company has undergone several leadership changes.", "Current leadership focuses on innovation."]]
            ],
            "answer": "{ceo}",
            "supporting_facts": [["{company}", 1]]
        },
        {
            "question": "What industry does {company} operate in?",
            "contexts": [
                ["{company}", ["{company} was founded by {founder}.", "The company operates in the {industry} industry."]],
                ["Industry Overview", ["The {industry} industry has grown rapidly.", "Many companies compete in this space."]]
            ],
            "answer": "{industry}",
            "supporting_facts": [["{company}", 1]]
        }
    ]
    
    # 数据填充
    companies = ["Apple Inc.", "Microsoft Corp.", "Google LLC", "Amazon Inc.", "Tesla Inc.", "Meta Platforms", "Netflix Inc."]
    founders = ["Steve Jobs", "Bill Gates", "Larry Page", "Jeff Bezos", "Elon Musk", "Mark Zuckerberg", "Reed Hastings"]
    ceos = ["Tim Cook", "Satya Nadella", "Sundar Pichai", "Andy Jassy", "Elon Musk", "Mark Zuckerberg", "Reed Hastings"]
    industries = ["technology", "software", "e-commerce", "automotive", "social media", "entertainment"]
    years = ["1976", "1975", "1998", "1994", "2003", "2004", "1997"]
    
    # 生成数据
    train_data = []
    val_data = []
    
    for i in range(1000):  # 生成1000个训练样本
        template = random.choice(templates)
        
        if "company1" in template["question"]:
            # 比较类问题
            idx1, idx2 = random.sample(range(len(companies)), 2)
            year1, year2 = int(years[idx1]), int(years[idx2])
            answer = companies[idx1] if year1 < year2 else companies[idx2]
            
            data = {
                "question": template["question"].format(
                    company1=companies[idx1], 
                    company2=companies[idx2]
                ),
                "contexts": [
                    [companies[idx1], [
                        f"{companies[idx1]} was founded in {years[idx1]} by {founders[idx1]}.",
                        f"It is a {random.choice(industries)} company."
                    ]],
                    [companies[idx2], [
                        f"{companies[idx2]} was founded in {years[idx2]} by {founders[idx2]}.",
                        f"It specializes in {random.choice(industries)}."
                    ]]
                ],
                "answer": answer,
                "supporting_facts": [[companies[idx1], 0], [companies[idx2], 0]]
            }
        else:
            # 单一问题
            idx = random.randint(0, len(companies)-1)
            
            if "CEO" in template["question"]:
                answer = ceos[idx]
                contexts = [
                    [companies[idx], [
                        f"{companies[idx]} is a technology company.",
                        f"{ceos[idx]} is the current CEO of {companies[idx]}."
                    ]],
                    ["Leadership", [
                        "The company has undergone several leadership changes.",
                        "Current leadership focuses on innovation."
                    ]]
                ]
            else:  # industry question
                answer = random.choice(industries)
                contexts = [
                    [companies[idx], [
                        f"{companies[idx]} was founded by {founders[idx]}.",
                        f"The company operates in the {answer} industry."
                    ]],
                    ["Industry Overview", [
                        f"The {answer} industry has grown rapidly.",
                        "Many companies compete in this space."
                    ]]
                ]
            
            data = {
                "question": template["question"].format(company=companies[idx], ceo=ceos[idx]),
                "contexts": contexts,
                "answer": answer,
                "supporting_facts": [[companies[idx], 1]]
            }
        
        if i < 800:
            train_data.append(data)
        else:
            val_data.append(data)
    
    # 转换为GDM-Net格式
    train_converted = convert_hotpotqa_format(train_data)
    val_converted = convert_hotpotqa_format(val_data)
    
    # 保存数据
    os.makedirs("data", exist_ok=True)
    save_json(train_converted, "data/hotpotqa_train.json")
    save_json(val_converted, "data/hotpotqa_val.json")
    
    print(f"✅ HotpotQA 风格数据集创建完成: {len(train_converted)} 训练样本, {len(val_converted)} 验证样本")
    return True


def convert_hotpotqa_format(data):
    """转换 HotpotQA 格式到 GDM-Net 格式"""
    converted = []
    
    for item in data:
        # 合并所有上下文
        document_parts = []
        entities = []
        entity_id = 0
        
        for title, sentences in item['contexts']:
            content = ' '.join(sentences)
            doc_part = f"{title}: {content}"
            start_pos = len(' '.join(document_parts))
            if document_parts:
                start_pos += 1  # 空格
            
            document_parts.append(doc_part)
            
            # 添加标题作为实体
            entities.append({
                "span": [start_pos, start_pos + len(title)],
                "type": "ENTITY",
                "text": title
            })
            entity_id += 1
        
        combined_doc = ' '.join(document_parts)
        
        # 创建关系
        relations = []
        if len(entities) >= 2:
            relations.append({
                "head": 0,
                "tail": 1,
                "type": "RELATED"
            })
        
        # 简单的答案分类
        answer_lower = item['answer'].lower()
        if any(word in answer_lower for word in ['apple', 'microsoft', 'google', 'amazon', 'tesla']):
            label = 0  # 公司
        elif any(word in answer_lower for word in ['cook', 'gates', 'pichai', 'bezos', 'musk']):
            label = 1  # 人名
        else:
            label = 2  # 其他
        
        converted.append({
            "document": combined_doc[:1500],  # 限制长度
            "query": item['question'],
            "entities": entities[:10],
            "relations": relations,
            "label": label,
            "metadata": {
                "source": "hotpotqa_style",
                "answer": item['answer'],
                "supporting_facts": item['supporting_facts']
            }
        })
    
    return converted


def download_docred_simple():
    """简化的 DocRED 下载"""
    print("📥 下载 DocRED 数据集...")
    
    # DocRED 的直接下载链接
    urls = {
        "train": "https://raw.githubusercontent.com/thunlp/DocRED/master/data/train_annotated.json",
        "dev": "https://raw.githubusercontent.com/thunlp/DocRED/master/data/dev.json"
    }
    
    try:
        for split, url in urls.items():
            print(f"下载 {split} 数据...")
            
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                converted = convert_docred_format(data[:100])  # 只取前100个样本
                
                filename = f"data/docred_{split}.json"
                save_json(converted, filename)
                print(f"✅ 保存 {filename}: {len(converted)} 样本")
            else:
                print(f"❌ 下载失败: {url}")
                return False
        
        print("✅ DocRED 下载完成")
        return True
        
    except Exception as e:
        print(f"❌ DocRED 下载失败: {e}")
        return False


def convert_docred_format(data):
    """转换 DocRED 格式"""
    converted = []
    
    for item in data:
        # 合并句子
        document = ' '.join([' '.join(sent) for sent in item['sents']])
        
        # 转换实体
        entities = []
        for i, entity_group in enumerate(item['vertexSet'][:10]):  # 限制实体数量
            mention = entity_group[0]  # 取第一个提及
            entities.append({
                "span": mention['pos'],
                "type": mention.get('type', 'MISC'),
                "text": mention['name']
            })
        
        # 转换关系
        relations = []
        for relation in item.get('labels', [])[:5]:  # 限制关系数量
            if relation['h'] < len(entities) and relation['t'] < len(entities):
                relations.append({
                    "head": relation['h'],
                    "tail": relation['t'],
                    "type": f"R{relation['r']}"
                })
        
        converted.append({
            "document": document[:1500],
            "query": f"What relations exist in this document?",
            "entities": entities,
            "relations": relations,
            "label": min(len(relations), 4),
            "metadata": {
                "source": "docred",
                "title": item.get('title', '')
            }
        })
    
    return converted


def save_json(data, filepath):
    """保存JSON文件"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"💾 已保存: {filepath} ({len(data)} 样本)")


def main():
    """主函数"""
    print("🎯 简化数据集下载器")
    print("=" * 40)
    
    print("可用数据集:")
    print("1. HotpotQA 风格数据集 (推荐)")
    print("2. DocRED 数据集")
    print("3. 两个都下载")
    
    choice = input("\n请选择 (1-3): ").strip()
    
    success = False
    
    if choice == "1":
        success = create_sample_hotpotqa()
    elif choice == "2":
        success = download_docred_simple()
    elif choice == "3":
        success1 = create_sample_hotpotqa()
        success2 = download_docred_simple()
        success = success1 or success2
    else:
        print("❌ 无效选择")
        return
    
    if success:
        print("\n🎉 数据集准备完成！")
        print("\n📋 下一步:")
        print("1. 检查 data/ 目录中的文件")
        print("2. 开始训练:")
        print("   python train/train.py --config config/optimized_config.yaml --mode train")
        print("3. 监控训练:")
        print("   tensorboard --logdir logs/")
    else:
        print("\n❌ 数据集下载失败")


if __name__ == "__main__":
    main()
