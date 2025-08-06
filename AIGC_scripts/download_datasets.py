"""
下载和转换推荐数据集的脚本
支持 HotpotQA, DocRED, 2WikiMultiHopQA, FEVER 等数据集
"""

import os
import json
import requests
import zipfile
import tarfile
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


class DatasetDownloader:
    def __init__(self, data_dir="datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def download_hotpotqa(self):
        """下载 HotpotQA 数据集"""
        print("📥 下载 HotpotQA 数据集...")
        
        try:
            # 使用 Hugging Face datasets 库
            dataset = load_dataset("hotpot_qa", "fullwiki")
            
            # 转换训练集
            train_data = self._convert_hotpotqa(dataset['train'])
            self._save_json(train_data[:8000], "data/hotpotqa_train.json")
            
            # 转换验证集
            val_data = self._convert_hotpotqa(dataset['validation'])
            self._save_json(val_data[:1000], "data/hotpotqa_val.json")
            
            print(f"✅ HotpotQA 下载完成: {len(train_data)} 训练样本, {len(val_data)} 验证样本")
            return True
            
        except Exception as e:
            print(f"❌ HotpotQA 下载失败: {e}")
            return False
    
    def _convert_hotpotqa(self, dataset) -> List[Dict]:
        """转换 HotpotQA 格式"""
        converted = []
        
        for item in tqdm(dataset, desc="转换 HotpotQA"):
            # 合并所有上下文文档
            documents = []
            for title, sentences in item['context']:
                doc_text = ' '.join(sentences)
                documents.append(f"{title}: {doc_text}")
            
            combined_doc = ' '.join(documents)
            
            # 提取实体（简化版本）
            entities = []
            entity_id = 0
            
            # 从支撑事实中提取实体
            for fact in item['supporting_facts']:
                title = fact[0]
                start_pos = combined_doc.find(title)
                if start_pos != -1:
                    entities.append({
                        "span": [start_pos, start_pos + len(title)],
                        "type": "ENTITY",
                        "text": title
                    })
                    entity_id += 1
            
            # 创建关系（基于支撑事实）
            relations = []
            if len(entities) >= 2:
                relations.append({
                    "head": 0,
                    "tail": 1,
                    "type": "SUPPORTS"
                })
            
            # 答案类型分类
            answer_type = 0  # yes/no
            if item['answer'].lower() in ['yes', 'no']:
                answer_type = 1 if item['answer'].lower() == 'yes' else 0
            else:
                answer_type = 2  # 其他答案
            
            converted.append({
                "document": combined_doc[:2000],  # 限制长度
                "query": item['question'],
                "entities": entities[:10],  # 限制实体数量
                "relations": relations,
                "label": answer_type,
                "metadata": {
                    "source": "hotpotqa",
                    "answer": item['answer'],
                    "supporting_facts": item['supporting_facts']
                }
            })
        
        return converted
    
    def download_docred(self):
        """下载 DocRED 数据集"""
        print("📥 下载 DocRED 数据集...")
        
        urls = {
            "train": "https://github.com/thunlp/DocRED/raw/master/data/train_annotated.json",
            "dev": "https://github.com/thunlp/DocRED/raw/master/data/dev.json"
        }
        
        try:
            for split, url in urls.items():
                print(f"下载 {split} 数据...")
                response = requests.get(url)
                response.raise_for_status()
                
                data = response.json()
                converted = self._convert_docred(data)
                
                if split == "train":
                    self._save_json(converted[:5000], "data/docred_train.json")
                else:
                    self._save_json(converted[:500], "data/docred_val.json")
            
            print("✅ DocRED 下载完成")
            return True
            
        except Exception as e:
            print(f"❌ DocRED 下载失败: {e}")
            return False
    
    def _convert_docred(self, dataset) -> List[Dict]:
        """转换 DocRED 格式"""
        converted = []
        
        for item in tqdm(dataset, desc="转换 DocRED"):
            # 合并句子
            document = ' '.join([' '.join(sent) for sent in item['sents']])
            
            # 转换实体
            entities = []
            for entity_group in item['vertexSet']:
                # 取第一个提及
                mention = entity_group[0]
                entities.append({
                    "span": mention['pos'],
                    "type": mention.get('type', 'MISC'),
                    "text": mention['name']
                })
            
            # 转换关系
            relations = []
            for relation in item.get('labels', []):
                relations.append({
                    "head": relation['h'],
                    "tail": relation['t'],
                    "type": f"R{relation['r']}"  # 关系ID转换为字符串
                })
            
            converted.append({
                "document": document[:2000],
                "query": f"What relations exist in this document about {item.get('title', 'the topic')}?",
                "entities": entities[:15],
                "relations": relations[:10],
                "label": min(len(relations), 4),  # 基于关系数量的简单分类
                "metadata": {
                    "source": "docred",
                    "title": item.get('title', '')
                }
            })
        
        return converted
    
    def download_2wikimultihopqa(self):
        """下载 2WikiMultiHopQA 数据集"""
        print("📥 下载 2WikiMultiHopQA 数据集...")
        
        try:
            dataset = load_dataset("2WikiMultihopQA", "2WikiMultihopQA")
            
            # 转换数据
            train_data = self._convert_2wiki(dataset['train'])
            val_data = self._convert_2wiki(dataset['validation'])
            
            self._save_json(train_data[:6000], "data/2wiki_train.json")
            self._save_json(val_data[:600], "data/2wiki_val.json")
            
            print(f"✅ 2WikiMultiHopQA 下载完成: {len(train_data)} 训练样本")
            return True
            
        except Exception as e:
            print(f"❌ 2WikiMultiHopQA 下载失败: {e}")
            return False
    
    def _convert_2wiki(self, dataset) -> List[Dict]:
        """转换 2WikiMultiHopQA 格式"""
        converted = []
        
        for item in tqdm(dataset, desc="转换 2WikiMultiHopQA"):
            # 合并上下文
            context_text = ""
            entities = []
            entity_id = 0
            
            for context in item['context']:
                title = context[0]
                content = ' '.join(context[1])
                context_text += f"{title}: {content} "
                
                # 添加标题作为实体
                start_pos = len(context_text) - len(content) - len(title) - 2
                entities.append({
                    "span": [start_pos, start_pos + len(title)],
                    "type": "ENTITY",
                    "text": title
                })
                entity_id += 1
            
            # 创建关系
            relations = []
            if len(entities) >= 2:
                relations.append({
                    "head": 0,
                    "tail": 1,
                    "type": "RELATED"
                })
            
            converted.append({
                "document": context_text[:2000],
                "query": item['question'],
                "entities": entities[:10],
                "relations": relations,
                "label": 0,  # 简化分类
                "metadata": {
                    "source": "2wikimultihopqa",
                    "answer": item['answer']
                }
            })
        
        return converted
    
    def download_fever(self):
        """下载 FEVER 数据集"""
        print("📥 下载 FEVER 数据集...")
        
        try:
            dataset = load_dataset("fever", "v1.0")
            
            # 只使用有标签的数据
            train_data = [item for item in dataset['train'] if item['label'] != 'NOT ENOUGH INFO']
            val_data = [item for item in dataset['validation'] if item['label'] != 'NOT ENOUGH INFO']
            
            converted_train = self._convert_fever(train_data)
            converted_val = self._convert_fever(val_data)
            
            self._save_json(converted_train[:8000], "data/fever_train.json")
            self._save_json(converted_val[:800], "data/fever_val.json")
            
            print(f"✅ FEVER 下载完成: {len(converted_train)} 训练样本")
            return True
            
        except Exception as e:
            print(f"❌ FEVER 下载失败: {e}")
            return False
    
    def _convert_fever(self, dataset) -> List[Dict]:
        """转换 FEVER 格式"""
        converted = []
        
        for item in tqdm(dataset, desc="转换 FEVER"):
            # 合并证据
            evidence_text = ""
            entities = []
            
            if item['evidence']:
                for evidence_set in item['evidence']:
                    for evidence in evidence_set:
                        if len(evidence) >= 3:
                            page_title = evidence[2]
                            evidence_text += f"{page_title}. "
                            
                            # 添加页面标题作为实体
                            start_pos = len(evidence_text) - len(page_title) - 2
                            entities.append({
                                "span": [start_pos, start_pos + len(page_title)],
                                "type": "EVIDENCE",
                                "text": page_title
                            })
            
            # 标签映射
            label_map = {"SUPPORTS": 1, "REFUTES": 0}
            label = label_map.get(item['label'], 2)
            
            converted.append({
                "document": evidence_text[:1500],
                "query": item['claim'],
                "entities": entities[:8],
                "relations": [],
                "label": label,
                "metadata": {
                    "source": "fever",
                    "original_label": item['label']
                }
            })
        
        return converted
    
    def _save_json(self, data: List[Dict], filepath: str):
        """保存JSON文件"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"💾 已保存: {filepath} ({len(data)} 样本)")


def main():
    """主函数"""
    print("🎯 GDM-Net 数据集下载器")
    print("=" * 50)
    
    downloader = DatasetDownloader()
    
    datasets = {
        "1": ("HotpotQA", downloader.download_hotpotqa),
        "2": ("DocRED", downloader.download_docred),
        "3": ("2WikiMultiHopQA", downloader.download_2wikimultihopqa),
        "4": ("FEVER", downloader.download_fever),
        "5": ("全部下载", lambda: all([
            downloader.download_hotpotqa(),
            downloader.download_docred(),
            downloader.download_2wikimultihopqa(),
            downloader.download_fever()
        ]))
    }
    
    print("请选择要下载的数据集:")
    for key, (name, _) in datasets.items():
        print(f"{key}. {name}")
    
    choice = input("\n请输入选择 (1-5): ").strip()
    
    if choice in datasets:
        name, download_func = datasets[choice]
        print(f"\n开始下载 {name}...")
        
        try:
            success = download_func()
            if success:
                print(f"\n🎉 {name} 下载完成！")
                print("\n📋 使用方法:")
                print("1. 检查 data/ 目录中的文件")
                print("2. 使用优化配置训练:")
                print("   python train/train.py --config config/optimized_config.yaml --mode train")
            else:
                print(f"\n❌ {name} 下载失败")
        except Exception as e:
            print(f"\n❌ 下载过程中出错: {e}")
    else:
        print("❌ 无效选择")


if __name__ == "__main__":
    main()
