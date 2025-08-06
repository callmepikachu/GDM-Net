"""
下载官方 HotpotQA 数据集
按照官方文档 https://hotpotqa.github.io/ 的方式下载
"""

import os
import json
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
import gzip


def download_file(url, filename):
    """下载文件并显示进度"""
    print(f"📥 下载 {filename}...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    
    print(f"✅ 下载完成: {filename}")


def download_official_hotpotqa():
    """下载官方 HotpotQA 数据集"""
    print("🎯 下载官方 HotpotQA 数据集")
    print("=" * 50)
    
    # 官方数据集URL (根据官网)
    urls = {
        "train": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json",
        "dev_distractor": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
        "dev_fullwiki": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json"
    }
    
    # 创建数据目录
    data_dir = Path("data/official_hotpotqa")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded_files = {}
    
    try:
        for split, url in urls.items():
            filename = str(data_dir / f"{split}.json")  # 转换为字符串

            if os.path.exists(filename):
                print(f"⚠️  文件已存在: {filename}")
                choice = input("是否重新下载? (y/N): ").strip().lower()
                if choice != 'y':
                    print(f"跳过下载: {split}")
                    with open(filename, 'r', encoding='utf-8') as f:
                        downloaded_files[split] = json.load(f)
                    continue

            try:
                download_file(url, filename)
                
                # 读取下载的文件
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    downloaded_files[split] = data
                    print(f"📊 {split}: {len(data)} 样本")
                    
            except Exception as e:
                print(f"❌ 下载失败 {split}: {e}")
                continue
        
        return downloaded_files
        
    except Exception as e:
        print(f"❌ 下载过程出错: {e}")
        return {}


def convert_official_hotpotqa(data, max_samples=None):
    """转换官方 HotpotQA 数据为 GDM-Net 格式"""
    print("🔄 转换数据格式...")
    
    converted = []
    
    # 限制样本数量以避免内存问题
    if max_samples:
        data = data[:max_samples]
    
    for i, item in enumerate(tqdm(data, desc="转换数据")):
        try:
            # 合并所有上下文段落
            document_parts = []
            entities = []
            entity_positions = {}
            
            for title, sentences in item['context']:
                # 合并句子
                content = ' '.join(sentences)
                doc_part = f"{title}: {content}"
                
                # 计算位置
                start_pos = len(' '.join(document_parts))
                if document_parts:
                    start_pos += 1  # 空格
                
                document_parts.append(doc_part)
                
                # 添加标题作为实体
                entities.append({
                    "span": [start_pos, start_pos + len(title)],
                    "type": "TITLE",
                    "text": title
                })
                
                # 记录实体位置，用于关系抽取
                entity_positions[title] = len(entities) - 1
            
            combined_doc = ' '.join(document_parts)
            
            # 从支撑事实创建关系
            relations = []
            supporting_titles = set()
            
            for fact in item['supporting_facts']:
                title = fact[0]
                supporting_titles.add(title)
            
            # 创建支撑实体之间的关系
            supporting_entities = [entity_positions[title] for title in supporting_titles if title in entity_positions]
            for i in range(len(supporting_entities)):
                for j in range(i+1, len(supporting_entities)):
                    relations.append({
                        "head": supporting_entities[i],
                        "tail": supporting_entities[j],
                        "type": "SUPPORTS"
                    })
            
            # 答案类型分类
            answer = item['answer'].lower()
            if answer in ['yes', 'no']:
                label = 1 if answer == 'yes' else 0
            else:
                # 根据答案长度简单分类
                if len(answer.split()) == 1:
                    label = 2  # 单词答案
                elif len(answer.split()) <= 3:
                    label = 3  # 短答案
                else:
                    label = 4  # 长答案
            
            converted_item = {
                "document": combined_doc[:2000],  # 限制长度避免内存问题
                "query": item['question'],
                "entities": entities[:15],  # 限制实体数量
                "relations": relations[:10],  # 限制关系数量
                "label": label,
                "metadata": {
                    "source": "official_hotpotqa",
                    "answer": item['answer'],
                    "supporting_facts": item['supporting_facts'],
                    "level": item.get('level', 'unknown'),
                    "type": item.get('type', 'unknown'),
                    "id": item.get('_id', f'sample_{i}')
                }
            }
            
            converted.append(converted_item)
            
        except Exception as e:
            print(f"⚠️  跳过样本 {i}: {e}")
            continue
    
    print(f"✅ 转换完成: {len(converted)} 样本")
    return converted


def save_converted_data(converted_data, split_name):
    """保存转换后的数据"""
    output_file = f"data/hotpotqa_{split_name}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 已保存: {output_file} ({len(converted_data)} 样本)")
    return output_file


def main():
    """主函数"""
    print("🎯 官方 HotpotQA 数据集下载器")
    print("基于官方文档: https://hotpotqa.github.io/")
    print("=" * 60)
    
    # 下载官方数据
    downloaded_files = download_official_hotpotqa()
    
    if not downloaded_files:
        print("❌ 没有成功下载任何数据集")
        return
    
    print(f"\n📊 下载统计:")
    for split, data in downloaded_files.items():
        print(f"  {split}: {len(data)} 样本")
    
    # 转换数据格式
    print(f"\n🔄 开始转换数据格式...")
    
    converted_files = []
    
    # 转换训练集 (限制样本数量)
    if 'train' in downloaded_files:
        print(f"\n处理训练集...")
        train_converted = convert_official_hotpotqa(downloaded_files['train'], max_samples=5000)
        train_file = save_converted_data(train_converted, 'official_train')
        converted_files.append(train_file)
    
    # 转换验证集
    if 'dev_distractor' in downloaded_files:
        print(f"\n处理验证集...")
        val_converted = convert_official_hotpotqa(downloaded_files['dev_distractor'], max_samples=1000)
        val_file = save_converted_data(val_converted, 'official_val')
        converted_files.append(val_file)
    
    # 显示数据样本
    if converted_files:
        print(f"\n📋 数据样本预览:")
        with open(converted_files[0], 'r', encoding='utf-8') as f:
            sample_data = json.load(f)
            if sample_data:
                sample = sample_data[0]
                print(f"问题: {sample['query']}")
                print(f"文档: {sample['document'][:200]}...")
                print(f"实体数量: {len(sample['entities'])}")
                print(f"关系数量: {len(sample['relations'])}")
                print(f"答案: {sample['metadata']['answer']}")
    
    print(f"\n🎉 官方 HotpotQA 数据集准备完成!")
    print(f"\n📋 下一步:")
    print(f"1. 检查转换后的数据: data/hotpotqa_official_*.json")
    print(f"2. 更新配置文件使用新数据:")
    print(f"   修改 config/optimized_config.yaml 中的 train_path 和 val_path")
    print(f"3. 开始训练:")
    print(f"   python train/train.py --config config/optimized_config.yaml --mode train")


if __name__ == "__main__":
    main()
