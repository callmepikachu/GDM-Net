#!/usr/bin/env python3
"""
预处理脚本：将HotpotQA数据集进行BERT tokenization并保存
消除训练时的CPU瓶颈
"""

import os
import json
import torch
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import Dict, Any, List
import argparse


def preprocess_dataset(
    input_path: str,
    output_dir: str,
    tokenizer_name: str = "models",
    max_length: int = 512,
    max_query_length: int = 64,
    batch_size: int = 64,
    use_gpu: bool = True
):
    """
    预处理数据集，进行tokenization并保存

    Args:
        input_path: 原始JSON数据文件路径
        output_dir: 输出目录
        tokenizer_name: tokenizer名称或路径
        max_length: 文档最大长度
        max_query_length: 查询最大长度
        batch_size: 批处理大小（GPU加速用）
        use_gpu: 是否使用GPU加速
    """
    
    print(f"🚀 开始预处理数据集: {input_path}")
    print(f"📁 输出目录: {output_dir}")

    # 检查GPU可用性
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载tokenizer
    print(f"📝 加载tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # 加载原始数据
    print(f"📖 加载原始数据...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📊 数据集大小: {len(data)} 样本")
    
    # 🚀 GPU加速的批量预处理
    tokenized_data = []

    # 准备批量数据
    queries = []
    documents = []
    original_samples = []

    for sample in data:
        query = sample.get('question', '')
        if isinstance(sample.get('context'), list):
            document = ' '.join(sample.get('context', []))
        else:
            document = sample.get('document', '')

        queries.append(query)
        documents.append(document)
        original_samples.append(sample)

    # 批量处理
    num_batches = (len(data) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="GPU Tokenizing"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(data))

        batch_queries = queries[start_idx:end_idx]
        batch_documents = documents[start_idx:end_idx]
        batch_originals = original_samples[start_idx:end_idx]

        try:
            # 🚀 批量tokenize查询（GPU加速）
            query_tokens = tokenizer(
                batch_queries,
                max_length=max_query_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            # 🚀 批量tokenize文档（GPU加速）
            doc_tokens = tokenizer(
                batch_documents,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            # 如果使用GPU，将张量移到GPU
            if use_gpu and torch.cuda.is_available():
                query_tokens = {k: v.to(device) for k, v in query_tokens.items()}
                doc_tokens = {k: v.to(device) for k, v in doc_tokens.items()}

            # 处理批次中的每个样本
            for i in range(len(batch_queries)):
                tokenized_sample = {
                    'query_input_ids': query_tokens['input_ids'][i].cpu(),
                    'query_attention_mask': query_tokens['attention_mask'][i].cpu(),
                    'doc_input_ids': doc_tokens['input_ids'][i].cpu(),
                    'doc_attention_mask': doc_tokens['attention_mask'][i].cpu(),
                    'original_sample': batch_originals[i]
                }
                tokenized_data.append(tokenized_sample)

        except Exception as e:
            print(f"⚠️  处理批次 {batch_idx} 时出错: {e}")
            # 回退到单个样本处理
            for i, (query, document, original) in enumerate(zip(batch_queries, batch_documents, batch_originals)):
                try:
                    query_tokens = tokenizer(query, max_length=max_query_length, padding='max_length', truncation=True, return_tensors='pt')
                    doc_tokens = tokenizer(document, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')

                    tokenized_sample = {
                        'query_input_ids': query_tokens['input_ids'].squeeze(0),
                        'query_attention_mask': query_tokens['attention_mask'].squeeze(0),
                        'doc_input_ids': doc_tokens['input_ids'].squeeze(0),
                        'doc_attention_mask': doc_tokens['attention_mask'].squeeze(0),
                        'original_sample': original
                    }
                    tokenized_data.append(tokenized_sample)
                except Exception as e2:
                    print(f"⚠️  跳过样本 {start_idx + i}: {e2}")
                    continue
    
    # 保存预处理结果
    output_file = os.path.join(output_dir, f"tokenized_{os.path.basename(input_path).replace('.json', '.pkl')}")
    
    print(f"💾 保存预处理结果到: {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump(tokenized_data, f)
    
    # 保存元数据
    metadata = {
        'num_samples': len(tokenized_data),
        'tokenizer_name': tokenizer_name,
        'max_length': max_length,
        'max_query_length': max_query_length,
        'original_file': input_path
    }
    
    metadata_file = os.path.join(output_dir, f"metadata_{os.path.basename(input_path).replace('.json', '.json')}")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ 预处理完成!")
    print(f"   - 处理样本数: {len(tokenized_data)}")
    print(f"   - 输出文件: {output_file}")
    print(f"   - 元数据文件: {metadata_file}")


def main():
    parser = argparse.ArgumentParser(description="预处理HotpotQA数据集进行tokenization")
    parser.add_argument("--input", required=True, help="输入JSON文件路径")
    parser.add_argument("--output_dir", default="/root/autodl-tmp/hotpotqa-pretokenized", 
                       help="输出目录")
    parser.add_argument("--tokenizer", default="models", help="Tokenizer名称或路径")
    parser.add_argument("--max_length", type=int, default=512, help="文档最大长度")
    parser.add_argument("--max_query_length", type=int, default=64, help="查询最大长度")
    parser.add_argument("--batch_size", type=int, default=64, help="批处理大小（GPU加速）")
    parser.add_argument("--no_gpu", action="store_true", help="禁用GPU加速")
    
    args = parser.parse_args()
    
    preprocess_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer,
        max_length=args.max_length,
        max_query_length=args.max_query_length,
        batch_size=args.batch_size,
        use_gpu=not args.no_gpu
    )


if __name__ == "__main__":
    main()
