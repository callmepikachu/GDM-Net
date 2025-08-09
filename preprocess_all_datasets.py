#!/usr/bin/env python3
"""
批量预处理所有HotpotQA数据集
"""

import os
import subprocess
import sys

def run_preprocessing():
    """运行所有数据集的预处理"""
    
    # 数据文件列表
    datasets = [
        "data/hotpotqa_official_train.json",
        "data/hotpotqa_official_val.json"
    ]
    
    output_dir = "/root/autodl-tmp/hotpotqa-pretokenized"
    tokenizer_path = "models"
    
    print("🚀 开始批量预处理HotpotQA数据集")
    print(f"📁 输出目录: {output_dir}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    for dataset_path in datasets:
        if not os.path.exists(dataset_path):
            print(f"⚠️  跳过不存在的文件: {dataset_path}")
            continue
        
        print(f"\n📝 处理数据集: {dataset_path}")
        
        # 运行预处理脚本
        cmd = [
            sys.executable, "preprocess_tokenization.py",
            "--input", dataset_path,
            "--output_dir", output_dir,
            "--tokenizer", tokenizer_path,
            "--max_length", "512",
            "--max_query_length", "64"
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ 成功处理: {dataset_path}")
            if result.stdout:
                print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"❌ 处理失败: {dataset_path}")
            print(f"错误: {e}")
            if e.stderr:
                print(f"错误详情: {e.stderr}")
    
    print(f"\n🎉 批量预处理完成!")
    print(f"📁 预处理文件保存在: {output_dir}")
    
    # 列出生成的文件
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        print(f"\n📋 生成的文件:")
        for file in sorted(files):
            file_path = os.path.join(output_dir, file)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"   - {file} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    run_preprocessing()
