"""
运行 GDM-Net 示例的简单脚本
"""

import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from gdmnet import GDMNet


def run_inference_example():
    """运行推理示例"""
    print("🧠 GDM-Net 推理示例")
    print("=" * 40)
    
    # 初始化模型
    print("📦 初始化模型...")
    model = GDMNet(
        bert_model_name='bert-base-uncased',
        hidden_size=768,
        num_entities=10,
        num_relations=20,
        num_classes=5
    )
    model.eval()
    
    # 初始化分词器
    print("📝 初始化分词器...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # 示例文档和查询
    examples = [
        {
            "document": "Apple Inc. is a technology company founded by Steve Jobs. Tim Cook is the current CEO.",
            "query": "Who is the CEO of Apple?"
        },
        {
            "document": "Microsoft Corporation was founded by Bill Gates. Satya Nadella is the current CEO.",
            "query": "Who founded Microsoft?"
        },
        {
            "document": "Tesla Inc. was founded by Elon Musk. The company focuses on electric vehicles.",
            "query": "What does Tesla focus on?"
        }
    ]
    
    print("🔍 开始推理...")
    
    for i, example in enumerate(examples, 1):
        print(f"\n--- 示例 {i} ---")
        print(f"文档: {example['document']}")
        print(f"查询: {example['query']}")
        
        # 分词
        doc_encoding = tokenizer(
            example['document'],
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        query_encoding = tokenizer(
            example['query'],
            max_length=64,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 推理
        with torch.no_grad():
            outputs = model(
                input_ids=doc_encoding['input_ids'],
                attention_mask=doc_encoding['attention_mask'],
                query=query_encoding['input_ids'],
                return_intermediate=True
            )
        
        # 获取结果
        logits = outputs['logits']
        probabilities = F.softmax(logits, dim=-1)
        prediction = torch.argmax(logits, dim=-1)
        
        print(f"预测类别: {prediction.item()}")
        print(f"置信度: {probabilities.max().item():.3f}")
        print(f"提取的实体数量: {len(outputs['entities'][0])}")
        print(f"提取的关系数量: {len(outputs['relations'][0])}")
        
        # 显示提取的实体（如果有）
        if outputs['entities'][0]:
            print("提取的实体:")
            for j, entity in enumerate(outputs['entities'][0][:3]):  # 只显示前3个
                print(f"  - 实体 {j+1}: 位置 {entity['start']}-{entity['end']}, 类型 {entity['type']}, 置信度 {entity['confidence']:.3f}")
        
        # 显示提取的关系（如果有）
        if outputs['relations'][0]:
            print("提取的关系:")
            for j, relation in enumerate(outputs['relations'][0][:3]):  # 只显示前3个
                print(f"  - 关系 {j+1}: 头实体 {relation['head']}, 尾实体 {relation['tail']}, 类型 {relation['type']}, 置信度 {relation['confidence']:.3f}")


def check_training_progress():
    """检查训练进度"""
    import os
    import glob
    
    print("\n📊 检查训练进度...")
    
    # 检查检查点文件
    checkpoint_files = glob.glob("checkpoints/*.ckpt")
    if checkpoint_files:
        print(f"找到 {len(checkpoint_files)} 个检查点文件:")
        for ckpt in checkpoint_files:
            size = os.path.getsize(ckpt) / (1024 * 1024)  # MB
            print(f"  - {os.path.basename(ckpt)} ({size:.1f} MB)")
    else:
        print("暂无检查点文件（训练可能仍在进行中）")
    
    # 检查日志文件
    log_dirs = glob.glob("logs/gdmnet/version_*")
    if log_dirs:
        print(f"找到 {len(log_dirs)} 个日志目录:")
        for log_dir in log_dirs:
            print(f"  - {log_dir}")
        print("\n💡 提示: 运行 'tensorboard --logdir logs/' 查看训练曲线")
    else:
        print("暂无日志文件")


def main():
    """主函数"""
    try:
        # 运行推理示例
        run_inference_example()
        
        # 检查训练进度
        check_training_progress()
        
        print("\n🎉 示例运行完成！")
        print("\n📋 后续操作:")
        print("1. 等待训练完成")
        print("2. 运行 'tensorboard --logdir logs/' 查看训练曲线")
        print("3. 使用训练好的模型进行推理")
        
    except Exception as e:
        print(f"❌ 运行示例时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
