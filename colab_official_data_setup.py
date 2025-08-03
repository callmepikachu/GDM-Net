"""
Google Colab 官方HotpotQA数据集配置脚本
专门为使用官方数据集优化的配置
"""

import json
import os
import yaml


def check_official_data():
    """检查官方数据文件"""
    print("🔍 检查官方HotpotQA数据集...")
    
    data_files = [
        'data/hotpotqa_official_train.json',
        'data/hotpotqa_official_val.json'
    ]
    
    all_exist = True
    total_samples = 0
    
    for file_path in data_files:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"✅ {file_path}: {len(data)} 样本")
            total_samples += len(data)
            
            # 显示真实的官方数据样本
            if data:
                sample = data[0]
                print(f"   文档: {sample['document'][:100]}...")
                print(f"   查询: {sample['query']}")
                print(f"   实体: {len(sample['entities'])}, 关系: {len(sample['relations'])}")
                print(f"   答案: {sample['metadata'].get('answer', 'N/A')}")
                print()
        else:
            print(f"❌ {file_path}: 文件不存在")
            all_exist = False
    
    if all_exist:
        print(f"🎉 官方数据集检查完成！总计 {total_samples} 样本")
        return True
    else:
        print("❌ 缺少官方数据文件")
        return False


def create_official_data_config():
    """创建针对官方数据优化的配置"""
    print("⚙️ 创建官方数据配置...")
    
    # 针对官方HotpotQA数据的优化配置
    config = {
        'seed': 42,
        'model': {
            'bert_model_name': 'bert-base-uncased',
            'hidden_size': 768,
            'num_entities': 8,  # 官方数据中的实体类型数
            'num_relations': 4,  # 官方数据中的关系类型数
            'num_classes': 5,    # 官方数据中的标签类别数
            'gnn_type': 'rgcn',
            'num_gnn_layers': 2,
            'num_reasoning_hops': 3,  # 适合多跳推理
            'fusion_method': 'gate',
            'learning_rate': 2e-5,
            'dropout_rate': 0.1
        },
        'data': {
            'train_path': 'data/hotpotqa_official_train.json',
            'val_path': 'data/hotpotqa_official_val.json', 
            'test_path': 'data/hotpotqa_official_val.json',
            'max_length': 512,  # 官方数据需要更长的序列
            'max_query_length': 64
        },
        'training': {
            'max_epochs': 10,
            'batch_size': 4,  # 官方数据更复杂，减少batch size
            'num_workers': 2,
            'accelerator': 'gpu',
            'devices': 1,
            'precision': 16,  # 混合精度节省内存
            'gradient_clip_val': 1.0,
            'accumulate_grad_batches': 2,  # 梯度累积补偿小batch size
            'val_check_interval': 0.25,  # 更频繁的验证
            'log_every_n_steps': 25,
            'checkpoint_dir': 'checkpoints',
            'early_stopping': True,
            'patience': 3
        },
        'logging': {
            'type': 'tensorboard',
            'save_dir': 'logs',
            'name': 'gdmnet-official-hotpotqa'
        }
    }
    
    # 保存配置
    with open('config/official_hotpotqa_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("✅ 官方数据配置: config/official_hotpotqa_config.yaml")
    return config


def analyze_official_data():
    """分析官方数据的特征"""
    print("📊 分析官方数据特征...")
    
    if not os.path.exists('data/hotpotqa_official_train.json'):
        print("❌ 训练数据不存在")
        return
    
    with open('data/hotpotqa_official_train.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 统计分析
    doc_lengths = []
    query_lengths = []
    entity_counts = []
    relation_counts = []
    label_counts = {}
    
    for item in data:
        doc_lengths.append(len(item['document']))
        query_lengths.append(len(item['query']))
        entity_counts.append(len(item['entities']))
        relation_counts.append(len(item['relations']))
        
        label = item['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"📋 数据统计:")
    print(f"   样本数量: {len(data)}")
    print(f"   平均文档长度: {sum(doc_lengths)/len(doc_lengths):.0f} 字符")
    print(f"   平均查询长度: {sum(query_lengths)/len(query_lengths):.0f} 字符")
    print(f"   平均实体数: {sum(entity_counts)/len(entity_counts):.1f}")
    print(f"   平均关系数: {sum(relation_counts)/len(relation_counts):.1f}")
    print(f"   标签分布: {label_counts}")
    
    # 推荐配置
    max_doc_len = max(doc_lengths)
    max_query_len = max(query_lengths)
    
    print(f"\n💡 推荐配置:")
    print(f"   max_length: {min(512, max_doc_len + 50)}")
    print(f"   max_query_length: {min(64, max_query_len + 10)}")
    print(f"   batch_size: 4 (GPU) 或 2 (内存不足时)")


def create_training_script():
    """创建官方数据训练脚本"""
    print("📝 创建训练脚本...")
    
    training_script = '''
# 官方HotpotQA数据集训练脚本

# 1. 检查环境
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 2. 设置环境变量
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 3. 开始训练
print("🚀 开始官方HotpotQA数据集训练...")
!python train/train.py --config config/official_hotpotqa_config.yaml --mode train

# 4. 启动TensorBoard监控
%load_ext tensorboard
%tensorboard --logdir logs/

print("✅ 训练启动完成！")
'''
    
    with open('train_official_hotpotqa.py', 'w') as f:
        f.write(training_script.strip())
    
    print("✅ 训练脚本: train_official_hotpotqa.py")


def show_colab_instructions():
    """显示Colab使用说明"""
    print("\n🎯 Google Colab 使用说明")
    print("=" * 50)
    
    print("📋 在Colab中执行以下步骤:")
    print()
    print("1. 🔧 运行环境设置:")
    print("   !python colab_official_data_setup.py")
    print()
    print("2. 🏋️ 开始训练:")
    print("   !python train/train.py --config config/official_hotpotqa_config.yaml --mode train")
    print()
    print("3. 📊 监控训练:")
    print("   %load_ext tensorboard")
    print("   %tensorboard --logdir logs/")
    print()
    print("4. 💾 保存结果到Drive:")
    print("   !cp -r checkpoints/* /content/drive/MyDrive/GDM-Net-Results/")
    print("   !cp -r logs/* /content/drive/MyDrive/GDM-Net-Results/")
    print()
    
    print("🎯 预期性能 (官方数据):")
    print("   - 训练时间: 2-4小时 (T4 GPU)")
    print("   - 内存使用: 6-8GB GPU")
    print("   - 预期准确率: 55-65%")
    print("   - 模型大小: ~550MB")


def main():
    """主函数"""
    print("🧠 GDM-Net 官方HotpotQA数据集 Colab配置")
    print("=" * 60)
    
    # 检查官方数据
    if not check_official_data():
        print("❌ 请确保官方数据文件存在")
        return False
    
    print()
    
    # 分析数据特征
    analyze_official_data()
    print()
    
    # 创建优化配置
    create_official_data_config()
    print()
    
    # 创建训练脚本
    create_training_script()
    print()
    
    # 显示使用说明
    show_colab_instructions()
    
    print("\n🎉 官方数据配置完成！")
    print("现在您可以在Colab上训练真正的学术级别GDM-Net模型了！")
    
    return True


if __name__ == "__main__":
    main()
