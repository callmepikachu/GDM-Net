"""
数据集推荐和比较脚本
帮助选择最适合的数据集
"""

import pandas as pd
from tabulate import tabulate


def show_dataset_comparison():
    """显示数据集比较表"""
    
    datasets = [
        {
            "数据集": "HotpotQA",
            "任务类型": "多文档问答",
            "样本数量": "90K",
            "难度": "⭐⭐⭐⭐",
            "匹配度": "⭐⭐⭐⭐⭐",
            "特点": "多跳推理,支撑事实",
            "下载大小": "~200MB",
            "推荐指数": "⭐⭐⭐⭐⭐"
        },
        {
            "数据集": "DocRED",
            "任务类型": "文档级关系抽取",
            "样本数量": "5K",
            "难度": "⭐⭐⭐⭐⭐",
            "匹配度": "⭐⭐⭐⭐⭐",
            "特点": "丰富实体关系,跨句推理",
            "下载大小": "~50MB",
            "推荐指数": "⭐⭐⭐⭐⭐"
        },
        {
            "数据集": "2WikiMultiHopQA",
            "任务类型": "多跳问答",
            "样本数量": "167K",
            "难度": "⭐⭐⭐⭐",
            "匹配度": "⭐⭐⭐⭐",
            "特点": "推理路径,Wikipedia",
            "下载大小": "~300MB",
            "推荐指数": "⭐⭐⭐⭐"
        },
        {
            "数据集": "FEVER",
            "任务类型": "事实验证",
            "样本数量": "185K",
            "难度": "⭐⭐⭐",
            "匹配度": "⭐⭐⭐",
            "特点": "证据检索,事实验证",
            "下载大小": "~400MB",
            "推荐指数": "⭐⭐⭐"
        },
        {
            "数据集": "MultiRC",
            "任务类型": "多句阅读理解",
            "样本数量": "6K",
            "难度": "⭐⭐⭐",
            "匹配度": "⭐⭐⭐",
            "特点": "多句推理,选择题",
            "下载大小": "~20MB",
            "推荐指数": "⭐⭐⭐"
        }
    ]
    
    df = pd.DataFrame(datasets)
    print("📊 数据集比较表")
    print("=" * 80)
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))


def show_detailed_recommendations():
    """显示详细推荐"""
    
    print("\n🎯 详细推荐分析")
    print("=" * 50)
    
    recommendations = [
        {
            "rank": 1,
            "dataset": "HotpotQA",
            "reason": "最适合GDM-Net的多文档推理架构",
            "pros": [
                "需要跨多个文档进行推理",
                "有明确的支撑事实标注",
                "问题复杂度适中",
                "数据量充足"
            ],
            "cons": [
                "需要较多计算资源",
                "数据预处理相对复杂"
            ],
            "best_for": "验证GDM-Net的核心能力"
        },
        {
            "rank": 2,
            "dataset": "DocRED",
            "reason": "专门为文档级关系抽取设计",
            "pros": [
                "丰富的实体和关系标注",
                "需要跨句子推理",
                "数据质量很高",
                "适合图神经网络"
            ],
            "cons": [
                "数据量相对较小",
                "任务相对专一"
            ],
            "best_for": "测试图记忆和关系推理能力"
        },
        {
            "rank": 3,
            "dataset": "2WikiMultiHopQA",
            "reason": "专门设计的多跳推理数据集",
            "pros": [
                "有推理路径标注",
                "基于Wikipedia真实数据",
                "数据量大"
            ],
            "cons": [
                "可能过于复杂",
                "需要大量计算资源"
            ],
            "best_for": "挑战模型的推理深度"
        }
    ]
    
    for rec in recommendations:
        print(f"\n🏆 第{rec['rank']}推荐: {rec['dataset']}")
        print(f"📝 推荐理由: {rec['reason']}")
        print("✅ 优点:")
        for pro in rec['pros']:
            print(f"   • {pro}")
        print("⚠️  缺点:")
        for con in rec['cons']:
            print(f"   • {con}")
        print(f"🎯 最适合: {rec['best_for']}")


def show_quick_start_guide():
    """显示快速开始指南"""
    
    print("\n🚀 快速开始指南")
    print("=" * 50)
    
    guides = {
        "初学者": {
            "推荐": "HotpotQA (小规模)",
            "命令": [
                "python download_datasets.py",
                "选择 1 (HotpotQA)",
                "python train/train.py --config config/optimized_config.yaml --mode train"
            ],
            "预期时间": "2-3小时训练",
            "内存需求": "4-6GB"
        },
        "研究者": {
            "推荐": "DocRED + HotpotQA",
            "命令": [
                "python download_datasets.py",
                "选择 5 (全部下载)",
                "分别训练不同数据集"
            ],
            "预期时间": "6-8小时训练",
            "内存需求": "6-8GB"
        },
        "工程师": {
            "推荐": "根据具体任务选择",
            "命令": [
                "先用小数据集验证",
                "再用大数据集训练",
                "使用GPU加速"
            ],
            "预期时间": "1-2天完整实验",
            "内存需求": "8GB+"
        }
    }
    
    for user_type, guide in guides.items():
        print(f"\n👤 {user_type}:")
        print(f"   📋 推荐: {guide['推荐']}")
        print(f"   ⏱️  预期时间: {guide['预期时间']}")
        print(f"   💾 内存需求: {guide['内存需求']}")
        print("   🔧 操作步骤:")
        for i, cmd in enumerate(guide['命令'], 1):
            print(f"      {i}. {cmd}")


def show_performance_expectations():
    """显示性能预期"""
    
    print("\n📈 性能预期")
    print("=" * 50)
    
    expectations = [
        {
            "数据集": "HotpotQA",
            "指标": "EM/F1",
            "基线": "45-55%",
            "GDM-Net预期": "55-65%",
            "训练时间": "2-4小时(CPU)"
        },
        {
            "数据集": "DocRED",
            "指标": "F1",
            "基线": "50-60%",
            "GDM-Net预期": "60-70%",
            "训练时间": "1-2小时(CPU)"
        },
        {
            "数据集": "FEVER",
            "指标": "准确率",
            "基线": "85-90%",
            "GDM-Net预期": "88-93%",
            "训练时间": "3-5小时(CPU)"
        }
    ]
    
    df = pd.DataFrame(expectations)
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))


def main():
    """主函数"""
    print("🎯 GDM-Net 数据集推荐系统")
    print("=" * 60)
    
    # 显示比较表
    show_dataset_comparison()
    
    # 显示详细推荐
    show_detailed_recommendations()
    
    # 显示快速开始指南
    show_quick_start_guide()
    
    # 显示性能预期
    show_performance_expectations()
    
    print("\n💡 最终建议:")
    print("1. 🥇 首选: HotpotQA - 最匹配GDM-Net设计理念")
    print("2. 🥈 次选: DocRED - 专业的关系抽取数据集")
    print("3. 🥉 备选: 2WikiMultiHopQA - 挑战推理能力")
    
    print("\n🚀 立即开始:")
    print("python download_datasets.py")


if __name__ == "__main__":
    try:
        main()
    except ImportError:
        print("❌ 缺少依赖包，正在安装...")
        import subprocess
        subprocess.run(["pip", "install", "tabulate", "pandas"])
        main()
