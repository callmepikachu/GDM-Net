#!/usr/bin/env python3
"""
移动所有AIGC生成的脚本到AIGC_scripts文件夹
"""

import os
import shutil
from pathlib import Path

def move_aigc_scripts():
    """移动所有AIGC生成的脚本文件"""
    
    # 确保目标文件夹存在
    aigc_dir = Path("AIGC_scripts")
    aigc_dir.mkdir(exist_ok=True)
    
    # 需要移动的脚本文件列表
    script_files = [
        "auto_setup.py",
        "check_gpu.py", 
        "colab_official_data_setup.py",
        "colab_quick_setup.py",
        "colab_troubleshooting.py",
        "convert_dataset.py",
        "create_small_dataset.py",
        "dataset_recommendations.py",
        "debug_data.py",
        "download_datasets.py",
        "download_official_hotpotqa.py",
        "final_device_fix.py",
        "final_fix.py",
        "fix_ddp_unused_parameters.py",
        "fix_multi_gpu_issues.py",
        "fix_network_issues.py",
        "memory_monitor.py",
        "optimization_guide.py",
        "quick_fix_strategy.py",
        "run_example.py",
        "simple_dataset_download.py",
        "simple_test.py",
        "test_device_fix.py",
        "test_installation.py",
        "test_with_env.py",
        "thorough_fix.py"
    ]
    
    moved_count = 0
    not_found_count = 0
    
    print("🔄 开始移动AIGC生成的脚本文件...")
    print("=" * 50)
    
    for script_file in script_files:
        source_path = Path(script_file)
        target_path = aigc_dir / script_file
        
        if source_path.exists():
            try:
                shutil.move(str(source_path), str(target_path))
                print(f"✅ 已移动: {script_file}")
                moved_count += 1
            except Exception as e:
                print(f"❌ 移动失败 {script_file}: {e}")
        else:
            print(f"⚠️ 文件不存在: {script_file}")
            not_found_count += 1
    
    print("=" * 50)
    print(f"📊 移动统计:")
    print(f"  ✅ 成功移动: {moved_count} 个文件")
    print(f"  ⚠️ 未找到: {not_found_count} 个文件")
    print(f"  📁 目标文件夹: {aigc_dir.absolute()}")
    
    # 列出AIGC_scripts文件夹中的所有文件
    if aigc_dir.exists():
        files_in_aigc = list(aigc_dir.glob("*.py"))
        print(f"\n📋 AIGC_scripts文件夹中的文件 ({len(files_in_aigc)} 个):")
        for file in sorted(files_in_aigc):
            print(f"  📄 {file.name}")

if __name__ == "__main__":
    move_aigc_scripts()
