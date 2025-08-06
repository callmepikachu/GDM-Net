"""
修复Hugging Face网络连接问题
解决模型下载失败的问题
"""

import os
import requests
import time
from transformers import BertModel, BertTokenizer


def test_network_connection():
    """测试网络连接"""
    print("🌐 测试网络连接...")
    
    urls_to_test = [
        "https://huggingface.co",
        "https://hf-mirror.com",
        "https://www.google.com"
    ]
    
    for url in urls_to_test:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print(f"✅ {url} - 连接成功")
            else:
                print(f"⚠️ {url} - 状态码: {response.status_code}")
        except Exception as e:
            print(f"❌ {url} - 连接失败: {e}")


def setup_mirror_sources():
    """设置镜像源"""
    print("🔧 设置镜像源...")
    
    # Hugging Face镜像源
    mirror_sources = [
        "https://hf-mirror.com",
        "https://huggingface.co"
    ]
    
    for mirror in mirror_sources:
        print(f"🔄 尝试镜像源: {mirror}")
        os.environ['HF_ENDPOINT'] = mirror
        
        try:
            # 测试下载一个小模型
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            print(f"✅ 镜像源 {mirror} 可用")
            return mirror
        except Exception as e:
            print(f"❌ 镜像源 {mirror} 失败: {e}")
            continue
    
    print("❌ 所有镜像源都不可用")
    return None


def download_bert_model():
    """下载BERT模型"""
    print("📥 下载BERT模型...")
    
    try:
        # 下载tokenizer
        print("📝 下载tokenizer...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        print("✅ tokenizer下载成功")
        
        # 下载模型
        print("🧠 下载BERT模型...")
        model = BertModel.from_pretrained('bert-base-uncased')
        print("✅ BERT模型下载成功")
        
        return True
        
    except Exception as e:
        print(f"❌ BERT模型下载失败: {e}")
        return False


def clear_cache_and_retry():
    """清理缓存并重试"""
    print("🧹 清理Hugging Face缓存...")
    
    # 清理缓存目录
    cache_dirs = [
        "~/.cache/huggingface",
        "/root/.cache/huggingface",
        "/tmp/huggingface"
    ]
    
    for cache_dir in cache_dirs:
        expanded_dir = os.path.expanduser(cache_dir)
        if os.path.exists(expanded_dir):
            try:
                import shutil
                shutil.rmtree(expanded_dir)
                print(f"✅ 清理缓存: {expanded_dir}")
            except Exception as e:
                print(f"⚠️ 清理缓存失败: {expanded_dir} - {e}")
    
    print("🔄 重新尝试下载...")
    return download_bert_model()


def fix_ssl_issues():
    """修复SSL证书问题"""
    print("🔒 修复SSL证书问题...")
    
    import ssl
    import urllib3
    
    # 禁用SSL警告
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # 设置SSL上下文
    ssl._create_default_https_context = ssl._create_unverified_context
    
    print("✅ SSL设置已修复")


def comprehensive_fix():
    """综合修复方案"""
    print("🛠️ Hugging Face网络问题综合修复")
    print("=" * 50)
    
    # 1. 测试网络连接
    test_network_connection()
    print()
    
    # 2. 修复SSL问题
    fix_ssl_issues()
    print()
    
    # 3. 设置镜像源
    working_mirror = setup_mirror_sources()
    print()
    
    # 4. 尝试下载模型
    if working_mirror:
        success = download_bert_model()
        if success:
            print("🎉 BERT模型下载成功！")
            return True
    
    # 5. 清理缓存重试
    print("🔄 尝试清理缓存...")
    success = clear_cache_and_retry()
    if success:
        print("🎉 清理缓存后下载成功！")
        return True
    
    # 6. 最终建议
    print("❌ 所有自动修复方案都失败了")
    print("💡 手动解决建议:")
    print("1. 重启Colab运行时")
    print("2. 检查网络连接")
    print("3. 稍后重试")
    print("4. 使用VPN或代理")
    print("5. 联系Colab支持")
    
    return False


def quick_fix():
    """快速修复方案"""
    print("⚡ 快速修复Hugging Face连接问题")
    
    # 设置环境变量
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'
    
    # 修复SSL
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    
    print("✅ 快速修复完成，请重新运行模型创建代码")


if __name__ == "__main__":
    # 运行综合修复
    success = comprehensive_fix()
    
    if not success:
        print("\n⚡ 尝试快速修复...")
        quick_fix()
