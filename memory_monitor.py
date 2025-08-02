"""
内存监控脚本
监控训练过程中的内存使用情况
"""

import psutil
import torch
import time
import matplotlib.pyplot as plt
from collections import deque
import threading


class MemoryMonitor:
    def __init__(self, max_points=100):
        self.max_points = max_points
        self.cpu_memory = deque(maxlen=max_points)
        self.gpu_memory = deque(maxlen=max_points)
        self.timestamps = deque(maxlen=max_points)
        self.monitoring = False
        
    def start_monitoring(self, interval=1.0):
        """开始监控"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("内存监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        print("内存监控已停止")
    
    def _monitor_loop(self, interval):
        """监控循环"""
        start_time = time.time()
        
        while self.monitoring:
            current_time = time.time() - start_time
            
            # CPU内存
            cpu_mem = psutil.virtual_memory()
            cpu_usage = cpu_mem.used / (1024**3)  # GB
            
            # GPU内存
            gpu_usage = 0
            if torch.cuda.is_available():
                gpu_usage = torch.cuda.memory_allocated() / (1024**3)  # GB
            
            self.cpu_memory.append(cpu_usage)
            self.gpu_memory.append(gpu_usage)
            self.timestamps.append(current_time)
            
            time.sleep(interval)
    
    def plot_memory_usage(self, save_path='memory_usage.png'):
        """绘制内存使用图"""
        if not self.timestamps:
            print("没有监控数据")
            return
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(list(self.timestamps), list(self.cpu_memory), 'b-', label='CPU Memory')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Memory Usage (GB)')
        plt.title('CPU Memory Usage')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        if torch.cuda.is_available():
            plt.plot(list(self.timestamps), list(self.gpu_memory), 'r-', label='GPU Memory')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Memory Usage (GB)')
            plt.title('GPU Memory Usage')
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'No GPU Available', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('GPU Memory Usage')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"内存使用图已保存到: {save_path}")
    
    def get_current_usage(self):
        """获取当前内存使用情况"""
        cpu_mem = psutil.virtual_memory()
        cpu_usage = cpu_mem.used / (1024**3)
        cpu_percent = cpu_mem.percent
        
        gpu_usage = 0
        gpu_percent = 0
        if torch.cuda.is_available():
            gpu_usage = torch.cuda.memory_allocated() / (1024**3)
            gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_percent = (gpu_usage / gpu_total) * 100 if gpu_total > 0 else 0
        
        return {
            'cpu_usage_gb': cpu_usage,
            'cpu_percent': cpu_percent,
            'gpu_usage_gb': gpu_usage,
            'gpu_percent': gpu_percent
        }


def optimize_memory_usage():
    """内存优化建议"""
    print("💡 内存优化建议:")
    print("1. 减少batch_size")
    print("2. 减少max_length")
    print("3. 使用gradient_checkpointing")
    print("4. 使用混合精度训练")
    print("5. 定期清理GPU缓存: torch.cuda.empty_cache()")


if __name__ == "__main__":
    monitor = MemoryMonitor()
    
    # 显示当前内存使用
    usage = monitor.get_current_usage()
    print(f"当前内存使用:")
    print(f"CPU: {usage['cpu_usage_gb']:.1f} GB ({usage['cpu_percent']:.1f}%)")
    print(f"GPU: {usage['gpu_usage_gb']:.1f} GB ({usage['gpu_percent']:.1f}%)")
    
    optimize_memory_usage()
