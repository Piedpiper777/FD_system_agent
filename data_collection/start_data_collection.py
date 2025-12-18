"""
数据采集统一启动脚本
负责启动设备监测数据采集，优先使用硬件采集，失败时回退到模拟数据生成

使用方法：
    python data_collection/start_data_collection.py

数据输出：
    edge/data/monitor/monitor_data.csv

采集策略：
    1. 优先启动硬件采集：data_collection/sensor_demo/log_adxl345.py (STM32 + ADXL345)
    2. 若硬件不可用，启动模拟数据生成：data_collection/simulated_data_generation/generate_monitor_data.py
    3. 用户可按 Ctrl+C 停止采集
"""

import subprocess
import sys
import time
from pathlib import Path

# 项目路径
PROJECT_ROOT = Path(__file__).resolve().parent
HARDWARE_SCRIPT = PROJECT_ROOT / 'sensor_demo' / 'log_adxl345.py'
SIMULATED_SCRIPT = PROJECT_ROOT / 'simulated_data_generation' / 'generate_monitor_data.py'


def try_hardware_collection():
    """尝试启动硬件采集脚本
    
    Returns:
        subprocess.Popen or None: 成功返回进程对象，失败返回 None
    """
    if not HARDWARE_SCRIPT.exists():
        print(f"[检测] 未找到硬件采集脚本: {HARDWARE_SCRIPT}")
        return None
    
    print("=" * 80)
    print("[数据采集] 检测到硬件采集脚本，尝试启动 STM32 + ADXL345 采集...")
    print(f"[数据采集] 脚本路径: {HARDWARE_SCRIPT}")
    print("=" * 80)
    
    try:
        # 启动硬件采集进程
        process = subprocess.Popen(
            [sys.executable, str(HARDWARE_SCRIPT)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # 等待一小段时间检查进程是否成功启动
        time.sleep(1.0)
        
        # 检查进程状态
        if process.poll() is not None:
            # 进程已退出，读取错误信息
            stdout, stderr = process.communicate()
            print(f"[失败] 硬件采集脚本启动失败")
            if stdout:
                print(f"标准输出:\n{stdout}")
            if stderr:
                print(f"错误输出:\n{stderr}")
            return None
        
        print(f"[成功] 硬件采集脚本已启动 (PID: {process.pid})")
        print("[提示] 按 Ctrl+C 停止采集")
        return process
        
    except Exception as e:
        print(f"[错误] 启动硬件采集脚本时出现异常: {e}")
        return None


def start_simulated_collection():
    """启动模拟数据生成脚本
    
    Returns:
        subprocess.Popen: 进程对象
    """
    if not SIMULATED_SCRIPT.exists():
        print(f"[错误] 未找到模拟数据生成脚本: {SIMULATED_SCRIPT}")
        sys.exit(1)
    
    print("=" * 80)
    print("[数据采集] 启动模拟数据生成...")
    print(f"[数据采集] 脚本路径: {SIMULATED_SCRIPT}")
    print("=" * 80)
    
    try:
        # 启动模拟数据生成进程
        process = subprocess.Popen(
            [sys.executable, str(SIMULATED_SCRIPT)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        print(f"[成功] 模拟数据生成已启动 (PID: {process.pid})")
        print("[提示] 按 Ctrl+C 停止采集")
        return process
        
    except Exception as e:
        print(f"[错误] 启动模拟数据生成时出现异常: {e}")
        sys.exit(1)


def monitor_process(process):
    """监控子进程并实时输出日志
    
    Args:
        process: subprocess.Popen 对象
    """
    try:
        # 实时读取并输出子进程的标准输出
        for line in iter(process.stdout.readline, ''):
            if line:
                print(line, end='')
        
        # 等待进程结束
        process.wait()
        
    except KeyboardInterrupt:
        print("\n[停止] 用户中断，正在停止数据采集...")
        process.terminate()
        try:
            process.wait(timeout=5)
            print("[停止] 数据采集已停止")
        except subprocess.TimeoutExpired:
            print("[停止] 进程未响应，强制终止...")
            process.kill()
            process.wait()
            print("[停止] 数据采集已强制停止")
    
    except Exception as e:
        print(f"[错误] 监控进程时出现异常: {e}")
        process.terminate()


def main():
    """主函数"""
    print("=" * 80)
    print("数据采集统一启动脚本")
    print("=" * 80)
    print("[信息] 目标输出文件: edge/data/monitor/monitor_data.csv")
    print("[信息] 采集频率: 10Hz (每0.1秒一次)")
    print("[信息] 数据格式: timestamp, col0(X), col1(Y), col2(Z), col3(温度)")
    print()
    
    # 优先尝试硬件采集
    process = try_hardware_collection()
    
    # 硬件采集失败，回退到模拟数据生成
    if process is None:
        print()
        print("[回退] 硬件采集不可用，将使用模拟数据生成")
        print("[提示] 如需使用硬件采集，请确保：")
        print("       1. STM32 设备已连接（CH340串口）")
        print("       2. 串口驱动已安装")
        print("       3. 串口未被其他程序占用")
        print()
        
        time.sleep(1)
        process = start_simulated_collection()
    
    # 监控进程运行
    monitor_process(process)


if __name__ == "__main__":
    main()
