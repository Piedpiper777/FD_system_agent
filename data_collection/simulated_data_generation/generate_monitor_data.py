"""
模拟监测数据持续生成脚本
持续生成模拟的设备监测数据并写入 monitor_data.csv
- 数据格式：timestamp, col0, col1, col2, col3 (X/Y/Z轴加速度 + 温度)
- 采集频率：10Hz（每0.1秒一次）
- 文件路径：edge/data/monitor/monitor_data.csv
"""

import csv
import time
import numpy as np
from datetime import datetime
from pathlib import Path

# 获取项目根目录和目标CSV文件路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EDGE_DIR = PROJECT_ROOT / 'edge'
CSV_FILE = EDGE_DIR / 'data' / 'monitor' / 'monitor_data.csv'

# 采集频率：10Hz = 每0.1秒一次
SAMPLE_INTERVAL = 0.1  # 秒

# 模拟数据参数
NOISE_LEVEL = 0.05  # 噪声水平
ANOMALY_PROB = 0.02  # 异常概率


def generate_normal_data(t):
    """生成正常工况下的模拟数据
    
    Args:
        t: 时间因子（用于生成周期性模式）
    
    Returns:
        tuple: (x, y, z, temp) 四维数据
    """
    # 基础信号 + 周期性分量 + 噪声
    x = 0.5 + 0.1 * np.sin(2 * np.pi * 0.5 * t) + np.random.normal(0, NOISE_LEVEL)
    y = 0.3 + 0.1 * np.cos(2 * np.pi * 0.3 * t) + np.random.normal(0, NOISE_LEVEL)
    z = 9.8 + 0.05 * np.sin(2 * np.pi * 0.2 * t) + np.random.normal(0, NOISE_LEVEL)
    temp = 25.0 + 2.0 * np.sin(2 * np.pi * 0.01 * t) + np.random.normal(0, 0.5)
    
    return x, y, z, temp


def generate_anomaly_data(t):
    """生成异常工况下的模拟数据（模拟振动异常）
    
    Args:
        t: 时间因子
    
    Returns:
        tuple: (x, y, z, temp) 四维数据
    """
    # 异常：振动幅度增大 + 高频分量
    x = 0.5 + 0.3 * np.sin(2 * np.pi * 0.5 * t) + 0.2 * np.sin(2 * np.pi * 5 * t) + np.random.normal(0, NOISE_LEVEL * 2)
    y = 0.3 + 0.3 * np.cos(2 * np.pi * 0.3 * t) + 0.2 * np.cos(2 * np.pi * 4 * t) + np.random.normal(0, NOISE_LEVEL * 2)
    z = 9.8 + 0.15 * np.sin(2 * np.pi * 0.2 * t) + np.random.normal(0, NOISE_LEVEL * 2)
    temp = 25.0 + 5.0 * np.sin(2 * np.pi * 0.01 * t) + np.random.normal(0, 1.0)  # 温度异常升高
    
    return x, y, z, temp


def main():
    """主函数：持续生成监测数据"""
    # 确保目录存在
    CSV_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # 检查并处理CSV文件表头
    file_exists = CSV_FILE.exists()
    
    # 打开（或创建）CSV 文件，以追加模式写入数据
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # 如果是新文件，写入表头
        if not file_exists:
            writer.writerow(["timestamp", "col0", "col1", "col2", "col3"])
            print(f"[模拟采集] 已创建新文件并写入表头: {CSV_FILE}")
        else:
            print(f"[模拟采集] 将追加数据到现有文件: {CSV_FILE}")

        print(f"[模拟采集] 开始生成模拟监测数据")
        print(f"[模拟采集] 采集频率: {1/SAMPLE_INTERVAL}Hz (每{SAMPLE_INTERVAL}秒一次)")
        print(f"[模拟采集] 数据格式: timestamp, X轴, Y轴, Z轴, 温度")
        print(f"[模拟采集] 按 Ctrl+C 结束")
        print("=" * 80)

        last_sample_time = time.time()
        record_count = 0
        t = 0  # 时间因子

        try:
            while True:
                current_time = time.time()
                
                # 控制采集频率：每0.1秒采集一次
                if current_time - last_sample_time < SAMPLE_INTERVAL:
                    time.sleep(0.01)  # 短暂休眠，避免CPU占用过高
                    continue
                
                last_sample_time = current_time
                t += SAMPLE_INTERVAL
                
                # 生成时间戳（ISO 8601格式）
                timestamp = datetime.now().isoformat()
                
                # 随机决定是否生成异常数据
                if np.random.random() < ANOMALY_PROB:
                    x, y, z, temp = generate_anomaly_data(t)
                    status = "异常"
                else:
                    x, y, z, temp = generate_normal_data(t)
                    status = "正常"
                
                # 写入CSV
                writer.writerow([timestamp, f"{x:.6f}", f"{y:.6f}", f"{z:.6f}", f"{temp:.2f}"])
                f.flush()  # 立即刷新到磁盘
                
                record_count += 1
                
                # 每50条记录打印一次进度
                if record_count % 50 == 0:
                    print(f"[模拟采集] 已生成 {record_count} 条记录 | "
                          f"最新: X={x:.3f}, Y={y:.3f}, Z={z:.3f}, T={temp:.1f}°C | "
                          f"状态: {status}")
        
        except KeyboardInterrupt:
            print(f"\n[模拟采集] 用户中断，共生成 {record_count} 条记录")
            print(f"[模拟采集] 数据已保存到: {CSV_FILE}")
        
        except Exception as e:
            print(f"\n[错误] 数据生成过程中出现异常: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
