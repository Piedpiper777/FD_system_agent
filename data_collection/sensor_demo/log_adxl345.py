"""
ADXL345传感器数据采集脚本
从串口读取ADXL345加速度计数据并保存到CSV文件
- 数据格式：timestamp, x, y, z
- 采集频率：10Hz（每0.1秒一次）
- 文件路径：edge/data/monitor/monitor_data.csv
"""

import serial
import serial.tools.list_ports
import csv
import time
from datetime import datetime
import os
from pathlib import Path

# ==== 配置参数 ====
# SERIAL_PORT = "COM4"      # Windows 例子：COM3 / COM4...
BAUDRATE = 115200         # 要和 STM32 里设置的一致
# 如果是 Linux：比如 "/dev/ttyUSB0" 或 "/dev/ttyACM0"

# 获取项目根目录（edge目录）
EDGE_DIR = Path(__file__).resolve().parent.parent.parent
CSV_FILE = EDGE_DIR / 'edge' / 'data' / 'monitor' / 'monitor_data.csv'

# 采集频率：10Hz = 每0.1秒一次
SAMPLE_INTERVAL = 0.1  # 秒

def find_ch340_port():
    """自动查找CH340串口"""
    for port in serial.tools.list_ports.comports():
        if 'CH340' in port.description.upper() or 'USB-SERIAL CH340' in port.description.upper():
            return port.device
    return None

def main():
    # 确保目录存在
    CSV_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # 自动查找CH340串口
    SERIAL_PORT = find_ch340_port()
    if SERIAL_PORT is None:
        print("[错误] 未找到CH340串口设备")
        print("[提示] 请确保CH340设备已连接")
        # 列出所有可用串口供参考
        print("=" * 80)
        print("[ADXL345采集] 可用串口列表:")
        for p in serial.tools.list_ports.comports():
            print(f"  {p.device}: {p.description}")
        print("=" * 80)
        return
    
    print(f"[ADXL345采集] 自动检测到CH340串口: {SERIAL_PORT}")
    
    # 列出可用串口
    print("=" * 80)
    print("[ADXL345采集] 可用串口列表:")
    for p in serial.tools.list_ports.comports():
        print(f"  {p.device}: {p.description}")
    print("=" * 80)
    
    try:
        # 打开串口
        ser = serial.Serial(
            SERIAL_PORT,
            BAUDRATE,
            timeout=1,
            dsrdtr=False,   # 避免打开串口时拉低 DTR 复位单片机
            rtscts=False
        )

        # CH340 在打开串口瞬间会拉低 DTR/RTS，导致 STM32 复位，这里立即关闭以保持程序运行
        ser.setDTR(False)
        ser.setRTS(False)
        time.sleep(0.2)  # 给单片机一点恢复时间
        ser.reset_input_buffer()

        print(f"[ADXL345采集] 串口已打开: {SERIAL_PORT} @ {BAUDRATE} baud")
    except Exception as e:
        print(f"[错误] 无法打开串口 {SERIAL_PORT}: {e}")
        print(f"[提示] 请检查串口是否被其他程序占用，或修改SERIAL_PORT配置")
        return

    # 检查并处理CSV文件表头
    file_exists = CSV_FILE.exists()
    
    # 打开（或创建）CSV 文件，以追加模式写入数据
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # 如果是新文件，写入表头
        if not file_exists:
            writer.writerow(["timestamp", "col0", "col1", "col2"])
            print(f"[ADXL345采集] 已创建新文件并写入表头: {CSV_FILE}")

        print(f"[ADXL345采集] 开始记录数据到 {CSV_FILE}")
        print(f"[ADXL345采集] 采集频率: {1/SAMPLE_INTERVAL}Hz (每{SAMPLE_INTERVAL}秒一次)")
        print(f"[ADXL345采集] 按 Ctrl+C 结束")
        print("=" * 80)

        last_sample_time = time.time()
        record_count = 0

        try:
            while True:
                current_time = time.time()
                
                # 控制采集频率：每0.1秒采集一次
                if current_time - last_sample_time < SAMPLE_INTERVAL:
                    time.sleep(0.01)  # 短暂休眠，避免CPU占用过高
                    continue
                
                # 读取一行（以 \n 结尾）
                line = ser.readline().decode("utf-8", errors="ignore").strip()

                if not line:
                    continue

                # 串口输出格式：计数,ax,ay,az（4列）
                # 第一列是计数（需要丢弃，替换为实时时间戳）
                # 后三列是加速度数据（ax, ay, az）需要保留
                parts = line.split(",")

                # 检查数据格式：至少需要4列（计数, ax, ay, az）
                if len(parts) < 4:
                    print(f"[警告] 丢弃异常行（列数不足，需要至少4列）: {line}")
                    continue

                try:
                    # 提取串口原始数据
                    count_from_serial = parts[0]  # 第一列：计数（将被丢弃）
                    # 提取后三列：ax, ay, az（丢弃第一列的计数）
                    ax = float(parts[1])
                    ay = float(parts[2])
                    az = float(parts[3])
                except (ValueError, IndexError) as e:
                    print(f"[警告] 丢弃异常行（数据格式错误）: {line}, 错误: {e}")
                    continue

                # 生成完整时间戳字符串（年月日时分秒毫秒）
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                
                # 写入CSV：timestamp, col0, col1, col2
                # col0=ax, col1=ay, col2=az
                row = [timestamp, ax, ay, az]
                writer.writerow(row)
                f.flush()  # 立即写盘，方便"实时"查看
                
                record_count += 1
                last_sample_time = current_time
                
                # 每100条记录打印一次，避免输出过多
                if record_count % 100 == 0:
                    print(f"[ADXL345采集] 已记录 {record_count} 条数据")

        except KeyboardInterrupt:
            print(f"\n[ADXL345采集] 收到停止信号")
            print(f"[ADXL345采集] 总共记录了 {record_count} 条数据")
        except Exception as e:
            print(f"\n[错误] 采集过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            ser.close()
            print(f"[ADXL345采集] 串口已关闭")

if __name__ == "__main__":
    main()
