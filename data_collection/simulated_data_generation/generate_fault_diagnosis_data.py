"""
模拟故障诊断数据生成脚本
生成三轴加速度数据，用于故障诊断任务
包含工况：转速
支持正常、内圈故障、外圈故障三类

输出到 collected/FaultDiagnosis 目录，文件名体现数据类型
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta


class SimulatedDataGenerator:
    """模拟数据生成器"""
    
    def __init__(self, sampling_rate: float = 100.0, random_seed: int = 42):
        """
        初始化数据生成器
        
        Args:
            sampling_rate: 采样率 (Hz)
            random_seed: 随机种子
        """
        self.sampling_rate = sampling_rate
        self.dt = 1.0 / sampling_rate
        np.random.seed(random_seed)
    
    def generate_normal_data(
        self,
        duration: float,
        rpm: float,
        base_amplitude: float = 1.0,
        noise_level: float = 0.1
    ) -> np.ndarray:
        """
        生成正常状态的三轴加速度数据
        
        Args:
            duration: 持续时间（秒）
            rpm: 转速（转/分钟）
            base_amplitude: 基础振幅
            noise_level: 噪声水平
            
        Returns:
            (N, 3) 数组，三列分别为 X, Y, Z 轴加速度
        """
        n_samples = int(duration * self.sampling_rate)
        t = np.arange(n_samples) * self.dt
        
        # 转速转换为角频率 (rad/s)
        omega = 2 * np.pi * rpm / 60.0
        
        # 生成三轴加速度
        # X轴：主要振动分量 + 谐波
        x = base_amplitude * (
            np.sin(omega * t) +
            0.3 * np.sin(2 * omega * t) +
            0.1 * np.sin(3 * omega * t)
        )
        
        # Y轴：相位偏移 + 不同谐波比例
        y = base_amplitude * (
            0.8 * np.sin(omega * t + np.pi/4) +
            0.2 * np.sin(2 * omega * t + np.pi/6) +
            0.1 * np.cos(omega * t)
        )
        
        # Z轴：垂直方向，振幅较小
        z = base_amplitude * (
            0.5 * np.sin(omega * t + np.pi/2) +
            0.2 * np.cos(2 * omega * t)
        )
        
        # 添加噪声
        x += np.random.normal(0, noise_level, n_samples)
        y += np.random.normal(0, noise_level, n_samples)
        z += np.random.normal(0, noise_level, n_samples)
        
        return np.column_stack([x, y, z])
    
    def generate_fault_data(
        self,
        duration: float,
        rpm: float,
        fault_type: str,
        base_amplitude: float = 1.0,
        noise_level: float = 0.1,
        fault_severity: float = 1.0
    ) -> np.ndarray:
        """
        生成故障状态的三轴加速度数据
        
        Args:
            duration: 持续时间（秒）
            rpm: 转速（转/分钟）
            fault_type: 故障类型 ('inner_ring', 'outer_ring')
            base_amplitude: 基础振幅
            noise_level: 噪声水平
            fault_severity: 故障严重程度 (1.0-5.0)
            
        Returns:
            (N, 3) 数组，三列分别为 X, Y, Z 轴加速度
        """
        n_samples = int(duration * self.sampling_rate)
        t = np.arange(n_samples) * self.dt
        omega = 2 * np.pi * rpm / 60.0
        
        # 首先生成正常数据
        normal_data = self.generate_normal_data(duration, rpm, base_amplitude, noise_level)
        x, y, z = normal_data[:, 0], normal_data[:, 1], normal_data[:, 2]
        
        # 根据故障类型添加故障特征
        if fault_type == 'inner_ring':
            # 内圈故障：高频冲击，特征频率约为 5.43 * 基频
            fault_freq = omega * 5.43
            impact = fault_severity * 0.6 * np.sin(fault_freq * t)
            x += impact
            y += impact * 0.85
            z += impact * 0.7
            # 振幅增大，X轴最明显
            x *= (1 + fault_severity * 0.35)
            y *= (1 + fault_severity * 0.28)
            z *= (1 + fault_severity * 0.22)
            # 添加周期性冲击
            impact_period = int(self.sampling_rate * 60 / rpm * 5.43)
            for i in range(0, n_samples, impact_period):
                if i < n_samples:
                    x[i] += fault_severity * 1.5 * np.random.uniform(0.8, 1.2)
                    y[i] += fault_severity * 1.2 * np.random.uniform(0.8, 1.2)
                    z[i] += fault_severity * 1.0 * np.random.uniform(0.8, 1.2)
            
        elif fault_type == 'outer_ring':
            # 外圈故障：高频冲击，特征频率约为 3.57 * 基频
            fault_freq = omega * 3.57
            impact = fault_severity * 0.5 * np.sin(fault_freq * t)
            x += impact
            y += impact * 0.9
            z += impact * 0.75
            # 振幅增大，相对内圈故障稍弱
            x *= (1 + fault_severity * 0.28)
            y *= (1 + fault_severity * 0.25)
            z *= (1 + fault_severity * 0.20)
            # 添加周期性冲击，频率低于内圈
            impact_period = int(self.sampling_rate * 60 / rpm * 3.57)
            for i in range(0, n_samples, impact_period):
                if i < n_samples:
                    x[i] += fault_severity * 1.2 * np.random.uniform(0.8, 1.2)
                    y[i] += fault_severity * 1.0 * np.random.uniform(0.8, 1.2)
                    z[i] += fault_severity * 0.9 * np.random.uniform(0.8, 1.2)
            
        else:
            raise ValueError(f"Unknown fault type: {fault_type}, must be 'inner_ring' or 'outer_ring'")
        
        return np.column_stack([x, y, z])
    
    def create_dataframe(
        self,
        data: np.ndarray,
        start_time: datetime = None,
        add_timestamp: bool = True
    ) -> pd.DataFrame:
        """
        将数据数组转换为DataFrame
        
        Args:
            data: (N, 3) 数据数组
            start_time: 起始时间
            add_timestamp: 是否添加时间戳列
            
        Returns:
            DataFrame with columns: [timestamp, X, Y, Z] or [X, Y, Z]
        """
        if start_time is None:
            start_time = datetime.now()
        
        n_samples = len(data)
        df = pd.DataFrame(data, columns=['X', 'Y', 'Z'])
        
        if add_timestamp:
            timestamps = [start_time + timedelta(seconds=i * self.dt) for i in range(n_samples)]
            df.insert(0, 'timestamp', timestamps)
        
        return df


def generate_fault_diagnosis_data(
    output_dir: Path,
    generator: SimulatedDataGenerator,
    num_per_class: int = 10,
    rpm_values: list = [1500, 2000, 2500],
    duration: float = 60.0
):
    """生成故障诊断数据（正常、内圈故障、外圈故障）"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fault_types = ['normal', 'inner_ring', 'outer_ring']
    file_count = 0
    
    for fault_type in fault_types:
        for i in range(num_per_class):
            for rpm in rpm_values:
                if fault_type == 'normal':
                    data = generator.generate_normal_data(duration, rpm)
                else:
                    data = generator.generate_fault_data(
                        duration, rpm, fault_type, fault_severity=3.0
                    )
                
                df = generator.create_dataframe(data)
                
                # 文件名格式：任务类型_标签_工况_序号.csv
                filename = f"fault_diagnosis_{fault_type}_rpm{rpm:.0f}_{i+1:03d}.csv"
                filepath = output_dir / filename
                df.to_csv(filepath, index=False)
                
                file_count += 1
                print(f"✓ 生成故障诊断数据: {filename}")
    
    print(f"\n故障诊断数据生成完成，共 {file_count} 个文件")


def main():
    # ==================== 配置参数 ====================
    # 输出目录（相对于项目根目录）
    OUTPUT_DIR = 'edge/data/collected/FaultDiagnosis'
    
    # 采样率 (Hz)
    SAMPLING_RATE = 100.0
    
    # 转速值列表 (RPM)
    RPM_VALUES = [1500, 2000, 2500]
    
    # 数据持续时间（秒）
    DURATION = 60.0
    
    # 每个类别生成的文件数
    NUM_PER_CLASS = 10
    
    # 随机种子
    RANDOM_SEED = 42
    # ================================================
    
    # 创建生成器
    generator = SimulatedDataGenerator(
        sampling_rate=SAMPLING_RATE,
        random_seed=RANDOM_SEED
    )
    
    # 确定输出目录
    project_root = Path(__file__).parent.parent
    output_dir = project_root / OUTPUT_DIR
    
    print("=" * 60)
    print("模拟故障诊断数据生成脚本")
    print("=" * 60)
    print(f"输出目录: {output_dir}")
    print(f"采样率: {SAMPLING_RATE} Hz")
    print(f"转速: {RPM_VALUES}")
    print(f"持续时间: {DURATION} 秒")
    print(f"每个类别文件数: {NUM_PER_CLASS}")
    print(f"随机种子: {RANDOM_SEED}")
    print("=" * 60)
    
    # 生成数据
    generate_fault_diagnosis_data(
        output_dir, generator,
        num_per_class=NUM_PER_CLASS,
        rpm_values=RPM_VALUES,
        duration=DURATION
    )
    
    print("\n" + "=" * 60)
    print("数据生成完成！")
    print("=" * 60)
    print("\n数据说明：")
    print("- 正常数据：正常状态的三轴加速度数据")
    print("- 内圈故障：内圈故障的三轴加速度数据")
    print("- 外圈故障：外圈故障的三轴加速度数据")
    print("\n数据特征：")
    print("- 不同转速：影响振动频率和振幅")
    print("- 内圈故障：高频冲击（5.43倍转频），X轴振幅显著增大")
    print("- 外圈故障：高频冲击（3.57倍转频），振幅增大相对较小")
    print("\n文件名格式：")
    print("- fault_diagnosis_{标签}_rpm{转速}_{序号}.csv")
    print("- 标签：normal, inner_ring, outer_ring")


if __name__ == '__main__':
    main()

