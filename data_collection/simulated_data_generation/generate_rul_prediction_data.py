"""
模拟RUL预测数据生成脚本
生成三轴加速度数据，用于RUL预测任务
包含工况：转速
支持内圈故障和外圈故障的退化过程

输出到 collected/RULPrediction 目录，文件名体现数据类型
数据已截断到故障点
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple
import json
from sklearn.preprocessing import MinMaxScaler


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
    
    def generate_rul_data(
        self,
        total_duration: float,
        rpm: float,
        fault_type: str,
        base_amplitude: float = 1.0,
        noise_level: float = 0.1,
        degradation_rate: float = 0.1
    ) -> Tuple[np.ndarray, int]:
        """
        生成RUL预测数据（从正常到故障的退化过程）
        
        改进点：
        1. 使用非线性退化模式（分段函数：早期慢，后期快）
        2. 随机化故障点位置（增加数据多样性）
        3. 使用 degradation_rate 参数控制退化速度
        4. 添加退化过程的随机波动
        5. 向量化处理提高效率
        
        Args:
            total_duration: 总持续时间（秒）
            rpm: 转速（转/分钟）
            fault_type: 故障类型
            base_amplitude: 基础振幅
            noise_level: 噪声水平
            degradation_rate: 退化速率系数（0.05-0.2，影响非线性退化的陡峭程度）
            
        Returns:
            (data, failure_index): 数据数组和故障点索引
        """
        n_samples = int(total_duration * self.sampling_rate)
        t = np.arange(n_samples) * self.dt
        omega = 2 * np.pi * rpm / 60.0
        
        # 生成基础正常数据
        data = self.generate_normal_data(total_duration, rpm, base_amplitude, noise_level)
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        
        # ========== 改进1: 非线性退化模式 ==========
        # 使用分段函数：前60%缓慢退化，后40%加速退化
        early_phase_ratio = 0.6
        early_phase_samples = int(n_samples * early_phase_ratio)
        late_phase_samples = n_samples - early_phase_samples
        
        # 早期阶段：缓慢线性增长（0 -> 1.5）
        severity_early = np.linspace(0, 1.5, early_phase_samples)
        
        # 后期阶段：加速增长（1.5 -> 5.0），使用指数函数
        # degradation_rate 控制加速程度：值越大，后期退化越快
        alpha = 0.5 + degradation_rate * 2.0  # 映射到 [0.5, 2.5]
        late_normalized = np.linspace(0, 1, late_phase_samples)
        severity_late = 1.5 + 3.5 * np.power(late_normalized, alpha)
        
        # 合并两个阶段
        severity_progression = np.concatenate([severity_early, severity_late])
        
        # ========== 改进2: 添加退化随机波动 ==========
        # 添加随机游走，模拟退化的不确定性
        random_walk = np.cumsum(np.random.normal(0, 0.03, n_samples))
        severity_progression = severity_progression + random_walk
        severity_progression = np.clip(severity_progression, 0, 5.0)
        
        # ========== 改进3: 随机化故障点 ==========
        # 故障阈值在合理范围内随机分布（70%-96%位置）
        failure_threshold = np.random.uniform(3.5, 4.8)
        failure_indices = np.where(severity_progression >= failure_threshold)[0]
        if len(failure_indices) > 0:
            failure_index = failure_indices[0]
        else:
            failure_index = int(n_samples * 0.9)  # 默认90%位置
        
        # ========== 改进4: 向量化处理故障特征 ==========
        # 根据故障类型添加渐进式故障特征
        if fault_type == 'inner_ring':
            fault_freq = omega * 5.43
            impact_period = int(self.sampling_rate * 60 / rpm * 5.43)
            
            # 向量化计算：一次性处理所有时间点
            # 故障特征仅在严重程度 > 0.1 时添加
            fault_mask = severity_progression > 0.1
            
            # 计算故障频率的冲击（向量化）
            impact_base = severity_progression * 0.12 * np.sin(fault_freq * t)
            impact_base[~fault_mask] = 0  # 早期正常阶段不添加故障特征
            
            x += impact_base
            y += impact_base * 0.85
            z += impact_base * 0.7
            
            # 振幅逐渐增大（向量化）
            amplitude_factor_x = np.ones(n_samples)
            amplitude_factor_x[fault_mask] = 1 + severity_progression[fault_mask] * 0.07
            x *= amplitude_factor_x
            
            amplitude_factor_y = np.ones(n_samples)
            amplitude_factor_y[fault_mask] = 1 + severity_progression[fault_mask] * 0.056
            y *= amplitude_factor_y
            
            amplitude_factor_z = np.ones(n_samples)
            amplitude_factor_z[fault_mask] = 1 + severity_progression[fault_mask] * 0.044
            z *= amplitude_factor_z
            
            # 周期性冲击（仍需循环，但只处理故障阶段）
            impact_indices = np.arange(0, n_samples, impact_period)
            impact_indices = impact_indices[impact_indices < n_samples]
            impact_indices = impact_indices[severity_progression[impact_indices] > 0.1]
            
            for i in impact_indices:
                severity = severity_progression[i]
                x[i] += severity * 0.3 * np.random.uniform(0.8, 1.2)
                y[i] += severity * 0.24 * np.random.uniform(0.8, 1.2)
                z[i] += severity * 0.2 * np.random.uniform(0.8, 1.2)
                    
        elif fault_type == 'outer_ring':
            fault_freq = omega * 3.57
            impact_period = int(self.sampling_rate * 60 / rpm * 3.57)
            
            # 向量化计算
            fault_mask = severity_progression > 0.1
            
            impact_base = severity_progression * 0.10 * np.sin(fault_freq * t)
            impact_base[~fault_mask] = 0
            
            x += impact_base
            y += impact_base * 0.9
            z += impact_base * 0.75
            
            # 振幅逐渐增大（向量化）
            x[fault_mask] *= (1 + severity_progression[fault_mask] * 0.056)
            y[fault_mask] *= (1 + severity_progression[fault_mask] * 0.05)
            z[fault_mask] *= (1 + severity_progression[fault_mask] * 0.04)
            
            # 周期性冲击
            impact_indices = np.arange(0, n_samples, impact_period)
            impact_indices = impact_indices[impact_indices < n_samples]
            impact_indices = impact_indices[severity_progression[impact_indices] > 0.1]
            
            for i in impact_indices:
                severity = severity_progression[i]
                x[i] += severity * 0.24 * np.random.uniform(0.8, 1.2)
                y[i] += severity * 0.2 * np.random.uniform(0.8, 1.2)
                z[i] += severity * 0.18 * np.random.uniform(0.8, 1.2)
        else:
            raise ValueError(f"Unknown fault type: {fault_type}, must be 'inner_ring' or 'outer_ring'")
        
        return np.column_stack([x, y, z]), failure_index
    
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


def generate_rul_prediction_data(
    labeled_dir: Path,
    meta_dir: Path,
    generator: SimulatedDataGenerator,
    rpm_values: list = [1000, 1500, 2500],
    train_duration: float = 300.0,
    test_ratio: float = 0.1,
    fault_type: str = 'inner_ring',
    num_train_per_rpm: int = 5,
    rul_unit: str = 'cycle',
    max_rul: int = 200
):
    """
    生成RUL预测数据，直接输出到labeled目录并生成元数据
    
    Args:
        labeled_dir: labeled数据输出目录
        meta_dir: 元数据输出目录
        generator: 数据生成器
        rpm_values: 转速值列表
        train_duration: 训练集持续时间（秒）
        test_ratio: 测试集时长相对于训练集的比例
        fault_type: 故障类型
        num_train_per_rpm: 每个转速生成的训练集数量
        rul_unit: RUL单位（'cycle', 'second', 'minute'）
        max_rul: RUL最大截断值
    """
    labeled_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    file_count = 0
    test_duration = train_duration * test_ratio
    
    # 创建MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # 全局unit计数器
    unit_id = 1
    
    for rpm in rpm_values:
        # 为每个转速生成多个训练unit
        for train_idx in range(num_train_per_rpm):
            degradation_rate = np.random.uniform(0.05, 0.2)
            train_data, train_failure_index = generator.generate_rul_data(
                train_duration, rpm, fault_type, degradation_rate=degradation_rate
            )
            
            # 截断到故障点
            train_data = train_data[:train_failure_index + 1]
            
            # 应用MinMax标准化
            train_data_normalized = scaler.fit_transform(train_data)
            
            # 创建DataFrame
            df = pd.DataFrame(train_data_normalized, columns=['X', 'Y', 'Z'])
            
            # 添加时间戳列
            start_time = datetime.now()
            timestamps = [start_time + timedelta(seconds=i * generator.dt) for i in range(len(df))]
            df.insert(0, 'timestamp', timestamps)
            
            # 保存CSV文件 - 使用unit_ID命名格式
            train_filename = f"unit_{unit_id:03d}_rpm{rpm:.0f}_{fault_type}.csv"
            train_filepath = labeled_dir / train_filename
            df.to_csv(train_filepath, index=False)
            
            # 生成元数据文件
            meta_data = {
                'filename': train_filename,
                'file_path': str(train_filepath),
                'unit_id': unit_id,  # 每个文件代表一个独立的unit（运行序列）
                'display_name': f"Unit-{unit_id:03d} (RPM{rpm} {fault_type})",
                'task_type': 'rul_prediction',
                'data_type': 'train',
                'created_at': datetime.now().isoformat(),
                'preprocessing': {
                    'method': 'minmax',
                    'feature_range': [0, 1],
                    'columns': ['X', 'Y', 'Z']
                },
                'tags_condition': [
                    {'key': '转速', 'value': str(rpm)}
                ],
                'rul_config': {
                    'failure_row_index': int(len(df) - 1),  # 最后一行是故障点
                    'rul_unit': rul_unit,
                    'max_rul': int(max_rul),
                    'fault_type': fault_type,
                    'degradation_rate': float(degradation_rate)
                },
                'unit_info': {
                    'description': '完整的run-to-failure退化序列，不可截断',
                    'independence': 'unit之间完全独立，不存在数据泄漏'
                },
                'data_info': {
                    'sampling_rate': float(generator.sampling_rate),
                    'duration': float(train_duration),
                    'total_samples': int(len(df)),
                    'failure_index': int(train_failure_index),
                    'rpm': int(rpm),
                    'features': ['X', 'Y', 'Z']
                }
            }
            
            meta_filename = train_filename.replace('.csv', '.json')
            meta_filepath = meta_dir / meta_filename
            with open(meta_filepath, 'w', encoding='utf-8') as f:
                json.dump(meta_data, f, indent=2, ensure_ascii=False)
            
            file_count += 1
            unit_id += 1  # 每生成一个训练unit，ID递增
            print(f"✓ 生成训练unit {file_count}: {train_filename} (Unit-{unit_id-1:03d}, RPM={rpm}, 长度={len(df)})")
            print(f"  - 数据长度: {len(df)}, 故障点: {len(df)-1}")
            print(f"  - 元数据: {meta_filename}")
        
        # 生成测试集（每个转速1个）
        degradation_rate = np.random.uniform(0.05, 0.2)
        test_data, test_failure_index = generator.generate_rul_data(
            test_duration, rpm, fault_type, degradation_rate=degradation_rate
        )
        
        # 截断到故障点
        test_data = test_data[:test_failure_index + 1]
        
        # 应用MinMax标准化
        test_data_normalized = scaler.fit_transform(test_data)
        
        # 创建DataFrame
        df = pd.DataFrame(test_data_normalized, columns=['X', 'Y', 'Z'])
        
        # 添加时间戳列
        start_time = datetime.now()
        timestamps = [start_time + timedelta(seconds=i * generator.dt) for i in range(len(df))]
        df.insert(0, 'timestamp', timestamps)
        
        # 保存CSV文件 - 使用unit_ID命名格式
        test_filename = f"unit_{unit_id:03d}_rpm{rpm:.0f}_{fault_type}.csv"
        test_filepath = labeled_dir / test_filename
        df.to_csv(test_filepath, index=False)
        
        # 生成元数据文件
        meta_data = {
            'filename': test_filename,
            'file_path': str(test_filepath),
            'unit_id': unit_id,  # 每个文件代表一个独立的unit（运行序列）
            'display_name': f"Unit-{unit_id:03d} (RPM{rpm} {fault_type})",
            'task_type': 'rul_prediction',
            'data_type': 'test',
            'created_at': datetime.now().isoformat(),
            'preprocessing': {
                'method': 'minmax',
                'feature_range': [0, 1],
                'columns': ['X', 'Y', 'Z']
            },
            'tags_condition': [
                {'key': '转速', 'value': str(rpm)}
            ],
            'rul_config': {
                'failure_row_index': int(len(df) - 1),  # 最后一行是故障点
                'rul_unit': rul_unit,
                'max_rul': int(max_rul),
                'fault_type': fault_type,
                'degradation_rate': float(degradation_rate)
            },
            'unit_info': {
                'description': '完整的run-to-failure退化序列，不可截断',
                'independence': 'unit之间完全独立，不存在数据泄漏'
            },
            'data_info': {
                'sampling_rate': float(generator.sampling_rate),
                'duration': float(test_duration),
                'total_samples': int(len(df)),
                'failure_index': int(test_failure_index),
                'rpm': int(rpm),
                'features': ['X', 'Y', 'Z']
            }
        }
        
        meta_filename = test_filename.replace('.csv', '.json')
        meta_filepath = meta_dir / meta_filename
        with open(meta_filepath, 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, indent=2, ensure_ascii=False)
        
        file_count += 1
        unit_id += 1  # 每生成一个unit，ID递增
        print(f"✓ 生成测试unit {file_count}: {test_filename} (Unit-{unit_id-1:03d}, RPM={rpm}, 长度={len(df)})")
    
    total_train = len(rpm_values) * num_train_per_rpm
    total_test = len(rpm_values)
    print(f"\n数据生成完成，共 {file_count} 个units ({total_train} 个训练units + {total_test} 个测试units)")
    print(f"数据目录: {labeled_dir}")
    print(f"元数据目录: {meta_dir}")


def main():
    # ==================== 配置参数 ====================
    # 输出目录（相对于项目根目录）
    LABELED_DIR = 'edge/data/labeled/RULPrediction'
    META_DIR = 'edge/data/meta/RULPrediction'
    
    # 采样率 (Hz)
    SAMPLING_RATE = 100.0
    
    # 转速值列表 (RPM) - 三种工况
    RPM_VALUES = [1000, 1500, 2500]
    
    # 训练集持续时间（秒）
    TRAIN_DURATION = 300.0
    
    # 测试集是训练集的比例
    TEST_RATIO = 0.1
    
    # 每个转速生成的训练集数量（增加数据多样性）
    NUM_TRAIN_PER_RPM = 5
    
    # 故障类型（'inner_ring' 或 'outer_ring'）
    FAULT_TYPE = 'inner_ring'
    
    # RUL单位（'cycle', 'second', 'minute'）
    RUL_UNIT = 'cycle'
    
    # RUL最大截断值
    MAX_RUL = 200
    
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
    labeled_dir = project_root / LABELED_DIR
    meta_dir = project_root / META_DIR
    
    print("=" * 70)
    print("RUL预测数据生成脚本 - 直接生成到labeled目录")
    print("=" * 70)
    print(f"数据输出目录: {labeled_dir}")
    print(f"元数据目录: {meta_dir}")
    print(f"采样率: {SAMPLING_RATE} Hz")
    print(f"转速: {RPM_VALUES}")
    print(f"故障类型: {FAULT_TYPE}")
    print(f"训练集持续时间: {TRAIN_DURATION} 秒")
    print(f"测试集持续时间: {TRAIN_DURATION * TEST_RATIO} 秒 (训练集的 {TEST_RATIO*100}%)")
    print(f"每个转速的训练集数量: {NUM_TRAIN_PER_RPM}")
    print(f"RUL单位: {RUL_UNIT}")
    print(f"RUL最大截断值: {MAX_RUL}")
    print(f"数据预处理: MinMax归一化 [0, 1]")
    print(f"随机种子: {RANDOM_SEED}")
    print("=" * 70)
    
    # 生成数据
    generate_rul_prediction_data(
        labeled_dir=labeled_dir,
        meta_dir=meta_dir,
        generator=generator,
        rpm_values=RPM_VALUES,
        train_duration=TRAIN_DURATION,
        test_ratio=TEST_RATIO,
        fault_type=FAULT_TYPE,
        num_train_per_rpm=NUM_TRAIN_PER_RPM,
        rul_unit=RUL_UNIT,
        max_rul=MAX_RUL
    )
    
    print("\n" + "=" * 70)
    print("数据生成完成！")
    print("=" * 70)
    print("\n数据说明：")
    print(f"- 每个文件代表一个独立的unit（运行序列）")
    print(f"- 总共生成 {len(RPM_VALUES) * (NUM_TRAIN_PER_RPM + 1)} 个units")
    print(f"- 文件命名格式：unit_XXX_rpmYYYY_fault_type.csv")
    print(f"- 每个unit是完整的run-to-failure退化轨迹，不可截断")
    print(f"- 训练units：完整的退化过程 (~{TRAIN_DURATION}秒)")
    print(f"- 测试units：较短的退化过程 (~{TRAIN_DURATION * TEST_RATIO}秒)")
    print(f"- 故障类型：{FAULT_TYPE}")
    print(f"- 数据已通过MinMax归一化到[0, 1]范围")
    print(f"- 元数据文件包含unit_id和完整的RUL配置信息")
    print("\n数据特征：")
    print("- 非线性退化：早期60%缓慢退化，后期40%加速退化")
    print("- 随机故障点：故障点位置在70%-96%范围内随机分布")
    print("- 随机退化速率：每个样本的退化速率随机化，增加多样性")
    print("- 退化随机波动：添加随机游走，模拟退化的不确定性")
    print("\n文件格式：")
    print("- CSV文件：timestamp, X, Y, Z（已归一化）")
    print("- JSON文件：包含RUL配置、工况信息、预处理参数等元数据")
    print("\n重要说明 - Unit级别独立性：")
    print("- ⚠️  每个文件是一个独立的unit，不能截断或混合")
    print("- ⚠️  数据划分应基于unit级别：train units, val units, test units")
    print("- ⚠️  不要使用比例划分（如70% train, 30% val），这会破坏退化轨迹完整性")
    print("- ⚠️  unit之间必须完全独立，避免信息泄漏")
    print("\n下一步：")
    print("- 在Web界面的训练页面分别选择：训练units、验证units、测试units")
    print("- 示例：选择unit 1-12作为训练集，unit 13-15作为验证集，unit 16-18作为测试集")
    print("- 确保同一个unit不会同时出现在训练集和验证集中")


if __name__ == '__main__':
    main()

