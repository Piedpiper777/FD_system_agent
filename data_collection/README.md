# 数据采集与示例指南（Data Collection Overview)

本指南统一说明项目的数据采集接口与演示资源，帮助你在无硬件/有硬件两种场景下快速打通数据流：
- 数据接口：系统核心只依赖 `edge/data/monitor/monitor_data.csv`。
- 示例演示：提供 STM32F103 + ADXL345 的采集示例与多种模拟数据生成脚本。
- 统一启动：通过 `start_data_collection.py` 一键启动数据采集（自动选择硬件或模拟）。

> 系统是硬件无关的。任何采集系统都可以接入，只要写入约定格式的 `monitor_data.csv`。

---

## 快速启动数据采集

**推荐方式**：使用统一启动脚本（自动选择硬件或模拟）

```bash
# 从项目根目录运行
python data_collection/start_data_collection.py
```

**工作流程**：
1. 优先尝试启动硬件采集（STM32 + ADXL345）
2. 若硬件不可用，自动回退到模拟数据生成
3. 数据统一输出到 `edge/data/monitor/monitor_data.csv`
4. 按 `Ctrl+C` 停止采集

**手动控制**（可选）：
```bash
# 仅启动硬件采集
python data_collection/sensor_demo/log_adxl345.py

# 仅启动模拟数据生成
python data_collection/simulated_data_generation/generate_monitor_data.py
```

---

## 1. 数据采集接口规范

系统通过统一的监测数据入口 `edge/data/monitor/monitor_data.csv` 读取数据并驱动前端监测与后端推理。

- 文件位置：`edge/data/monitor/monitor_data.csv`
- 推荐 CSV 格式：
  ```csv
  timestamp,col0,col1,col2,col3
  2025-12-01T10:00:00.000,0.123,0.456,9.789,25.3
  2025-12-01T10:00:00.100,0.124,0.457,9.790,25.3
  ...
  ```
- 字段说明：
  - `timestamp`：时间戳（ISO 8601格式）；
  - `col0, col1, col2`：三轴加速度（X/Y/Z）；
  - `col3`：温度（可选，根据场景扩展）。
- 采集频率：推荐 10Hz（每0.1秒一次）
- 基本原则：
  - 连续写入（追加模式），系统将实时读取并渲染；
  - 列名与列顺序遵循项目配置；
  - 采样率、单位、量纲请在元数据或配置中记录，便于训练与分析。

---

## 2. 有硬件：STM32 + ADXL345 演示平台

目录：`data_collection/sensor_demo/`

- 内容概览：
  - `F103_UART_LED_TEST.ioc` / `Core/` / `Drivers/` 等 STM32 工程文件；
  - `STM32F103RCTX_FLASH.ld` 等链接与启动文件；
  - `log_adxl345.py`：上位机采集脚本（串口读取 ADXL345 数据并写入 CSV）。

- 启动方式：
  ```bash
  # 方式1：通过统一启动脚本（推荐）
  python data_collection/start_data_collection.py
  
  # 方式2：直接运行硬件采集脚本
  python data_collection/sensor_demo/log_adxl345.py
  ```

- 使用思路：
  1. 将 ADXL345 连接至 STM32（I2C/SPI），设置采样率；
  2. 通过串口/USB（CH340）将数据传至上位机（边缘服务器）；
  3. 脚本自动检测 CH340 串口并读取数据；
  4. 数据写入 `edge/data/monitor/monitor_data.csv`；
  5. 启动 `edge/app.py`，页面即可显示实时波形并驱动推理。

- 重要提示：
  - 这是一个**演示**工程，方便理解接入流程；
  - 实际项目可替换为 PLC、工控机或自研板卡，只需输出 `monitor_data.csv` 即可。
  - 确保安装 CH340 驱动和 `pyserial` 库：`pip install pyserial`

---

## 3. 无硬件：模拟数据生成脚本

目录：`data_collection/simulated_data_generation/`

- 脚本列表：
  - `generate_monitor_data.py`：**持续生成监测数据**（写入 `monitor_data.csv`，用于实时监测演示）；
  - `generate_anomaly_detection_data.py`：异常检测数据生成（正常/故障工况，批量生成）；
  - `generate_rul_prediction_data.py`：RUL预测数据（Unit级，run-to-failure退化序列）；
  - `generate_fault_diagnosis_data.py`：故障诊断数据生成（多类故障特征）。

### 3.1 实时监测数据生成（推荐用于演示）

**用途**：持续生成模拟监测数据，支持实时监测和推理演示

**启动方式**：
```bash
# 方式1：通过统一启动脚本（推荐，自动回退）
python data_collection/start_data_collection.py

# 方式2：直接运行模拟生成脚本
python data_collection/simulated_data_generation/generate_monitor_data.py
```

**特点**：
- 持续运行，每 0.1 秒生成一条数据（10Hz）
- 自动追加到 `edge/data/monitor/monitor_data.csv`
- 模拟正常工况 + 随机异常（2% 概率）
- 包含四维数据：X/Y/Z轴加速度 + 温度
- 按 `Ctrl+C` 停止

### 3.2 批量训练数据生成

**用途**：生成用于模型训练和评估的数据集

**生成异常检测数据**：
```bash
cd data_collection/simulated_data_generation
python generate_anomaly_detection_data.py
```

**生成 RUL 预测数据**：
    ```bash
    cd data_collection/simulated_data_generation
    python generate_rul_prediction_data.py
    ```

- 输出位置（示例）：
  - 异常检测：`edge/data/collected/AnomalyDetection/`
  - RUL预测：`edge/data/labeled/RULPrediction/` 与 `edge/data/meta/RULPrediction/`
  - 故障诊断：`edge/data/processed/FaultDiagnosis/` 与 `edge/data/meta/FaultDiagnosis/`

---

## 4. Edge 应用的自动回退逻辑

`edge/app.py` 在启动时将尝试自动使用硬件采集脚本或提醒使用模拟数据：

- 优先尝试：`data_collection/sensor_demo/log_adxl345.py`
- 如果脚本不存在或串口失败：
  - 在终端提示使用：`data_collection/simulated_data_generation/generate_anomaly_detection_data.py`
  - 系统仍可正常运行，因为只依赖 `edge/data/monitor/monitor_data.csv`

这保证了在无硬件的情况下，系统也能通过模拟数据完整体验监测、异常检测、故障诊断与 RUL 预测等功能。

---

## 5. 推荐接入与部署方式

- **快速体验**：先运行模拟脚本生成数据，再启动 `cloud` 与 `edge` 服务，通过前端进行训练、推理与可视化。
- **现场部署**：将采集逻辑固化在各自 PLC/采集设备中，只在边缘服务器上运行本项目；通过网络/共享目录等方式，将数据统一写入 `monitor_data.csv`。
- **多场景扩展**：可在 `data_collection/simulated_data_generation` 中扩展更多脚本，或者将更多硬件演示工程放入 `data_collection/sensor_demo`。

---

更多背景与系统整体说明，请参考：
- `README.md`（项目总览与快速开始）
- `introduction.md`（项目背景、创新点与技术架构）
- `edge/README.md` 与 `cloud/README.md`（各组件说明）
