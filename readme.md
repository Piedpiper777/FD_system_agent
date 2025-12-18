# 运筹与优化团队工业设备智能健康管理平台

## 一键启动（推荐）

> **说明**：使用一键启动脚本可以快速启动所有服务，无需手动操作。

### 使用方式

#### 方式一：交互式菜单（推荐）

```bash
# 在项目根目录执行
./start.sh
```

脚本会显示交互式菜单，支持以下操作：
- 启动Edge服务
- 启动Cloud服务
- 启动数据采集
- 启动所有服务（Edge + 数据采集）
- 停止所有服务
- 查看服务状态
- 查看日志

#### 方式二：命令行参数

```bash
# 启动Edge服务
./start.sh start-edge

# 启动Cloud服务
./start.sh start-cloud

# 启动数据采集
./start.sh start-data

# 启动所有服务（Edge + 数据采集）
./start.sh start-all

# 停止所有服务
./start.sh stop
# 或使用独立的停止脚本
./stop.sh

# 查看服务状态
./start.sh status
```

### 功能特性

- ✅ 自动检测conda环境
- ✅ 自动激活虚拟环境
- ✅ 后台运行服务
- ✅ 日志文件管理（保存在 `logs/` 目录）
- ✅ 进程状态监控
- ✅ 优雅停止服务
- ✅ 彩色输出提示

### 注意事项

1. **首次使用前**，请确保已创建虚拟环境：
   ```bash
   # Edge环境
   conda create -n edge python=3.10.19
   conda activate edge
   cd edge
   pip install -r requirements.txt
   
   # Cloud环境（如需要）
   conda create -n cloud python=3.10.19
   conda activate cloud
   cd cloud
   pip install -r requirements.txt
   ```

2. **日志文件位置**：`logs/` 目录
   - `logs/edge.log` - Edge服务日志
   - `logs/cloud.log` - Cloud服务日志
   - `logs/data_collection.log` - 数据采集日志

3. **停止服务**：使用 `./start.sh stop` 或按 `Ctrl+C`（在交互模式下）

## 开发环境一键启动

> **说明**：开发环境通常在同一台设备上运行所有服务，使用此脚本可一键启动 Edge、Cloud 和数据采集三个服务。

### 使用方法

```bash
# 后台模式启动（默认，推荐）
./dev_start.sh

# 前台模式启动（可以看到实时日志提示）
./dev_start.sh --foreground
# 或
./dev_start.sh -f
```

### 功能特点

- ✅ **一键启动三个服务**：Edge、Cloud、数据采集同时启动
- ✅ **自动环境检测**：自动检测并激活对应的conda环境
- ✅ **进程管理**：自动管理进程，支持优雅停止
- ✅ **状态监控**：启动后显示所有服务的运行状态
- ✅ **日志管理**：所有日志统一保存在 `logs/` 目录
- ✅ **信号处理**：按 `Ctrl+C` 自动停止所有服务

### 启动流程

1. 检查conda环境
2. 依次启动 Edge 服务
3. 依次启动 Cloud 服务
4. 依次启动数据采集服务
5. 显示所有服务状态和访问地址

### 停止服务

```bash
# 使用停止脚本
./stop.sh

# 或使用start.sh的停止功能
./start.sh stop
```

### 查看日志

```bash
# 查看所有日志
tail -f logs/*.log

# 查看单个服务日志
tail -f logs/edge.log
tail -f logs/cloud.log
tail -f logs/data_collection.log
```

## 快速开始（手动启动）

### 1. 云端部署(云服务器)

(1)进入cloud目录</br>
```bash
cd cloud
```
(2)创建虚拟环境</br>
```bash
conda create -n cloud python=3.10.19
```
(3)激活虚拟环境</br>
```bash
conda activate cloud
```
(4)配置虚拟环境</br>
```bash
pip install -r requirements.txt
```
注意：mindspore可能需要自行从官网安装</br>

(5)配置环境变量</br>

详见[cloud/.env](cloud/.env)</br>

(6)运行程序</br>
```bash
python app.py
```
### 2. 边缘端部署（本地电脑）
(1)进入edge目录</br>
```bash
cd edge
```
(2)创建虚拟环境</br>
```bash
conda create -n edge python=3.10.19
```
(3)激活虚拟环境</br>
```bash
conda activate edge
```
(4)配置虚拟环境</br>
```bash
pip install -r requirements.txt
```
注意：mindspore可能需要自行从官网安装</br>

(5)配置环境变量</br>

详见[edge/.env](edge/.env)</br>

(6)运行程序</br>
```bash
python app.py
```
### 3. 数据采集启动（可选，独立进程，此处仅为演示）

> **说明**：数据采集已与 Edge 服务解耦，需要单独启动。如果不启动数据采集，系统仍可正常运行（使用已有的历史数据）。

```bash
# 启动数据采集（优先硬件，失败时自动回退到模拟数据生成）
python data_collection/start_data_collection.py
```

**数据采集策略**：
- **优先**：STM32 + ADXL345 硬件采集（需连接 CH340 串口）
- **回退**：模拟数据生成（无需硬件）
- **输出**：统一写入 `edge/data/monitor/monitor_data.csv`
- **频率**：10Hz（每 0.1 秒一次）

**手动控制**：
```bash
# 仅硬件采集（如果硬件可用）
python data_collection/sensor_demo/log_adxl345.py

# 仅模拟数据生成
python data_collection/simulated_data_generation/generate_monitor_data.py
```

### 4. 前端运行
(1)直接在edge端浏览器运行
浏览器访问
```
http://127.0.0.1:5000
```
(2)移动设备浏览器连接到
```
http://[edge端IPv4地址]:5000"
```
- 需确保与edge端在同一网络下
- 获取电脑/服务器的IP地址：</br>
  - Windows: 打开cmd或者powershell，输ipconfig
查找"IPv4 地址"；</br>
  - Linux/Mac: 打开终端，输入ifconfig或ip addr show查找inet地址。