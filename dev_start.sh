#!/bin/bash

# ============================================
# 工业设备智能健康管理平台 - 开发环境一键启动脚本
# 同时启动 Edge、Cloud 和数据采集服务
# ============================================

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EDGE_DIR="$PROJECT_ROOT/edge"
CLOUD_DIR="$PROJECT_ROOT/cloud"
DATA_COLLECTION_DIR="$PROJECT_ROOT/data_collection"
LOGS_DIR="$PROJECT_ROOT/logs"

# 创建日志目录
mkdir -p "$LOGS_DIR"

# 日志文件
EDGE_LOG="$LOGS_DIR/edge.log"
CLOUD_LOG="$LOGS_DIR/cloud.log"
DATA_COLLECTION_LOG="$LOGS_DIR/data_collection.log"

# PID文件
EDGE_PID="$LOGS_DIR/edge.pid"
CLOUD_PID="$LOGS_DIR/cloud.pid"
DATA_COLLECTION_PID="$LOGS_DIR/data_collection.pid"

# ============================================
# 工具函数
# ============================================

print_header() {
    echo -e "${CYAN}"
    echo "=========================================="
    echo "  工业设备智能健康管理平台"
    echo "  开发环境一键启动"
    echo "=========================================="
    echo -e "${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

print_service() {
    echo -e "${MAGENTA}▶ $1${NC}"
}

# 检查conda是否安装
check_conda() {
    if ! command -v conda &> /dev/null; then
        print_error "未检测到conda，请先安装Anaconda或Miniconda"
        exit 1
    fi
    print_success "检测到conda"
}

# 检查虚拟环境是否存在
check_env() {
    local env_name=$1
    
    if conda env list | grep -q "^${env_name} "; then
        return 0
    else
        return 1
    fi
}

# 初始化conda
init_conda() {
    eval "$(conda shell.bash hook)"
}

# 检查进程是否运行
is_running() {
    local pid_file=$1
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0
        else
            rm -f "$pid_file"
            return 1
        fi
    fi
    return 1
}

# 停止进程
stop_process() {
    local name=$1
    local pid_file=$2
    
    if is_running "$pid_file"; then
        local pid=$(cat "$pid_file")
        print_info "停止 ${name} (PID: ${pid})..."
        kill "$pid" 2>/dev/null
        sleep 1
        if ps -p "$pid" > /dev/null 2>&1; then
            kill -9 "$pid" 2>/dev/null
        fi
        rm -f "$pid_file"
    fi
}

# 清理函数
cleanup() {
    echo ""
    print_warning "正在停止所有服务..."
    stop_process "数据采集服务" "$DATA_COLLECTION_PID"
    stop_process "Cloud服务" "$CLOUD_PID"
    stop_process "Edge服务" "$EDGE_PID"
    print_info "所有服务已停止"
    exit 0
}

# 注意：不在全局设置 EXIT trap，只在需要时设置
# 后台模式下不应该在正常退出时清理服务

# ============================================
# 启动函数
# ============================================

# 启动Edge服务
start_edge() {
    if is_running "$EDGE_PID"; then
        print_warning "Edge服务已在运行中，跳过启动"
        return 0
    fi
    
    print_service "启动Edge服务..."
    
    if [ ! -d "$EDGE_DIR" ]; then
        print_error "Edge目录不存在: $EDGE_DIR"
        return 1
    fi
    
    init_conda
    if ! check_env "edge"; then
        print_error "Edge虚拟环境不存在，请先创建: conda create -n edge python=3.10.19"
        return 1
    fi
    
    cd "$EDGE_DIR" || return 1
    conda activate edge
    
    # 后台启动
    nohup python app.py > "$EDGE_LOG" 2>&1 &
    local pid=$!
    echo $pid > "$EDGE_PID"
    
    sleep 2
    
    if is_running "$EDGE_PID"; then
        print_success "Edge服务启动成功 (PID: ${pid})"
        print_info "  访问地址: http://127.0.0.1:5000"
        return 0
    else
        print_error "Edge服务启动失败，请查看日志: $EDGE_LOG"
        rm -f "$EDGE_PID"
        return 1
    fi
}

# 启动Cloud服务
start_cloud() {
    if is_running "$CLOUD_PID"; then
        print_warning "Cloud服务已在运行中，跳过启动"
        return 0
    fi
    
    print_service "启动Cloud服务..."
    
    if [ ! -d "$CLOUD_DIR" ]; then
        print_error "Cloud目录不存在: $CLOUD_DIR"
        return 1
    fi
    
    init_conda
    if ! check_env "cloud"; then
        print_error "Cloud虚拟环境不存在，请先创建: conda create -n cloud python=3.10.19"
        return 1
    fi
    
    cd "$CLOUD_DIR" || return 1
    conda activate cloud
    
    # 后台启动
    nohup python app.py > "$CLOUD_LOG" 2>&1 &
    local pid=$!
    echo $pid > "$CLOUD_PID"
    
    sleep 2
    
    if is_running "$CLOUD_PID"; then
        print_success "Cloud服务启动成功 (PID: ${pid})"
        return 0
    else
        print_error "Cloud服务启动失败，请查看日志: $CLOUD_LOG"
        rm -f "$CLOUD_PID"
        return 1
    fi
}

# 启动数据采集
start_data_collection() {
    if is_running "$DATA_COLLECTION_PID"; then
        print_warning "数据采集服务已在运行中，跳过启动"
        return 0
    fi
    
    print_service "启动数据采集服务..."
    
    if [ ! -d "$DATA_COLLECTION_DIR" ]; then
        print_error "数据采集目录不存在: $DATA_COLLECTION_DIR"
        return 1
    fi
    
    init_conda
    if ! check_env "edge"; then
        print_error "Edge虚拟环境不存在（数据采集使用edge环境）"
        return 1
    fi
    
    cd "$PROJECT_ROOT" || return 1
    conda activate edge
    
    # 后台启动
    nohup python "$DATA_COLLECTION_DIR/start_data_collection.py" > "$DATA_COLLECTION_LOG" 2>&1 &
    local pid=$!
    echo $pid > "$DATA_COLLECTION_PID"
    
    sleep 2
    
    if is_running "$DATA_COLLECTION_PID"; then
        print_success "数据采集服务启动成功 (PID: ${pid})"
        return 0
    else
        print_error "数据采集服务启动失败，请查看日志: $DATA_COLLECTION_LOG"
        rm -f "$DATA_COLLECTION_PID"
        return 1
    fi
}

# ============================================
# 主程序
# ============================================

main() {
    print_header
    
    # 检查conda
    check_conda
    
    # 检查参数
    local run_mode="background"
    if [ "$1" == "--foreground" ] || [ "$1" == "-f" ]; then
        run_mode="foreground"
    fi
    
    echo ""
    print_info "启动模式: ${run_mode}"
    echo ""
    
    # 启动所有服务
    local success_count=0
    local total_count=3
    
    start_edge && success_count=$((success_count + 1))
    sleep 1
    
    start_cloud && success_count=$((success_count + 1))
    sleep 1
    
    start_data_collection && success_count=$((success_count + 1))
    
    echo ""
    echo "=========================================="
    if [ $success_count -eq $total_count ]; then
        print_success "所有服务启动成功！"
    else
        print_warning "部分服务启动失败 (${success_count}/${total_count})"
    fi
    echo "=========================================="
    echo ""
    
    # 显示服务信息
    print_info "服务状态:"
    echo ""
    
    if is_running "$EDGE_PID"; then
        local pid=$(cat "$EDGE_PID")
        print_success "  Edge服务: 运行中 (PID: ${pid})"
        print_info "    访问地址: http://127.0.0.1:5000"
    else
        print_error "  Edge服务: 未运行"
    fi
    
    if is_running "$CLOUD_PID"; then
        local pid=$(cat "$CLOUD_PID")
        print_success "  Cloud服务: 运行中 (PID: ${pid})"
    else
        print_error "  Cloud服务: 未运行"
    fi
    
    if is_running "$DATA_COLLECTION_PID"; then
        local pid=$(cat "$DATA_COLLECTION_PID")
        print_success "  数据采集服务: 运行中 (PID: ${pid})"
    else
        print_error "  数据采集服务: 未运行"
    fi
    
    echo ""
    print_info "日志文件位置:"
    echo "  - Edge服务: $EDGE_LOG"
    echo "  - Cloud服务: $CLOUD_LOG"
    echo "  - 数据采集: $DATA_COLLECTION_LOG"
    echo ""
    
    if [ "$run_mode" == "foreground" ]; then
        # 前台模式：设置 EXIT trap，按 Ctrl+C 时清理
        trap cleanup INT TERM EXIT
        
        print_info "前台模式：按 Ctrl+C 停止所有服务"
        print_info "实时查看日志，使用以下命令："
        echo ""
        echo "  tail -f $EDGE_LOG"
        echo "  tail -f $CLOUD_LOG"
        echo "  tail -f $DATA_COLLECTION_LOG"
        echo ""
        print_info "或使用多终端查看所有日志："
        echo "  tail -f $LOGS_DIR/*.log"
        echo ""
        
        # 等待用户中断
        while true; do
            sleep 1
            # 检查所有服务是否还在运行
            if ! is_running "$EDGE_PID" && ! is_running "$CLOUD_PID" && ! is_running "$DATA_COLLECTION_PID"; then
                print_warning "所有服务已停止"
                break
            fi
        done
    else
        # 后台模式：只捕获中断信号，不捕获 EXIT
        # 这样脚本正常退出时不会清理服务
        trap 'echo ""; print_warning "检测到中断信号，但服务将继续在后台运行"; print_info "使用 ./stop.sh 停止服务"; exit 0' INT TERM
        
        print_info "后台模式：服务已在后台运行"
        print_info "使用以下命令停止服务："
        echo ""
        echo "  ./stop.sh"
        echo "  或"
        echo "  ./start.sh stop"
        echo ""
        print_info "查看实时日志："
        echo "  tail -f $LOGS_DIR/*.log"
        echo ""
    fi
}

# 运行主程序
main "$@"

