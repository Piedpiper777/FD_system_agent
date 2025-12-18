#!/bin/bash

# ============================================
# 工业设备智能健康管理平台 - 一键启动脚本
# ============================================

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
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
    echo "  一键启动脚本"
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
    local env_dir=$2
    
    if conda env list | grep -q "^${env_name} "; then
        print_success "虚拟环境 '${env_name}' 已存在"
        return 0
    else
        print_warning "虚拟环境 '${env_name}' 不存在"
        return 1
    fi
}

# 激活conda环境
activate_env() {
    local env_name=$1
    local env_dir=$2
    
    # 初始化conda
    eval "$(conda shell.bash hook)"
    
    if check_env "$env_name" "$env_dir"; then
        print_info "激活虚拟环境: ${env_name}"
        conda activate "$env_name"
        if [ $? -eq 0 ]; then
            print_success "虚拟环境激活成功"
            return 0
        else
            print_error "虚拟环境激活失败"
            return 1
        fi
    else
        print_error "请先创建虚拟环境: conda create -n ${env_name} python=3.10.19"
        return 1
    fi
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
    local log_file=$3
    
    if is_running "$pid_file"; then
        local pid=$(cat "$pid_file")
        print_info "停止 ${name} (PID: ${pid})..."
        kill "$pid" 2>/dev/null
        sleep 2
        if ps -p "$pid" > /dev/null 2>&1; then
            kill -9 "$pid" 2>/dev/null
        fi
        rm -f "$pid_file"
        print_success "${name} 已停止"
    else
        print_warning "${name} 未运行"
    fi
}

# ============================================
# 启动函数
# ============================================

# 启动Edge服务
start_edge() {
    if is_running "$EDGE_PID"; then
        print_warning "Edge服务已在运行中"
        return 0
    fi
    
    print_info "启动Edge服务..."
    
    # 检查edge目录
    if [ ! -d "$EDGE_DIR" ]; then
        print_error "Edge目录不存在: $EDGE_DIR"
        return 1
    fi
    
    # 激活环境
    cd "$EDGE_DIR" || exit 1
    if ! activate_env "edge" "$EDGE_DIR"; then
        return 1
    fi
    
    # 检查requirements
    if [ ! -f "$EDGE_DIR/requirements.txt" ]; then
        print_warning "未找到requirements.txt，跳过依赖检查"
    fi
    
    # 后台启动
    nohup python app.py > "$EDGE_LOG" 2>&1 &
    local pid=$!
    echo $pid > "$EDGE_PID"
    
    sleep 3
    
    if is_running "$EDGE_PID"; then
        print_success "Edge服务启动成功 (PID: ${pid})"
        print_info "访问地址: http://127.0.0.1:5000"
        print_info "日志文件: $EDGE_LOG"
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
        print_warning "Cloud服务已在运行中"
        return 0
    fi
    
    print_info "启动Cloud服务..."
    
    # 检查cloud目录
    if [ ! -d "$CLOUD_DIR" ]; then
        print_error "Cloud目录不存在: $CLOUD_DIR"
        return 1
    fi
    
    # 激活环境
    cd "$CLOUD_DIR" || exit 1
    if ! activate_env "cloud" "$CLOUD_DIR"; then
        return 1
    fi
    
    # 检查requirements
    if [ ! -f "$CLOUD_DIR/requirements.txt" ]; then
        print_warning "未找到requirements.txt，跳过依赖检查"
    fi
    
    # 后台启动
    nohup python app.py > "$CLOUD_LOG" 2>&1 &
    local pid=$!
    echo $pid > "$CLOUD_PID"
    
    sleep 3
    
    if is_running "$CLOUD_PID"; then
        print_success "Cloud服务启动成功 (PID: ${pid})"
        print_info "日志文件: $CLOUD_LOG"
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
        print_warning "数据采集服务已在运行中"
        return 0
    fi
    
    print_info "启动数据采集服务..."
    
    # 检查数据采集目录
    if [ ! -d "$DATA_COLLECTION_DIR" ]; then
        print_error "数据采集目录不存在: $DATA_COLLECTION_DIR"
        return 1
    fi
    
    # 使用edge环境（数据采集与edge在同一环境）
    cd "$PROJECT_ROOT" || exit 1
    if ! activate_env "edge" "$EDGE_DIR"; then
        return 1
    fi
    
    # 后台启动
    nohup python "$DATA_COLLECTION_DIR/start_data_collection.py" > "$DATA_COLLECTION_LOG" 2>&1 &
    local pid=$!
    echo $pid > "$DATA_COLLECTION_PID"
    
    sleep 2
    
    if is_running "$DATA_COLLECTION_PID"; then
        print_success "数据采集服务启动成功 (PID: ${pid})"
        print_info "日志文件: $DATA_COLLECTION_LOG"
        return 0
    else
        print_error "数据采集服务启动失败，请查看日志: $DATA_COLLECTION_LOG"
        rm -f "$DATA_COLLECTION_PID"
        return 1
    fi
}

# ============================================
# 停止函数
# ============================================

stop_all() {
    echo ""
    print_info "正在停止所有服务..."
    stop_process "数据采集服务" "$DATA_COLLECTION_PID" "$DATA_COLLECTION_LOG"
    stop_process "Cloud服务" "$CLOUD_PID" "$CLOUD_LOG"
    stop_process "Edge服务" "$EDGE_PID" "$EDGE_LOG"
    echo ""
    print_success "所有服务已停止"
}

# ============================================
# 状态检查
# ============================================

check_status() {
    echo ""
    print_info "服务运行状态:"
    echo ""
    
    if is_running "$EDGE_PID"; then
        local pid=$(cat "$EDGE_PID")
        print_success "Edge服务: 运行中 (PID: ${pid})"
        print_info "  访问地址: http://127.0.0.1:5000"
    else
        print_warning "Edge服务: 未运行"
    fi
    
    if is_running "$CLOUD_PID"; then
        local pid=$(cat "$CLOUD_PID")
        print_success "Cloud服务: 运行中 (PID: ${pid})"
    else
        print_warning "Cloud服务: 未运行"
    fi
    
    if is_running "$DATA_COLLECTION_PID"; then
        local pid=$(cat "$DATA_COLLECTION_PID")
        print_success "数据采集服务: 运行中 (PID: ${pid})"
    else
        print_warning "数据采集服务: 未运行"
    fi
    
    echo ""
}

# ============================================
# 日志查看
# ============================================

view_logs() {
    echo ""
    print_info "选择要查看的日志:"
    echo "1) Edge服务日志"
    echo "2) Cloud服务日志"
    echo "3) 数据采集日志"
    echo "4) 所有日志"
    echo "0) 返回"
    read -p "请选择 [0-4]: " choice
    
    case $choice in
        1)
            if [ -f "$EDGE_LOG" ]; then
                tail -f "$EDGE_LOG"
            else
                print_warning "日志文件不存在"
            fi
            ;;
        2)
            if [ -f "$CLOUD_LOG" ]; then
                tail -f "$CLOUD_LOG"
            else
                print_warning "日志文件不存在"
            fi
            ;;
        3)
            if [ -f "$DATA_COLLECTION_LOG" ]; then
                tail -f "$DATA_COLLECTION_LOG"
            else
                print_warning "日志文件不存在"
            fi
            ;;
        4)
            print_info "显示所有日志（最后20行）..."
            echo ""
            echo "=== Edge服务日志 ==="
            tail -n 20 "$EDGE_LOG" 2>/dev/null || echo "无日志"
            echo ""
            echo "=== Cloud服务日志 ==="
            tail -n 20 "$CLOUD_LOG" 2>/dev/null || echo "无日志"
            echo ""
            echo "=== 数据采集日志 ==="
            tail -n 20 "$DATA_COLLECTION_LOG" 2>/dev/null || echo "无日志"
            ;;
        0)
            return
            ;;
        *)
            print_error "无效选择"
            ;;
    esac
}

# ============================================
# 主菜单
# ============================================

show_menu() {
    echo ""
    echo -e "${CYAN}请选择操作:${NC}"
    echo "1) 启动Edge服务"
    echo "2) 启动Cloud服务"
    echo "3) 启动数据采集"
    echo "4) 启动所有服务（Edge + 数据采集）"
    echo "5) 停止所有服务"
    echo "6) 查看服务状态"
    echo "7) 查看日志"
    echo "0) 退出"
    echo ""
}

# ============================================
# 主程序
# ============================================

main() {
    print_header
    
    # 检查conda
    check_conda
    
    # 处理命令行参数
    if [ $# -gt 0 ]; then
        case $1 in
            start-edge)
                start_edge
                exit $?
                ;;
            start-cloud)
                start_cloud
                exit $?
                ;;
            start-data)
                start_data_collection
                exit $?
                ;;
            start-all)
                start_edge
                sleep 2
                start_data_collection
                exit $?
                ;;
            stop)
                stop_all
                exit $?
                ;;
            status)
                check_status
                exit $?
                ;;
            *)
                echo "用法: $0 [start-edge|start-cloud|start-data|start-all|stop|status]"
                exit 1
                ;;
        esac
    fi
    
    # 交互式菜单
    while true; do
        show_menu
        read -p "请选择 [0-7]: " choice
        echo ""
        
        case $choice in
            1)
                start_edge
                ;;
            2)
                start_cloud
                ;;
            3)
                start_data_collection
                ;;
            4)
                print_info "启动所有服务..."
                start_edge
                sleep 2
                start_data_collection
                ;;
            5)
                stop_all
                ;;
            6)
                check_status
                ;;
            7)
                view_logs
                ;;
            0)
                print_info "退出脚本"
                exit 0
                ;;
            *)
                print_error "无效选择，请重新输入"
                ;;
        esac
    done
}

# 捕获Ctrl+C
trap 'echo ""; print_warning "检测到中断信号"; stop_all; exit 0' INT TERM

# 运行主程序
main "$@"

