#!/bin/bash

# ============================================
# 工业设备智能健康管理平台 - 停止脚本
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
LOGS_DIR="$PROJECT_ROOT/logs"

# PID文件
EDGE_PID="$LOGS_DIR/edge.pid"
CLOUD_PID="$LOGS_DIR/cloud.pid"
DATA_COLLECTION_PID="$LOGS_DIR/data_collection.pid"

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
        sleep 2
        if ps -p "$pid" > /dev/null 2>&1; then
            kill -9 "$pid" 2>/dev/null
        fi
        rm -f "$pid_file"
        print_success "${name} 已停止"
        return 0
    else
        print_warning "${name} 未运行"
        return 1
    fi
}

# 主程序
echo -e "${CYAN}"
echo "=========================================="
echo "  停止所有服务"
echo "=========================================="
echo -e "${NC}"
echo ""

print_info "正在停止所有服务..."
echo ""

stopped=0
stop_process "数据采集服务" "$DATA_COLLECTION_PID" && stopped=$((stopped + 1))
stop_process "Cloud服务" "$CLOUD_PID" && stopped=$((stopped + 1))
stop_process "Edge服务" "$EDGE_PID" && stopped=$((stopped + 1))

echo ""
if [ $stopped -gt 0 ]; then
    print_success "已停止 ${stopped} 个服务"
else
    print_warning "没有运行中的服务"
fi

