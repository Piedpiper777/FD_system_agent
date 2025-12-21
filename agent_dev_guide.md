# 多智能体开发说明
## 项目部署与本地配置❗❗❗
（在readme.md文件中已有说明，此处再次说明）
1. 本地安装git（网上搜）
2. 本地新建一个文件夹，如project（自己做好文件管理即可）
3. 在这个文件夹路径下运行 git clone https://github.com/Piedpiper777/FD_system_agent.git
4. 安装conda（网上搜）
5. 配置conda环境
- Edge环境
```
conda create -n edge python=3.10.19
conda activate edge
cd edge
pip install -r requirements.txt
```
- Cloud环境
```
conda create -n cloud python=3.10.19
conda activate cloud
cd cloud
pip install -r requirements.txt
```
6. 运行程序，根目录下运行以下命令即可（也可按需启动，参考readme.md文件）：
```
./dev_start.sh
```
7. 退出时停止程序（切记❗）
```
./stop.sh
```
# ❗❗❗以下内容仅供参考，可按照自己的理解开发
## 目标与场景
- 支撑多人并行开发：每人负责一个子智能体（如异常检测、故障诊断等），独立迭代与验证。
- 前端有一个常驻悬浮对话窗口，作为统一交互入口；后端有总智能体负责路由/编排各子智能体。
- 子智能体在开发环境下可独立运行与测试（独立与用户交互），最终通过统一协议接入总智能体。

## 架构概览
- 悬浮对话窗口（前端组件）：页面常驻、会话独立，可嵌入任意页，不随跳转销毁。
- 总智能体（网关/编排层）：负责意图识别、路由、并发调用、结果合并、熔断/降级、审计日志。
- 子智能体（领域能力）：对外暴露统一 API（推理/训练/工具），可本地或容器化部署。
- 通信方式：HTTP/WS 调用为主，可选接入消息队列（如 NATS/Kafka）用于长耗时任务和状态轮询。

## 协议与数据约定
- 通用请求字段：
  - `trace_id`：全链路追踪；前端生成，贯穿总/子智能体。
  - `session_id`：同一会话标识；悬浮窗本地存储。
  - `user_id`：用户标识（匿名用 uuid），用于审计/限流。
  - `intent`：意图标签或 NL 文本，由总智能体解析后下发。
  - `payload`：与子智能体相关的结构化参数。
  - 可选 `context`：上文摘要、对话历史截断版本。
- 通用响应字段：
  - `status`: `ok` | `error` | `partial`
  - `data`: 业务结果
  - `error`: `{ code, message, detail? }`
  - `usage`（可选）：tokens/耗时/资源信息
- 错误码示例：`INVALID_INPUT`，`TIMEOUT`，`UPSTREAM_ERROR`，`RATE_LIMITED`。
- 流式返回（WS/SSE）：
  - `event`: `token` | `status` | `done` | `error`
  - `seq`: 递增序号
  - `trace_id`, `session_id`
  - `delta`: 文本增量或进度信息

## 悬浮对话窗口规范
- 形态：封装为可嵌入组件（建议 Web Component 或 React 组件），独立样式命名空间。
- 常驻：使用 `localStorage/IndexedDB` 存储 `session_id`、会话摘要；路由跳转后恢复。
- 通信：默认 HTTP/WS 到总智能体，仅暴露单一后端入口。
- 体验：支持流式输出、打字机效果、进度提示、超时/重试提示；发送时展示 loading/禁用状态。
- 多会话：允许用户切换会话或清除历史；内部对每个会话维护上下文摘要（避免过长上下文）。
- 可配置项：后端 URL、鉴权 token、主题、语言、日志级别（仅调试）。

## 临时文件中转（开发占位方案）
- 前端悬浮窗输入写入 `edge/data/chat/input.txt`（JSONL，每行字段 `timestamp/session_id/trace_id/text`）。
- 前端输出展示 `edge/data/chat/output.txt`（纯文本或由子智能体写入的结果），支持轮询刷新。
- 子智能体本地开发时，可直接读取 `input.txt`、写回 `output.txt` 做对接验证；后续接入总智能体时，替换读写逻辑为真实 API，不改前端。

## 子智能体开发规范
- API 形态：建议至少提供
  - `POST /invoke`：同步/流式推理调用（支持 `stream=true`）。
  - `POST /jobs`：长耗时任务提交，返回 `job_id`。
  - `GET /jobs/{job_id}`：查询任务状态/结果。
- 输入校验：严格验证 `payload` schema，返回标准错误码。
- 资源隔离：训练与推理解耦；推理服务尽量轻量，依赖以容器/虚拟环境声明。
- 观测性：统一 `trace_id`，结构化日志（json），指标（qps、latency、error_rate），可选链路追踪（OpenTelemetry）。
- 安全：鉴权（token/JWT）、输入过滤（防 prompt 注入）、速率限制、最小权限访问外部资源。
- 配置：通过 `.env` 或配置文件暴露端口、模型路径、超时、并发限制；避免硬编码。
- 测试：单元测试覆盖核心推理逻辑；契约测试校验接口协议；可提供 mock 响应供总智能体集成测试。

## 总智能体编排要点
- 路由策略：基于意图分类/关键词/规则，必要时回退到“澄清”子流程。
- 并发与合并：支持扇出并发调用多个子智能体，按优先级或最先完成返回；合并结果时附来源与置信度。
- 可靠性：超时、重试（带抖动）、熔断、降级（返回缓存/简化结果），并记录审计日志。
- 状态管理：长耗时任务使用异步 job + 轮询/回调；短请求用同步/流式。
- 策略注入：预留 hook 便于添加新子智能体或规则，不影响现有路由。

## 开发流程建议
1) 先实现悬浮窗 + 总智能体“壳”与 mock 接口，跑通端到端交互。
2) 每个子智能体在本地/容器内独立开发，提供稳定的 `/invoke` 与 job 接口。
3) 编写契约测试（对 mock 与真实服务都运行），避免集成期协议不一致。
4) 集成阶段逐个替换 mock 为真实服务，开启链路追踪，观察 qps/latency。
5) 预发布/灰度：开启限流与熔断，收集日志与指标再放量。

## 本地调试与环境
- 推荐在独立 Conda/venv 中运行子智能体；端口约定避免冲突（示例：总智能体 7000，子智能体递增）。
- 如需全链路调试，可用 `docker-compose` 或脚本同时启动总智能体 + 若干子智能体 mock。
- 日志位置与格式约定：`logs/{service}.log`，JSON 行日志，包含 `trace_id/session_id/level`.

## 安全与合规
- 鉴权：前端携带 token，网关校验并下传；内部服务可使用内网 ACL 或 mTLS。
- 数据：对敏感字段做脱敏/最小留存；长日志中避免记录原始敏感输入。
- 风控：输入长度/类型限制，防止 prompt 注入与异常 payload。

## 交付物清单（每个子智能体）
- `README`/接口文档：描述能力、依赖、部署、示例请求/响应。
- 接口契约（OpenAPI/JSON Schema）与错误码说明。
- 可运行的本地启动方式（脚本/compose），以及健康检查接口。
- 测试用例：单测 + 契约测试；可选负载/回归测试脚本。
- 日志/指标接入说明（trace_id 贯通）。

