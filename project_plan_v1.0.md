## DeepSeek R1 推理服务项目方案

**版本:** 1.0
**日期:** 2024-07-27

**项目目标:** 构建一个可弹性伸缩、高性能的 **完整 671B 参数 DeepSeek R1** 推理服务。该服务利用 NVIDIA Dynamo 作为上层分布式协调框架，**vLLM 结合 Ray 作为底层跨节点推理引擎 (采用流水线并行和张量并行)**，并将服务部署在 **多个 RunPod H100 GPU Pods** 上。

---

### 1. 环境准备 (Environment Setup)

此阶段的目标是准备好所有必需的软件、硬件资源、模型存储、配置和工具，为后续的服务搭建和测试奠定基础。

**1.1 软件栈确认与获取:**

*   **核心框架与引擎:**
    *   **NVIDIA Dynamo (分布式框架):**
        *   **来源:** 从官方 GitHub 仓库 `ai-dynamo/dynamo` 获取最新稳定版源代码 ([https://github.com/ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo))。
        *   **依赖:** Rust 工具链 (Cargo), Python 3.x, Docker, `kubectl` (若用 K8s 部署), `etcd` (作为 Dynamo 依赖的分布式 KV 存储)。
        *   **构建:** 遵循官方文档编译 Dynamo 核心组件 (Planner, Router, Worker Agents 等)。
    *   **vLLM (推理引擎):**
        *   **来源:** 使用 `pip install vllm` 安装最新稳定版。
        *   **依赖:** Python 3.x, PyTorch (CUDA-enabled, 版本需与 RunPod GPU 驱动兼容), Transformers, CUDA Toolkit (版本需与 PyTorch 和 RunPod GPU 驱动匹配)。
    *   **Ray (分布式执行后端):**
        *   **来源:** 使用 `pip install ray[default]` 安装最新稳定版。
        *   **依赖:** Python 3.x。
*   **模型:**
    *   **DeepSeek R1 (671B):**
        *   **来源:** 确认 Hugging Face Hub (`deepseek-ai/DeepSeek-R1`) 或其他来源提供的 **完整 671B 权重**。需验证格式 (可能是 FP8 或需转为 BF16) 与 vLLM 加载兼容性。
        *   **存储策略:** **模型权重将通过一次性下载存储在 RunPod 网络存储卷 (Volume) 中，以避免重复下载和加速 Pod 启动。**
        *   **格式:** 确保为 vLLM 支持的格式（通常为 Hugging Face Transformers 格式）。
        *   **Tokenizer:** 获取配套的 Tokenizer 文件。
*   **平台与工具:**
    *   **RunPod (GPU 平台):**
        *   **账户:** 注册 RunPod 账户并获取 API Key。
        *   **客户端:** 安装 `runpod-python` 库 (`pip install runpod`) 或准备直接调用 GraphQL API。
    *   **容器化:** Docker Engine (最新稳定版)。
    *   **版本控制:** Git。
    *   **Python 环境:** `venv` 或 `conda`。
    *   **编排:** **Kubernetes (如 K3s) 或手动脚本/RunPod API 调用**，用于管理构成 Ray 集群的多个 Pods。
    *   **监控:**
        *   **Metrics:** Prometheus (Server), Grafana (Dashboard)。
        *   **Agents:** Prometheus Node Exporter, `nvidia-dcgm-exporter` (获取详细 GPU 指标), vLLM metrics endpoint, 自定义应用 metrics exporter (如 FastAPI 的 Prometheus exporter)。
    *   **日志:**
        *   **聚合:** Loki 或 Elasticsearch。
        *   **收集:** Promtail, Fluentd, 或 Fluent Bit。
    *   **负载测试:**
        *   **工具:** Locust (Python, 易于编程和模拟复杂场景), k6 (Go/JS, 高性能), 或自定义 Python `asyncio` 脚本。

**1.2 基础设施设置 (RunPod):**

*   **GPU 选型:** **H100 80GB GPU** 是首选。**估算所需 Pod 数量:** 基于 671B R1 模型 VRAM 需求 (FP16 约 >1.3TB，BF16 类似，FP8 减半但仍巨大) 和 H100 80GB VRAM，计算运行模型所需的最小 GPU 总数 (例如，BF16 可能需要 17+ 卡)。确定需要启动多少个 RunPod 实例 (例如，若选 8xH100 Pod，则需 3+ 个 Pod)。
*   **网络规划:**
    *   确保 RunPod Pods 之间（Dynamo 组件、P/D 节点、API 服务）可以低延迟通信。配置 Pod 间网络。
    *   配置外部访问（如通过 RunPod 的 TCP Endpoint 或自定义入口）到 API 服务。
    *   设置必要的防火墙/安全组规则。
*   **存储配置:**
    *   **镜像仓库:** 配置 Docker Hub、NVIDIA NGC 或私有容器仓库访问权限，用于存储和拉取自定义构建的 Docker 镜像。
    *   **模型权重存储 (关键步骤):**
        1.  **创建网络存储卷:** 在 RunPod 控制台创建持久化网络存储卷，分配足够空间（例如 500GB - 1TB），选择正确区域。
        2.  **创建临时下载 Pod:**
            *   **目的:** 提供一个临时环境以下载模型到网络存储卷。
            *   **方法 (RunPod UI):**
                *   导航到 RunPod Secure Cloud 或 Community Cloud。
                *   选择一个基础的 GPU 或 CPU Pod 模板（例如 "RunPod Pytorch" 或一个简单的 Ubuntu 模板）。不需要强大的 GPU，能联网和运行 Python/Shell 即可。
                *   在配置选项中，**附加 (Attach)** 第 1 步创建的网络存储卷，并指定挂载路径 (例如 `/model_storage`)。
                *   设置适当的容器磁盘和卷磁盘大小（如果需要临时存储空间）。
                *   启动 Pod。
            *   **方法 (RunPod CLI/API):** 使用相应的命令或 API 调用创建 Pod，确保包含挂载网络存储卷的参数。
        3.  **在临时 Pod 中安装必要工具并下载模型:**
            *   **连接到临时 Pod:** 使用 SSH 或 Web Terminal 连接到刚创建的 Pod。
            *   **安装下载工具 (如果基础镜像没有):**
                ```bash
                # 更新包列表
                apt-get update
                # 安装 pip (如果需要)
                apt-get install -y python3-pip
                # 安装 Hugging Face Hub 库
                pip install huggingface_hub
                # (可选) 如果使用 Git LFS 方法下载，安装 git 和 git-lfs
                # apt-get install -y git git-lfs && git lfs install --system --skip-repo
                ```
            *   **执行模型下载:** 使用 `huggingface_hub` Python 库或 `huggingface-cli` 命令行工具，将 DeepSeek R1 模型权重**下载到之前指定的挂载卷路径下** (例如 `/model_storage/deepseek-r1`)。
                ```python
                # 示例 Python 代码 (在 Pod 内运行)
                from huggingface_hub import snapshot_download
                model_id = "deepseek-ai/DeepSeek-V2" # 替换为实际模型 ID
                local_dir = "/model_storage/deepseek-r1" # 下载到挂载的卷
                print(f"Downloading {model_id} to {local_dir}...")
                try:
                    snapshot_download(
                        repo_id=model_id,
                        local_dir=local_dir,
                        local_dir_use_symlinks=False,
                        # token="YOUR_HF_TOKEN" # 如果需要认证，取消注释并填入
                    )
                    print("Download complete.")
                except Exception as e:
                    print(f"Error downloading model: {e}")
                ```
                *   **注意:** 如果模型需要认证，请先在 Hugging Face 网站生成 Token，并在下载时提供 (通过 `token` 参数或 `huggingface-cli login`)。
            *   **(新增) 格式转换 (如果需要):** 如果下载的是 FP8 权重，而计划使用 BF16 运行 vLLM，需在此临时 Pod 中或之后步骤中运行 DeepSeek 提供的转换脚本 (`fp8_cast_bf16.py`) 将权重转换为 BF16 并存回 Volume。
            *   **确认下载:** 检查挂载卷路径下是否包含完整的模型文件。
        4.  **清理临时 Pod:** **确认下载完成后**，可以安全地停止并删除这个临时 Pod。网络存储卷及其中的模型数据将保留。
    *   **(可选) 其他持久卷:** 若 Dynamo 配置、应用日志等需要持久化存储，可根据需要创建并挂载额外的网络存储卷。

*   **服务组件 Pod 创建说明:** 请注意，用于运行 Dynamo 控制平面、P/D 节点和 API 服务的 Pod **将在后续的"2. 服务搭建"阶段创建**，它们将使用在"1.3 容器镜像构建"中定义的自定义 Docker 镜像。

**1.3 容器镜像构建:**

*   **基础镜像:** 使用 **RunPod 官方的 PyTorch 或 CUDA 基础镜像**，确保 CUDA/PyTorch 版本与 vLLM/Ray 兼容。
*   **Ray + vLLM + Dynamo Worker 节点镜像 (`ray-vllm-dynamo-worker`):**
    *   基于选定的基础镜像构建。
    *   **安装核心依赖:**
        *   `pip install vllm ray[default]`
        *   安装其他必要 Python 包 (如 `huggingface_hub`, `prometheus_client`)。
    *   **集成 Dynamo Agent:** 将编译好的 Dynamo Worker Agent 添加到镜像中。
    *   **包含监控 Agents:** (同前) 如 `nvidia-dcgm-exporter`。
    *   **重要:** (同前) 不包含模型权重。
    *   **包含启动脚本:** 脚本负责：
        *   启动 Ray Worker 进程，并连接到 Ray Head 节点（地址通过环境变量传入）。
        *   (可选，看部署模式) 可能需要在此脚本中根据 Ray 的角色启动 vLLM 引擎实例 (通过 vLLM Python API 或命令行，并传入模型路径、并行参数等)。
        *   启动 Dynamo Worker Agent。
*   **Dynamo 控制平面镜像 (`dynamo-controller`):**
    *   包含编译好的 Dynamo Planner, Router 等核心组件。
    *   包含其运行所需的配置文件和启动脚本。
*   **API 服务镜像 (`api-service`):**
    *   包含 FastAPI 应用代码及其所有 Python 依赖。
    *   包含 Prometheus exporter 以暴露 API 指标。
*   **推送镜像:** 将以上所有构建好的自定义镜像推送到在 1.2 中配置好的容器镜像仓库。

---

### 2. 服务搭建 (Service Construction)

此阶段的目标是利用环境准备阶段的成果，在 RunPod 上部署并组装一个功能完整的、基于 Dynamo 和 vLLM 的 DeepSeek R1 推理服务系统。

**2.1 单节点 vLLM 服务验证 (使用网络存储):**

*   **目的:** 确认 vLLM 在目标 RunPod GPU 上能成功从网络存储卷加载并运行 DeepSeek R1。
*   **步骤:**
    1.  在选定的 RunPod GPU 实例上启动一个 Pod，使用 `dynamo-vllm-worker` 镜像（仅运行 vLLM 部分，不启动 Dynamo Agent）。
    2.  **将包含模型权重的网络存储卷附加到此 Pod** (例如挂载到 `/persistent_storage`)。
    3.  配置 vLLM 启动参数，使其**从挂载的 Volume 中加载模型** (例如，模型路径设置为 `/persistent_storage/deepseek-r1`)。
    4.  确认 DeepSeek R1 模型成功加载，无 OOM 或路径错误。
    5.  通过端口转发或 vLLM 暴露的 API 发送测试请求，验证基本推理功能。
    6.  记录模型加载时间和空载显存占用。

**2.2 NVIDIA Dynamo 核心组件部署:**

*   **目的:** 部署 Dynamo 分布式框架的控制平面。
*   **步骤:**
    1.  部署 `etcd` 服务（若 Dynamo 需要外部 etcd）。可作为 RunPod Pod 运行。
    2.  使用 `dynamo-controller` 镜像，部署 Dynamo Router(s) Pod(s)，并在配置中指定其角色为 Router。
    3.  使用 `dynamo-controller` 镜像，部署 Dynamo Planner(s) Pod(s)，并在配置中指定其角色为 Planner。
    4.  确保 Router 和 Planner 配置了正确的 `etcd` 地址，并能够相互发现和通信。

**2.3 P/D 节点 (vLLM Workers) 部署与集成:**

*   **目的:** 部署执行实际推理任务的 GPU 节点，并将其纳入 Dynamo 的管理。
*   **步骤:**
    1.  使用 `dynamo-vllm-worker` 镜像，启动初始数量（例如，各 1 个）的 P 节点 Pod 和 D 节点 Pod。
    2.  **关键配置:**
        *   **挂载模型存储卷:** 将包含 DeepSeek R1 权重的网络存储卷附加到所有 P/D 节点 Pod (例如挂载到 `/persistent_storage`)。
        *   **配置启动参数:** 通过环境变量或配置文件传入：
            *   Dynamo Router/Planner 的地址。
            *   节点的角色 (Prefill 或 Decode)。
            *   **vLLM 模型路径:** 指向挂载卷中的模型目录 (例如 `/persistent_storage/deepseek-r1`)。
            *   其他 vLLM 参数 (如 `gpu_memory_utilization`, `max_num_seqs`)。
            *   RunPod GPU 设备信息。
    3.  确认 Pod 内的 vLLM 服务和 Dynamo Agent 均成功启动。
    4.  通过 Dynamo 的日志或管理接口（若有）检查 P/D 节点是否已成功注册并处于活动状态。

**2.4 配置 P/D 分离 (Disaggregated Serving):**

*   **目的:** 在 Dynamo 中启用并配置 P/D 分离（分解服务）的核心优化特性。
*   **步骤:**
    1.  **Router 配置:** 在 Dynamo Router 的配置中启用 KV Cache 感知路由模式。
    2.  **Planner 配置:**
        *   设置 P:D 资源比例为 1:1。
        *   配置 RunPod API Key 及相关参数（目标 GPU 类型、区域、`dynamo-vllm-worker` 镜像 ID），使 Planner 能够管理 RunPod 资源。
        *   定义基础的自动伸缩策略（例如基于队列长度或资源利用率，具体阈值将在测试阶段调优）。
    3.  **(可选) KV Cache Manager 配置:** 根据需求和性能测试结果，考虑配置 KV Cache 卸载策略（例如，是否启用卸载，卸载目标是 CPU RAM 还是更慢的存储）。

**2.5 API 入口服务部署:**

*   **目的:** 提供标准化的、面向用户的服务接口，并将请求转发给 Dynamo。
*   **步骤:**
    1.  开发 FastAPI 应用：
        *   实现 OpenAI 兼容的 API endpoint (如 `/v1/chat/completions`)。
        *   处理请求：接收用户输入 -> 调用 Dynamo Router 的 gRPC 或 HTTP 接口发起推理任务 -> (异步) 等待 Dynamo 返回的最终结果 -> 将结果格式化并返回给用户。
        *   集成 Prometheus exporter 以暴露 API 层面的关键指标（请求计数、延迟、错误率等）。
    2.  使用 `api-service` 镜像将 FastAPI 应用部署到 RunPod (通常部署在 CPU Pod 即可，根据负载也可考虑增加实例)。
    3.  配置 RunPod TCP Endpoint 或其他网络入口，将外部流量正确路由到 API 服务实例。

**2.6 监控与日志系统集成:**

*   **目的:** 建立全面的可观测性，以便于性能分析、问题排查和运维管理。
*   **步骤:**
    1.  部署监控和日志基础设施：Prometheus 服务器、Grafana 服务器、Loki (日志聚合) 和 Promtail (日志收集) 或等效替代方案。
    2.  **配置 Prometheus Scrape Targets:** 确保 Prometheus 能够抓取以下所有目标的 metrics endpoints:
        *   Dynamo Router(s) 和 Planner(s)。
        *   所有活动的 P/D 节点 (包括 vLLM metrics, Node Exporter metrics, DCGM-Exporter metrics)。
        *   API 服务实例。
    3.  **配置日志收集:** 配置 Promtail 或 Fluentd/Fluent Bit 从所有相关 Pod (Dynamo 组件, P/D 节点, API 服务) 收集标准输出/错误日志以及应用内部日志，并将它们发送到 Loki 或 Elasticsearch。
    4.  **创建 Grafana Dashboards:** 设计并创建以下关键仪表板:
        *   **服务概览:** QPS, 端到端延迟分布 (P50/P90/P99), 请求成功率, 活跃 P/D 节点数, 总 TPS, 估算的 RunPod 成本。
        *   **Dynamo 状态:** Router 请求队列长度, KV 缓存命中率 (若可观测), Planner 调度决策频率和延迟, P/D 节点注册状态。
        *   **vLLM & GPU 性能:** 每个 P/D 节点的 GPU 计算利用率 (%), GPU 显存使用量 (MB/GB) 和利用率 (%), GPU 显存带宽利用率 (%) (若 DCGM exporter 提供), vLLM 内部处理队列长度 (running/waiting sequences), vLLM KV 缓存块使用率 (%), TTFT 和 ITL 分布。
        *   **API 服务性能:** API 请求 QPS, 延迟分布, 错误率 (按 endpoint 区分)。
        *   **系统资源:** 各 Pod 的 CPU 利用率 (%), RAM 使用量 (MB/GB), 网络 I/O (MB/s)。
        *   **日志查询:** 在 Grafana 中集成 Loki 或 Elasticsearch 数据源，方便关联查询日志。

---

### 3. 性能测试 (Performance Testing)

此阶段的目标是通过模拟真实负载，系统地评估服务的性能、稳定性、弹性和成本效益，识别瓶颈并进行针对性优化。

**3.1 测试工具与环境:**

*   **负载生成工具:** 推荐使用 **Locust**，因其基于 Python，易于编写复杂的测试逻辑（如模拟多轮对话、自定义请求头等），并支持分布式运行以产生高并发负载。k6 也是一个高性能的选择。
*   **部署位置:** 将 Locust master 和 worker 实例部署在与 RunPod 服务相同区域的 CPU 实例上，以最小化网络延迟对测试结果的影响。
*   **测试数据集:**
    *   **Prompt 类型:** 准备包含不同长度 Prompt 的混合数据集，例如：短 (<50 tokens), 中 (50-500 tokens), 长 (>500 tokens)，并能调整各类 Prompt 的比例。
    *   **生成长度:** 测试固定 `max_tokens` 和不同 `max_tokens` 组合的场景。
    *   **多轮对话数据:** 构造模拟多轮对话的请求序列，其中后续请求包含前几轮的 context，用于有效测试 KV 缓存机制。

**3.2 测试方案与场景:**

*   **场景 1: 基准性能评估 (Latency & Throughput)**
    *   **目的:** 确定服务在不同并发水平下的基础响应时间和处理能力。
    *   **方法:**
        *   **低并发测试:** 从低并发开始 (例如 1-10 个并发用户)，使用混合 Prompt 数据集发送请求，精确测量端到端延迟、TTFT (Time To First Token) 和 ITL (Inter-Token Latency) 的分布 (P50, P90, P99)。
        *   **容量爬坡测试:** 逐步增加并发用户数（例如，每次增加 10 或 20 个用户），持续监控 QPS (Queries Per Second), TPS (Tokens Per Second), 以及上述延迟指标。找到系统处理能力饱和、延迟开始急剧上升的"拐点"。
    *   **观测重点:** 端到端延迟、TTFT、ITL、QPS、TPS、GPU 计算/显存利用率、vLLM 等待队列长度。
*   **场景 2: P/D 分离与 KV 缓存效果验证**
    *   **目的:** 验证 P/D 分离架构和 Dynamo KV 缓存感知路由在特定场景下的优化效果。
    *   **方法:**
        *   **长 Prompt 场景:** 使用主要包含长 Prompt (>500 tokens) 的数据集进行测试，观察 P 节点（计算密集）和 D 节点（访存密集）的 GPU 资源利用率是否符合预期，对比与短 Prompt 场景的资源消耗模式。
        *   **多轮对话场景:** 使用模拟多轮对话的数据集，对比启用 Dynamo KV 缓存感知路由与（如果可能）禁用或使用简单轮询路由时的 TTFT 和端到端延迟。
    *   **观测重点:** TTFT, 端到端延迟, P/D 节点各自的 GPU 计算利用率和显存带宽使用率, Dynamo Router KV 缓存命中率指标 (如果 Dynamo 提供)。
*   **场景 3: 压力与稳定性测试**
    *   **目的:** 评估服务在持续高负载下的稳定运行能力和资源消耗情况。
    *   **方法:** 选择接近容量拐点（例如场景 1 中确定的拐点负载的 80%-90%）的并发水平，使用混合数据集持续运行负载至少 1-2 小时。
    *   **观测重点:** 请求成功率是否保持稳定 (接近 100%)，延迟指标是否在可接受范围内波动，GPU/CPU/内存使用是否存在持续、无限制增长（可能表明资源泄漏），检查所有组件（Dynamo, vLLM, API 服务）的日志中是否有错误、异常或警告信息。
*   **场景 4: 弹性伸缩能力测试**
    *   **目的:** 验证 Dynamo Planner 与 RunPod API 集成实现的自动伸缩功能是否按预期工作。
    *   **方法:**
        *   **增压测试 (Scale-Up):** 从低并发负载开始，模拟流量的快速增长（例如，每 5 分钟增加 50% 的并发用户），观察 Dynamo Planner 是否及时检测到负载增加并通过 RunPod API 创建新的 P/D Pod 对 (保持 1:1 比例)。记录从负载增加到新 Pod 可用并开始处理请求的时间。
        *   **减压测试 (Scale-Down):** 从高并发负载开始，模拟流量的快速下降（例如，每 10 分钟减少 50% 的并发用户），观察 Dynamo Planner 是否在满足冷却期 (Cooldown Period) 后，识别到冗余资源并调用 RunPod API 销毁多余的 P/D Pod 对。
    *   **观测重点:** Grafana 仪表板上活跃 P/D 节点数量的变化曲线，RunPod 控制台/API 返回的 Pod 状态变化，系统整体 QPS/TPS 在伸缩过程中的响应情况。

**3.3 性能观测指标 (通过 Grafana Dashboard 集中查看):**

*   **端到端请求指标 (来自 Load Tester & API Service):**
    *   QPS (Queries Per Second - 实际处理速率)
    *   请求成功率 (%)
    *   端到端延迟分布 (Histogram / Percentiles: P50, P90, P99)
    *   TTFT (Time To First Token) 分布 (Histogram / Percentiles)
    *   ITL (Inter-Token Latency - 平均毫秒/Token)
*   **吞吐量指标:**
    *   总 TPS (Tokens Per Second - 系统整体)
    *   平均 Per-GPU TPS (区分 P 节点和 D 节点，理解各自贡献)
*   **vLLM & GPU 指标 (每个 P/D 节点):**
    *   GPU 计算利用率 (%) (来自 `nvidia-dcgm-exporter`)
    *   GPU 显存使用量 (MB/GB) & 显存利用率 (%) (来自 `nvidia-dcgm-exporter`)
    *   GPU 显存带宽利用率 (%) (若 DCGM exporter 提供)
    *   vLLM KV Cache 占用率 (% 或 blocks) (来自 vLLM metrics endpoint)
    *   vLLM 处理队列长度 (Running / Waiting Sequences) (来自 vLLM metrics endpoint)
*   **NVIDIA Dynamo 指标:**
    *   Router 请求队列长度
    *   Router KV Cache 命中/未命中率 (%) (若 Dynamo 暴露此指标)
    *   Planner 调度的活跃 P/D Worker 数量
    *   Planner 调度延迟 (ms) (若 Dynamo 暴露此指标)
    *   (若启用) KV Cache 卸载/加载活动量和延迟 (若 Dynamo 暴露)
*   **系统资源指标 (每个 Pod):**
    *   CPU 利用率 (%) (来自 Node Exporter)
    *   容器 RAM 使用量 (MB/GB) (来自 Node Exporter / cAdvisor)
    *   网络发送/接收速率 (MB/s) (来自 Node Exporter)

**3.4 结果分析与调优:**

*   **瓶颈识别:** 基于观测到的指标数据，定位性能瓶颈：
    *   **高延迟/长 TTFT:** 可能是 P 节点计算饱和、vLLM 调度延迟高、Dynamo Router 处理慢、网络传输慢（特别是 KV 缓存传递）或 API 服务本身处理慢。
    *   **低 TPS:** 可能是 D 节点访存/计算瓶颈、vLLM KV 缓存容量不足导致频繁换出、ITL 过高（模型生成速度慢）或并发处理能力受限。
    *   **显存不足 (OOM):** vLLM 配置参数 (`gpu_memory_utilization`, `max_num_seqs`) 设置过高，或模型本身对显存需求超出预期。
    *   **高错误率:** 检查导致错误的组件日志（API 服务、Dynamo、vLLM），可能是配置错误、资源耗尽、代码 Bug 或网络问题。
    *   **伸缩不及时:** Planner 的监控间隔、触发阈值设置不当，RunPod Pod 启动/销毁本身耗时，或 RunPod API 调用失败。
*   **调优策略:**
    *   **vLLM 参数:** 调整 `gpu_memory_utilization`（权衡显存占用与并发能力），`max_num_seqs`（直接影响并发请求数），`max_model_len`（支持的最大序列长度），尝试启用/禁用特定 vLLM 优化选项。
    *   **Dynamo 配置:** 调整 Router/Planner 的线程数、队列大小等参数，优化 KV 缓存管理策略（如调整卸载阈值和目标），调整伸缩策略的敏感度（阈值、冷却时间）。
    *   **资源调整 (RunPod):** 更换性能更强的 GPU 类型（例如更高显存带宽或计算能力），增加 P/D 节点对的数量（水平扩展）。
    *   **模型优化:** 如果可行且业务允许，探索使用量化版本的 DeepSeek R1 模型（如 AWQ, GPTQ, FP8），这可以显著降低显存占用并可能加速计算，但需 vLLM 支持且可能轻微影响模型精度。
    *   **网络优化:** 检查 RunPod Pod 间网络延迟和带宽，考虑网络配置优化（若 RunPod 提供相关选项）。
*   **迭代过程:** 每次应用调优措施后，重新执行相关的性能测试场景，对比关键指标的变化，验证优化效果。持续迭代，直到满足预设的性能目标（如 P99 延迟 < X ms, 系统 TPS > Y tokens/s）或在成本预算内达到最佳性能。

---

**最终交付物:**

*   生产环境可用的 Docker 镜像（包括 `dynamo-vllm-worker`, `dynamo-controller`, `api-service`）及其对应的 Dockerfile。
*   部署所需的配置文件和脚本（例如，用于配置 Dynamo 的 YAML 文件，启动 P/D 节点和 API 服务的脚本，可能的 Kubernetes YAML 文件）。
*   用于负载测试的 Locustfile (或 k6 脚本) 和测试数据集。
*   一份详细的性能测试报告，包含测试场景、配置、关键结果图表和瓶颈分析。
*   Grafana Dashboard 的 JSON 定义文件，方便导入和复现监控视图。
*   一份全面的运维手册，包括：
    *   详细的部署步骤。
    *   关键组件的配置说明。
    *   监控指标的解释和推荐的告警阈值。
    *   常见问题的排查指南 (FAQ)。
    *   服务更新和维护流程。

--- 