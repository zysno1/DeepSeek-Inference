## DeepSeek R1 (671B) RunPod 推理服务实验方案

**Version:** 1.1 (中文版)
**Objective:** Implement and evaluate a distributed inference service for the full DeepSeek R1 (671B) model on RunPod H100 NVL GPUs. The architecture utilizes vLLM with Ray for cross-node model parallelism and NVIDIA Dynamo for higher-level request scheduling, aiming for a logically disaggregated Prefill/Decode (P/D) serving pattern.
**Platform:** RunPod Cloud
**Target GPU:** NVIDIA H100 NVL (94GB VRAM)
**Target Model:** DeepSeek R1 671B

**Critical Assumption:** This plan assumes the availability of **stable, vLLM-compatible FP8 weights** for DeepSeek R1 (671B), accessible via Hugging Face Hub (e.g., `deepseek-ai/DeepSeek-R1`). FP8 weights (approx. 671 GB) are necessary to fit the model within feasible H100 NVL VRAM limits. With 94GB per GPU, an 8-GPU Pod offers 752 GB total VRAM, making single-pod deployment extremely tight even for just the weights, and necessitating multi-node for KV cache. If only BF16/FP16 weights (~1.3TB+) are available, the required GPU count will double or more, significantly impacting feasibility and cost. **Verification of FP8 weight availability and vLLM compatibility is the absolute first step.**

---

### 1. 环境准备

**Goal:** Provision RunPod resources, configure software, and download model artifacts.

**1.1 RunPod 账户与访问:**
    *   Ensure an active RunPod account.
    *   Generate a RunPod API Key for programmatic access (if needed for automation).
    *   Generate an SSH key pair locally (<code>ssh-keygen</code>). Note the public key content (<code>cat ~/.ssh/id_rsa.pub</code>).

**1.2 RunPod 网络卷 (Network Volume):**
    *   Create a RunPod Network Volume in the desired region.
    *   **Required Size:** Minimum 1 TB (to accommodate ~671GB FP8 weights + tokenizer + configs + potential KV cache swap + buffer).
    *   Note the Volume ID and region.

**1.3 模型下载至网络卷:**
    *   **Objective:** 将模型权重下载到持久化的网络卷。
    *   **Procedure:**
        1.  Launch a **temporary utility Pod** (any basic template, e.g., `RunPod Pytorch`).
        2.  **Attach** the Network Volume created in 1.2 during Pod creation (e.g., mount at `/workspace`).
        3.  通过 SSH 连接到该工具 Pod。
        4.  安装必要工具: <code>pip install huggingface_hub[cli]</code>
        5.  **(可选) 登录 Hugging Face:** <code>huggingface-cli login</code> (如果模型是受限访问的)。
        6.  切换到卷挂载点: <code>cd /workspace</code>
        7.  创建模型目录: <code>mkdir -p models/deepseek-r1</code>
        8.  进入目录: <code>cd models/deepseek-r1</code>
        9.  下载模型文件:
            ```bash
            # 下载 FP8 权重 (如果 FP8 文件命名有特定模式，请调整 include/exclude 参数)
            huggingface-cli download deepseek-ai/DeepSeek-R1 \
                --repo-type model \
                --include "*.fp8" \
                --local-dir . --local-dir-use-symlinks False

            # 下载必要的非权重文件 (config.json, tokenizer*, etc.)
            huggingface-cli download deepseek-ai/DeepSeek-R1 \
                --repo-type model \
                --exclude "*.bin" --exclude "*.safetensors" --exclude "*.pt" --exclude "*.fp8" \
                --local-dir . --local-dir-use-symlinks False
            ```
        10. 验证下载内容 (<code>ls -lh .</code>)。确保 <code>config.json</code>, tokenizer 文件, 和 FP8 权重文件都存在。
        11. **Terminate** the temporary utility Pod. The model remains on the Network Volume.

**1.4 推理 Pod 定义:**
    *   **GPU Type:** H100 NVL (94 GB).
    *   **Instance Configuration:** Target **8xH100 NVL per Pod**. Note: The full R1 (FP8 weights ~671 GB) requires VRAM exceeding a single 8xH100 NVL Pod's capacity (752 GB) once KV cache and activations are considered. Multi-node requires minimum 2 Pods (16 GPUs total, 1504 GB VRAM). Initial validation (Section 2) uses one Pod but acknowledges its limitations.
    *   **Template:** Use `RunPod vLLM` template (select appropriate CUDA version, e.g., 12.1).
    *   **Network Volume:** Configure Pods to **attach** the Network Volume (from 1.2) containing the model (e.g., mount at `/workspace`). Ensure read/write access if needed for logs/cache.
    *   **Environment Variables:**
        *   `PUBLIC_KEY`: Set value to your **full** SSH public key content (for SSH access).
        *   *(Specific vLLM/Ray/Dynamo env vars will be set during deployment)*。
    *   **Ports:** 暴露端口 `22` (SSH), `8000` (vLLM 默认 API), 可能还有 Ray 端口 (`6379` head, `8265` dashboard), 以及 Dynamo 相关端口。

---

### 2. 单机验证 (8xH100 NVL Pod)

**Goal:** Verify basic model loading, FP8 compatibility (if used), and minimal inference functionality on a single RunPod 8xH100 NVL instance before attempting multi-node setup. Acknowledge severe VRAM/KV cache limitations in this configuration (752 GB total VRAM vs ~671GB FP8 weights).

**2.1 启动单个 Pod:**
    *   按照 1.4 的定义创建 **一个** RunPod Pod。
    *   等待 Pod 初始化完成。

**2.2 SSH 连接 Pod:**
    *   使用 Pod 的连接信息 (IP, 端口 22) 和您的私钥进行 SSH 连接。

**2.3 安装/验证工具:**
    *   确认模板中已包含 <code>vllm</code>, <code>ray</code> (后续步骤可能需要), <code>pip</code>, <code>python</code>。
    *   如果需要，安装缺失的依赖 (<code>pip install ...</code>)。

**2.4 启动 vLLM 服务 (张量并行):**
    *   执行 vLLM API 服务器启动命令:
        ```bash
        python -m vllm.entrypoints.api_server \
            --model /workspace/models/deepseek-r1 \
            --tensor-parallel-size 8 \
            --dtype float8_e4m3fn \
            --kv-cache-dtype fp8 \
            --gpu-memory-utilization 0.85 \
            --max-model-len 8192 \
            --trust-remote-code \
            --port 8000
            # 根据需要添加其他相关 vLLM 参数 (注意降低 memory utilization)
        ```
    *   监控日志，检查是否成功加载，有无 OOM 错误。使用 <code>nvidia-smi</code> 检查 VRAM 使用情况。

**2.5 最低限度推理测试:**
    *   在 Pod 上的 *另一个终端* (或如果 8000 端口已转发/暴露，可从本地使用 <code>curl</code>):
        ```bash
        curl http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "/workspace/models/deepseek-r1",
            "prompt": "DeepSeek R1 is",
            "max_tokens": 10,
            "temperature": 0.1
        }'
        ```
    *   验证是否返回了有效的文本补全。

**2.6 预期结果:**
    *   Model loads successfully on 8xH100 NVL within the ~752 GB VRAM (weights only ~671GB), leaving very little room for KV cache.
    *   Basic inference generates tokens (likely only possible for very short sequences/low batch size).
    *   *This validates the model format compatibility and core vLLM functionality on the target hardware, highlighting the necessity of multi-node.*

---

### 3. 多机分布式推理 (2x Pods, 共 16xH100 NVL)

**Goal:** Deploy DeepSeek R1 across two 8xH100 NVL Pods (16 GPUs total, 1504 GB VRAM) using vLLM+Ray for distributed execution, with Dynamo for request routing/scheduling.

**3.1 启动 Pod 集群:**
    *   按照 1.4 的定义创建 **两个** RunPod Pod (Pod A, Pod B)，确保两者都挂载 **同一个** 包含模型的网络卷。
    *   验证 Pod 均在运行，且可通过 SSH 访问。
    *   确保 Pod A 和 Pod B 之间网络互通 (需要时检查 RunPod VPC/安全组设置)。记录它们的内部 IP 地址。

**3.2 Setup Ray Cluster:**
    *   **在 Pod A (Head 节点) 上:**
        ```bash
        ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265
        ```
    *   **在 Pod B (Worker 节点) 上:**
        ```bash
        # 将 <Pod_A_IP> 替换为 Pod A 的实际内部 IP
        ray start --address=<Pod_A_IP>:6379
        ```
    *   **验证集群 (在 Pod A 上):** <code>ray status</code>。应显示有 2 个节点，共 16 块可用 GPU。

**3.3 通过 Ray 启动分布式 vLLM 服务:**
    *   **在 Pod A 上 (或通过 Ray Job 提交):** 启动配置为 Ray 分布式的 vLLM API 服务器。
        ```bash
        python -m vllm.entrypoints.api_server \
            --model /workspace/models/deepseek-r1 \
            --tensor-parallel-size 16 \
            --distributed-executor-backend ray \
            --dtype float8_e4m3fn \
            --kv-cache-dtype fp8 \
            --gpu-memory-utilization 0.90 \
            --max-model-len 128000 \
            --trust-remote-code \
            --port 8000
            # 添加其他相关 vLLM/Ray 参数
        ```
    *   Monitor Ray Dashboard (Pod A 的 8265 端口，可能需要端口转发) 以及两个 Pod 上的 vLLM 日志，确认分布式启动成功。检查所有 16 块 GPU 的 VRAM 使用情况。

**3.4 Dynamo 设置与集成 (概念性):**
    *   *(Dynamo 的具体安装/设置细节高度依赖其架构，此处省略。假定 Dynamo 组件 (Planner, Router, Agent) 已安装/可用。)*
    *   部署 Dynamo Planner 和 Router 组件 (可在单独的 CPU Pod 上，或资源允许时与 vLLM Pod 共存)。
    *   配置 Dynamo Router 将请求转发到 vLLM+Ray 服务的端点 (例如，Pod A 的 IP 地址和 8000 端口，假定有内部负载均衡或直接访问)。
    *   配置 Dynamo Planner 使用 RunPod 凭证和策略以实现可能的自动伸缩 (如果需要)。
    *   配置 Dynamo 的调度逻辑以实现所需的 P/D 请求处理模式 (例如，优先处理 P 阶段，管理 D 阶段迭代)。*这是"逻辑上的"P/D 分离。*

**3.5 服务测试:**
    *   向 **Dynamo Router 的入口点** 发送推理请求。
    *   验证请求被 16-GPU 的 vLLM+Ray 集群正确处理，并通过 Dynamo 返回结果。
    *   检查 Dynamo 组件、Pod A 和 Pod B 上的日志。

---

### 4. 性能测试与评估

**Goal:** 定量测量部署的 16xH100 NVL 分布式服务的吞吐量、延迟和成本。

**4.1 Test Setup:**
    *   **Tool:** Use Locust or a similar load generation tool. Deploy Locust workers on CPU instances in the same RunPod region.
    *   **Target:** Send requests to the Dynamo Router endpoint.
    *   **Metrics Collection:** Ensure monitoring stack (Prometheus, Grafana, potentially Loki) is collecting metrics from vLLM, Ray, GPUs (DCGM-Exporter), Dynamo, and the load generator.

**4.2 Key Performance Indicators (KPIs):**
    *   **吞吐量:**
        *   每秒请求数 (RPS)。
        *   每秒输出 Token 数 (TPS) - 所有并发请求的总和。
    *   **延迟:**
        *   首个 Token 时间 (TTFT) 分布 (P50, P90, P99)。
        *   单个输出 Token 时间 (TPOT) / Token 间延迟 (ITL) 分布 (P50, P90, P99)。
        *   端到端请求延迟分布 (P50, P90, P99)。
    *   **资源利用率:**
        *   单块 GPU VRAM 使用率 (%)。
        *   单块 GPU 计算利用率 (%)。
        *   Pod 间网络带宽 (如果可测量)。
        *   Ray 集群资源使用情况。
    *   **成本:**
        *   每小时 RunPod 成本 (<code>2 * 8xH100_NVL_pod每小时费用</code> + CPU Pod 费用)。
        *   估算的每百万输出 Token 成本。

**4.3 测试场景:**
    *   **Varying Concurrency:** Ramp up concurrent users sending requests with moderate prompt/output lengths. Identify saturation point. Measure KPIs at different concurrency levels.
    *   **Varying Input/Output Length:** Test scenarios with short prompts/long outputs, long prompts/short outputs, and long prompts/long outputs (approaching 128K context if feasible). Analyze impact on TTFT, TPOT, and TPS.
    *   **Varying Batch Size (Implicit):** Monitor how vLLM's internal batching adapts under different concurrency/request patterns.
    *   **Stress Test:** Run sustained load near saturation point for an extended period (e.g., 1 hour) to check stability and resource leakage.

**4.4 评估:**
    *   Analyze collected metrics and Grafana dashboards.
    *   Compare performance against project goals or benchmarks.
    *   Identify primary bottlenecks (e.g., VRAM capacity for KV cache, inter-GPU communication bandwidth via Ray, compute limits, Dynamo scheduling overhead).
    *   Calculate cost-effectiveness metrics.
    *   Document findings and potential optimization areas.

---