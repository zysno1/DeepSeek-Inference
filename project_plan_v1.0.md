## DeepSeek R1 (671B) Inference Service Experiment on RunPod

**Version:** 1.1
**Objective:** Implement and evaluate a distributed inference service for the full DeepSeek R1 (671B) model on RunPod H200 GPUs. The architecture utilizes vLLM with Ray for cross-node model parallelism and NVIDIA Dynamo for higher-level request scheduling, aiming for a logically disaggregated Prefill/Decode (P/D) serving pattern.
**Platform:** RunPod Cloud
**Target GPU:** NVIDIA H200 (141GB VRAM)
**Target Model:** DeepSeek R1 671B

**Critical Assumption:** This plan assumes the availability of **stable, vLLM-compatible FP8 weights** for DeepSeek R1 (671B), accessible via Hugging Face Hub (e.g., `deepseek-ai/DeepSeek-R1`). FP8 weights (approx. 671 GB) are necessary to fit the model within feasible H200 VRAM limits. If only BF16/FP16 weights (~1.3TB+) are available, the required GPU count will double, significantly impacting feasibility and cost. **Verification of FP8 weight availability and vLLM compatibility is the absolute first step.**

---

### 1. Environment Preparation

**Goal:** Provision RunPod resources, configure software, and download model artifacts.

**1.1 RunPod Account & Access:**
    *   Ensure an active RunPod account.
    *   Generate a RunPod API Key for programmatic access (if needed for automation).
    *   Generate an SSH key pair locally (`ssh-keygen`). Note the public key content (`cat ~/.ssh/id_rsa.pub`).

**1.2 RunPod Network Volume:**
    *   Create a RunPod Network Volume in the desired region.
    *   **Required Size:** Minimum 1 TB (to accommodate ~671GB FP8 weights + tokenizer + configs + potential KV cache swap + buffer).
    *   Note the Volume ID and region.

**1.3 Model Download to Network Volume:**
    *   **Objective:** Download model weights to the persistent Network Volume.
    *   **Procedure:**
        1.  Launch a **temporary utility Pod** (any basic template, e.g., `RunPod Pytorch`).
        2.  **Attach** the Network Volume created in 1.2 during Pod creation (e.g., mount at `/workspace`).
        3.  SSH into the utility Pod.
        4.  Install necessary tools: `pip install huggingface_hub[cli]`
        5.  **(Optional) Login to Hugging Face:** `huggingface-cli login` (if model is gated).
        6.  Navigate to the volume mount point: `cd /workspace`
        7.  Create model directory: `mkdir -p models/deepseek-r1`
        8.  Navigate into directory: `cd models/deepseek-r1`
        9.  Download model files:
            ```bash
            huggingface-cli download deepseek-ai/DeepSeek-R1 \
                --local-dir . \
                --local-dir-use-symlinks False \
                --include="*.fp8" # Adjust include/exclude pattern if FP8 files have specific naming
            # Also download essential non-weight files (config.json, tokenizer*, etc.)
            # Example: huggingface-cli download deepseek-ai/DeepSeek-R1 --include="*.json" --include="*.model" --local-dir . --local-dir-use-symlinks False
            ```
        10. Verify download contents (`ls -lh .`).
        11. **Terminate** the temporary utility Pod. The model remains on the Network Volume.

**1.4 Inference Pod Definition:**
    *   **GPU Type:** H200 (141 GB).
    *   **Instance Configuration:** Target **8xH200 per Pod**. Note: The full R1 requires >1128GB (8xH200 VRAM) even with FP8 due to KV cache. Multi-node requires minimum 2 Pods (16 GPUs total). Initial validation (Section 2) uses one Pod.
    *   **Template:** Use `RunPod vLLM` template (select appropriate CUDA version, e.g., 12.1).
    *   **Network Volume:** Configure Pods to **attach** the Network Volume (from 1.2) containing the model (e.g., mount at `/workspace`). Ensure read/write access if needed for logs/cache.
    *   **Environment Variables:**
        *   `PUBLIC_KEY`: Set value to your **full** SSH public key content (for SSH access).
        *   *(Specific vLLM/Ray/Dynamo env vars will be set during deployment)*.
    *   **Ports:** Expose port `22` (SSH), `8000` (vLLM default API), potentially Ray ports (`6379`, `8265`), and Dynamo ports.

---

### 2. Single-Node Validation (8xH200 Pod)

**Goal:** Verify basic model loading, FP8 compatibility (if used), and minimal inference functionality on a single RunPod 8xH200 instance before attempting multi-node setup. Acknowledge severe KV cache limitations in this configuration.

**2.1 Launch Single Pod:**
    *   Create **one** RunPod Pod as defined in 1.4.
    *   Wait for the Pod to initialize.

**2.2 SSH into Pod:**
    *   Use the Pod's connection info (IP, Port 22) and your private SSH key.

**2.3 Install/Verify Tools:**
    *   Confirm `vllm`, `ray` (if needed for later steps), `pip`, `python` are present from the template.
    *   Install any missing dependencies if required (`pip install ...`).

**2.4 Start vLLM Server (Tensor Parallel):**
    *   Execute the vLLM API server startup command:
        ```bash
        python -m vllm.entrypoints.api_server \\
            --model /workspace/models/deepseek-r1 \\
            --tensor-parallel-size 8 \\
            --dtype float8_e4m3fn \\ # Explicitly specify FP8 dtype if applicable
            --kv-cache-dtype fp8 \\  # Specify KV cache dtype if using FP8 cache
            --gpu-memory-utilization 0.90 \\ # Adjust as needed
            --max-model-len 8192 \\ # Start with a smaller context length for validation
            --trust-remote-code \\ # Often needed for complex models
            --port 8000
            # Add other relevant vLLM args as needed
        ```
    *   Monitor logs for successful loading and potential OOM errors. Check `nvidia-smi` for VRAM usage.

**2.5 Minimal Inference Test:**
    *   From *another terminal* on the Pod (or use `curl` from local if port 8000 is forwarded/exposed):
        ```bash
        curl http://localhost:8000/v1/completions \\
        -H "Content-Type: application/json" \\
        -d '{
            "model": "/workspace/models/deepseek-r1",
            "prompt": "DeepSeek R1 is",
            "max_tokens": 10,
            "temperature": 0.1
        }'
        ```
    *   Verify a valid text completion is returned.

**2.6 Expected Outcome:**
    *   Model loads successfully on 8xH200 within the ~1128 GB VRAM (weights only ~671GB).
    *   Basic inference generates tokens.
    *   *This validates the model format compatibility and core vLLM functionality on the target hardware.*

---

### 3. Multi-Node Distributed Inference (2x Pods, 16xH200 Total)

**Goal:** Deploy DeepSeek R1 across two 8xH200 Pods (16 GPUs total) using vLLM+Ray for distributed execution, with Dynamo for request routing/scheduling.

**3.1 Launch Pod Cluster:**
    *   Create **two** RunPod Pods (Pod A, Pod B) as defined in 1.4, ensuring both mount the **same** Network Volume containing the model.
    *   Verify Pods are running and SSH access works for both.
    *   Ensure network connectivity between Pod A and Pod B (check RunPod VPC/Security Group settings if needed). Note their internal IP addresses.

**3.2 Setup Ray Cluster:**
    *   **On Pod A (Head Node):**
        ```bash
        ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265
        ```
    *   **On Pod B (Worker Node):**
        ```bash
        # Replace <Pod_A_IP> with the actual internal IP of Pod A
        ray start --address=<Pod_A_IP>:6379
        ```
    *   **Verify Cluster (on Pod A):** `ray status`. Should show 2 nodes, 16 GPUs total available.

**3.3 Start Distributed vLLM Service via Ray:**
    *   **On Pod A (or via Ray Job Submission):** Launch the vLLM API server configured for Ray distribution.
        ```bash
        python -m vllm.entrypoints.api_server \\
            --model /workspace/models/deepseek-r1 \\
            --tensor-parallel-size 16 \\ # Total GPUs across the cluster
            --distributed-executor-backend ray \\
            --dtype float8_e4m3fn \\ # Specify FP8 if used
            --kv-cache-dtype fp8 \\ # Specify FP8 if used
            --gpu-memory-utilization 0.90 \\
            --max-model-len 128000 \\ # Target full context length
            --trust-remote-code \\
            --port 8000
            # Add other relevant vLLM/Ray args
        ```
    *   Monitor Ray Dashboard (port 8265 on Pod A, potentially needs port forwarding) and vLLM logs on both Pods for successful distributed startup. Check VRAM usage on all 16 GPUs.

**3.4 Dynamo Setup & Integration (Conceptual):**
    *   *(Dynamo installation/setup details depend heavily on its specific architecture and are omitted here. Assume Dynamo components (Planner, Router, Agent) are installed/available.)*
    *   Deploy Dynamo Planner and Router components (potentially on separate CPU Pods or alongside vLLM Pods if resources allow).
    *   Configure Dynamo Router to forward requests to the vLLM+Ray service endpoint (e.g., Pod A's IP on port 8000, assuming internal load balancing or direct access).
    *   Configure Dynamo Planner with RunPod credentials and policies for potential auto-scaling (if desired).
    *   Configure Dynamo's scheduling logic to implement the desired P/D request handling pattern (e.g., prioritizing P-phase, managing D-phase iterations). *This is the "logical" P/D separation.*

**3.5 Service Test:**
    *   Send inference requests to the **Dynamo Router's entry point**.
    *   Verify requests are correctly processed by the 16-GPU vLLM+Ray cluster and results are returned via Dynamo.
    *   Check logs on Dynamo components, Pod A, and Pod B.

---

### 4. Performance Testing & Evaluation

**Goal:** Quantitatively measure the throughput, latency, and cost of the deployed 16xH200 distributed service.

**4.1 Test Setup:**
    *   **Tool:** Use Locust or a similar load generation tool. Deploy Locust workers on CPU instances in the same RunPod region.
    *   **Target:** Send requests to the Dynamo Router endpoint.
    *   **Metrics Collection:** Ensure monitoring stack (Prometheus, Grafana, potentially Loki) is collecting metrics from vLLM, Ray, GPUs (DCGM-Exporter), Dynamo, and the load generator.

**4.2 Key Performance Indicators (KPIs):**
    *   **Throughput:**
        *   Requests Per Second (RPS).
        *   Output Tokens Per Second (TPS) - aggregate across all concurrent requests.
    *   **Latency:**
        *   Time To First Token (TTFT) distribution (P50, P90, P99).
        *   Time Per Output Token (TPOT) / Inter-Token Latency (ITL) distribution (P50, P90, P99).
        *   End-to-End Request Latency distribution (P50, P90, P99).
    *   **Resource Utilization:**
        *   GPU VRAM Usage per GPU (%).
        *   GPU Compute Utilization per GPU (%).
        *   Network Bandwidth between Pods (if measurable).
        *   Ray Cluster resource usage.
    *   **Cost:**
        *   RunPod Cost per Hour (`2 * cost_of_8xH200_pod/hr` + CPU pods).
        *   Estimated Cost per Million Output Tokens.

**4.3 Test Scenarios:**
    *   **Varying Concurrency:** Ramp up concurrent users sending requests with moderate prompt/output lengths. Identify saturation point. Measure KPIs at different concurrency levels.
    *   **Varying Input/Output Length:** Test scenarios with short prompts/long outputs, long prompts/short outputs, and long prompts/long outputs (approaching 128K context if feasible). Analyze impact on TTFT, TPOT, and TPS.
    *   **Varying Batch Size (Implicit):** Monitor how vLLM's internal batching adapts under different concurrency/request patterns.
    *   **Stress Test:** Run sustained load near saturation point for an extended period (e.g., 1 hour) to check stability and resource leakage.

**4.4 Evaluation:**
    *   Analyze collected metrics and Grafana dashboards.
    *   Compare performance against project goals or benchmarks.
    *   Identify primary bottlenecks (e.g., VRAM capacity for KV cache, inter-GPU communication bandwidth via Ray, compute limits, Dynamo scheduling overhead).
    *   Calculate cost-effectiveness metrics.
    *   Document findings and potential optimization areas.

---