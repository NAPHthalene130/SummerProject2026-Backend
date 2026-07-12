# AutoDL 云端部署指南

## 架构

```
浏览器/Android App  →  AutoDL 实例 (GPU + FastAPI + YOLO)  →  摄像头 RTSP
                              ↕
                          MySQL (可选，可复用本地数据库)
```

## 部署步骤

### 1. 创建 AutoDL 实例

在 autodl.com 选择 GPU 实例（推荐 RTX 4090 / A100），镜像选择 **PyTorch 2.x + CUDA 12.x**。

### 2. 上传代码

```bash
# 在 AutoDL Jupyter 终端或 SSH 中：
git clone https://github.com/NAPHthalene130/SummerProject2026-Backend.git
cd SummerProject2026-Backend
git checkout Experiment2
```

### 3. 一键部署

```bash
bash scripts/deploy_autodl.sh
```

脚本会自动安装依赖、检测 GPU、启动 API。

### 4. 配置文件

编辑 `~/summer2026-config/config.yaml` 配置摄像头和数据库。

### 5. 访问

- **API 文档**: `https://connect.6006.autodl.team/docs`
- **健康检查**: `https://connect.6006.autodl.team/health`
- **视频流**: 前端直接连接上述地址，WebRTC 自动协商

### 6. 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SP2026_PORT` | `8000` | API 端口 |
| `SP2026_ENABLE_STREAMING` | `true` | 启用视频流 |
| `SP2026_ENABLE_TRAFFIC_ANALYST` | `true` | 启用交通分析 |
| `SP2026_ICE_SERVERS` | 空 | TURN/STUN 服务器配置 |
| `SP2026_CORS_ORIGINS` | `["*"]` | CORS 跨域 |

### 7. WebRTC 视频流

在云环境中，WebRTC 需要 STUN/TURN 服务器穿透 NAT：

**免费 STUN**（直接可用，无需配置）：
```
SP2026_ICE_SERVERS=stun:stun.l.google.com:19302
```

**TURN 服务器**（如自建 coturn）：
```
SP2026_ICE_SERVERS=turn:your-turn.com:3478,username=user,credential=pass
```

多个服务器用 `;` 分隔：
```
SP2026_ICE_SERVERS=stun:stun.l.google.com:19302;turn:turn.example.com:3478,username=user,credential=pass
```

### 8. Docker 部署（可选）

```bash
docker build -t summer2026-api .
docker run --gpus all -p 8000:8000 summer2026-api
```
