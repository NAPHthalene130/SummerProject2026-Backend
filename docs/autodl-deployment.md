# AutoDL 云端部署指南

## 架构

```
浏览器/Android App  →  AutoDL 实例 (GPU + FastAPI + YOLO)
                               ↕
                    校园网机器 → RTSP 中继 (Info/rtsp_relay.py)
                               ↕
                          摄像头 RTSP 流
```

## 拓扑说明

- **摄像头** 在校园网内（局域网），云端无法直接访问 RTSP
- **校园网机器**（你的本机）运行 `rtsp_relay.py`，将 RTSP 转为 HTTP MJPEG
- **AutoDL 云端** 通过 HTTP 访问校园网机器的中继端口获取视频帧
- **YOLO 推理、WebRTC 推流** 全部在云端完成

## 部署步骤

### 1. 校园网机器：启动 RTSP 中继

```bash
cd SummerEx
python Info/rtsp_relay.py
```

中继默认监听 `http://0.0.0.0:8888`，每路摄像头可通过 `http://<本机IP>:8888/cam-{id}` 访问 MJPEG 流。

**确保本机 8888 端口能被云端访问**（需要公网 IP 或内网穿透，如 frp/ngrok）。

### 2. 创建 AutoDL 实例

在 autodl.com 选择 GPU 实例，镜像选 **PyTorch 2.x + CUDA 12.x**。

### 3. 上传代码到云端

```bash
git clone https://github.com/NAPHthalene130/SummerProject2026-Backend.git
cd SummerProject2026-Backend
git checkout Experiment2
```

### 4. 配置摄像头 URL 指向中继

编辑 `~/summer2026-config/config.yaml` 的摄像头地址：

```yaml
cameras:
  - id: "cam-01"
    name: "苏州街-海淀南路"
    url: "http://YOUR_RELAY_IP:8888/cam-01"
    # RTSP 无法直连，改为 HTTP MJPEG 中继
```

### 5. 一键部署

```bash
bash scripts/deploy_autodl.sh
```

### 6. 访问

- **API 文档**: `https://connect.6006.autodl.team/docs`
- **健康检查**: `https://connect.6006.autodl.team/health`

### 7. 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SP2026_PORT` | `8000` | API 端口 |
| `SP2026_ENABLE_STREAMING` | `true` | 启用视频流 |
| `SP2026_ICE_SERVERS` | (含 TURN) | STUN/TURN 服务器 |

### 8. 本地前端的 Vite 配置

```powershell
$env:VITE_API_TARGET="https://connect.6006.autodl.team"
npm run dev
```

## RTSP 中继详细说明

运行在 `Info/rtsp_relay.py`：

- 自动读取 `Info/cameras.yaml` 或后端 `config.yaml` 的摄像头配置
- 每路摄像头独立线程捕获帧，以 JPEG 格式缓存
- HTTP 端点 `/{cam_id}` 输出 MJPEG 流
- 可用 `http://<IP>:8888/` 查看摄像头列表
- 可用 `http://<IP>:8888/health` 检查状态
