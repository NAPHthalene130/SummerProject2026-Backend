# 云端部署与 Android 通信

## 架构

```text
Android 真机 -> HTTPS 域名 -> Nginx/Caddy -> FastAPI:8000 -> MySQL
```

Android 只需要访问公开的 HTTPS 域名，例如 `https://traffic-api.example.com`。不要把 MySQL 端口开放给手机。

## 1. 启动云端服务

服务器安装 Docker 和 Docker Compose 后：

```bash
cp .env.cloud.example .env
# 修改 .env 中的数据库密码
docker compose -f docker-compose.cloud.yml up -d --build
curl http://127.0.0.1:8000/health
```

默认使用 API-only 模式，不启动 RTSP、YOLO 和交通分析智能体。需要这些服务时，再将对应环境变量改为 `true` 并提供摄像头及模型配置。

CI/CD 会在 `dev`、`main` 分支测试通过后将镜像发布到 GHCR。云服务器使用已发布的不可变镜像时，在 `.env` 中设置完整镜像名（建议使用 `<branch>-<sha>` 标签），登录 GHCR 后执行：

```bash
docker compose -f docker-compose.cloud.yml pull api
docker compose -f docker-compose.cloud.yml up -d --no-build api
curl http://127.0.0.1:8000/health
```

未设置 `SP2026_API_IMAGE` 时，Compose 仍使用本地镜像名，原有的 `up -d --build` 部署方式不受影响。

## 2. 配置 HTTPS

生产环境必须为域名配置可信 HTTPS 证书。以 Caddy 为例：

```caddyfile
traffic-api.example.com {
  reverse_proxy 127.0.0.1:8000
}
```

防火墙只开放 `80`、`443` 和运维所需的 SSH 端口。`8000` 可以仅监听内网，或由安全组限制访问。

## 3. 配置 Android

打开 Android 应用的“我的 -> 云端服务器设置”，输入：

```text
https://traffic-api.example.com
```

点击“测试并保存”。应用会请求 `/health`，成功后工单列表、人员派发、状态回传和图片都使用这个地址。

也可以在构建 APK 时提供默认地址：

```powershell
.\gradlew.bat :app:assembleDebug -PTRAFFIC_API_BASE_URL=https://traffic-api.example.com
```

## 4. 云端环境变量

- `SP2026_DB_HOST`
- `SP2026_API_IMAGE`
- `SP2026_DB_PORT`
- `SP2026_DB_USERNAME`
- `SP2026_DB_PASSWORD`
- `SP2026_DB_NAME`
- `SP2026_ENABLE_STREAMING`
- `SP2026_ENABLE_TRAFFIC_ANALYST`
- `SP2026_LLM_URL`
- `SP2026_LLM_API_KEY`
- `SP2026_LLM_MODEL_NAME`

不要将真实密码、API Key 或生产 `config.yaml` 提交到 Git。
