# 云端部署与 Android 通信

## 架构

```text
Android 真机 -> HTTPS 域名 -> Nginx/Caddy -> FastAPI:8000 -> MySQL
```

Android 只需要访问公开的 HTTPS 域名，例如 `https://traffic-api.example.com`。不要把 MySQL 端口开放给手机。

## 1. 启动云端服务

服务器安装 Docker 和 Docker Compose 后：

```bash
# 1. 从模板创建 config.yaml（含 RTSP/YOLO/LLM/摄像头等全部配置）
cp config-template.yaml config.yaml
# 修改 config.yaml 中的数据库密码、API Key 等敏感信息

# 2. 创建 docker-compose 环境变量文件（仅数据库凭证和镜像名）
cat > .env << 'EOF'
MYSQL_ROOT_PASSWORD=your-strong-root-password
SP2026_DB_USERNAME=summer_project
SP2026_DB_PASSWORD=your-strong-db-password
SP2026_DB_NAME=summer_project_2026
EOF

# 3. 启动服务
docker compose -f docker-compose.cloud.yml up -d --build
curl http://127.0.0.1:8000/health
```

所有应用配置（流媒体、RTSP、YOLO、LLM、摄像头列表等）统一在 `config.yaml` 中管理。`docker-compose.cloud.yml` 仅通过环境变量注入数据库连接信息，其余配置由容器内的 `config.yaml` 提供。

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

## 4. 云端配置

所有应用配置统一在 `config.yaml` 中管理（streaming、RTSP、YOLO、LLM、Embedding、摄像头列表等）。Docker Compose 仅注入数据库连接信息：

- `MYSQL_ROOT_PASSWORD` — MySQL root 密码
- `SP2026_DB_USERNAME` / `SP2026_DB_PASSWORD` / `SP2026_DB_NAME` — 应用数据库凭证
- `SP2026_API_IMAGE` — 使用的容器镜像名（可选）

不要将真实密码、API Key 或生产 `config.yaml` 提交到 Git。
