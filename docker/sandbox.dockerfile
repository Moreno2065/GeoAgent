# =============================================================================
# GeoAgent Sandbox Docker 镜像
# =============================================================================
# 用途：在独立容器中执行 LLM 生成的 Python 代码，实现与宿主机的物理隔离。
#
# 构建：
#   docker build -f docker/sandbox.dockerfile -t geoagent-sandbox:latest .
#
# 运行（配合 docker-compose.yml）：
#   docker compose -f docker/docker-compose.yml up -d
#
# 安全设计：
#   - 非 root 用户 (geoagent) 运行代码
#   - 只读文件系统 + 白名单写入目录（/workspace/outputs）
#   - 网络隔离（无 INTERNET 网络，仅 host 或自定义内网）
#   - 内存限制 512MB，CPU 限制 1 核
#   - 资源限制通过 docker-compose.yml 的 resources 字段指定
# =============================================================================

FROM python:3.11-slim

# ---- 安全加固：创建非 root 用户 ----
RUN groupadd --gid 1000 geoagent \
    && useradd --uid 1000 --gid geoagent --shell /bin/bash geoagent

# ---- 安装最小运行时依赖 ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgdaljni3 \
        libproj22 \
        libgeos-c1v5 \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=geoagent:geoagent py_requirements.txt /tmp/py_requirements.txt
RUN pip install --no-cache-dir -r /tmp/py_requirements.txt \
    && rm /tmp/py_requirements.txt

# ---- 工作目录结构 ----
WORKDIR /workspace
RUN mkdir -p /workspace/outputs /workspace/uploads && \
    chown -R geoagent:geoagent /workspace

# ---- 写输出权限（仅 outputs 目录） ----
RUN chmod 755 /workspace/outputs

# ---- 限制可用系统命令（删除危险二进制文件） ----
RUN rm -f /usr/bin/wget /usr/bin/curl /usr/bin/nc /usr/bin/netcat \
           /usr/bin/python3 /usr/local/bin/python3 \
           /usr/sbin/reboot /usr/sbin/poweroff /usr/sbin/shutdown \
           /usr/bin/apt* /usr/bin/dpkg* 2>/dev/null || true

# ---- 复制沙盒服务端代码 ----
COPY --chown=geoagent:geoagent src/geoagent/sandbox/ /workspace/sandbox/
USER geoagent

# ---- 资源限制在 docker-compose.yml 中指定 ----
# 容器层面不设限制（由 docker-compose 统一管理）

EXPOSE 8765

ENV PYTHONUNBUFFERED=1
ENV SANDBOX_PORT=8765
ENV SANDBOX_OUTPUT_DIR=/workspace/outputs
ENV SANDBOX_MAX_BYTES=536870912   # 512 MB

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import socket; s=socket.socket(); s.connect(('localhost', 8765)); s.close()" || exit 1

# ---- 默认启动沙盒服务 ----
CMD ["python", "-m", "sandbox.server"]
