# MOSS-TTS-Nano ONNX HTTP 服务 — linux/amd64（与 GitHub Actions platforms 一致）
#
# FaaS / 冷启动：权重在构建期 COPY 进 /models，进程只读本地 ONNX，不做运行时下载；默认无需 PVC、无需挂载对象存储。
# CI 在合并构建前把实体模型放进构建上下文的 models/；本地 docker build 前也需自备 models/（勿用指向 HF 缓存的 symlink）。
# CI 已 pin huggingface_hub>=1 并校验体积与无 symlink，避免「镜像极小但缺权重」的假镜像。

ARG ONNXRUNTIME_VERSION=1.19.2

# -----------------------------------------------------------------------------
FROM ubuntu:22.04 AS builder

ARG ONNXRUNTIME_VERSION
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    libsentencepiece-dev \
    pkg-config \
    xz-utils \
  && rm -rf /var/lib/apt/lists/*

# 勿预先创建 /opt/onnxruntime：否则 mv 会把 tarball 目录挪成 /opt/onnxruntime/onnxruntime-linux-x64-*/，
# include 将不在 prefix/include，CMake 会报 INTERFACE_INCLUDE_DIRECTORIES 路径不存在。
RUN set -eux; \
  curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz" \
    | tar xz -C /tmp; \
  mv "/tmp/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}" /opt/onnxruntime; \
  mkdir -p /opt/onnxruntime/lib/pkgconfig; \
  { \
    echo "prefix=/opt/onnxruntime"; \
    echo "exec_prefix=\${prefix}"; \
    echo "libdir=\${exec_prefix}/lib"; \
    echo "includedir=\${prefix}/include"; \
    echo ""; \
    echo "Name: libonnxruntime"; \
    echo "Description: ONNX Runtime"; \
    echo "Version: ${ONNXRUNTIME_VERSION}"; \
    echo "Libs: -L\${libdir} -lonnxruntime"; \
    echo "Cflags: -I\${includedir}"; \
  } > /opt/onnxruntime/lib/pkgconfig/libonnxruntime.pc

ENV PKG_CONFIG_PATH=/opt/onnxruntime/lib/pkgconfig

WORKDIR /src/cpp
COPY cpp/ ./

RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  && cmake --build build -j"$(nproc)"

# -----------------------------------------------------------------------------
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
    libsentencepiece0 \
  && rm -rf /var/lib/apt/lists/*

# ONNX Runtime CPU 共需要主库与 providers_shared
COPY --from=builder /opt/onnxruntime/lib/libonnxruntime.so* /usr/local/lib/
COPY --from=builder /opt/onnxruntime/lib/libonnxruntime_providers_shared.so* /usr/local/lib/
COPY --from=builder /src/cpp/build/moss_tts_onnx_server /usr/local/bin/moss_tts_onnx_server

# 完整打入镜像（无外挂存储）；布局与 README 中 models/ 一致
COPY models /models

RUN ldconfig

ENV MOSS_TTS_MODEL_DIR=/models
EXPOSE 18083

# 监听 0.0.0.0；默认使用镜像内 /models（FaaS 勿配 volume；仅调试时可挂载覆盖同路径）
CMD ["moss_tts_onnx_server", "--model-dir", "/models", "--host", "0.0.0.0", "--port", "18083", "--threads", "4"]
