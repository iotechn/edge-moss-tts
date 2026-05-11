# MOSS-TTS-Nano ONNX HTTP 服务 — linux/amd64（与 GitHub Actions platforms 一致）
# CI 在构建前下载 models/ 并 COPY 进镜像；本地单独构建时需先有 models/ 目录。

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

RUN set -eux; \
  mkdir -p /opt/onnxruntime/lib/pkgconfig; \
  curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz" \
    | tar xz -C /tmp; \
  mv "/tmp/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}" /opt/onnxruntime; \
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

# 与 README 一致的目录布局（Action 中下载到构建上下文的 models/）
COPY models /models

RUN ldconfig

ENV MOSS_TTS_MODEL_DIR=/models
EXPOSE 18083

# 监听 0.0.0.0；模型已在镜像内 /models，也可用挂载覆盖该目录
CMD ["moss_tts_onnx_server", "--model-dir", "/models", "--host", "0.0.0.0", "--port", "18083", "--threads", "4"]
