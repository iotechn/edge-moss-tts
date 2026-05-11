# MOSS-TTS-Nano ONNX：C++ HTTP 服务

本仓库提供基于 **ONNX Runtime C++ API** 的 **MOSS-TTS-Nano** 推理与极简 HTTP 接口，对应上游官方 ONNX 部署说明：[OpenMOSS/MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano)（`infer_onnx.py` / `app_onnx.py` 使用的同一套权重与 `browser_poc_manifest.json`）。

## 可行性结论（简要） 

- **可以**：官方已发布 **MOSS-TTS-Nano-100M-ONNX** 与 **MOSS-Audio-Tokenizer-Nano-ONNX**，推理阶段不依赖 PyTorch，ONNX Runtime（含 C++）可直接加载 `.onnx` 与外部 `.data`。
- **本实现默认路径**：与上游默认一致，`manifest["generation_defaults"]["sample_mode"] == "fixed"` 时使用 **`moss_tts_local_fixed_sampled_frame.onnx`** 逐帧生成音频 token，再配合 **`moss_tts_decode_step.onnx`** 更新全局隐藏状态与 KV；解码侧使用 **`moss_audio_tokenizer_decode_full.onnx`** 一次性还原波形。
- **与 Python 版本的差异**：未接入 **WeTextProcessing** 等重型文本规范化管线，仅做空白折叠与简单中英标点处理；追求与官方 demo 完全一致时，建议继续用上游 `app_onnx.py`，或以本服务为骨架自行接入规范化库。

## 目录布局（模型）

推荐使用与上游一致的目录（`browser_poc_manifest.json` 内 `codec_meta` 为相对路径）：

```text
models/
├── MOSS-TTS-Nano-100M-ONNX/
│   ├── browser_poc_manifest.json   # 或在 models/ 根目录放一份
│   ├── tts_browser_onnx_meta.json
│   ├── tokenizer.model
│   ├── moss_tts_prefill.onnx
│   ├── moss_tts_global_shared.data
│   └── … 其他 onnx / data …
└── MOSS-Audio-Tokenizer-Nano-ONNX/
    ├── codec_browser_onnx_meta.json
    ├── moss_audio_tokenizer_encode.onnx
    └── … 其他 onnx / data …
```

权重可从 Hugging Face 获取：

- [OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX)
- [OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX)

也可使用 `git clone` / `huggingface-cli download` 将上述两个目录放到本地的 `models/` 下。

```bash
# 如果是国内环境，才需要HF_ENDPOINT，Github Action不需要
HF_ENDPOINT=https://hf-mirror.com hf download OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX --local-dir models/MOSS-TTS-Nano-100M-ONNX
HF_ENDPOINT=https://hf-mirror.com hf download OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX --local-dir models/MOSS-Audio-Tokenizer-Nano-ONNX
```

## Docker 镜像（FaaS / 无外挂存储）

`Dockerfile` 在构建阶段将 **`models/` 完整 `COPY` 到镜像内 `/models`**，默认 `CMD` 使用 `--model-dir /models`。服务进程**不会在运行时从网络拉取权重**，因此 **无需挂载 PVC、对象存储或宿主机目录**，适合希望 **冷启动尽量快** 的 FaaS（仅剩容器拉镜像与本机读盘加载 ONNX 的开销）。

**GitHub Actions**：在 **`ubuntu-latest` Runner** 上 `actions/checkout`（`lfs: true`）并执行 **`git lfs pull`**，随后在流水线内 `docker build`——**不使用你的个人电脑上的 `models/`**。合并进 `main` 触发的构建镜像 job 与本地无关。

**本地**镜像调试：请安装 [Git LFS](https://git-lfs.com/) 并 `git lfs pull`，确认 **`du -sh models` 约 700MB+** 且 **`find models -type l` 为空** 后再 `docker build`。

若镜像体积仍只有约 **450–500MB**，几乎都是 **`docker build` 上下文里仍是 LFS 指针**（未拉实体）。请在仓库根执行 `git lfs pull`，或用 `head -1 models/MOSS-TTS-Nano-100M-ONNX/moss_tts_global_shared.data` 自查：若出现 `version https://git-lfs.github.com/spec/v1` 即未拉全。云厂商「仅 Dockerfile 构建」若默认 `git clone` 不带 LFS，需在构建步骤显式启用 LFS 或改用含实体 `models/` 的上下文。

### Docker 构建与运行示例

在仓库根目录构建（`linux/amd64`，与 CI 一致；Apple Silicon 本机可先 `docker build --platform linux/amd64 …`）：

```bash
docker build -t moss-tts-nano:onnx .
```

前台运行并将容器内 **18083** 映射到本机（镜像默认监听 `0.0.0.0:18083`，无需额外挂载 `models`）：

```bash
docker run --rm -p 18083:18083 moss-tts-nano:onnx
```

后台常驻并命名容器：

```bash
docker run -d --name moss-tts -p 18083:18083 moss-tts-nano:onnx
```

覆盖默认线程数等启动参数时，在镜像名后写明完整命令（会替换镜像内的 `CMD`）：

```bash
docker run --rm -p 18083:18083 moss-tts-nano:onnx \
  moss_tts_onnx_server --model-dir /models --host 0.0.0.0 --port 18083 --threads 8
```

健康检查与合成 WAV（宿主机执行）：

```bash
curl -sS http://127.0.0.1:18083/health
curl -sS -X POST 'http://127.0.0.1:18083/tts' \
  --data-urlencode 'text=欢迎使用 MOSS-TTS-Nano ONNX 服务。' \
  --data-urlencode 'voice=Junhao' \
  -o out.wav
```

调试时如需用宿主机上的 `models/` 覆盖镜像内权重，可加只读挂载（一般不必）：

```bash
docker run --rm -p 18083:18083 -v "$(pwd)/models:/models:ro" moss-tts-nano:onnx \
  moss_tts_onnx_server --model-dir /models --host 0.0.0.0 --port 18083 --threads 4
```

## 依赖

- **CMake** ≥ 3.16，**C++17**
- **ONNX Runtime**：已通过 **pkg-config** 找到 `libonnxruntime`（macOS 示例：`brew install onnxruntime`）
- **SentencePiece**：`brew install sentencepiece`
- 构建时会自动拉取 **nlohmann/json**、**cpp-httplib**（需网络）

## 编译

```bash
cd cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

生成的可执行文件：`cpp/build/moss_tts_onnx_server`。

若 pkg-config 找不到 ONNX Runtime，请先安装并确保 `PKG_CONFIG_PATH` 包含其 `.pc` 所在目录（Homebrew 一般会配置好）。

## 运行

```bash
export MOSS_TTS_MODEL_DIR=/path/to/models   # 或使用 --model-dir
./build/moss_tts_onnx_server --model-dir /path/to/models --host 127.0.0.1 --port 18083 --threads 4
```

参数说明：

| 参数 | 含义 |
|------|------|
| `--model-dir` | 含 `browser_poc_manifest.json` 的目录（或其上级 `models`，见引擎内候选路径） |
| `--host` | 监听地址，默认 `127.0.0.1` |
| `--port` | 端口，默认 `18083` |
| `--threads` | ONNX Runtime intra-op 线程数 |

## HTTP 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查，`{"ok":true}` |
| GET | `/voices` | 列出 `manifest` 内的内置音色 JSON |
| POST | `/tts` | 表单 `application/x-www-form-urlencoded`，返回 `audio/wav` |

### `/tts` 表单字段

| 字段 | 必填 | 说明 |
|------|------|------|
| `text` | 是 | 合成文本 |
| `voice` | 否 | 内置音色名（见 `/voices`）；省略则用列表第一个 |
| `prompt_wav` | 否 | **服务器本机路径** 的参考 WAV（16-bit PCM 或 float32；会自动线性重采样到 48 kHz、按模型通道复制/混音）。用于音色克隆时请确保路径对服务端进程可读 |
| `seed` | 否 | RNG 种子（整数），默认 `1234` |
| `max_new_frames` | 否 | 覆盖 manifest 中的最大生成帧数 |

### 调用示例

```bash
curl -sS -X POST 'http://127.0.0.1:18083/tts' \
  --data-urlencode 'text=欢迎使用 MOSS-TTS-Nano ONNX 服务。' \
  --data-urlencode 'voice=Junhao' \
  -o out.wav
```

指定本机参考音频（路径需在服务端存在）：

```bash
curl -sS -X POST 'http://127.0.0.1:18083/tts' \
  --data-urlencode 'text=Hello from ONNX Runtime C++.' \
  --data-urlencode 'prompt_wav=/absolute/path/to/reference.wav' \
  -o out_clone.wav
```

## 说明与限制

- 当前 HTTP 实现为 **单进程、请求互斥调用引擎**，适合演示与轻量集成；高并发请做多进程或队列。
- **`prompt_wav` 按路径读取** 便于本地部署；若需上传二进制文件，可自行改用 multipart 或前置对象存储。
- 若上游 manifest 改为 **非 `fixed`** 采样路径（例如依赖 `local_cached_step` / `local_greedy_frame` 组合），需要在本仓库 `engine.cpp` 中补齐对应分支（逻辑可参考上游 `ort_cpu_runtime.py`）。

## 许可证

模型与上游代码许可以 [MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano) 仓库为准；本 C++ 示例代码可按你需要自行约定许可。
