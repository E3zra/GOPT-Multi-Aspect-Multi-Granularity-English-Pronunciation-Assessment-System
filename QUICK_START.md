# 🚀 GOPT 快速开始指南 - 持久化版本

## 📌 新特性

✅ **完全持久化**: 重建 Docker 镜像不会丢失配置  
✅ **自动设置**: 容器启动时自动创建模型链接  
✅ **修复集成**: 所有 bug 修复已内置到镜像中  
✅ **开箱即用**: 无需手动配置，直接使用  

---

## 🎯 一分钟快速测试

```bash
# 1. 重建镜像（包含所有修复）
docker-compose build

# 2. 启动容器
docker-compose up -d

# 3. 测试你的音频（修改路径和文本）
python test_user_audio.py
```

**就是这么简单！** 🎉

---

## 📂 项目结构

```
gopt/
├── docker/
│   ├── entrypoint.sh          ← 新增：自动设置脚本
│   ├── setup_models.sh         ← 模型下载脚本
│   └── ...
├── models/
│   └── librispeech/            ← LibriSpeech 模型（~2GB，持久化）
├── audio_input/                ← 放置你的音频文件
├── audio_output/               ← 处理结果输出
├── process_custom_audio_fixed.sh  ← 修复的处理脚本
├── test_user_audio.py          ← 测试脚本
├── Dockerfile                  ← 已更新：包含持久化配置
└── docker-compose.yml          ← Docker 编排文件
```

---

## 🔧 首次安装

### 步骤 1: 构建镜像

```bash
docker-compose build
```

⏱️ **时间**: 20-40 分钟（首次）

### 步骤 2: 启动容器

```bash
docker-compose up -d
```

### 步骤 3: 下载 ASR 模型

```bash
# 如果 models/librispeech/ 目录为空，运行：
docker exec gopt-pipeline bash /workspace/scripts/setup_models.sh
```

⏱️ **时间**: 10-30 分钟（仅首次，~2GB 下载）  
💾 **位置**: `./models/librispeech/` （宿主机，持久化）

### 步骤 4: 验证安装

```bash
docker logs gopt-pipeline
```

✅ **成功标志**:
```
[1/4] Setting up LibriSpeech model links...
   ✓ Model links created successfully
   ✓ tdnn_sp link verified
   ✓ lang link verified
[2/4] Checking GOPT pretrained model...
   ✓ GOPT pretrained model found
...
Initialization complete! Container ready for use.
```

---

## 🎵 测试音频

### 方法 1: 使用 Python 脚本

编辑 `test_user_audio.py` 中的音频路径：

```python
audio_file = r"D:\你的路径\音频.m4a"
transcript = "你的文本内容"
```

运行：

```bash
python test_user_audio.py
```

### 方法 2: 使用 Web 界面

```bash
# 启动 Web 服务器
python web_server.py --port 8080

# 浏览器打开
http://localhost:8080
```

上传音频文件，输入文本，点击"Process"即可。

---

## 🔄 重建镜像（所有修复都会保留）

```bash
# 停止容器
docker-compose down

# 重建镜像
docker-compose build

# 启动容器
docker-compose up -d

# 验证
docker logs gopt-pipeline
```

**注意**: 
- ✅ LibriSpeech 模型会保留（在 `./models/`）
- ✅ 所有修复自动应用（内置在镜像中）
- ✅ 容器启动时自动设置链接

---

## 📊 结果解读

### 评分标准（0-2 分制）

| 分数范围 | 等级 | 说明 |
|---------|------|------|
| 1.6-2.0 | ✅ 优秀 | 接近母语水平 |
| 1.2-1.6 | ⚠️ 良好 | 有提升空间 |
| 0.8-1.2 | ⚠️ 一般 | 需要练习 |
| 0.0-0.8 | ❌ 较差 | 需要重点改进 |

### 评估维度

- **Accuracy** (准确度): 发音是否准确
- **Completeness** (完整性): 是否发出所有音素
- **Fluency** (流畅度): 说话是否流畅，有无停顿
- **Prosodic** (韵律): 语调、重音是否自然
- **Total** (总分): 综合评分

---

## 🆘 常见问题

### Q1: 重建后还需要重新下载模型吗？

**A**: ❌ **不需要！** 模型存储在宿主机的 `./models/` 目录，通过 volume 挂载，重建镜像不会丢失。

### Q2: 容器启动很慢？

**A**: 首次启动可能需要 1-2 分钟，因为要：
- 创建符号链接
- 验证模型
- 检查依赖

这是正常的。查看日志：`docker logs gopt-pipeline`

### Q3: 处理音频失败？

**A**: 检查以下几点：
1. 容器是否运行：`docker ps`
2. 启动日志是否有错误：`docker logs gopt-pipeline`
3. 模型是否存在：`ls models/librispeech/`
4. 文件路径是否正确

### Q4: 支持哪些音频格式？

**A**: 
- ✅ WAV, MP3, M4A, FLAC, OGG, AAC, WMA
- 自动转换为 16kHz 单声道 WAV
- 最大文件大小：50MB

### Q5: 可以批量处理吗？

**A**: 可以！修改 `test_user_audio.py` 添加循环：

```python
audio_files = [
    ("audio1.m4a", "transcript 1"),
    ("audio2.m4a", "transcript 2"),
    # ...
]

for audio, text in audio_files:
    process_audio(audio, text)
```

---

## 📝 文件说明

### 核心文件

| 文件 | 说明 | 持久化 |
|-----|------|--------|
| `docker/entrypoint.sh` | 容器启动脚本，自动设置链接 | ✅ 内置在镜像 |
| `process_custom_audio_fixed.sh` | 修复的音频处理脚本 | ✅ 内置在镜像 |
| `Dockerfile` | 镜像构建文件，已更新 | ✅ 源代码 |
| `test_user_audio.py` | 测试脚本 | ✅ 源代码 |
| `models/librispeech/` | ASR 模型（~2GB） | ✅ Volume 挂载 |

### 修复内容

1. ✅ **自动模型链接**: `entrypoint.sh` 在容器启动时自动创建
2. ✅ **GOP 提取修复**: 使用 `--nj 1` 避免并行作业错误
3. ✅ **路径修复**: 直接使用 `data/train` 和 `data/test`
4. ✅ **脚本集成**: 修复的脚本内置在镜像中

---

## 🎓 进阶使用

### 调整资源限制

编辑 `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: '8.0'      # 增加 CPU
      memory: 16G      # 增加内存
```

### 启用 GPU 支持

```yaml
runtime: nvidia
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

### 自定义音频处理

修改 `/workspace/gopt/process_custom_audio.sh` 中的参数。

---

## 📚 更多文档

- 📖 [完整重建指南](REBUILD_GUIDE.md)
- 🔧 [模型设置指南](MODEL_SETUP_GUIDE.md)
- 🐳 [Docker 使用指南](DOCKER_USAGE.md)
- 🎵 [音频处理指南](PROCESS_YOUR_AUDIO_GUIDE.md)

---

## ✅ 检查清单

安装后验证：

- [ ] Docker 容器运行中 (`docker ps`)
- [ ] 启动日志无错误 (`docker logs gopt-pipeline`)
- [ ] 模型链接已创建（查看日志中的 "✓ Model links created"）
- [ ] Python 依赖正常 (`docker exec gopt-pipeline python3 -c "import torch"`)
- [ ] 音频处理成功 (`python test_user_audio.py`)

---

## 🎉 就是这样！

现在你可以：
1. ✅ 随意重建 Docker 镜像
2. ✅ 所有修复自动应用
3. ✅ 模型不会丢失
4. ✅ 容器启动即可使用

**祝你英语发音进步！** 🚀

