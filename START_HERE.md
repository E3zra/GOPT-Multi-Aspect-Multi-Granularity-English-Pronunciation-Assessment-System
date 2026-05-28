# GOPT Web Interface - Getting Started

> **⚠️ DEPRECATED / OUTDATED**
> 
> This document is written in Chinese and contains **outdated information** including references to non-existent files:
> - `WEB_INTERFACE_QUICKSTART.md` [OUTDATED — file does not exist]
> - `WEB_INTERFACE_GUIDE.md` [OUTDATED — file does not exist]
> - `TUTORIAL_BEGINNER.md` [OUTDATED — file does not exist]
> - `pip install -r requirements.txt` is WRONG for host-side setup (missing host packages)
> 
> **For the current, verified setup guide in English, see [START_LOCAL.md](START_LOCAL.md).**

---


## ✅ 实施已完成

GOPT Web 界面已完全实现并可以使用！这是一个现代化的网页应用，让你可以通过浏览器上传音频文件并获得发音评估结果。

---

## 🚀 快速开始（3 步）

### 步骤 1: 启动 Docker 容器

```bash
docker start gopt-pipeline
```

### 步骤 2: 安装依赖（首次使用）

```bash
pip install -r requirements.txt
```

### 步骤 3: 启动 Web 服务器

**推荐方式（使用启动脚本）：**

Linux/Mac:
```bash
chmod +x start_web_interface.sh
./start_web_interface.sh
```

Windows:
```cmd
start_web_interface.bat
```

**或直接运行：**

```bash
python web_server.py
```

### 步骤 4: 打开浏览器

```
http://localhost:8080
```

---

## 📁 项目文件说明

### 核心文件（已创建）

| 文件 | 说明 |
|------|------|
| `web_server.py` | FastAPI Web 服务器（509 行） |
| `static/index.html` | 前端界面（228 行） |
| `static/style.css` | 样式表（675 行） |
| `static/app.js` | JavaScript 逻辑（550 行） |

### 文档文件（已创建）

| 文件 | 说明 |
|------|------|
| `WEB_INTERFACE_QUICKSTART.md` | **快速启动指南** ⭐ 推荐先看 |
| `WEB_INTERFACE_GUIDE.md` | 完整使用指南（详细） |
| `WEB_INTERFACE_IMPLEMENTATION_SUMMARY.md` | 技术实施总结 |
| `START_HERE.md` | 本文件 |

### 工具文件（已创建）

| 文件 | 说明 |
|------|------|
| `test_web_interface.py` | 自动化测试脚本 |
| `start_web_interface.sh` | Linux/Mac 启动脚本 |
| `start_web_interface.bat` | Windows 启动脚本 |

---

## 🎯 主要功能

✨ **现代化界面**
- 拖放文件上传
- 实时进度显示
- 美观的结果展示
- 响应式设计（支持移动端）

✨ **完整流程**
- 音频文件上传（支持 WAV, MP3, M4A, FLAC, OGG, AAC, WMA）
- 自动格式转换（16kHz 单声道）
- 文本转录输入
- 后台自动处理
- 多维度评分展示

✨ **用户友好**
- 自动格式转换
- 清晰的错误提示
- 详细的使用说明
- 健康状态检查

---

## 📖 使用流程

### 1. 准备音频文件

**支持的格式：**
- **WAV, MP3, M4A, FLAC, OGG, AAC, WMA**
- 自动转换为 16kHz 单声道
- 时长: 5-30 秒
- 大小: 最大 50MB

**✨ 新功能：自动格式转换**
系统会自动将上传的音频转换为正确格式，无需手动转换！

**手动转换（可选）：**
```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

### 2. 上传并输入文本

1. 打开浏览器访问 `http://localhost:8080`
2. 拖放或选择 .wav 文件
3. 输入音频转录文本（全大写、无标点）
4. 点击"开始评估"

**文本格式示例：**
```
正确: HELLO WORLD HOW ARE YOU TODAY
错误: Hello, world! How are you today?
```

### 3. 等待处理

- 处理时间: 通常 3-5 分钟
- 实时进度显示
- 自动状态更新

### 4. 查看结果

评估结果包含：
- **句子级评分**: 准确度、完整度、流畅度、韵律、总分
- **音素级评分**: 每个音素的详细评分
- **原始数据**: 完整 JSON 格式
- **下载功能**: 保存结果供后续分析

---

## 🧪 运行测试

在使用前，可以运行测试脚本验证配置：

```bash
python test_web_interface.py
```

测试内容：
- ✓ 文件结构完整性
- ✓ 依赖包安装
- ✓ Docker 容器状态
- ✓ 服务器启动
- ✓ 配置正确性

---

## ⚠️ 常见问题

### Q: Docker 容器未运行？

**错误**: "Docker container 'gopt-pipeline' is not running"

**解决**:
```bash
docker start gopt-pipeline
```

### Q: 缺少依赖？

**错误**: "ModuleNotFoundError: No module named 'fastapi'"

**解决**:
```bash
pip install -r requirements.txt
```

### Q: 文件格式错误？

**错误**: "Only .wav files are supported"

**解决**: 使用 FFmpeg 转换音频格式
```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

### Q: 转录文本错误？

**错误**: "Word XXX not in lexicon"

**原因**: 包含标点、缩写或特殊字符

**解决**:
- 移除标点符号
- 展开缩写（IT'S → IT IS）
- 使用标准英文单词

---

## 📚 文档导航

根据你的需求选择合适的文档：

| 如果你想... | 查看文档 |
|------------|---------|
| 快速开始使用 | `WEB_INTERFACE_QUICKSTART.md` ⭐ |
| 了解详细功能 | `WEB_INTERFACE_GUIDE.md` |
| 了解技术细节 | `WEB_INTERFACE_IMPLEMENTATION_SUMMARY.md` |
| 查看 API 文档 | 启动后访问 `http://localhost:8080/docs` |
| 了解主项目 | `README.md` |
| 初学者教程 | `TUTORIAL_BEGINNER.md` |

---

## 🔗 API 端点

Web 服务器提供以下 REST API：

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 主页面 |
| `/health` | GET | 健康检查 |
| `/upload` | POST | 上传文件 |
| `/status/{task_id}` | GET | 查询状态 |
| `/result/{task_id}` | GET | 获取结果 |
| `/download/{task_id}` | GET | 下载结果 |
| `/docs` | GET | API 文档 |

---

## 🎓 技术架构

### 前端
- HTML5 + CSS3 + Vanilla JavaScript
- 响应式设计
- 无外部依赖

### 后端
- FastAPI（Web 框架）
- Uvicorn（ASGI 服务器）
- aiofiles（异步文件操作）
- Docker 容器集成

### 数据流
```
浏览器 ←→ Web 服务器 ←→ Docker 容器 ←→ GOPT 模型
```

---

## 🚧 未来功能（已预留接口）

- [ ] **ASR 自动转录** - 无需手动输入文本（方案 C）
- [ ] 批量处理多个文件
- [ ] 历史记录和结果比较
- [ ] 更详细的可视化图表
- [ ] 导出 PDF 报告
- [ ] 移动应用

---

## 💡 使用提示

1. **音频质量很重要** - 清晰的录音会得到更准确的评分
2. **文本必须匹配** - 转录文本要与实际发音完全一致
3. **避免噪音** - 在安静环境录音效果最好
4. **标准发音** - 模型基于标准英语发音训练
5. **适当长度** - 5-30 秒的音频效果最佳

---

## 🆘 获取帮助

如果遇到问题：

1. ✅ 查看 **常见问题** 部分（上方）
2. ✅ 运行测试脚本: `python test_web_interface.py`
3. ✅ 查看完整指南: `WEB_INTERFACE_GUIDE.md`
4. ✅ 检查浏览器控制台（F12）
5. ✅ 查看服务器日志（终端输出）
6. ✅ 检查 Docker 日志: `docker logs gopt-pipeline`

---

## 🎉 开始你的发音评估之旅！

现在一切就绪，你可以：

1. ✅ 启动服务器
2. ✅ 打开浏览器
3. ✅ 上传音频
4. ✅ 获得评估

**祝使用愉快！** 🚀

---

## 📞 反馈和支持

如有任何问题或建议，请：
- 查看项目文档
- 提交 GitHub Issue
- 联系项目维护者

**版本**: 1.0.0  
**状态**: ✅ 完全可用  
**最后更新**: 2024年11月

