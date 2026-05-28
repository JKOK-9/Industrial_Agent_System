# LLM Fine-Tuning Platform

基于 Flask + Vue 的前后端分离大模型微调平台。后端负责模型下载、训练任务、数据集校验和 LLaMA-Factory 调用；前端负责基座模型管理、LoRA 微调提交、日志查看和微调模型管理。

## 目录

```text
backend/     Flask API
frontend/    Vue 3 + Vite 前端
runtime/     模型、数据集、训练配置、输出与日志
```

## 环境

项目按 `conda` 的 `agent` 环境验证：

```powershell
conda activate agent
python -m pip install -r requirements.txt
```

真实训练前安装 LLaMA-Factory：

```powershell
python -m pip install llamafactory
```

如果 LLaMA-Factory 是源码安装，可设置：

```powershell
$env:LLAMAFACTORY_WORKDIR="D:\path\to\LLaMA-Factory"
$env:LLAMAFACTORY_CLI="llamafactory-cli"
```

## 启动

后端：

```powershell
conda activate agent
python -m flask --app backend.app run --host 127.0.0.1 --port 5000
```

前端：

```powershell
cd frontend
npm install
npm run dev
```

打开 <http://127.0.0.1:5173>。

## 功能

- 基座模型：Hugging Face 下载、ModelScope 下载、本地路径登记、删除
- 微调训练：选择基座模型、上传数据、配置 LoRA 参数、提交异步训练、查看日志
- 微调模型：查看训练成功后的模型产物、删除模型目录
- 下载任务：展示最近下载状态和失败原因

## 微调数据格式

当前支持三种上传格式：

- Alpaca JSON / JSONL：`instruction` 和 `output` 必填，`input`、`system`、`history` 可选
- ShareGPT JSON / JSONL：`conversations` 必填
- OpenAI Messages JSON / JSONL：`messages` 必填

平台会为每次训练任务生成独立的 `dataset_info.json`，并通过 `llamafactory-cli train <config.yaml>` 启动训练。

## UI 规范落地

前端样式按 `系统UI设计规范（WEB版）.pdf` 调整：

- 品牌主色使用 Blue 6 `#0057FF`
- 中性色以 `#FAFAFA`、`#F0F2F5`、`#DFE2E6`、`#687182`、`#1E2226` 为基础
- 控件默认高度 32px，按钮分为主按钮、次按钮、文字按钮
- 布局采用中后台信息密度，基于 4px 间距和 24 栅格思路
- 图标使用 16px 线性图标
