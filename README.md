# minimal-agent-cli

这是一个基于 `minimal_agent` 的本地 agent harness，用于运行 `commit` 分类任务和批量评测分类效果。

## 快速开始

安装依赖：

```bash
python3 -m pip install -e .
```

在项目根目录配置模型环境变量：

```dotenv
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-5.4
```

## 单次 Commit 分类

[my-task.txt](my-task.txt) 只保留本次运行输入。任务文件中使用块式输入指定目标仓库和目标 `commit`：

```text
[REPOSITORY_PATH]
/path/to/repo

[COMMIT]
<commit-sha>
```

运行：

```bash
python3 run_task.py my-task.txt --agent commit_eval
```

说明：

- 当任务文件中包含 `[REPOSITORY_PATH]` 时，脚本会自动切换到目标仓库
- `--cwd` 只在任务文件没有指定仓库时作为兜底
- 最终输出的 JSON 包含 `primary_label`、`confidence`、`reason`、`intent_items`

## 批量评测

运行：

```bash
python3 eval_commit_classifier.py --sample-size 10 --seed 42 --output-dir eval_runs
```

说明：

- 默认 agent 为 `commit_eval`
- 加上 `--show-events` 可以打印每个 case 的推理和工具轨迹
- 结果会写入 `eval_runs/` 下的 `.jsonl` 和 `_summary.json`

## 相关文件

| 路径 | 作用 |
| --- | --- |
| [my-task.txt](my-task.txt) | 单次 `commit` 分类输入文件，保存仓库路径和 `commit` id |
| [run_task.py](run_task.py) | 单任务运行入口 |
| [eval_commit_classifier.py](eval_commit_classifier.py) | 批量评测入口 |
| `minimal_agent/main.py` | 通用 CLI 入口，负责组装 `project`、`model`、`tools`、`session` 和运行循环 |
| `minimal_agent/agent.py` | `commit_eval` agent 定义，以及固定的 commit 分类规则 |
| `minimal_agent/config.py` | 加载模型、步数限制、验证和 LSP 配置 |
| `minimal_agent/project.py` | 解析项目根目录、状态目录和 session 目录 |
| `minimal_agent/model.py` | OpenAI 兼容模型适配层 |
| `minimal_agent/prompt.py` | 组装发送给模型的 system/user/messages 与工具 schema |
| `minimal_agent/runtime/loop.py` | ReAct 主循环，负责 reasoning、tool call、final 和 verify 阶段切换 |
| `minimal_agent/runtime/verify.py` | 处理最终验证步骤 |
| `minimal_agent/session/store.py` | 持久化 session 状态、事件、todo 和 summary |
| `minimal_agent/analysis/service.py` | 聚合分析能力，供工具和运行时共享 |
| `minimal_agent/tool/registry.py` | 注册并分发所有内置工具 |
| `minimal_agent/tool/git_commit.py` | `git_commit_show` 工具 |
| `minimal_agent/tool/git_read_file.py` | 历史文件读取工具 |
| `minimal_agent/tool/rag.py` | 相似提交检索工具 |
