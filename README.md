# commit-agent

这是一个 agent harness，其中 `commit_eval` 用于运行 `commit 分类` 任务，并支持批量评测该类别任务的效果。

## 快速开始

安装依赖，推荐使用 Python3.12：

```bash
python3 -m pip install -e .
```

在项目根目录创建文件 .env，并配置模型环境变量：

```dotenv
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=xxx
OPENAI_MODEL=xxx

# RAG embedding
SILICONFLOW_API_KEY=your_siliconflow_api_key
RAG_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-4B
RAG_EMBEDDING_DIM=1024
# SILICONFLOW_EMBEDDING_URL=https://api.siliconflow.cn/v1/embeddings
```

## 单次 Commit 分类

[my-task.txt](my-task.txt) 只保留本次运行输入。任务文件中使用块式输入指定目标仓库（需要提前 clone）和目标 `commit`：

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


## 批量评测

运行：

```bash
python3 eval_agent_jsonl.py data/eval_small_resampled_v3.jsonl \
  --task-file my-task.txt \
  --agent commit_eval \
  --max-workers 10 \
  --repo-root /path/to/repos \
  --output-dir eval_runs/agent_jsonl

# 或使用内置的随机抽样：
python3 eval_agent_jsonl.py \
  --auto-sample-30 \
  --sample-seed 123 \
  --task-file my-task.txt \
  --agent commit_eval \
  --max-workers 10 \
  --repo-root /path/to/repos \
  --output-dir eval_runs/agent_jsonl
```

说明：

- 默认 agent 为 `commit_eval`
- `input_jsonl` 和 `--auto-sample-30` 二选一
- `--repo-root` 指向本地仓库根目录
- 加上 `--show-events` 可以打印每个 case 的推理和工具轨迹
- 结果默认写入 `eval_runs/agent_jsonl/` 下的 `.jsonl` 和 `_summary.json`

## 相关文件

| 路径 | 作用 |
| --- | --- |
| [my-task.txt](my-task.txt) | 单次 `commit` 分类输入文件，保存仓库路径和 `commit` id |
| [run_task.py](run_task.py) | 单任务运行入口 |
| [eval_agent_jsonl.py](eval_agent_jsonl.py) | 批量评测入口 |
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
