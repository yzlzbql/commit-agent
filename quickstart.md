# Quickstart

本文档说明当前这套 commit 分类评测的最小工作流：如何准备小规模测试集、如何运行评测、如何根据结果迭代 prompt / 流程，以及何时切换到最终大规模测试。

## 1. 数据介绍

评测数据位于 [data](/data2/opencode/examples/minimal-agent-cli/data)，kernel_tagged_commits 和 oracle_commits 是数据源。

### 1.1 `oracle_commits.jsonl`

[oracle_commits.jsonl](/data2/opencode/examples/minimal-agent-cli/data/oracle_commits.jsonl) 是通用 commit 数据集，保存了 commit 元信息与标签信息。常见字段包括：

- `repo`
- `commit_id`
- `tag`
- `commit_message`
- `author_name`
- `author_email`
- `author_date`
- 部分样本还带有 `patch`

当前实验流程里，实际从这个文件中抽取：

- `test`
- `docs`
- `refactor`

### 1.2 `kernel_tagged_commits.jsonl`

[kernel_tagged_commits.jsonl](/data2/opencode/examples/minimal-agent-cli/data/kernel_tagged_commits.jsonl) 是内核相关的带标签 commit 数据集。

当前实验流程里，从这个文件中抽取：

- `feat`
- `fix`

### 1.3 当前分类标签集合

当前评测只考虑 5 个标签：

- `feat`
- `fix`
- `refactor`
- `docs`
- `test`

## 2. 小规模迭代流程

每一轮 prompt / 流程迭代，都先跑一个小规模测试集，而不是直接跑大规模最终集。

### 2.1 抽样规则

每轮小规模测试固定抽取 `250` 条：

- 从 `oracle_commits.jsonl` 中抽取
  - `test` 各 `50`
  - `docs` 各 `50`
  - `refactor` 各 `50`
- 从 `kernel_tagged_commits.jsonl` 中抽取
  - `feat` 各 `50`
  - `fix` 各 `50`

总计：

- `5 * 50 = 250`

## 3. 运行评测

### 3.1 评测脚本

主脚本是 [eval_agent_jsonl.py](/data2/opencode/examples/minimal-agent-cli/eval_agent_jsonl.py)。

### 3.2 运行命令

示例命令：

```bash
python3 eval_agent_jsonl.py data/eval_small_resampled_v3.jsonl --task-file my-task.txt --agent commit_eval --max-workers 10 --repo-root /data2/yzl/huyanglin-repo
```

如果你使用的是每类 `50` 条的中等规模测试集，可以改成：

```bash
python3 eval_agent_jsonl.py data/eval_50_each.jsonl --task-file my-task.txt --agent commit_eval --max-workers 10 --repo-root /data2/yzl/huyanglin-repo
```

自动抽取
```bash
  python3 eval_agent_jsonl.py \
    --auto-sample-30 \
    --sample-seed 123 \
    --task-file my-task.txt \
    --agent commit_eval \
    --max-workers 10 \
    --repo-root /data2/yzl/huyanglin-repo
  ```

参数说明：

- `data/...jsonl`
  - 输入评测集
- `--task-file my-task.txt`
  - 当前待迭代的 prompt
- `--agent commit_eval`
  - commit 分类专用 agent
- `--max-workers 10`
  - 并发数
- `--repo-root /data2/huyanglin-repo`
  - 本地仓库根目录

## 4. 结果查看

评测结果默认输出到 [eval_runs/agent_jsonl](/data2/opencode/examples/minimal-agent-cli/eval_runs/agent_jsonl)。

每轮会生成两个文件：

- `xxx.jsonl`
  - 每条样本的评测结果
- `xxx_summary.json`
  - 汇总指标

汇总里重点关注：

- `accuracy`
- `accuracy_by_label`
- `eval_status_counts`
- `run_status_counts`

## 5. 迭代方式

每一轮都按下面的闭环进行：

1. 准备一轮小规模测试集
2. 运行 `eval_agent_jsonl.py`
3. 查看整体 accuracy 和各标签 accuracy 
4. 抽取错例和失败 case
5. 迭代修改：
   - `my-task.txt`
   - 必要时调整工具使用策略
   - 必要时修复 runtime / tool 行为
6. 用同一批小集或重新抽样的小集复测

推荐优先分析这三类问题：

- `feat` 被吸到 `fix` / `refactor`
- `docs` / `test` 被误判为代码类标签
- `parse_error` / `run_error`

## 6. 何时进入最终测试

当满足下面任一条件时，可以停止小规模迭代，切换到最终测试：

- 每个 label 的 accuracy 都大于 `80%`
- 或者整体 accuracy 超过 `85%`

## 7. 最终测试集规则

最终测试集从两个原始数据文件中按如下规则构建：

- 从 `oracle_commits.jsonl` 中抽取
  - `docs` `200`
  - `test` `200`
  - `refactor` `150`
- 从 `kernel_tagged_commits.jsonl` 中抽取
  - `feat` `200`
  - `fix` `200`

说明：

- `refactor` 在当前数据里只有 `150` 条可用，因此最终集不是 `1000` 条，而是 `950` 条
- 最终总量为：
  - `200 + 200 + 150 + 200 + 200 = 950`

## 8. 推荐实践

- 小规模阶段优先优化稳定性，先把 `parse_error` 和 `run_error` 压低
- 再处理标签边界问题，尤其是 `feat / fix / refactor`
- 不要只盯整体 accuracy ，要看各标签是否均衡
- 每次 prompt 改动后，先在小集上验证，再决定是否进入大集
