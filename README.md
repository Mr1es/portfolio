# 二面作业

校科联二面 · 三份作业合集。

| 目录 | 项目 | 说明 |
|---|---|---|
| `harness/` | 华小牛 Agent Harness | 自己搭的 Agent：LLM 客户端封装、工具系统（计算器 / 文件读写 / 网络请求）、对话记忆、ReAct 主循环 |
| `myllm-tring/` | MyLLM Tring | 另一个 LLM 相关作业，来自 [Mr1es/MyLLMTring](https://github.com/Mr1es/MyLLMTring) |
| `yolo/` | yolo | 目标检测相关实验，源自 [Linmoqian/yolo_train](https://github.com/Linmoqian/yolo_train) |

## harness 怎么跑

```bash
cd harness
python main.py
```

需要在**仓库根目录**放一个 `.env`（不进版本库），里面三行：

```
OPENAI_API_KEY=你的key
STEP_BASE_URL=https://api.stepfun.com/step_plan/v1
STEP_MODEL=step-3.7-flash
```

依赖：`pip install openai python-dotenv requests`。建议用虚拟环境，仓库根的 `.env` 和虚拟环境都已在 `.gitignore` 里。

## 说明

- `yolo/` 的上游是 [Linmoqian/yolo_train](https://github.com/Linmoqian/yolo_train)，本仓库里的 `yolo/` 是在它基础上做的练习，非原创项目。
- `harness/` 编写过程中参考了公开教程，并使用 AI 辅助调试；代码由本人整理、运行并逐行理解。
