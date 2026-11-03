# 能力与使用方式

InfiniCoder 提供两条执行路径。Runtime 适合日常编程、受限源码修改和可恢复会话；Native 适合需要 shell 操作的 Linux 项目，通过调用方授权的沙箱执行。二者使用同一个任务入口选择配置，不会自动互换权限或会话。

| 任务需要 | 使用方式 | 能力与边界 |
| --- | --- | --- |
| 常规实现与 debug | `profile="general"`，或默认 `auto` | 分析文件、检索源码、修改、检查反馈和独立验收 |
| 多文件项目 | 明确允许修改的范围与检查命令 | 版本化符号、静态调用、结构和 use-def；按模块及符号展开上下文。复杂动态调用可能未知 |
| 算子开发 | `profile="operator"` 或领域路由 | shape、dtype、stride、精度与测量指导；受限模式可生成候选。GPU 正确性和性能需项目实际运行环境 |
| 训练与推理框架 | `profile="framework"` 或领域路由 | 数值、状态、所有权、并发、资源生命周期及阶段检查指导；不自动提供分布式集群 |
| 小上下文模型 | 提供匹配的本地 tokenizer，设置上下文预算 | 分层选择源码；可显式选择任务历史管理。选择或压缩上下文不保证任务成功率提升 |
| 澄清、需求更新、暂停恢复 | Runtime CLI / Python API | 结构化事件、差异、累计预算及恢复配置校验。Native 是单次执行 |

`auto` 根据任务语义和有限的仓库信息选择领域流程，不会启动多智能体或切换模型。基础路径在多文件范围等条件满足时要求计划；调用方也可直接提供里程碑。领域卡、规划和记忆均需与当前代码及真实检查结合。

`infinicoder inspect` 与 Python `inspect_task` 返回选择原因。`solve` 将实际激活情况写入任务状态目录的 `capabilities.json`。选择模块、观察到模块执行、内容实际送达模型和任务验收通过是不同状态。

高级的跨任务经验、性能搜索、候选分支和协作需要额外的显式合同与配置，默认不全部开启。`automatic_workers` 是实现模式选择，不表示自动运行多智能体。静态分析不能替代运行，模型自报成功不能替代最终验收。

## InfiniLM 接入

InfiniLM 在独立环境中启动，InfiniCoder 连接其 `/v1/chat/completions` 接口：

```sh
export INFINICODER_API_BASE=http://127.0.0.1:8000/v1
export INFINICODER_MODEL=your-served-model-id
# 若服务需要认证，另设 INFINICODER_API_KEY
infinicoder --workspace examples/workspace --spec examples/task.json \
  --state-dir ../infinicoder-session --allow-commands
```

模型名称须使用服务实际加载的 ID。InfiniCoder 不安装推理引擎或下载权重；某个模型能否由 InfiniLM 加载，取决于服务版本和后端支持。

Native 使用文本动作协议，无需启用服务端工具调用解析器。它不发送 `tools` 或 `strict` JSON schema；工具调用响应和未完整结束的动作不会被执行。该版本的客户端适配经过协议与执行测试，未据此宣称完成了目标模型在真实 InfiniLM 推理后端上的能力评测。

Native 需要 Linux、可用的 bubblewrap、任务 Python 环境与匹配的 tokenizer。无法建立隔离时不会退回宿主 shell。完整参数与示例见 [API](api.md)。
