# InfiniCoder

InfiniCoder 是面向通用编程、算子开发和训练推理框架的代码智能体。
它将版本化代码分析、按需展开的上下文、任务规划、代码修改与独立验收连接到同一个执行流程。

当前版本为 **0.8.0a11 开发者预览**。支持 CLI 与 Python API，直接连接 InfiniLM 的兼容 HTTP 服务。模型权重和推理引擎由独立服务提供。源码已按功能包组织，入口、调用链、启用条件和逐模块索引见[项目代码说明](docs/guide/codebase.md)。

提供代码分析与检索、按领域选择的开发流程、受限修改、检查反馈和独立验收。它们的启用条件见[能力说明](docs/guide/capabilities.md)。当前仍是开发者预览，不承诺所有任务均能完成；本次依赖替换不代表新的模型能力评测结论。

## 安装

Python 3.10 或更高版本。在下载并解压的源码目录中运行：

```sh
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv/Scripts/Activate.ps1
python -m pip install .
infinicoder --version
```

核心 Runtime 使用 Python 标准库。按需安装额外依赖：

tokenizer、Native 及 full 额外依赖要求 **Python 3.11 或更高版本**。核心 Runtime 仍支持 Python 3.10。InfiniCoder 不依赖 LiteLLM 或 mini-SWE-agent 安装包。

| 安装方式 | 用途和前提 |
| --- | --- |
| `pip install '.[tokenizer]'` | 使用本地模型 tokenizer 精确计算内容 token；服务端聊天模板仍需匹配 |
| `pip install '.[native]'` | Clang C/C++/CUDA 静态分析；项目需提供正确的编译数据库和头文件 |
| `pip install '.[native_cpu]'` | InfiniCoder Native 执行与本地 tokenizer；需要 Linux 与可用的 bubblewrap |
| `pip install '.[full]'` | 上述 Python 依赖及原生工具校验依赖；不安装 GPU 驱动、CUDA 或系统沙箱 |

GPU 算子与框架运行还需要任务自身的 PyTorch、Triton、CUDA 等环境。InfiniCoder 不携带模型权重。

## 开始一次任务

先启动 [InfiniLM](https://github.com/InfiniTensor/InfiniLM) 的聊天服务，或使用已有兼容端点。以下环境变量由调用方设置，密钥不要写进任务文件或提交到 Git：

```sh
export INFINICODER_API_BASE=http://127.0.0.1:8000/v1
export INFINICODER_MODEL=your-served-model-id
export INFINICODER_API_KEY=your-key-if-required
```

PowerShell 对应写法为 `$env:INFINICODER_MODEL='your-served-model-id'`。

仓库提供一个可复制的任务例子。它限定只能修改一个实现文件，用已有测试验收：

```sh
infinicoder --workspace examples/workspace --spec examples/task.json --state-dir ../infinicoder-session --profile general --context-limit 16384 --temperature 0 --allow-commands --interactive --progress
```

`--allow-commands` 允许执行工作区代码。请在可信项目的独立副本中运行；对于不可信仓库使用外部操作系统沙箱。Runtime 的路径权限不等同于操作系统隔离。

任务 JSON 写明需求、允许修改的路径和验收命令。没有独立验收时，智能体会请求补充，不能自行宣布成功。无需将测试结果交给模型自行判断。

## 会话与交互

```sh
infinicoder-session status --state-dir ../infinicoder-session
infinicoder-session diff --state-dir ../infinicoder-session
infinicoder-session pause --state-dir ../infinicoder-session
```

使用原任务、配置与状态目录，给运行命令加 `--resume` 继续。回答澄清可加 `--answer '补充约束'`；修改需求或验收使用显式 `--revise`。恢复会保留已消费预算，并校验保存的合同与配置。不存在的会话会报错。

CLI 最终结果以 JSON 写到 stdout，进度写到 stderr。退出码：0 成功，1 任务未成功（包括暂停、等待输入或预算耗尽），2 调用或配置错误，130 未被会话处理的键盘中断。已被会话结算的中断按保存的任务状态退出。完整状态和事件保存在状态目录。

## Python API

```python
from infinicoder import (
    Check, TaskSpec, ModelConfig, CompatibleClient, RuntimeConfig, create_runtime,
)

client = CompatibleClient(ModelConfig(
    base_url="http://127.0.0.1:8000/v1", model="your-served-model-id", temperature=0,
))
runtime = create_runtime(
    "examples/workspace", client, "../infinicoder-session",
    config=RuntimeConfig(profile="general", context_limit=16384, allow_commands=True),
)
task = TaskSpec(
    "实现保序去重，保持输入不变。",
    allowed_paths=["solution.py"],
    checks=[Check("acceptance", ["python", "-m", "unittest", "discover", "-s", "tests", "-q"])],
)
state = runtime.execute(task)
print(state["status"], state["reason"])
```

## 能力与扩展

新任务可先用 `infinicoder inspect --workspace ... --spec ...` 查看模块选择，再用 `infinicoder solve` 执行。Python 对应 `inspect_task`、`solve` 与 `SolvePolicy`。需要暂停恢复时使用 Runtime 会话入口。

- 多视图代码记忆：文件、符号、静态调用、结构、引用和 use-def；源码位置及哈希绑定事实，修改后重新分析。
- 层级检索：先路由模块，再展开相关符号和源码。未知的动态调用、别名和解析失败会保留为未知。
- 决策与实现：支持带依赖和修改范围的计划、阶段检查、需求修订及最终完整验收。
- 专用流程：算子侧关注 shape、dtype、stride、精度和真实测量；框架侧关注状态恢复、数值、所有权、并发与资源生命周期。

这些是已实现的机制，不代表每种机制已获得稳定成功率提升。`auto` 是 Runtime 内的领域路由，不会自动进入实验路由或 Native 的 `cpu-auto`。`automatic_workers` 选择文件/函数实现模式，不表示自动启动多智能体；经验库和可信性能搜索也需另行配置。完整启用范围见[能力状态表](docs/guide/capabilities.md)。高级选项见 `infinicoder --advanced-help`；历史属性导入仍兼容，高级适配器可从 `infinicoder.advanced` 导入。

[API 与支持范围](docs/guide/api.md) · [架构与扩展](docs/guide/architecture.md) · [贡献指南](CONTRIBUTING.md) · [安全边界](SECURITY.md)

Apache-2.0，Copyright 2026 InfiniTensor。第三方依赖遵循各自许可证，见 [NOTICE](NOTICE)。
