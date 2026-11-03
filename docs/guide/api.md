# API 与支持范围

本版保留顶层公共导入及 CLI 名称，内部模块迁入功能包。旧 `infinicoder.engine` 对应 `infinicoder.runtime.engine`，旧 `infinicoder.model_client` 对应 `infinicoder.models.model_client`；新代码优先使用 `from infinicoder import ...`。完整目录和维护位置见[项目代码说明](codebase.md)。旧会话的源码/配置身份不会自动改写；需要完成旧会话时使用原冻结版本。

普通 Runtime `solve(..., resume=True)` 使用稳定的原始预算验证配置，初始化时间仍计入截止；改变模型或总预算仍被拒绝。底层 `Runtime.execute` 可选 `started_at` 为本次调用开始的 Unix 时间，用于把调用方初始化纳入预算，不改变配置身份。来自不兼容版本的会话不能直接恢复；请先保留旧状态备份。

模型端点须直接返回响应；HTTP 重定向按配置错误处理，不自动把认证头或源码发送到另一个地址。调用方确认真实地址后显式修改配置。


## 任务入口

新任务可使用 `inspect_task` / `solve`，CLI 对应 `infinicoder inspect` / `solve`。`SolvePolicy` 集中选择 backend、domain、context、planning；同一选择逻辑服务预览和实际执行。Native 的 `enhancements=False` 是同工具基线，不能与显式记忆/规划选项矛盾使用。`solve` 不清空 TaskSpec 字段、不扩大权限，不自动改用另一个后端。

`inspect` 只预览任务路由与主要模块，不调用模型或检查器，也不证明外部脚本/环境可用。外部 public/final 检查绑定在 solve 校验后加入最终 capabilities.json；不要把预览中的未绑定状态当成执行时关闭。

输入、环境前提与 `capabilities.json` 字段解释见[能力说明](capabilities.md)。普通 `RuntimeConfig/create_runtime` 与旧入口继续兼容。Native 可通过 `python_environment` 选择独立于控制程序的项目虚拟环境，对应 CLI `--python-env`；该目录在沙箱中只读，并须位于可编辑工作区之外。

Runtime 新增显式的隔离检查配方：同时传入 `public_check_script` 与 `check_script`。前者在修复过程中反馈，后者只在模型循环结束后独立运行一次；脚本位于工作区外并在首次请求前固定。模型不能执行任意 shell，检查使用 Linux + bubblewrap 和 `python_environment`。这条配方要求新会话，不能与 `TaskSpec.checks`、`allow_commands=True` 或 `resume=True` 混用；需要交互恢复时使用原有 TaskSpec 检查合同。两条合同不会互相覆盖。

该配方返回原 Runtime `status`（公共检查状态）、独立 `success` 和 `independent_acceptance`。任务成功要求最终检查及非授权源码身份检查同时通过；取消、暂停或待输入时不启动 final。公共检查受任务总截止约束；独立 final 最多 180 秒，实际耗时与源码身份另行记录。CLI 退出码依据独立 success。`tokenizer_path` 也适用于 Runtime，不能与自定义 estimator 同时指定。

## 常用配置

`RuntimeConfig` 集中常用会话配置，`create_runtime` 复用原有 Runtime，不创建第二套智能体循环。

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| profile | auto | auto/general/operator/framework/legacy；领域路由会记录依据 |
| max_steps / max_model_calls | 60 / 60 | 会话动作与模型调用上限；恢复不重置 |
| time_limit | 1800 秒 | 累计运行时间上限 |
| request_timeout | 180 秒 | 单次请求上限，受剩余会话预算约束 |
| context_limit | 16384 tokens | 调用方声明的服务容量，必须按实际服务修改 |
| allow_commands | False | 是否允许执行工作区代码与命令 |
| code_memory | True | 版本绑定、分层代码记忆 |
| development_workflows | True | 算子/框架指导流程 |
| automatic_workers | True | 原有条件式实现工作器选择 |
| domain_compiler | True | 满足前提时使用领域候选生成器；仍需独立验证 |
| fusion_expert | False | 受限 linear + bias + tanh GELU 融合候选，需 operator 路由与完整模块工作器 |

默认值、条件触发、手动实验及历史模块的完整区别见[当前能力与启用状态](capabilities.md)。`automatic_workers` 不会启动协作智能体；`code_memory` 不等于 Native 的 `task_working_memory`，也不会自动建立跨任务经验库。

输出 token 来自 `ModelConfig.max_output_tokens`，默认 2048。`create_runtime` 将它与 context_limit 一起传入 `ModelProfile`，不根据模型名称猜容量。`TokenEstimator` 可接入服务所用 tokenizer；未提供时使用启发式估算，不能视作准确上界。CLI 的 `--tokenizer-path` 仅加载本地文件且禁止 remote code。

新 `infinicoder` CLI 复用 RuntimeConfig 的 16384 默认上下文。CLI 采样温度默认 0，ModelConfig 默认 None（服务端默认）；要复现相同运行，请如首页示例显式设置温度与上下文。高级 `python -m infinicoder.runtime.engine` 与旧 `run` 保留原字符预算派生行为，不等同于新 CLI 的默认值。

`Runtime` 和旧 `run` 的高级参数保持兼容。`__all__` 只列常用 API；原顶层高级属性仍可显式导入，也可改用 `infinicoder.advanced`。`from infinicoder import *` 的集合已收缩，旧 wildcard 使用者应改为显式导入。高级接口在预览阶段可能调整，跨能力组合并不都经过效果验证。

## 任务合同

`TaskSpec.task` 为需求；`allowed_paths` 是精确文件或以 `/` 结尾的目录前缀，不能使用 glob。`checks` 是调用方冻结的独立验收，`required_artifacts` 是必须存在的交付文件。

模型提出 `finish` 只会请求运行时验收。诊断 `verify` 可提供反馈，但不能代替最终验收。模型生成的测试和自己打印的 PASSED 不是成功证明。

状态目录必须位于可编辑工作区外。状态中包含任务、工具输出、历史与源码差异，按项目源代码的敏感级别保存；不要提交状态目录。

## 会话操作

`runtime.execute(spec)` 新建；`runtime.execute(spec, resume=True)` 恢复。旧 `run(..., resume=True)` 必须明确提供已有 state_dir。

通过 `runtime.store.request_control('pause')` 或 `'cancel'` 请求协作式控制；CLI 用 `infinicoder-session pause/cancel`。控制在模型或动作边界生效，不保证立即中止正在执行的远程请求。

澄清答案以 `spec.amendments` 中的用户记录追加，CLI 支持 `--answer` 与 `--interactive`。需求或验收整体修改必须调用显式 revision 流程，保留备份及累计预算。事件通过 `event_handler` 和会话 JSONL 提供，CLI progress 复用相同事件。

恢复会校验会话版本、任务合同、检查器与策略身份。新 CLI/create_runtime 另保存模型设置、上下文和常用预算限制的配置摘要，恢复前比较，防止无意更换模型或抬高上限；只保存哈希，不写入密钥，并允许 API key 轮换。调用方仍需提供原配置，系统不会从摘要还原设置。任意高级参数组合并非全部已绑定，仍须保持原配置。

旧会话缺少该摘要时，新入口明确拒绝假装已经验证；需继续使用原版本或高级 Runtime/run，不能自动给旧会话补一个新摘要。不支持的会话 schema 使用 `infinicoder-session migrate` 显式迁移，先备份；schema 迁移不会补齐原本不存在的配置证据。代码索引可重建。不要手工改写状态哈希来绕过校验。

## 两种执行后端

| 项目 | Runtime 会话 | Native 高级单次执行 |
| --- | --- | --- |
| 入口 | infinicoder / create_runtime | infinicoder-native / advanced.run_native |
| 会话恢复、任务修订、里程碑 | 支持 | 不支持；合同含不支持字段时拒绝 |
| 平台 | 核心可在 Windows/Linux 运行 | CPU 隔离执行需要 Linux + bubblewrap |
| 默认 profile | auto | cpu（代码工作台，非 plain 基线） |
| 默认输出预算 | 2048 | 4096 |
| 结果 | 详细 state dict | CoreOutcome，分别记录验收和执行退出 |
| 验收 | 冻结检查器/外部 verifier | 调用方绑定的独立 final |

二者通过已有执行分派工作，能力与合同不互换；不会静默丢字段或自动跨后端恢复。Native 还支持多种研究 profile，先查看 `infinicoder-native --help` 和各适配器输入合同。CPU 隔离不可用时不能回退到宿主 shell。
