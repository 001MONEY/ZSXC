# Deep Agents 课件

本课讲 LangChain 的 **Deep Agents SDK**：用 `create_deep_agent` 搭一套面向长任务的 agent harness，并学会用 **Skills** 按需装入流程和领域知识。

综合案例是深度研究 Agent：主代理规划任务，把检索交给子代理，按 skill 写出带来源的报告。

文档依据（2026-11 官方文档）：

- https://docs.langchain.com/oss/python/deepagents/overview
- https://docs.langchain.com/oss/python/deepagents/customization
- https://docs.langchain.com/oss/python/deepagents/backends
- https://docs.langchain.com/oss/python/deepagents/skills
- https://docs.langchain.com/oss/python/deepagents/human-in-the-loop
- https://docs.langchain.com/oss/python/deepagents/deep-research

版本以 `deepagents>=0.7` 为基线。文中单独标出更高版本才有的能力。

---

## 课程地图

| 讲 | 题目 | 学员带走的能力 |
| --- | --- | --- |
| 1 | Harness 与 `create_deep_agent` | 说清 Deep Agents 在栈里的位置，跑通第一个 agent，读懂默认中间件 |
| 2 | 虚拟文件系统、Backend、权限 | 选对文件放在哪，并限制读写范围 |
| 3 | 上下文、子代理、人在回路 | 让长任务留在上下文窗口里，并把危险工具交给人批 |
| 4 | Skills 基础 | 写出能被发现、能被按需读入的 `SKILL.md` |
| 5 | Skills 进阶 | 多来源、pin、子代理技能、读写权限、脚本 |
| 6 | 深度研究 Agent | 把工具、规划、子代理和 skill 合成一个研究流程 |

---

## 第 1 讲　Harness 与 `create_deep_agent`

### 目标

- 说出 Deep Agents、LangChain、LangGraph 各自负责什么。
- 用 `create_deep_agent` 跑通一次工具调用。
- 指出「只传 model」时栈里已经有什么，哪些能力要额外打开。

### Deep Agents 是什么

Deep Agents 是一套 **agent harness**。它跑在 LangChain 的 agent 构件和 LangGraph 运行时上面，用来做多步骤、会写文件、会把任务分出去的长任务。

三层可以这样记：

| 层 | 产品 | 课堂上怎么用 |
| --- | --- | --- |
| 运行时 | LangGraph | 持久执行、checkpoint、interrupt、streaming |
| 最小 harness | LangChain `create_agent` | 模型、工具、提示、中间件，自己拼 |
| 意见明确的 harness | Deep Agents `create_deep_agent` | 文件系统、子代理、摘要、技能、记忆已经接好 |

官方给它的能力分成四块：

1. **执行环境**：工具、虚拟文件系统、可选沙箱、解释器
2. **上下文管理**：Skills、Memory、摘要、大结果卸载、prompt cache
3. **委托**：子代理；任务清单要自己打开
4. **控制**：工具执行前的人工审批

同一套 harness 有两种跑法。本课用 **Deep Agents SDK**，进程自己托管。Managed Deep Agents 由 LangSmith 托管 Agent Server、沙箱和调度，本课不展开。

### 安装与最小例子

```bash
pip install deepagents
```

```python
from deepagents import create_deep_agent


def get_weather(city: str) -> str:
    """查询城市天气。"""
    return f"{city} 今天晴。"


agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[get_weather],
    system_prompt="你是天气助手，回答前先调用工具。",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "上海天气怎么样？"}]}
)
print(result["messages"][-1].content)
```

`create_deep_agent` 返回的是编译好的 LangGraph 图。调用方式和 LangGraph 一样：输入里带 `messages`，输出里带回完整消息列表。

工具要有清晰的名字和文档字符串。模型靠这些决定何时调用。

### 创建参数

课堂上先记住这张表，后面每一讲都会回到其中几项。

| 参数 | 作用 |
| --- | --- |
| `model` | `provider:model` 字符串，或已经初始化的聊天模型 |
| `system_prompt` | 主代理指令 |
| `tools` | 自己的函数、LangChain 工具，或 MCP 工具 |
| `backend` | 虚拟文件放在哪。默认是线程内的 `StateBackend` |
| `permissions` | 按路径限制读和写 |
| `subagents` | 自定义子代理 |
| `middleware` | 追加或按名字替换中间件 |
| `interrupt_on` | 指定工具在执行前暂停 |
| `skills` | 技能目录，按需加载 |
| `memory` | `AGENTS.md` 路径，启动时整份载入 |
| `response_format` | 结构化输出 |
| `checkpointer` / `store` | 线程内持久化 / 跨线程存储 |
| `state_schema` / `context_schema` | 自定义图状态、单次运行上下文 |

### 只传 model 时，栈里有什么

官方把「只传 model」叫 **bare stack**。主代理通常按这个顺序装中间件：

1. `FilesystemMiddleware`：`ls`、`read_file`、`write_file`、`edit_file`、`glob`、`grep` 等
2. `SubAgentMiddleware`：默认挂一个 general-purpose 子代理，主代理因此有 `task` 工具
3. `SummarizationMiddleware`：对话变长后压缩历史
4. `PatchToolCallsMiddleware`：中断或参数损坏后，补齐悬空的工具调用
5. Prompt caching：Anthropic 与 Bedrock 生效，其他模型上空转
6. Harness profile 附带的提供商中间件
7. `UnsupportedContentMiddleware`：模型不支持的多模态内容改成文字占位
8. `ModelRefusalMiddleware`：模型拒绝时抛 `ModelRefusalError`
9. Profile 里声明要排除的工具，在这里从工具列表拿掉

所以第一个天气 agent **已经能读写虚拟文件，也已经能把任务交给通用子代理**。学员如果没传 `skills`、`memory`、`interrupt_on`，那三项就不在栈里。

### 传了参数之后，完整栈怎么排

官方 full stack 在 bare stack 的骨架上插入可选项。合并规则：你传入的中间件如果 `.name` 和内置项相同，就地替换；其余插在 `PatchToolCallsMiddleware` 之后、profile / prompt cache / memory 之前。

完整顺序：

1. `FilesystemMiddleware`（带上 `permissions` 时，读写真的在这里拦截）
2. `SubAgentMiddleware`（至少有一个同步子代理时才有；默认的 general-purpose 算一个）
3. `SummarizationMiddleware`
4. `PatchToolCallsMiddleware`
5. `AsyncSubAgentMiddleware`（配置了异步子代理才有）
6. 你传入的 `middleware`
7. Harness profile 附加项
8. `SkillsMiddleware`（传了 `skills` 才有）
9. Prompt caching
10. `MemoryMiddleware`（传了 `memory` 才有）
11. `HumanInTheLoopMiddleware`（传了 `interrupt_on` 才有）
12. `UnsupportedContentMiddleware`
13. `ModelRefusalMiddleware`
14. 按 profile 去掉被排除的工具

两个容易讲错的点：

- 任务清单 **不是** 默认栈的一部分。v0.7 起要自己加 `TodoListMiddleware`，代理才会有 `write_todos`。
- 子代理图里没有 `SubAgentMiddleware`。`task` 只出现在主代理上，避免子代理再无限派生子代理。

### 怎么关掉默认子代理

不想要 `task` 时，用 harness profile 关掉 general-purpose，并且不要再传同步 `subagents`。文件系统中间件本身不能从栈里拔掉；想少给模型看文件工具，用 `excluded_tools`，或给 `FilesystemMiddleware` 传 `tools` 白名单。白名单里必须保留 `read_file`。

## 第 2 讲　虚拟文件系统、Backend、权限

### 目标

- 列出内置文件工具，并说出 `execute` 什么时候才出现。
- 按「线程内 / 本机磁盘 / 跨线程 / 按路径分流」选择 backend。
- 用 `FilesystemPermission` 写出先匹配先生效的读写规则。

### 文件工具是上下文的底座

Deep Agents 把很多能力都放在同一套虚拟文件系统上：大工具结果的卸载、Skills、Memory、沙箱脚本、研究报告。模型看到的是路径和工具，存到哪里由 backend 决定。

| 工具 | 作用 |
| --- | --- |
| `ls` | 列目录，带大小和修改时间 |
| `read_file` | 按行读文本；也支持图片、音视频、PDF、PPT |
| `write_file` | 新建或覆盖 |
| `edit_file` | 精确字符串替换 |
| `delete` | 删文件，或递归删目录。需要 `deepagents>=0.7` |
| `glob` | 按模式找文件 |
| `grep` | 在文件内容里搜索 |
| `execute` | 跑 shell。只有沙箱或 `LocalShellBackend` 会把它交给模型 |

backend 不支持删除时，`delete` 会自动从工具列表消失。`execute` 同理。

`FilesystemMiddleware(tools=[...])` 可以只暴露子集。`read_file` 必须留在列表里，否则创建 agent 时会报错。这个白名单需要 `deepagents>=0.7`。自定义的 `tools=` 不受这个白名单影响。

### 内置 backend

| Backend             | 数据存放在哪                                                                                        | 适合                                        |
| ------------------- | --------------------------------------------------------------------------------------------- | ----------------------------------------- |
| `StateBackend`（默认）  | LangGraph state，跟线程走塞进这次对话的状态，不产生磁盘文件。配了 `checkpointer` 之后，同一个 `thread_id` 的下一轮还能读到；换一个线程就没了 | 草稿、当次任务的中间文件。配了 checkpointer 才能跨轮次留在同一线程里 |
| `FilesystemBackend` | 本机某个绝对路径 `root_dir`别的线程、下次启动也能读，适合记忆和共享技能                                                    | 本地开发、让代理改项目文件                             |
| `StoreBackend`      | LangGraph Store                                                                               | 跨线程的记忆、共享技能库                              |
| `ContextHubBackend` | LangSmith Context Hub 仓库                                                                      | 不想单独备 Store，又要持久文件                        |
| Sandbox             | 隔离环境                                                                                          | 要装依赖、跑测试、调用 CLI                           |
| `LocalShellBackend` | 宿主机 shell                                                                                     | 只在受控的开发机上用                                |
| `CompositeBackend`  | 按路径前缀分流                                                                                       | `/memories/`、`/skills/` 持久，其余仍在线程 state 里 |

`FilesystemBackend` 的 `root_dir` 必须是绝对路径。默认 `virtual_mode=False` 时，`root_dir` **并不能**挡住 `..`、`~` 或跑出根目录的绝对路径。要做路径限制，设 `virtual_mode=True`。

`LocalShellBackend` 和直接的本机 `FilesystemBackend` 都能读到密钥文件。官方建议：Web API 用 `StateBackend`、`StoreBackend` 或沙箱；本机磁盘只用于 CLI 和 CI，并配合人工审批。

### 按路径分流

线程里的草稿留在 state，跨会话的记忆放到 store：

```python
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    store=store,
    backend=CompositeBackend(
        default=StateBackend(),
        routes={"/memories/": StoreBackend()},
    ),
)
```

代理对 `/memories/profile.md` 的读写进 store，对 `/drafts/report.md` 的读写留在当前线程。LangSmith Deployment 会自己提供 store，本地演示用 `InMemoryStore`。

项目文件和代理内部文件要分开时，把项目目录单独做成一条路由，默认 backend 继续用 `StateBackend`。这样摘要卸载、对话产物不会写进业务仓库。

### 权限规则

`permissions` 是一组 `FilesystemPermission`。每条规则有：

- `operations`：`"read"`、`"write"`，或两者
- `paths`：glob，例如 `/workspace/**`、`**/.env`
- `mode`：`"allow"`、`"deny"`、`"interrupt"`

规则从上到下匹配，**第一条命中的规则生效**。没有任何规则命中时，操作被允许。所以「只许动工作区」要先写允许，再写一条更宽的拒绝；「技能库只读、个人技能可写」要把更具体的拒绝放在更宽的拒绝前面。

```python
from deepagents import FilesystemPermission, create_deep_agent

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    permissions=[
        FilesystemPermission(
            operations=["read", "write"],
            paths=["/workspace/**"],
            mode="allow",
        ),
        FilesystemPermission(
            operations=["read", "write"],
            paths=["/**"],
            mode="deny",
        ),
    ],
)
```

权限作用在内置文件工具上。沙箱里的 `execute` 可以跑任意命令，**不走**这套路径规则。自定义校验用 backend 的 policy hook。`mode="interrupt"` 会在写入前暂停，恢复方式和第 3 讲的人在回路相同，需要 `deepagents>=0.6.8`，并且要有 checkpointer。

### 代码执行：沙箱和解释器

两条路，课堂要分开讲：

- **沙箱**：实现 `SandboxBackendProtocolV2` 后，模型得到 `execute`，可以装包、跑测试、用操作系统上的文件。提供商包括 LangSmith、AgentCore、Daytona 等。
- **解释器**：增加 `eval`，在隔离的 QuickJS 里跑 JavaScript，用来做循环、批处理和确定性变换。解释器没有 shell、没有装包、没有文件系统和网络。

### 本讲练习

1. 用默认 backend 让代理写 `/notes/today.md`，换一个新的 `thread_id` 再读，观察文件还在不在。
2. 加上 `CompositeBackend`，把 `/memories/` 路由到 `StoreBackend`，用两个线程写同一路径，说明哪一类文件跨线程还在。
3. 写一条权限：拒绝一切对 `**/.env` 的读。让代理读 `.env`，看工具结果。

---

## 第 3 讲　上下文、子代理、人在回路

### 目标

- 说明长任务怎样同时用「卸载、摘要、隔离、长期文件」保住上下文。
- 配置同步子代理，说清 `isolated` 和 `fork`。
- 在工具执行前暂停，并用 `approve` / `edit` / `reject` 恢复。

### 上下文的四段流水

官方把一次长任务的上下文看成四段：

1. **输入**：系统提示、memory、skill 摘要、工具说明
2. **压缩**：大工具结果写到虚拟文件，过长历史由 `SummarizationMiddleware` 摘要
3. **隔离**：子代理在自己的上下文里做重活，只交回一份报告
4. **长期**：要跨线程保留的内容写进 store 或 `AGENTS.md`

Anthropic 和 Bedrock 上，静态系统提示（基础指令、memory、skill 摘要）默认走 prompt cache。其他提供商这条中间件空转。换 skill 集合会改变系统提示，从而让该线程的 cache 失效。

Memory 用 `memory=["./AGENTS.md"]` 打开。这些文件在启动时载入，适合项目约定和稳定偏好。代理也可以用 `edit_file` 改它们。和 Skills 的差别放到第 4 讲的对照表里讲。

规划工具单独打开：

```python
from deepagents import create_deep_agent
from langchain.agents.middleware import TodoListMiddleware

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    middleware=[TodoListMiddleware()],
)
```

`write_todos` 把任务记在 agent state 里，状态是 `pending`、`in_progress`、`completed`。适合步骤多、或界面要展示进度的任务。v0.7 之前它曾默认开启，现在必须显式加入。

### 同步子代理

主代理通过内置的 `task` 工具派活。同步子代理会挡住主代理，直到交回 **一份** 报告。子代理不能往主对话里连续发多条消息。

默认有一个 general-purpose 子代理。自定义时传字典：

| 字段 | 要求 | 说明 |
| --- | --- | --- |
| `name` | 必填 | `task` 用这个名字选择子代理，也会出现在消息元数据里 |
| `description` | 必填 | 主代理靠它决定何时委派。写成具体动作 |
| `system_prompt` | `isolated` 时必填 | 不继承主代理的系统提示 |
| `mode` | 可选 | `"isolated"`（默认）或 `"fork"` |
| `tools` | 可选 | 写出后 **整表替换** 继承来的工具。只留这一子任务需要的 |
| `model` | 可选 | 省略则用主代理的模型 |
| `middleware` | 可选 | 不继承主代理额外加的中间件 |
| `interrupt_on` | 可选 | 覆盖主代理的审批配置。需要 checkpointer |
| `skills` | 可选 | 不继承主代理的技能。general-purpose 才会继承 |
| `response_format` | 可选 | 子代理的结构化结果 |

```python
from langchain.agents.middleware import TodoListMiddleware
from deepagents import create_deep_agent

'''
字典这条仍是 Deep Agent 的子代理，默认的文件工具、摘要那些还在，你加的中间件再并进去。同名的就地替换，其余插进栈里。它不继承主代理额外加的中间件，所以主代理有 TodoListMiddleware，子代理也不会自动有。
'''
researcher = {
    "name": "researcher",
    "description": "检索一个明确主题，交回带来源的要点。一次只接受一个主题。",
    "system_prompt": "你只做检索。先搜索，再交回要点和 URL。",
    "tools": [web_search],
    "middleware": [TodoListMiddleware()],
}

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    subagents=[researcher],
)

'''
已经用 create_agent 搭好一张图时，用 CompiledSubAgent 包一下再传入。runnable 必须是编译好的图，并且 state 里要有 messages
'''
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from deepagents import CompiledSubAgent, create_deep_agent

researcher_graph = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[web_search],
    system_prompt="你只做检索。先搜索，再交回要点和 URL。",
    middleware=[TodoListMiddleware()],
)

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    subagents=[
        CompiledSubAgent(
            name="researcher",
            description="检索一个明确主题，交回带来源的要点。一次只接受一个主题。",
            runnable=researcher_graph,
        )
    ],
)
```

`mode` 决定子代理看得到什么：

- `isolated`：只看见这次委派的任务。重检索、长网页放这里，主代理上下文保持短。
- `fork`：继承父对话和父系统提示。适合必须接着上文做判断的子任务。`system_prompt` 在 fork 里是可选的补充说明。

主代理要在 **同一次** 回复里发出多个 `task` 调用，子任务才会并行。每个子代理仍只返回一份结果。

异步子代理是另一套工具：launch、check、update、cancel。主代理可以在后台任务跑着的时候继续和用户说话。本课的研究案例用同步子代理；异步作为了解项。

### 人在回路

`interrupt_on` 把指定工具的执行变成一次暂停。必须传入 checkpointer，否则无法停住再恢复。

四种决定：

| type | 效果 |
| --- | --- |
| `approve` | 按原参数执行 |
| `edit` | 先改参数再执行。改动过大时，模型可能重新规划 |
| `reject` | 不执行，把说明交回模型。拒绝有副作用的工具用这个 |
| `respond` | 把人的话当成工具结果，工具本身不跑。用于「向用户提问」这类工具 |

`respond` 的文本会被模型当成一次成功的工具结果。拒绝删除、拒绝发信，用 `reject` 并写明不要重试。

```python
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from deepagents import create_deep_agent

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    interrupt_on={
        "write_file": True,  #表示每次写文件都暂停，四种决定都能用
        "edit_file": {"allowed_decisions": ["approve", "edit", "reject"]}, #那条收窄了选项，只留 approve、edit、reject。
    },
    checkpointer=MemorySaver(),
)

config = {"configurable": {"thread_id": str(uuid7())}}
result = agent.invoke(
    {"messages": [{"role": "user", "content": "把笔记写到 /notes/a.md"}]},
    config=config,
)

if result.get("__interrupt__"):
    pending = result["__interrupt__"][0].value
    for action in pending["action_requests"]:
        print(action["name"], action["args"])

    result = agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config=config,
    )
```

一次暂停里可能有多个 `action_requests`。`decisions` 要按同样顺序各给一条。`edit` 的形状是：

```python
{
    "type": "edit",
    "edited_action": {
        "name": "write_file",
        "args": {"file_path": "/notes/a.md", "content": "改过的正文"},
    },
}
```

`reject` 带 `message`，告诉模型工具没执行、下一步做什么。恢复时必须用同一个 `config`。

`InterruptOnConfig` 还可以加 `when`。谓词返回 `False` 时这次调用直接执行，不会进入待审列表。

权限规则里的 `mode="interrupt"` 和 `interrupt_on` 走同一套恢复流程。前者按路径拦截写入，后者按工具名拦截。

### 本讲练习

1. 打开 `TodoListMiddleware`，让代理把「列三个标题、各写一段、合成一个文件」记成待办，观察 state 里的 todo。
2. 加一个只带搜索工具的 `researcher`。比较主代理消息条数：检索发生在子代理里，主对话只多出一份报告。
3. 对 `write_file` 开启审批，分别恢复 `approve` 和 `reject`，看文件是否出现、模型下一句怎么说。

---

## 第 4 讲　Skills 基础

### 目标

- 按 Agent Skills 规范摆放目录和 `SKILL.md`。
- 讲清三级加载：元数据、正文、附属文件。
- 把 Skills、Memory、Tools 分到三种不同的用途上。

### Skills 解决什么问题

长流程如果全部写进 `system_prompt`，每次调用都在为用不到的步骤付 token。Skills 把流程、领域知识和附属材料放进目录：启动时模型只看见每份技能的名字和描述，判断相关之后才去读正文。

适合放进 skill 的内容：

- 多步流程
- 工具该按什么顺序用、材料从哪来
- 和脚本绑在一起的固定步骤
- 格式、风格、必须跑测试这类守则

稳定且每次都相关的约定，放 Memory（`AGENTS.md`）。要真正执行的动作，做成 Tools。

| | Skills | Memory | Tools |
| --- | --- | --- | --- |
| 用途 | 按任务取出的流程和知识 | 每次都要的持久上下文 | 可调用的程序 |
| 何时进入上下文 | 模型判断相关，或被 pin | 启动时 | 每一轮都在工具列表里 |
| 格式 | 子目录中的 `SKILL.md` | `AGENTS.md` | 函数 |
| 多个来源 | 同名时后者覆盖前者 | 内容合并 | 创建 agent 时绑定 |

代理也可以在工作中改写 skill，于是 skill 又能充当「用到才读」的记忆。组织制度和批准过的技能库应保持只读，第 5 讲讲怎么封死写入。

### 目录必须多一层

`skills=` 指向 **技能集合目录**。它的子目录才是一份 skill。直接指向含有 `SKILL.md` 的那一层，什么都加载不到。

```text
skills/
  langgraph-docs/
    SKILL.md
    scripts/
      fetch_docs.py
    references/
      api-patterns.md
    assets/
      report-template.md
```

`scripts/`、`references/`、`assets/` 是规范里的三个可选目录。Deep Agents 在发现阶段和激活阶段都不会自动读它们，只有 `SKILL.md` 点了名，模型才会去读或执行。

### `SKILL.md`

文件以 YAML frontmatter 开头，后面是给模型看的步骤。Deep Agents 遵循 Agent Skills 规范。单个文件超过 10 MB 会被跳过。

| 字段 | 是否必填 | 约束 |
| --- | --- | --- |
| `name` | 必填 | 小写字母、数字、连字符，1–64 字符，并且等于父目录名 |
| `description` | 必填 | 做什么、什么时候用。最多 1024 字符 |
| `license` | 可选 | 许可证 |
| `compatibility` | 可选 | 环境要求，最多 500 字符 |
| `metadata` | 可选 | 自定义键值 |
| `allowed-tools` | 可选 | 空格分隔的预批准工具，实验字段 |

`description` 是发现阶段模型唯一能看到的句子。要同时写清能力和触发词。相关技能的描述必须能区分开；描述重叠时，模型会选错或犹豫。技能数量变多之后，宁可合并，把细节放进 `references/`。

正文建议控制在约 500 行、5000 token 以内。结构用步骤、判断标准、输入输出例子、要上报的边界情况。对附属文件的引用只做一层，避免模型连续读三四次才碰到真正的材料。

示例：

```md
---
name: langgraph-docs
description: >-
  回答 LangGraph 的用法和概念时使用。用户提到 StateGraph、checkpoint、
  interrupt 或 LangGraph 文档时启用。
---

# langgraph-docs

## 步骤

1. 用 `fetch_url` 读取 https://docs.langchain.com/llms.txt 。
2. 从索引里选 2 到 4 个最相关的 URL。
3. 再抓取这些页面，先给结论，再给必要上下文，并附上页面链接。

## 附属文件

- `references/api-patterns.md`：常见图 API 的短例，用户问到具体 API 时再读。
```

### 挂到 agent 上

路径使用正斜杠，相对 backend 根目录。省略 `skills` 则完全不加载。同名技能以后出现的来源为准。

用本机目录时：

```python
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

backend = FilesystemBackend(root_dir="/绝对路径/my-project", virtual_mode=True)

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    backend=backend,
    skills=["/skills/"],
)
```

`root_dir` 下面已经存在的技能，`FilesystemBackend` 可以直接读。`virtual_mode=True` 把访问限制在这个根目录内。

默认的 `StateBackend` 没有磁盘。技能内容要在调用时放进 `files`，并用 `create_file_data()` 包装。裸字符串不会被当成文件。

```python
from deepagents.backends.utils import create_file_data

skill_md = open("skills/langgraph-docs/SKILL.md", encoding="utf-8").read()

result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "LangGraph 的 checkpoint 做什么？"}],
        "files": {
            "/skills/langgraph-docs/SKILL.md": create_file_data(skill_md),
        },
    },
    config={"configurable": {"thread_id": "1"}},
)
```

这里的 `skills` 参数仍然是集合目录，例如 `["/skills/"]`，文件键要落在这个目录的子路径上。

`StoreBackend` 用 `backend.upload_files()` 预先写入，再把同一个 backend 交给 `create_deep_agent`。

SDK 只加载你传进去的路径。它不会自动扫描 `~/.deepagents` 或 `~/.agents`。

### 三级加载

| 级别 | 载入什么 | 何时 |
| --- | --- | --- |
| 1. 元数据 | frontmatter 的 `name`、`description` | 启动时，每一份已配置的技能 |
| 2. 正文 | `SKILL.md` 去掉 frontmatter 后的正文 | 技能被调用时，经 `read_file` |
| 3. 资源 | `scripts/`、`references/`、`assets/` | 正文要求读取或执行时 |

`SkillsMiddleware` 负责前两级，并放在你的中间件之后、prompt cache 之前，这样它看到的是摘要之后的对话和真正被调用的模型。第三级由模型按正文自己去读。

有 checkpointer 时，技能列表在线程里只加载一次，之后的模型调用复用这份列表。没有 checkpointer 的本地运行每次都会重新加载，所以改文件会「好像自动生效」。部署到带持久 checkpointer 的环境后，同一线程会一直拿着旧列表。重新加载在第 5 讲。

### 本讲练习

1. 建一个 `skills/pdf-notes/`，描述里写明「用户提到 PDF、表格抽取时使用」。正文只写三步，把字段表挪到 `references/fields.md`。
2. 用一个含糊描述（「处理文档」）再跑同一句用户请求，比较模型有没有去读 `SKILL.md`。
3. 故意把 `skills=` 指到 `.../pdf-notes/` 本身，确认启动时技能列表为空，再改回父目录。

---

## 第 5 讲　Skills 进阶

### 目标

- 按角色选择技能源，或让同一路径在不同租户下指向不同库。
- 在用户点名技能时 pin，在技能文件变更后让线程重新加载。
- 给子代理单独的技能，并控制技能目录只读、可写或写入前要审批。
- 说明 skill 脚本要进沙箱才能执行。

### 谁能看见哪些技能

在创建 agent 之前决定 `skills` 列表，适合「同一套文件，按角色发不同路径」：

```python
SKILLS_BY_ROLE = {
    "engineering": ["/skills/engineering/"],
    "support": ["/skills/support/"],
}

def create_agent_for_user(role: str):
    return create_deep_agent(
        model="anthropic:claude-sonnet-4-6",
        skills=SKILLS_BY_ROLE.get(role, []),
    )
```

多个库一起可见时，按优先级从低到高传入。同名技能以后者为准：

```python
agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    skills=["/skills/org/", "/skills/team/", "/skills/request/"],
)
```

同一路径要按用户、租户或环境落到不同文件时，路径保持 `skills=["/skills/"]`，由 backend 的 namespace 隔离：

```python
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    skills=["/skills/"],
    backend=CompositeBackend(
        default=StateBackend(),
        routes={
            "/skills/": StoreBackend(
                namespace=lambda rt: (
                    rt.server_info.assistant_id,
                    rt.server_info.user.identity,
                ),
            ),
        },
    ),
)
```

组织级共享库用 org id 做 namespace，并配只读规则。用户自己的库用 user id。Store 里的键会去掉 `/skills/` 前缀，种子数据写成 `/company-policies/SKILL.md` 这种路径，并带上 `content` 和 `encoding`。

### 重新加载

需要 `deepagents>=0.7.16`。线程里已经缓存的技能列表，用 `skills_metadata: None` 清掉，下一次运行会重新扫描来源。

```python
config = {"configurable": {"thread_id": "1"}}

agent.invoke(
    {
        "messages": [{"role": "user", "content": "按最新的查阅步骤回答。"}],
        "skills_metadata": None,
    },
    config=config,
)
```

也可以在两次运行之间 `agent.update_state(config, {"skills_metadata": None})`。

空列表的含义是「扫描过，结果是零份技能」，线程后半段会一直没有技能。要重新扫描必须传 `None`。

中间件里若要重置，放在 `after_agent`。放在 `after_model` 会让 **这一轮剩余的模型调用** 看到一份空的技能列表。重新加载如果发现技能集合变了，系统提示改变，prompt cache 失效；集合没变则提示不变。

### Pin：用户已经点名了哪份技能

需要 `deepagents>=0.7.23`。模型默认自己决定读哪份 `SKILL.md`。用户在界面里输入 `/langgraph-docs` 或从菜单选中时，应用把名字放进 `pinned_skills`。Deep Agents 不会自己从消息正文里解析斜杠命令，解析是应用的事。

```python
result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "怎么给图加 checkpointer？"}],
        "pinned_skills": ["langgraph-docs"],
    },
    config={"configurable": {"thread_id": "1"}},
)
```

`SkillsMiddleware` 在下一次模型调用前读取对应 `SKILL.md`，去掉 frontmatter，按传入顺序各追加一条 `HumanMessage`。消息上 `additional_kwargs["lc_source"] == "pinned_skill"`，`additional_kwargs["skill"]` 里有 `name`、`path`、`description`，界面可以只显示标签。

Pin 之后这条消息按当时的正文留在对话里。以后再改磁盘上的 `SKILL.md`，旧消息不变，因此已有的 prompt cache 仍然有效。再次 pin 会再追加一份当时的正文。找不到的名字会被跳过并记调试日志。Pin 也会像模型自己读过正文一样，把该技能声明的工具交出来。

### 子代理的技能不自动继承

- general-purpose 子代理继承 `create_deep_agent(..., skills=...)`。
- 自定义子代理不继承。要在它自己的字典里写 `skills`。
- 两边的技能状态互相不可见。

```python
research_subagent = {
    "name": "researcher",
    "description": "按研究技能检索一个主题并交回要点。",
    "system_prompt": "遵循已启用的研究技能。只交回要点和 URL。",
    "tools": [web_search],
    "skills": ["/skills/researcher/"],
}

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    skills=["/skills/main/"],
    subagents=[research_subagent],
)
```

主代理和 general-purpose 使用 `/skills/main/`。`researcher` 只使用 `/skills/researcher/`。

### 只读、可写、写入前审批

能看见路径，和允许改文件，是两件事。

只读的批准库：路由到存有正式技能的 backend，再拒绝该路径上的写。

```python
from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    store=store,
    skills=["/skills/"],
    backend=CompositeBackend(
        default=StateBackend(),
        routes={
            "/skills/": StoreBackend(
                namespace=lambda rt: ("curated-skills", rt.context.org_id),
            ),
        },
    ),
    permissions=[
        FilesystemPermission(
            operations=["write"],
            paths=["/skills/**"],
            mode="deny",
        ),
    ],
)
```

又要批准库、又要个人可写库时：两条路由，`skills` 两个路径都传入，只对 `/skills/approved/**` 写 `deny`。代理用 `write_file` 和 `edit_file` 更新可写路径下的 `SKILL.md`。跨会话的一般性偏好另走 `/memories/`，那是 Memory，不是技能格式。

写入前要人看过：

```python
from langgraph.checkpoint.memory import MemorySaver

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    skills=["/skills/editable/"],
    permissions=[
        FilesystemPermission(
            operations=["write"],
            paths=["/skills/**"],
            mode="interrupt",
        ),
    ],
    checkpointer=MemorySaver(),
)
```

若是「所有文件写入都要批」，用 `interrupt_on={"write_file": True, "edit_file": True}`。恢复方式与第 3 讲相同。

共享记忆有注入风险：一个用户写进技能或记忆，另一个用户下次会把它当指令。共享库默认只读，由应用代码写入；必须让代理写共享路径时，加上 interrupt。

### 脚本要有 shell 才跑得起来

没有 `execute` 时，skill 是被动说明：模型读完步骤，再用已有工具去做。把固定逻辑放进 `scripts/`，并在正文里写明何时运行，模型就可以执行已经测过的脚本，而不必每次把脚本重新写一遍。

模型可以从任何 backend 读脚本，但 **执行** 需要沙箱提供的 shell。技能文件如果在沙箱外面，容器里看不到。做法是用中间件在 `abefore_agent` 把 store 里的技能上传进沙箱，在 `after_agent` 把沙箱里改过的文件写回 store。解释器的 `eval` 不能用来跑这些脚本。

### 排错清单

| 现象 | 先检查 |
| --- | --- |
| 技能从不触发 | `description` 是否写了触发场景；是否该改为 pin |
| 启动时一份技能都没有 | `skills` 是否指向集合目录的上一级；StateBackend 是否用 `create_file_data` 放进了 `files` |
| 改了 `SKILL.md` 仍走旧步骤 | 有 checkpointer 时是否把 `skills_metadata` 设成了 `None` |
| 附属文件找不到 | 正文有没有写相对路径；引用是否超过一层 |
| 子代理不会用主代理的技能 | 自定义子代理要自己写 `skills` |
| 脚本报无法执行 | backend 是不是沙箱；文件有没有被传进容器 |
| 写入被拒绝或一直停住 | 权限规则顺序；interrupt 是否配了 checkpointer |

### 本讲练习

1. 准备 `/skills/org/` 和 `/skills/team/` 里同名的 `report-style`，确认生效的是列表中靠后的那份。
2. 用 `pinned_skills` 强制启用一份描述写得很泛的技能，看模型是否直接按正文做。
3. 给 `/skills/**` 加写拒绝，让代理「改进这份技能」，确认 `SKILL.md` 没有被改掉。

---

## 第 6 讲　深度研究 Agent

### 目标

把前五讲收成一个研究代理：

1. 用 `write_todos` 拆开问题。
2. 检索只发生在子代理里，主代理上下文只接收要点。
3. 报告的结构和引用格式放在 skill 里，按需读取。
4. 最终报告写到虚拟文件 `/final_report.md`。

官方有一篇完整教程：[Build a deep research agent](https://docs.langchain.com/oss/python/deepagents/deep-research)。那篇用 Tavily 发现 URL，再把网页转成 Markdown，并给了很长的委派限额提示。下面是课堂用的缩短版，API 与教程一致，提示词按本课重写。生产流程以官方那篇为准。

### 流程

```text
用户问题
  → 主代理 write_todos
  → 主代理把原题写到 /research_request.md
  → 主代理 task() 给 researcher（比较类问题一次发多个 task）
  → 子代理 web_search，交回要点和 URL
  → 主代理判断要不要读 research-report 技能
  → 主代理写 /final_report.md
  → 主代理读回请求文件，核对问题是否被覆盖
```

检索放在子代理里，是因为网页正文很长。隔离之后，主代理只看到压缩过的发现。报告版式放在 skill 里，是因为闲聊和检索不需要这份版式；用户真的要报告时再读。

### 依赖

```bash
pip install deepagents tavily-python
export TAVILY_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

搜索可以换成课程提供的任何检索函数。Tavily 只是官方教程里的一个具体工具。

### 搜索工具

```python
import os

from langchain.tools import tool
from tavily import TavilyClient

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


@tool
def web_search(query: str) -> str:
    """搜索网页。返回标题、URL 和摘要。一次查询只回答一个具体问题。"""
    data = tavily_client.search(query, max_results=3)
    blocks = []
    for item in data.get("results", []):
        blocks.append(
            f"## {item.get('title', '')}\nURL: {item.get('url', '')}\n\n{item.get('content', '')}"
        )
    if not blocks:
        return f"没有找到与「{query}」相关的结果。"
    return "\n\n---\n\n".join(blocks)
```

官方教程还会把 URL 再抓成全文。全文更适合核对引用，也更容易撑满上下文，正好用来演示「大工具结果会被卸载到虚拟文件」。课堂若时间不够，摘要版已经能把子代理讲清楚。

### 报告技能

```text
skills/
  research-report/
    SKILL.md
```

```md
---
name: research-report
description: >-
  把已收集的研究发现写成最终报告时使用。用户要求报告、综述、对比文，
  或主代理准备写入 /final_report.md 时启用。
---

# research-report

只在研究发现已经交回之后使用本技能。不要用它代替检索。

## 步骤

1. 读 `/research_request.md`，列出必须回答的点。
2. 合并子代理交回的 URL。同一个 URL 只保留一个编号。
3. 用 `write_file` 把报告写到 `/final_report.md`。
4. 再读一遍该文件，核对每个必答点都有对应段落和引用。

## 报告结构

- 标题
- 直接回答（一段）
- 分点发现，每点末尾用 `[n]`
- 若是对比题：各自概述、对照、结论
- 参考文献：编号、标题、URL

没有来源的句子不要写成事实。来源不够时，在文末列出缺口。
```

### 主代理

```python
from datetime import datetime
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware import TodoListMiddleware

ROOT = Path(__file__).resolve().parent

RESEARCHER_PROMPT = """你是检索子代理。
一次只研究委派说明里的那一个主题。
必须调用 web_search。
交回：3 到 6 条要点、每条对应的 URL、你还没覆盖的缺口。
不要写最终报告，不要再派生子任务。
今天的日期：{date}
"""

ORCHESTRATOR_PROMPT = """你是研究主编。
对每个研究请求按顺序做：
1. 用 write_todos 列出检索任务。简单事实题只列 1 条；明确的对比题按对比对象各列 1 条。
2. 用 write_file 把用户原题写到 /research_request.md。
3. 用 task 把检索交给 researcher。需要并行时，在同一次回复里发出多个 task。
4. 你自己不要调用 web_search。
5. 材料够了就按 research-report 技能写 /final_report.md。
6. 读回 /research_request.md，确认每个要求都被报告覆盖。
最多做 3 轮委派。来源不足就在报告里写明缺口。
"""

research_subagent = {
    "name": "researcher",
    "description": "检索一个明确主题并交回要点和 URL。一次只给一个主题。",
    "system_prompt": RESEARCHER_PROMPT.format(date=datetime.now().strftime("%Y-%m-%d")),
    "tools": [web_search],
}

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[web_search],
    system_prompt=ORCHESTRATOR_PROMPT,
    subagents=[research_subagent],
    middleware=[TodoListMiddleware()],
    backend=FilesystemBackend(root_dir=str(ROOT), virtual_mode=True),
    skills=["/skills/"],
)


if __name__ == "__main__":
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "比较 Deep Agents 的子代理隔离和把所有检索都放在主代理里，对上下文长度有什么差别。",
                }
            ]
        }
    )
    print(result["messages"][-1].content)
```

对照前几讲，这个文件里每一样东西的来历：

| 代码 | 来自 |
| --- | --- |
| `FilesystemBackend(..., virtual_mode=True)` | 第 2 讲。技能和报告都在项目根下，路径出不了根目录 |
| `skills=["/skills/"]` | 第 4 讲。指向集合目录，真正的技能在 `skills/research-report/` |
| `TodoListMiddleware` | 第 3 讲。v0.7 起要自己加，才会有 `write_todos` |
| `subagents=[research_subagent]` | 第 3 讲。`isolated` 是默认。子代理的工具表只有 `web_search` |
| 主代理的 `tools` 仍带 `web_search` | 工具要先挂在主代理上，子代理才能在覆盖列表里使用它。提示词要求主编自己不要调用 |
| 未传 `interrupt_on` | 课堂演示让写入直接发生。加上之后，`/final_report.md` 会先暂停等人批 |

自定义子代理不继承 `/skills/`。本例子让 **主代理** 写报告，所以报告技能放在主代理上是对的。若改成子代理写报告，要在 `research_subagent` 里加 `"skills": ["/skills/research-report/"]` 的上一级，也就是 `"/skills/"`，或把报告技能单独放进只给子代理的集合目录。

### 课堂演示时看什么

1. 待办是不是先出现，再开始 `task`。
2. 主代理消息里有没有大段网页。大段应留在子代理轨迹里。
3. 模型在写报告之前有没有 `read_file` 读 `SKILL.md`。没有的话，先看 `description` 是否包含「报告」一类触发词，或在这次调用里传 `pinned_skills: ["research-report"]`。
4. `/final_report.md` 里的编号和 URL 能不能对上子代理交回的来源。

多轮修改报告时加上 `MemorySaver` 和固定的 `thread_id`。技能文件若在线程中途被改过，下一次调用前把 `skills_metadata` 设为 `None`。

### 本讲练习

1. 把用户题换成「只问一个事实」。看待办条数和 `task` 次数是不是停在 1。
2. 换成「A、B、C 三家对比」。确认多个 `task` 出现在同一次模型回复里。
3. 给 `write_file` 加 `interrupt_on`，在恢复前修改即将写入的标题，再批准。
4. 删掉 `skills`，比较报告结构是否还遵守参考文献格式。再 pin `research-report`，看结构是否回来。

---

## 讲完时学员应能回答的问题

1. 只传 `model` 时，代理为什么已经能写文件、已经能 `task`？
2. `StateBackend` 和 `StoreBackend` 分别在什么边界上保留文件？
3. 权限规则没有命中时，读写会发生什么？
4. 为什么重检索放在 `isolated` 子代理里？
5. `reject` 和 `respond` 分别把什么交回给模型？
6. `skills=[".../my-skill/"]` 和 `skills=[".../skills/"]` 哪一个能加载？
7. 三级加载各自由谁执行？
8. 有 checkpointer 时，改了 `SKILL.md` 为什么线程里还是旧的？清空该用 `None` 还是 `[]`？
9. 自定义子代理怎样才能看见一份技能？
10. 深度研究里，工具、todo、子代理、skill、虚拟文件各承担哪一步？
