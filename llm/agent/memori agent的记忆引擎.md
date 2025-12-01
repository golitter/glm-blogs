> memori：让AI代理不再遗忘！Memori 是面向开发者的开源记忆引擎，为更智能、更高效的AI应用提供持久上下文。可以从交互中学习，并跨会话保持上下文

双系统：

- **主动模式**：
  - 一次性工作记忆：每个会话开始时，仅注入一次最核心的记忆
  - 后台分析：每6个小时分析一次对话模式，进行深度学习
  - 核心记忆提升：将关键的个人事实提升至即时访问状态
  - 类人记忆：模拟短期记忆
  - 性能优化：极低的令牌消耗
- **自动模式**：
  - 动态上下文搜索：分析每一次查询
  - 全库搜索：智能搜索整个记忆数据库
  - 上下文感知注入：每次调用llm时，精确注入3-5个最相关的记忆
  - 检索代理：由ai驱动的搜索策略
  - 丰富上下文：使用更多的token，换取最大化上下文感知能力

通过`memori.enable()`注册回调函数，拦截所有llmapi调用，再请求/响应双向注入记忆数据。

三层代理架构：

1. **对话内容结构化提取**：使用openai structured outputs api提取内容
2. **智能记忆检索**：llm理解查询意图；多策略搜索；5分钟查询缓存
3. **长期记忆优化**：后台分析记忆模式，提升重要记忆到短期上下文

# 使用

创建：

```shell
conda create -n agent_mmy python=3.10
uv pip install memorisdk
```



接入deepseek api：

```python
from memori.core.providers import ProviderConfig

DEEPSEEK_API_KEY = "sk-abc"

deepseek_config = ProviderConfig.from_custom(
    base_url="https://api.deepseek.com" ,
    api_key=DEEPSEEK_API_KEY,
    model="deepseek-chat"
)
client = deepseek_config.create_client()

try:
    print("正在向 DeepSeek API 发送请求...")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "你好！请介绍一下你自己。"},
        ],
        temperature=0.7,
    )

    print("\n--- DeepSeek API 响应 ---")
    print(response.choices[0].message.content)

except Exception as e:
    print(f"请求 DeepSeek API 时出错: {e}")

```

之后看看memori性能：

```python
from memori.core.providers import ProviderConfig
from memori import Memori
DEEPSEEK_API_KEY = "sk-abc"

deepseek_config = ProviderConfig.from_custom(
    base_url="https://api.deepseek.com" ,
    api_key=DEEPSEEK_API_KEY,
    model="deepseek-chat"
)

memori = Memori(
    database_connect="sqlite:///deepseek_memory.db",
    provider_config=deepseek_config,
    conscious_ingest=True
)

memori.enable()
client = deepseek_config.create_client()
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你好，我叫田乐蒙！请介绍一下你自己。"},
    ],
    temperature=0.7,
) 

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你还记得我是谁吗？"},
    ],
    temperature=0.7,
)
print("\n--- DeepSeek API 响应 ---")
print(response.choices[0].message.content)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "我喜欢用FastAPI、Mongodb和React来开发应用，你觉得怎么样？"},
    ],
    temperature=0.7,
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你对我的印象是什么？"},
    ],
    temperature=0.7,
)
print("\n--- DeepSeek API 响应 ---")
print(response.choices[0].message.content)
memori.disable()
```

输出：

```shell

--- DeepSeek API 响应 ---
当然记得！你是**田乐蒙**，我们刚刚才打过招呼呢！😊 很高兴再次和你聊天！有什么我可以帮你的吗？

--- DeepSeek API 响应 ---
田乐蒙，根据我们的对话，我对你的印象是：

**技术热情者** 🔧
- 你喜欢使用现代技术栈（FastAPI、MongoDB、React）
- 对全栈开发有明确的偏好和思考
- 技术选择偏向高效、灵活的开发方式

**积极交流者** 💬
- 主动介绍自己并开启对话
- 对技术讨论表现出兴趣
- 愿意分享自己的技术偏好

**有条理的学习者** 📚
- 从简单的问候到技术讨论，对话有自然的递进
- 似乎在探索或确认技术方向

总的来说，你给我的印象是一个对现代Web开发充满热情、思路清晰的技术爱好者！而且从你选择的技术栈来看，你很可能注重开发效率和项目的可维护性。

我很好奇，你目前是在学习这些技术，还是在用它们做具体的项目呢？😊
```

感觉还是可以的。



## 使用memori、langchain、deepseek

```python
from memori.core.providers import ProviderConfig
from memori import Memori, create_memory_tool
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent # langchain > 1.0.0时
from langchain.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field

DEEPSEEK_API_KEY = "sk-abc"

deepseek_config = ProviderConfig.from_custom(
    base_url="https://api.deepseek.com" ,
    api_key=DEEPSEEK_API_KEY,
    model="deepseek-chat"
)

# 初始化 Memori 以实现持久化记忆
memory_system = Memori(
    database_connect="sqlite:///langchain_example_memory.db",
    provider_config=deepseek_config,
    conscious_ingest=True,
    namespace="langchain_example",
)


# 启用记忆系统
memory_system.enable()

# 为代理创建记忆工具
memory_tool = create_memory_tool(memory_system)

class MemorySearchInput(BaseModel):
    """记忆搜索工具的输入。"""

    query: str = Field(
        description="在记忆中搜索的内容 (例如: '关于 AI 的过往对话', '用户偏好')"
    )


class MemorySearchTool(BaseTool):
    """用于搜索代理记忆的 LangChain 工具。"""

    name: str = "search_memory"
    description: str = (
        "在代理的记忆中搜索过去的对话和信息。"
        "使用此工具来回忆之前的交互、用户偏好和上下文。"
    )
    args_schema: type[BaseModel] = MemorySearchInput

    def _run(
        self,
        query: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """使用该工具搜索记忆。"""
        try:
            if not query.strip():
                return "请提供搜索查询"

            result = memory_tool.execute(query=query.strip())
            return str(result) if result else "未找到相关记忆"

        except Exception as e:
            return f"记忆搜索错误: {str(e)}"

memory_search_tool = MemorySearchTool()
# 初始化 LLM
llm = ChatDeepSeek(
    api_key= DEEPSEEK_API_KEY,
    model="deepseek-chat",
    temperature=0,
)
# 创建提示模板
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一个有帮助的 AI 助手，能够记住过去的对话和用户偏好。请始终先检查你的记忆，以提供个性化和有上下文的回复。

指令:
1. 首先，使用 search_memory 工具在你的记忆中搜索相关的过往对话
2. 使用任何相关的记忆来提供个性化的回复
3. 提供有帮助且有上下文的回答
4. 保持对话友好

如果这是第一次对话，请自我介绍并说明你会记住我们的对话。""",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

agent = create_openai_tools_agent(llm, [memory_search_tool], prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[memory_search_tool],
    verbose=False,
    handle_parsing_errors=True,
    max_iterations=5,
)

def chat_with_memory(user_input: str) -> str:
    """使用具有记忆增强功能的 LangChain 代理处理用户输入"""
    try:
        # 使用用户输入运行代理
        result = agent_executor.invoke(
            {
                "input": user_input,
                "chat_history": [],  # 我们改用 Memori 来实现持久化记忆
            }
        )

        # 获取回复内容
        response_content = result.get(
            "output", "抱歉，我无法生成回复。"
        )

        # 将对话存储在记忆中
        memory_system.record_conversation(
            user_input=user_input, ai_output=response_content
        )

        return response_content

    except Exception as e:
        error_msg = f"抱歉，我遇到了一个错误: {str(e)}"
        return error_msg

while True:
    try:
        # 获取用户输入
        user_input = input("您: ").strip()

        # 检查退出命令
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("\nAI: 再见！我会记住我们的对话，以备下次使用。🤖✨")
            break

        if not user_input:
            continue

    
        # 从记忆增强型代理获取回复
        response = chat_with_memory(user_input)

        print(f"AI: {response}\n")

    except KeyboardInterrupt:
        print("\n\nAI: 再见！我会记住我们的对话，以备下次使用。🤖✨")
        break
    except Exception as e:
        print(f"\n错误: {str(e)}")
        print("请重试。\n")

print("\n📊 会话摘要:")
print("- 记忆数据库: langchain_example_memory.db")
print("- 命名空间: langchain_example")
print("\n您的记忆已保存，并将在未来的会话中可用！")
# 1. 你好，我叫田乐蒙，我喜欢用FastAPI、Mongodb和LangChain构建应用程序。
# 2. 我是谁，我喜欢什么？
```





输出：

```shell
您: 你好，我叫田乐蒙，我喜欢用FastAPI、Mongodb和LangChain构建应用程序。
AI: 你好田乐蒙！很高兴认识你！我是你的AI助手，我会记住我们的对话和你的偏好，以便为你提供更个性化和有上下文的帮助。

听起来你对FastAPI、MongoDB和LangChain很熟悉，这是一个很棒的组合！FastAPI是一个现代、快速的Python Web框架，MongoDB是灵活的NoSQL数据库，而LangChain则是构建LLM应用的强大工具。你用这个技术栈构建过什么有趣的应用吗？

我会记住你的名字和技术偏好，这样在未来的对话中，我可以更好地理解你的背景，为你提供更相关的建议和帮助。

有什么我可以帮你的吗？无论是关于FastAPI开发、MongoDB优化、LangChain应用，还是其他技术问题，我都很乐意协助你！

您: 我是谁，我喜欢什么？
AI: 根据我的记忆，你是**田乐蒙**（Tian Lemeng）。你喜欢使用**FastAPI、MongoDB和LangChain**来构建应用程序。

很高兴认识你！看起来你是一位对现代Web开发技术很感兴趣的开发者。FastAPI是一个很棒的Python Web框架，MongoDB是流行的NoSQL数据库，而LangChain则是构建AI应用的重要工具。这个技术栈很适合构建智能的、数据驱动的Web应用。

有什么关于这些技术的问题，或者你想讨论什么项目想法吗？我很乐意帮助你！

您: 
```











[GibsonAI/Memori: Open-Source Memory Engine for LLMs, AI Agents & Multi-Agent Systems (github.com)](https://github.com/GibsonAI/Memori)

[Docs | Memori – The memory fabric for enterprise AI (memorilabs.ai)](https://memorilabs.ai/docs/)

[Memori/examples/integrations/langchain_example.py at main · GibsonAI/Memori (github.com)](https://github.com/GibsonAI/memori/blob/main/examples/integrations/langchain_example.py)

[GibsonAI/Memori | DeepWiki](https://deepwiki.com/GibsonAI/Memori)