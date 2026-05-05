https://www.youtube.com/watch?v=v4F1gFy-hqg

https://www.bilibili.com/video/BV1avo9B7EiX

## 软件基础比以往任何时候都更重要



### 1

```text
The Ai didn't do what I want
```

《程序员修炼之道》中“没有人能确切知道自己想要什么”。与ai之间存在沟通隔阂。在多人协作中，大家会有一个短暂的相似设计理念。需要与ai达成某种共同认知。

- `/grill-me`：ai进行提问，试图达成共识。
- `/write-a-prd`：转成产品需求文档
- `/write-issues`：转为任务事项

tips：

```text
Before You Code, Reach A Shared Design Concept
```



### 2

```text
The AI is way too verbose
```

有了通用语言，开发者之间的对话和代码的表达都源自同一个领域模型。

- `/ubiquitous-language`：扫描代码库，查找术语。

tips：

```text
Create A Shared Language With The AI
```



### 3

```text
Code That Doesn't Work
```

利用反馈循环。



### 4

```text
Doing way too much
```

始终采取微小、审慎的步骤。反馈的频率就是你的速度限制。永远不要承担过大的任务。

采用测试驱动开发tdd。为了方便测试等，尽量使用深层模块。

> 深层模块：通过简单接口封装大量功能
>
> 浅层模块：功能不多，接口复杂
>
> ai非常容易且擅长写浅层模块。



### 5

```text
AI Doesn't Understand My Code
```

- `/improve-codebase-architecture`：改进代码库



### 6

```text
My Brain Hurts
```



tips：

```text
Design the interface, delegate the implementation
```







🚀 核心干货提炼： 

- 💡 拒绝“逃避编程”：单纯靠修改规格说明来生成代码会导致“垃圾代码堆积”，优秀的软件基础是发挥AI价值的前提。 
- 🔍 GRIME 技能：通过不断深入询问计划细节，与AI达成“共同设计概念（Design Concept）”，解决AI理解偏差的问题。 
- 🗣️ 建立普适语言：利用领域驱动设计（DDD）思想，为开发者与AI构建统一的术语表，消除沟通鸿沟。 
- 🧪 TDD 与反馈循环：利用测试驱动开发强迫AI进行小步迭代，避免其因“行驶过快”而陷入逻辑盲区。 
- 🏗️ 设计接口，委托实现：通过构建“深度模块”而非“浅层模块”，将复杂的内部实现交给AI，人类只需负责设计清晰的边界与接口。