主要的大更新

- **异步工具调用**：LLM理解**异步**概念，之前是通过harness来模拟LLM理解异步，现在LLM原生就理解了。可以通过API进行操作。
- **运行中转向**：通过websocket，用户可以在LLM工作期间追加纠正或新要求。这个也是**LLM原生支持的**。同上，之前是通过harness来实现。
- **对话中调整推理强度并保留缓存**
- **安全监控与边界意识**

感觉主要是前两个的更新似乎很重要，前两个之前是通过harness来实现的功能，现在有一部分成为了LLM的原生能力。虽然LLM原生理解异步和运行中转向，但是还是需要实现对应harness的，只是相对容易些。

另一个大的更新就是computer use能力：

| Benchmark             | GPT-6 Astra | GPT-5.6 Sol | Claude 对比           |
| --------------------- | ----------- | ----------- | --------------------- |
| **ScreenSpot-Pro**    | **92.7%**   | 76.9%       | Claude Fable 5：87.3% |
| **OSWorld 2.0**       | **72.6%**   | 65.7%       | Claude Opus 5：70.2%  |
| **Agents’ Last Exam** | **59.3%**   | 53.6%       | Claude Opus 5：55.5%  |

astra的的视觉定位＋GUI操作的组合能力更强。

目前的GUI Agent、mcp（大多是为了将软件能力暴露给agent进行执行的相关工具）并不太需要了，直接让agent(astra)操作电脑就行，而且token会越来越便宜，这个一定是**可行的**。



关于通用强LLM会不会吞并harness，我想是会吞并一部分的，但是吞并整个harness是绝无可能的。因为很多需求并不像coding那样确切，或者说要唯一确定的coding内容（命名、注释、**唯一功能逻辑**等）agent并不能实现，而其他需求往往就是这**唯一确定的coding内容**。所以skills、mcp等依旧有价值。一些典藏提示词也有价值，例如：去ai味的、gptimage2调整风格的提示词，尽可能的提高**抽SSR**的概率。



[(99+ 封私信 / 68 条消息) 卡卡罗特 的想法: GPT Astra 这次引入了两个极其重要、但很容易被忽略的能力：1. 原生异步工具调用2. 原生消息 steer这里我… - 知乎](https://www.zhihu.com/pin/2079354306327732271?native=1&scene=share&share_code=J3JowBVjXzlT&utm_psn=2079585775075734016)

[刚刚，GPT-6正式发布！OpenAI：欢迎来到AGI时代](https://mp.weixin.qq.com/s/ICv0Dra1PFr7sxAGqO_MVA)