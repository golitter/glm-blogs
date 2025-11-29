qwen3系列包括dense和MoE架构模型。qwen3的关键创新是**将思考模式和非思考模式集成到一个统一的框架中，并能够根据用户需求进行动态切换**，还引入了**思考预算**：允许用户在推理过程中动态分配计算资源，从而平衡延迟及性能。

# 预训练

> 与qwen2.5相比，qwen3显著扩展了训练数据的规模和多样性。
>
> 为了进一步扩展预训练数据语料库，采用qwen2.5-vl模型对大量类pdf文档进行文本识别，然后使用qwen2.5模型对识别出来的文本进行优化，提高质量。
>
> 同时还采用qwen2.5 qwen2.5-math qwen2.5-coder用不同格式合成数据。

预训练阶段分为3个阶段：

1. **通用阶段**：使用超过30t个token进行训练，序列长度为4096个token，数据包含119种语言和方言。
2. **推理阶段**：增加了STEM、代码、推理和合成数据的比例来优化此阶段的数据。使用大约5t更高质量的token进行预训练。
3. **长上下文阶段**：在最后一个预训练阶段，会收集高质量的长上下文语料库，将qwen3的上下文扩展到32768个token。

# 后训练

后训练两个核心目标：

1. **思考控制**：整合“思考”和“非思考”模式，让用户灵活地选择模型是否进行推理，并且可以通过为思考过程指定token预算来控制思考地深度。
2. **强到弱蒸馏**：简化轻量级模型地后训练过程，利用来自大模型的知识，大幅度降低构建更小规模模型所需的计算成本。

![image-20251129142914065](qwen3%20%E6%8A%80%E6%9C%AF%E6%8A%A5%E5%91%8A.assets/image-20251129142914065.png)

后训练包含四个阶段，前两个阶段专注于模型的思考能力，后两个阶段目的是将“非思考”模式整合到模型当中。

## long-CoT 冷启动

策划了一个全面的数据集，数据集中的每个问题都与经过验证的参考答案或代码的测试用例相匹配。

数据集的构建包括两个过滤阶段：query过滤和response过滤。

使用qwen2.5-72b-instruct来识别和去除不容易验证的query。还去除了q2.5-72b-ins在不使用CoT推理情况下就可以正确回答的query，**防止模型不使用深层次推理只依赖表面猜测**。同时，还对每个query进行标注领域，以平衡数据集领域。

之后使用qwen-32b为每个过滤后的query生成N个候选response。当q32b无法正确生成解决方案时，会人工评估这些response的准确性。

对于pass@N为正数的query，会应用更严格的过滤标准，以去除response

1. 答案不正确的response
2. 包含大量重复的response
3. 没有充足推理的情况下明显表现出猜测行为的response
4. 在思考内容和总结内容之间表现出不一致的response
5. 涉及不恰当语言混和或风格转变的response
6. 被怀疑与潜在验证集过于相似的response

之后从数据集中精心选择的子集用于冷启动训练。

**目标**：让模型学会推理模式，而不过分强调推理性能，可以确保模型在后续阶段具有更强大的灵活性和改进潜力。

## reasoning rl

推理rl阶段使用的query需要满足：

1. 没有在冷启动阶段使用过
2. 对于冷启动模型来说是可学习的
3. 尽可能具有挑战性的
4. 涵盖广泛领域

最终收集了3995个query-verifier pair，使用GRPO来更新模型参数。

实验发现：**使用大batch size和每个query大量rollout进行off-policy训练，有助于提高采样效率，对训练过程有益**。

> - **query-verifier pair**：由一个query和一个verifier组成的配对。在大模型强化学习中常用于自动评估模型输出是否正确。
> - **rollout**：大模型rl中，从某个初始状态开始，让模型按照当前策略生成一系列动作，直到终止状态的完整过程。
> - **off-policy**：离策略。用于更新策略的数据，可以不是由当前策略直接生成的，而是来自历史策略、其他或预存的经验池。

## thinking mode fusion

思维模式融合阶段的目标是让“非思考”能力整合到当前开发的“思考”模型中，降低同时部署思考和非思考的成本和复杂性。

在推理模型上进行sft。sft数据集结合了思考和非思考数据，为了确保模型性能不会因额外的sft受到影响，非思考数据是使用第二阶段模型本身对第一阶段query进行拒绝采样生成的。

会在用户query或system设定中，分别引入`/think`和`/no_think`标志，允许模型通过模板中的空思维块来遵循用户的输入并选择恰当的模式。

对于非思考模式样本，会保留一个空思维链，从而确保模型内部的格式一致性，并允许开发人员通过在模板中连接加入空思维链来防止模型思考。

> 默认情况下，模型以思考模型允许。因此会添加一些思考模式训练样本，其中query中不包括`/think`标志。对于更复杂的多轮对话，会在query中随机插入多个`/think`和`/no_think`标志，模型遵循最后遇到的一个标志。

思考融合模式的另一个优点：**偶像学会在非思考和思考模式下回复，会自然地2开发出处理中间情况地能力-基于不完整思维链生成response**，这种能力使得模型可以控制**思考预算**。当模型地思考长度达到用户定义地阈值时，可以手动停止思考过程并插入停止思考指令。

## general rl

通用rl阶段目的是广泛提升模型在各种场景下地能力和稳定性。qwen建立了一套复杂地奖励系统，涵盖超过多个不同任务，每个任务都有定制的评分标准。

核心能力提升：

- **指令跟随**：确保模型能够准确解释和遵循用户指令，包括与内容、格式、输出结构化
- **指令遵循**：期望模型遵守特定的格式约定，在“思考”和“非思考”模型之间切换
- **偏好对其**：对于开放query，侧重于提高模型回复的有用性、参与度和风格
- **agent能力**：涉及训练模型通过指定接口正确调用工具
- **专业场景能力**：针对上下文定制的任务

qwen使用了三种不同类型的奖励：

- **基于规则的奖励**：在reasoning rl阶段使用。可以高精度地评估模型输出的正确性，防止奖励hacking行为
- **基于模型的奖励（带参考答案）**：为每个query提高参考答案，并使用qwen2.5-72b-ins根据此答案对模型的输出进行评分
- **基于模型的奖励（无参考答案）**：利用人类偏好数据，训练一个奖励模型来为模型的response分配标量的分数。

## 强到弱蒸馏

强到弱蒸馏专门用于优化轻量级模型。蒸馏过程主要分为两个主要阶段：

- **off-policy**：初始阶段，结合教师模型在思考和非思考模式下生成的输出进行response蒸馏
- **on-policy**：在次阶段，学硕模型为微调生成on-policy序列。采样prompt，学生模型以思考或非思考生成response。通过将学生模型的logits与教师模型的logits对齐来微调学生模型，最小化KL散度。

# 推理

使用官网提供的transformers库的例子，默认`enable_thinking=True`。

prompt为"介绍一下你自己。"，默认会推理

改为“介绍一下你自己。/no_think”，为**空思维链**

```python
prompt = "介绍一下你自己。/no_think"
messages = [
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True, # Switches between thinking and non-thinking modes. Default is True.
)
```

将`enable_thinking`设置为False，prompt添加`/think`标志也不会进行推理：

```python
prompt = "介绍一下你自己。/think"
messages = [
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False, # Switches between thinking and non-thinking modes. Default is True.
)
```

`enable_thinking`是硬标签，`/think`和`/_no_think`是软标签。



**思考预算**

该功能属于qwen3模型的定制，开源框架中不可用，需要在阿里云百炼api上使用。

对于开源框架，可以通过两次生成来实现这个功能：

1. 第一次生成时，生成的token数量达到思考预算，检查思考过程是否完成。如果没有完成，追加上`early_stopping_text`提示。
2. 第二次生成时，继续生成直到内容结束或达到长度上限。







[2025 LLM 技术报告(4)：Qwen3 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/1907098454011942106)

[Qwen3/Qwen3_Technical_Report.pdf at main · QwenLM/Qwen3 (github.com)](https://github.com/QwenLM/Qwen3/blob/main/Qwen3_Technical_Report.pdf)

[快速开始 - Qwen](https://qwen.readthedocs.io/zh-cn/latest/getting_started/quickstart.html)