传统的监督学习方法存在三个局限性：

1. 数据质量完全决定训练质量：模型只能模仿训练数据，难以超越
2. 缺乏探索能力：只是被动的学习人类提供的路径
3. 难以优化长期目标：无法精确优化多步推理的中间过程

强化学习通过让智能体自主生成多个候选答案并根据正确性获得奖励。

Agentic RL的核心思想：**将llm作为可学习策略，嵌入智能体的感知-决策-执行循环，通过强化学习优化多步任务表现**。

## llm训练全景

强大的llm通常要经历两个主要阶段：**预训练（pre-training）和后训练（post-training）**。

### 预训练

llm训练的第一阶段，目标是让模型学习语言的基本规律和世界知识。这个阶段使用海量的文本数据，通过自监督学习的方式训练模型。常见的预训练任务是因果语言建模，即下一个词预测。

预训练阶段数据量大、计算成本高，学到的是通用语言理解和生成能力。

### 后训练

后训练通常包括三个步骤

#### 监督微调（SFT）

目标是让模型学会遵循指令和对话格式，训练数据是$(prompt, completion)$对，训练目标与预训练类似，仍然是最大化正确输出的概率：
$$
\mathcal{L}_{\text{SFT}} = -\sum_{i=1}^{N} \log P(y_i \mid x_i; \theta)
$$
$x_i$是输入提示$(prompt)$，$y_i$是期望的输出，$N$是训练样本数量。监督微调的特定是数据量较小，需要人工标注、快速见效，主要学习任务格式和基本能力。

#### 奖励建模（RM）

监督微调后的模型可以遵循指令，但生成的回答质量参差不齐。需要一个方式来评估回答的质量，因此采用奖励模型。

奖励模型的训练数据是偏好对比数据，包含同一个问题的两个回答，一个好，一个差。奖励模型的训练目标是学习人类的偏好：
$$
\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x,y_w,y_l)} \left[ \log \sigma \left( r_\phi(x,y_w) - r_\phi(x,y_l) \right) \right]
$$
其中，$r_{\phi}(x,y)$是奖励模型，输入是$(提示，回答)$对，输出是质量分数；$y_w$是更好的回答，$y_l$是更差的回答，$\sigma$是sigmoid函数，目标是让奖励模型给更好的回答更高的分数。

#### 强化学习微调

有了奖励模型后，可以用强化学习来优化语言模型，让其生成更高质量的回答。最经典的算法是PPO(Proximal Policy Optimization)，训练目标为：
$$
\mathcal{L}_{\text{PPO}} = \mathbb{E}_{x,y \sim \pi_\theta} \left[ r_\phi(x,y) \right] - \beta \cdot D_{\text{KL}} \left( \pi_\theta \parallel \pi_{\text{ref}} \right)
$$
其中$\pi_\theta$是当前策略，即语言模型，$\pi_{ref}$是参考策略，可以是SFT模型，$r_{\phi}(x,y)$是奖励模型的评分，$D_{KL}$是KL散度，目的是为了防止模型偏离太远，$\beta$是平衡系数。这个目标函数的含义是：最大化奖励，但是不要偏离原始模型太远。



传统的RLHF（Reinforcement Learning from Human Feedback）需要大量人工标注偏好数据，成本高昂。为了降低成本，有人提出了RLAIF（Reinforcement Learning from AI Feedback），用强大的模型替代人类标注员。

## Agentic RL

传统的后训练主要关注单论对话的质量优化：给定一个用户问题，模型生成一个回答，然后根据回答的质量获得奖励。这种方式适合优化对话助手，但是对于多步推理、工具使用、长期规划的任务来说，不ok。

Agentic RL是一种新的范式，将llm视为一个可学习的策略，嵌入在一个顺序决策循环中。在这个框架下，智能体需要在动态环境中与外部世界交互，执行多步行动来完成复杂任务，获得中间反馈来指导后续决策，优化长期累积奖励而非单步奖励。

强化学习是基于马尔可夫决策过程（Markov Decision Process，MDP）框架进行形式化的。MDP是由五元组（S，A，P，R，$\gamma$定义：

- 状态空间$S$
- 行动空间$A$
- 状态转移函数$P(s^{`}|s,\alpha)$
- 奖励函数$R(s,\alpha)$
- 折扣因子$\gamma$

| 对比 | 后训练          | Agentic RL                             |
| ---- | --------------- | -------------------------------------- |
| 状态 | 单一提示        | 动态                                   |
| 行动 | 文本生成        | 文本＋工具＋环境操作                   |
| 转移 | 无转移          | 状态随行动变化                         |
| 奖励 | 单步 $r(s_0,y)$ | 积累$\sum_t \gamma^t r(s_t, \alpha_t)$ |
| 时间 | $T = 1$         | $T \gg 1$                              |
| 目标 | 短期            | 长期                                   |

Agentic RL的目标是赋予llm智能体六大核心能力：

- **推理**：从给定信息中逻辑地得出结论。

  强化学习地优势在于通过试错学习有效的推理策略，发现训练数据中没有的推理路径，学会何时需要深度思考、何时可以快速回答。

- **工具使用**：学会合适需要使用工具、选择哪个工具、如何组合多个工具。

- **记忆**：哪些信息值得记住、何时更新记忆、何时删除过时信息。

- **规划**：学会动态的规划：通过试错发现有效的行动序列，学会权衡短期和长期收益。

- **自我改进**：自我反思：识别自己的错误、分析失败原因，调整策略。

- **感知**：理解多模态信息的能力。



数据集和奖励函数是强化学习训练的两大基石。

- **数据集**：定义智能体要学习的任务
- **奖励函数**：定义什么是好的行为

原始格式是直接来自于数据集，包含问题question和答案answer（含推理过程），适合人类阅读。SFT格式用于监督微调，将问题转为对话格式的prompt，完整解答作为completion

```json
{
    "prompt": "<|im_start|>user\nNatalia sold clips to 48 of her friends...<|im_end|>\n<|im_start|>assistant\n",
    "completion": "Let me solve this step by step.\n\nStep 1: ...\n\nFinal Answer: 72<|im_end|>"
}
```

RL格式用于强化学习，只提供问题和正确答案，不提供解题过程

```json
{
    "prompt": "<|im_start|>user\nNatalia sold clips to 48 of her friends...<|im_end|>\n<|im_start|>assistant\n",
    "ground_truth": "72"
}
```



强化学习中，PPO（Proximal Policy Optimization）是最经典的算法之一。PPO通过限制策略更新的幅度，保证训练的稳定性。但是PPO在llm训练中：需要训练Value Model，增加了训练复杂度和显存占用；需要维护四个模型：Policy Model、Reference Model、Value Model、Reward Model，工程实现复杂，训练不稳定，容易出现奖励崩塌或策略退化。

GRPO（Group Relative Policy Optimization）是一种简化的PPO变体，专门为llm设计。GRPO的额核心思想是：不需要Value Model，使用组内相对奖励代替绝对奖励；简化训练流程，只需要Policy Model和Reference Model。

GRPO训练循环：

1. **采样阶段**：对于每个问题，使用当前策略生成多个答案。这些答案构成一个“组”，用于计算相对奖励。
2. **奖励计算**：对每个生成的答案计算奖励$r_i$。奖励可以是准确率、长度惩罚、步骤奖励或它们的组合。
3. **相对奖励**：计算组内平均奖励，然后计算相对奖励。这样做的好处是减少奖励方差，训练更稳定
4. **策略更新**：使用相对奖励更新策略，同时添加KL散度惩罚，防止策略偏离参考模型太远



【部分。。。】





```shell
conda create --name agentic_rl python=3.10
conda activate agentic_rl

uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "hello-agents[rl]==0.2.5"

```































[第十一章 Agentic-RL (datawhalechina.github.io)](https://datawhalechina.github.io/hello-agents/#/./chapter11/第十一章 Agentic-RL)