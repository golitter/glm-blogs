https://zhuanlan.zhihu.com/p/2000568085258712069

https://juejin.cn/post/7615455795724648483

https://github.com/Fission-AI/OpenSpec



**AI“写的快”远远不够，关键在于“写的对”**。需要从vibe coding过渡到强调规范、可复现、可靠交付的新阶段，SDD（Specification Driven Development，规范驱动开发）这是这一转变的核心方法论。

SDD 核心理念：

1. 规范先行，而非文档补写。开发始于对“做什么”和“为什么”的清晰定义——包括业务规则、合规约束、成功标准等。
2. 分阶段验证，拒绝模糊推进。SDD 将开发拆解为 Specify → Plan → Tasks → Implement 四个明确阶段。每个阶段产出物（规范、技术方案、任务清单）必须经人工确认后，才进入下一阶段。这确保 AI 始终在正确轨道上运行。
3. 规范即上下文，赋能多AI协同。



主流开源SDD工具：Spec Kit，BMAD，OpenSpec。

> SpecKit适用于从零到一的。
>
> OpenSpec可以无缝融入现有开发流程。
>
> BMAD是较为基础的框架，可能需要大量实现、集成相关工具。
>
> **核心诉求：如何在不改造现有系统、不绑定单一 AI 工具的前提下，快速建立一套可协作、可追溯、可审计的 SDD 工作流。**

OpenSpec：

- 轻量可嵌入
- 多AI友好
- 变更可管理



安装：

```shell
npm install -g @fission-ai/openspec@latest
```

初始化：

```shell
openspec init
```

选择对应的aicoding工具，之后会生成对应的skills等配置。

默认只启动核心命令，如果要使用完整功能需要切换配置文件。

启动aicoding，将会出现所有`/opsx:命令`。

## 相关概念

- change，变更：一次独立的开发任务，对应一个需求或功能点。每个变更在`changes/`目录下拥有独立目录。
- artifact，工件：变更过程中的产出物，包括提案、规范、设计、任务清单、代码实现、验证报告等。
- specs，规范：记录项目全局或模块的设计约定，存储在`openspec/specs/`中，可以被多个变更共享。
- fast-forward，快进：一次性生成规划阶段的所有工作，跳过逐步创建的中间步骤。



## 核心指令

- `/opsx:explore`，探索：只读模式讨论需求，不生成文件
- `opsx:propose`，提案：一步创建变更并生成所有规划制品。
- `opsx:apply`，执行：根据任务清单编写代码
- `opsx:archive`，归档：归档单个已完成的变更



默认只暴露上面四个核心指令，开启非核心指令需要手动设置：

```shell
openspec config profile
```

选择`Delivery and workflows`之后，可以进行选择其他的指令：

```shell
? Select workflows to make available:
 [x] Propose change
 [x] Explore ideas
 [ ] New change
 [ ] Continue change
 [x] Apply tasks
 [ ] Fast-forward
 [ ] Sync specs
 [x] Archive change
❯[ ] Bulk archive
 [ ] Verify change
 [ ] Onboard

Archive multiple completed changes together
Space to toggle, Enter to confirm
```

之后执行更新操作即可：
```shell
openspec update
```

此时再输入`/opsx:`就可以看见其他非核心指令。



## 开发流程

`/opsx:explore` 建立全局规范（如代码风格、API设计原则）等并存放到`openspec/specs`中。

之后：

```shell
/opsx:propose 规划
/opsx:apply 实施
/opsx:verify 验证
/opsx:archive	归档
```

`/opsx:archive`时如果发现还没有进行同步，会询问是否要执行`/opsx:sync`。在验证和归档间加一个同步效果应该更好。





