https://openai.com/zh-Hans-CN/index/harness-engineering/

最重要的是：**人类的时间和注意力**。

重新定义工程师的角色。



## 要给 Codex 的是一张地图，而不是一本 1,000 页的说明书

原因：

- 过多的指导反而变得无效且难以核查。

这里将`AGENTS.md`作为**内容目录**。代码库的知识库被放到`docs/`目录中，`AGENTS.md`（行数较少）主要作为地图，可以通过`docs/`进行**渐进式披露**。

> 将系统的更多部分转化为智能体可以检查、验证并直接修改的形式，可以直接提高杠杆效应 — 这不仅适用于 Codex，也适用于其他智能体（例如[Aardvark](https://openai.com/zh-Hans-CN/index/introducing-aardvark/)) 也在参与代码库的开发。



## 规范架构

仅仅靠文档是没有办法保持agent生成的连贯性。通过强制执行不变量，而非对实施过程进行微观管理。



agent生成会不可避免导致**漂移**，可能需要定期手动修改。

