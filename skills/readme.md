# Skills

Claude Code 自定义技能集合。

---

## course — 分阶段学习助手

一个通用的分阶段学习辅助技能，帮助用户系统学习任意技术主题。根据用户的基础和目标生成学习计划，按阶段给出详细学习内容，学完后自动生成精简的重点摘要笔记。

```
/course plan          # 交互式问答，生成学习计划
/course content       # 生成当前阶段学习内容
/course summary       # 学完后生成重点摘要笔记
/course status        # 查看当前进度
/course               # 自动判断下一步该做什么
```

## python-dl-development

用于 Python 深度学习/机器学习项目的开发规范 skill，要求代码以模块方式运行，使用 src 结构和清晰导入，函数具备严格类型注解，配置和结构化数据优先用 Pydantic 表达，遵循 PEP 8，Ruff 仅在项目已有配置时使用，并避免硬编码绝对路径和实验环境配置。


## taste skill

给AI编程助手注入审美与工程规范”的提示/技能文件集合，让AI写出来的前端更像设计师做出来的。

https://github.com/Leonxlnx/taste-skill



## ui-ux-pro-max-skill

给AI编程助手装的“设计外挂”，能在写代码前自动生成行业专属的设计系统，让AI直接输出专业级UI。

https://github.com/nextlevelbuilder/ui-ux-pro-max-skill



### https://github.com/mattpocock/skills

专门为 AI 编程助手（特别是 Claude）提供结构化的工程技能。



## agent-skills (Addy Osmani)

生产级 AI 编程助手工程技能集，覆盖软件开发全生命周期。包含 24 个技能，7 个斜杠命令（/spec、/plan、/build、/test、/review、/code-simplify、/ship），4 个专家角色（代码审查、测试、安全、性能）。每个技能都是带步骤、验证门和"反合理化"表的结构化工作流。

https://github.com/addyosmani/agent-skills



## w-skills

AI 编程助手技能框架，当前包含 30 个技能，按功能领域分为：规划与设计（头脑风暴、PRD）、前端开发（React 最佳实践、shadcn、Ant Design 等 11 个）、文档与办公、AI 内容生成（Seedance 视频、Gemini 图像）、开发工具（代码审查、PR 创建、Web 测试等）、全栈开发（React/Node、ASP.NET Core、Tauri v2）等。支持 `npx skills find/add` 发现和安装技能。

https://github.com/dbDev-code/w-skills



## academic-research-skills

面向 Claude Code 的学术研究全流程技能套件（research → write → review → revise → finalize）。包含深度研究（13 智能体 7 模式）、学术论文撰写（12 智能体 10 模式）、论文评审（7 智能体多视角评审，0-100 评分）和 10 阶段流水线编排器。支持 APA 7.0 / Chicago / IEEE 等引用格式，支持中英文双语。

https://github.com/Imbad0202/academic-research-skills
