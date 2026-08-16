[deepseek-ai/deepseek-harness: DeepSeek Harness: Everything is a Plugin.](https://github.com/deepseek-ai/deepseek-harness)

> 目前还不完善，后面可能会破坏性更新，不能用于正式稳定的使用，可以用来学习。

官方建议的启动方式：

```shell
npx @deepseek-ai/dsh web
```

全局安装：

```shell
npm install -g @deepseek-ai/dsh
```

使用：

```shell
dsh web
```

不支持`0.0.0.0`，在服务器里面可以用：

```shell
dsh --profile web --port 2345
```

之后使用端口转发即可。



DSH和Pi的对比：

**DSH 与 Pi 不是简单的竞品复制关系，而是从相似起点走向不同复杂度。Pi 更像 Neovim 式可扩展编码工具，DSH 更像由插件构成的 agent 操作系统/应用容器。**不过 DSH 目前仍是开发者预览，实际稳定性和生态成熟度暂时不能只按架构上限判断。

> 业务流程必须可控、可编排，选 LangGraph；定制一个自主开发 Agent，选 Pi；定制整套 Agent 基础设施，选 DSH。





[karminski/deepseek-v4-pro-0813-scaffold-stop-3x2-validation: 用于验证 deepseek-v4-pro-0813 特定提示词过拟合 与 模型停止控制问题是否存在因果关系](https://github.com/karminski/deepseek-v4-pro-0813-scaffold-stop-3x2-validation)

[插件 | DeepSeek Harness](https://deepseekdocs.com/docs/user-guide/plugins)

[我做了一个harness架构图 不是很完善 分享给大家可视化理解harness架构 · deepseek-ai/deepseek-harness · Discussion #905](https://github.com/deepseek-ai/deepseek-harness/discussions/905)