[动手学Agent应用开发（已满） - Datawhale](https://www.datawhale.cn/activity/483/learn/220/5062)

[🐫_CAMEL_Creating_Your_First_Agent.ipynb - Colab (google.com)](https://colab.research.google.com/drive/1VPTonWyiauk7OEnsIeEt1gZKwgDvqLHz?usp=sharing)



agent包含Tools、Memory、Planning、Action模块。

多个agent协同处理任务要比单个agent处理的效果要好。

camel社区的最开始多智能体架构是`Role Playing`（角色扮演）：用户输入query到AI User agent，这个agent会跟Assistant agent进行协作。在Assistant agent部分可以调用多个sub agent，这些sub agent是用来做特定的任务的特化功能agent。

现在camel使用的为`Workforce`（细化分层）：用户输入query到Coordinator agent，该agent对query进行分析提炼任务。Task Manager agent将提炼的任务进行拆解由各个任务agent进行处理。

早期的`Role Playing`核心思想是通过角色扮演，模拟现实的演进来处理问题。

现在的`Workforce`更偏向于分层的结构化，通过更好的结合agent能力来处理问题。