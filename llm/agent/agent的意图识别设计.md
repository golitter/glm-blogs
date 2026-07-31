[Weixin Official Accounts Platform](https://mp.weixin.qq.com/s/BlF50Z143CfJjeBsr-YrHQ)

生产环境通常会采用级联架构，让不同方法各自处理擅长的流量：规则处理确定性命令和安全拦截，轻量分类器覆盖稳定高频意图，Embedding 负责候选召回，LLM 处理模糊表达、多轮状态和长尾请求，RAG 动态提供业务定义与案例，规划器负责多任务拆解。每一层还要保留拒识、追问和升级通道。



> 大多数agent业务场景相对单一，用不了太多意图识别trick。如果要用到，**应该要考虑新建agent**。

