消息队列（Message Queue，MQ）是一种用于在不同系统或组件之间传递消息的异步通信机制。

> 基本的结构：
>
> ```shell
> 生产者 Producer → 消息队列 Queue → 消费者 Consumer
> ```
>
> 
>
> 消息队列通过中间存储和异步投递，让生产者与消费者不必同时在线，也不必直接依赖彼此。
>
> 它主要解决以下问题：
>
> - 异步处理
> - 系统解耦
> - 流量削峰
> - 故障隔离
> - 消息可靠传递



Kafka 是一个分布式事件流平台。它把不断发生的事件持久化成一条日志，让多个系统能够独立、并行地读取和处理。

优点：

- 解耦：订单系统不需要知道有哪些下游。
- 异步：发布事件后即可继续处理请求。
- 削峰：高峰期消息先积压在 Kafka，下游按能力消费。
- 可恢复：消费者故障恢复后，可以接着处理。
- 扩展：增加一个新订阅者，不需要修改生产者。



基本概念：

- **topic**：主题，消息归类的基本单元
- **partitions**：消息分区，通过偏移量offset来指定消息的位置
- **replicas**：分区副本，每个分区可以有多个副本，一个leader多个follower
- **broker**：集群，kafka由一个或多个broker组成集群，每个broker就是一个kafka实例
- **zookeeper**：协调多个broker，存储元数据

> 旧架构：Broker + ZooKeeper
> 新架构（kafka4.0）：Broker + KRaft Controller



![image-20260915160800974](%E6%B6%88%E6%81%AF%E9%98%9F%E5%88%97kafka.assets/image-20260915160800974.png)





【16分钟彻底学会Kafka（消息队列、分布式系统架构、千亿量级的日志处理）】 https://www.bilibili.com/video/BV17hdXYZE78/?share_source=copy_web&vd_source=790a5ca2f47c821c3a8b320e22e891fb

[(99+ 封私信 / 80 条消息) 人人都说kafka，何为kafka？？？三分钟kafka原理详解~ - 知乎](https://zhuanlan.zhihu.com/p/1976637853099397655)