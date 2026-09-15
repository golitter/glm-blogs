> GPT-5.6 Sol润色

# Kafka 消息队列

消息队列（Message Queue，MQ）通过中间存储和异步投递，让生产者与消费者不必同时在线，也不必直接依赖彼此。

```text
生产者 Producer → 消息队列 → 消费者 Consumer
```

核心价值：**异步处理、系统解耦、流量削峰、故障缓冲**。

---

## Kafka 是什么

Kafka 是一个**分布式事件流平台**。它将事件持久化到日志中，允许多个系统独立、并行地读取和处理。

- **解耦**：生产者无需知道有哪些下游系统
- **异步**：发布消息后即可继续处理请求
- **削峰**：高峰期消息先积压，下游按能力消费
- **可恢复**：消费者通过 Offset 从上次位置继续处理
- **可扩展**：增加新订阅者时通常无需修改生产者

Kafka 与传统队列的一个重要区别是：**消息被消费后不会立即删除**，而是按保留时间或容量策略清理。

---

## 核心概念

| 概念 | 含义 |
|------|------|
| **Producer** | 向 Topic 发布消息的生产者 |
| **Consumer** | 从 Topic 读取并处理消息的消费者 |
| **Topic** | 消息的逻辑分类，如 `order-created` |
| **Partition** | Topic 的物理分片，用于并行读写和水平扩展 |
| **Offset** | 消息在某个 Partition 内的位置编号 |
| **Broker** | Kafka 集群中的一个服务节点 |
| **Replica** | Partition 的副本，用于故障恢复 |
| **Consumer Group** | 共同分担消息的一组 Consumer |

Offset 只在所属 Partition 内有意义，定位一条消息需要 `Topic + Partition + Offset`。

---

## 分区与顺序

- 一个 Topic 可以拆分成多个 Partition，分布在不同 Broker 上
- 多个 Partition 可以并行读写，但 Partition 越多，调度和副本管理成本也越高
- Kafka 只保证**单个 Partition 内有序**，不保证整个 Topic 全局有序
- 需要保序的消息应使用相同 Key，使其进入同一 Partition

---

## 副本与可用性

每个 Partition 可以有多个副本：

- **Leader**：处理该 Partition 的读写请求
- **Follower**：从 Leader 同步数据，Leader 故障时可被选为新 Leader
- **ISR**：与 Leader 保持同步的副本集合

### acks 写入确认

`acks` 决定生产者发送消息后，需要等待多少副本确认：

| 取值 | 成功条件 | 特点 |
|------|----------|------|
| `acks=0` | 不等待 Broker 确认 | 延迟最低，但消息可能丢失 |
| `acks=1` | Leader 写入成功 | 性能和可靠性折中；副本同步前 Leader 故障仍可能丢失 |
| `acks=all` | 所有 ISR 副本确认 | 可靠性最高，但延迟相对更高；`-1` 与 `all` 等价 |

`acks=all` 还需要配合 `min.insync.replicas`，后者规定至少需要多少个 ISR 副本才允许写入。若 ISR 数量不足，Broker 会拒绝写入，避免消息只落到少数副本上。

常见的高可靠组合：

```properties
replication.factor=3
min.insync.replicas=2
acks=all
```

即 3 个副本、至少 2 个副本保持同步，生产者才将写入视为成功。

---

## Consumer Group

- 同一消费组内，一个 Partition 同一时刻只分配给一个 Consumer
- Consumer 数量超过 Partition 数量时，多出的 Consumer 会空闲
- 不同消费组互不影响，可以各自完整消费同一 Topic
- Consumer 加入、离开或 Partition 变化时，可能触发分区重新分配（Rebalance）

```text
Topic（3 个 Partition）
  ├─ Consumer Group A：Consumer-1 处理 P0/P1，Consumer-2 处理 P2
  └─ Consumer Group B：独立消费 P0/P1/P2
```

---

## 消息语义

| 语义 | 特点 |
|------|------|
| **At most once** | 最多一次，可能丢失，不会重复 |
| **At least once** | 至少一次，不易丢失，但可能重复 |
| **Exactly once** | 精确一次，需要幂等生产者、事务等机制配合 |

实际业务常采用 **At least once + 消费端幂等**，避免重复消息造成重复扣款、重复创建订单等问题。

---

## ZooKeeper 与 KRaft

- **旧架构**：Broker + ZooKeeper，ZooKeeper 负责集群元数据和协调
- **Kafka 4.0+**：只支持 KRaft，由 KRaft Controller 集群管理元数据，不再依赖 ZooKeeper

> 下图是 **ZooKeeper 时代的旧架构示意图**，主要用于理解 Broker、Partition、副本和 Consumer Group。

![Kafka 旧架构示意图](%E6%B6%88%E6%81%AF%E9%98%9F%E5%88%97kafka.assets/image-20260915160800974.png)

---

## 易错点

- Broker 是集群节点，不是集群本身
- Offset 是 Partition 内的位置，不是全局消息 ID
- Kafka 只保证 Partition 内有序
- 消费成功不等于消息被删除
- 副本、`acks`、Offset 提交和消费幂等共同决定整体可靠性
- Kafka 4.0 已移除 ZooKeeper 模式

---

## 参考资料

- [Apache Kafka 官方文档](https://kafka.apache.org/documentation/)
- [16 分钟彻底学会 Kafka](https://www.bilibili.com/video/BV17hdXYZE78/)
- [Kafka 原理详解](https://zhuanlan.zhihu.com/p/1976637853099397655)
