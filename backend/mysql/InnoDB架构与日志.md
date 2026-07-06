# InnoDB架构与日志

---

## InnoDB vs MyISAM

| 特性 | InnoDB | MyISAM |
|------|--------|--------|
| 事务 | ✅ 支持（ACID） | ❌ 不支持 |
| 锁粒度 | 行锁（并发高） | 表锁（并发低） |
| 外键 | ✅ 支持 | ❌ 不支持 |
| 崩溃恢复 | ✅ redo log 恢复 | ❌ 需手动修复 |
| MVCC | ✅ 支持 | ❌ 不支持 |
| 聚簇索引 | ✅ 主键即聚簇索引 | ❌ 都是非聚簇索引 |
| 存储文件 | .ibd（数据+索引） | .MYD（数据）+ .MYI（索引） |
| COUNT(*) | 需遍历（MVCC 下行数不确定） | 直接存储行数，极快 |

- MySQL 5.5 起默认引擎为 InnoDB，绝大多数场景不需要换
- 修改引擎 `ALTER TABLE t ENGINE = InnoDB` 会锁表重建，大表慎用

---

## InnoDB 内存结构

### Buffer Pool（缓冲池）

- 缓存数据页和索引页，减少磁盘 IO，是 InnoDB 最重要的内存结构
- 默认 128MB，生产环境建议设为物理内存的 60%~80%
- LRU 链表分两段：Young 区（热数据，前 5/8）+ Old 区（冷数据，后 3/8）
- 新读入的页先放 Old 区，1 秒内再次访问才升到 Young 区，防止全表扫描污染热数据
- 命中率 = 1 - (磁盘读次数 / 逻辑读次数)，生产目标 > 99%

### Change Buffer（写缓冲）

- 缓存非唯一二级索引的 DML 操作，等后续读取时再合并到索引页
- 只对非唯一二级索引生效：唯一索引需立刻检查唯一性，主键顺序写入本来就快
- 适合写多读少的场景

### Log Buffer（日志缓冲）

- 缓存 redo log 记录，事务提交时刷到磁盘的 redo log 文件
- 刷盘策略由 `innodb_flush_log_at_trx_commit` 控制：

| 值 | 行为 | 安全性 |
|----|------|--------|
| 0 | 每秒刷盘一次 | 可能丢 1 秒数据 |
| 1 | 每次提交都刷盘 | 最安全（**生产必须用**） |
| 2 | 每次提交写到 OS 缓存，每秒 fsync | OS 崩溃会丢数据 |

---

## Redo Log（重做日志）

- 保证事务**持久性**（Durability）
- 核心思想：WAL（Write-Ahead Logging），先写日志再刷数据
- 记录物理日志：哪个页做了什么修改
- 固定大小、循环写（ib_logfile0 + ib_logfile1），write pos 追着 checkpoint 写
- 顺序写比随机写快很多（SSD 差 10 倍，HDD 差 100 倍）

---

## Undo Log（回滚日志）

- 保证事务**原子性**（Atomicity）：回滚时恢复数据
- 支持 MVCC：提供历史版本供快照读
- INSERT 记录主键（回滚时 DELETE），DELETE 记录整行（回滚时 INSERT），UPDATE 记录旧值（反向 UPDATE）
- 事务提交后 undo log 保留一段时间供 MVCC 使用，无事务引用时被 purge 线程清理

---

## 三种日志对比（高频考点）

| | Redo Log | Undo Log | Binlog |
|---|---|---|---|
| 作用 | 崩溃恢复，保证持久性 | 回滚事务，保证原子性 | 主从复制，数据备份 |
| 层级 | InnoDB 引擎层 | InnoDB 引擎层 | MySQL Server 层 |
| 内容 | 物理日志（哪个页改了啥） | 逻辑日志（反向 SQL） | 逻辑日志（SQL/行变更） |
| 写入方式 | 循环写，固定大小 | 随事务产生，可清理 | 追加写，文件递增 |
| 事务时机 | 事务中持续写 | 事务中持续写 | 事务提交时一次性写入 |

---

## 一条 SQL 的执行流程

### 查询流程

连接器（认证权限）→ 查询缓存（8.0 已移除）→ 解析器（词法+语法分析）→ 优化器（选索引、定执行计划）→ 执行器（权限检查、调引擎接口）→ 存储引擎（Buffer Pool → 磁盘）→ 返回结果

### 更新流程（核心！面试必考）

1. **读数据**：Buffer Pool 命中直接用，未命中则读磁盘
2. **写 Undo Log**：记录旧值，用于回滚和 MVCC
3. **更新 Buffer Pool**：修改内存中的数据页，该页变为「脏页」
4. **写 Redo Log（prepare）**：记录物理修改，刷到磁盘
5. **写 Binlog**：记录逻辑变更，刷到磁盘
6. **提交 Redo Log（commit）**：将 redo log 标记为 commit，事务完成
7. **后台异步刷脏页**：Buffer Pool 中的脏页在合适时机由后台线程刷到磁盘

### 两阶段提交（2PC）

```
Redo Log (prepare) → 写 Binlog → Redo Log (commit)
     ← 第一阶段 →                ← 第二阶段 →
```

**为什么需要？** 保证 redo log 和 binlog 一致性。如果只用一阶段：
- 先写 redo log 后写 binlog，崩溃后主库恢复数据但从库没有 → 主从不一致
- 先写 binlog 后写 redo log，崩溃后主库没恢复但从库执行了 → 主从不一致
- 两阶段提交：崩溃后检查 binlog 是否有对应记录，有则提交，无则回滚 ✅

---

## Binlog（二进制日志）

- MySQL Server 层日志，记录所有数据修改操作
- 主要用途：**主从复制**、**数据恢复**（PITR）

### 三种格式

| 格式 | 记录内容 | 优缺点 |
|------|----------|--------|
| STATEMENT | SQL 语句 | 日志量小，但 NOW()/UUID() 等函数主从不一致 |
| ROW | 行数据变更 | 日志量大，数据一致性最好（**生产推荐**） |
| MIXED | 默认 STATEMENT，不安全时切 ROW | 折中方案 |

- `SHOW BINARY LOGS` 查看文件列表
- `SHOW MASTER STATUS` 查看当前文件和位置
- `SHOW BINLOG EVENTS` 查看事件内容

---

## 主从复制

### 复制三步骤

1. **Master 写 Binlog**：主库数据变更记录到 binlog
2. **Slave IO 线程拉取**：从库 IO 线程连接主库，拉取 binlog 存为 relay log（中继日志）
3. **Slave SQL 线程执行**：从库 SQL 线程读取 relay log 并执行，同步数据

### 复制模式

| 模式 | 行为 | 特点 |
|------|------|------|
| 异步复制（默认） | 主库不等从库确认就返回 | 性能好，可能丢数据 |
| 半同步复制 | 等至少一个从库确认收到 binlog | 折中 |
| 全同步复制 | 所有从库都执行完才返回 | 太慢，基本不用 |

### 读写分离

- 写操作走主库，读操作走从库
- **注意从库延迟**：异步复制不是实时同步，`SHOW SLAVE STATUS` 中 `Seconds_Behind_Master` 查看延迟
- 写后立即读的业务应走主库

### PITR（基于时间点的恢复）

1. 恢复最近全量备份（mysqldump）
2. 用 mysqlbinlog 重做备份时间点到目标时间点的 binlog

---

## 面试速查

| 问题 | 答案 |
|------|------|
| InnoDB vs MyISAM？ | InnoDB 支持事务、行锁、MVCC、崩溃恢复；MyISAM 不支持 |
| Buffer Pool 作用？ | 缓存数据页和索引页，减少磁盘 IO |
| Buffer Pool LRU？ | 分 Young/Old 两段，防止全表扫描污染热数据 |
| Change Buffer？ | 缓存非唯一二级索引的修改，减少随机 IO |
| Redo Log 作用？ | 保证持久性，WAL 先写日志再刷数据 |
| Undo Log 作用？ | 保证原子性（回滚）+ MVCC（版本链） |
| 三种日志区别？ | Redo（物理/引擎）、Undo（逻辑/引擎）、Binlog（逻辑/Server） |
| 更新 SQL 执行流程？ | 读数据→写 undo log→改 Buffer Pool→写 redo log(prepare)→写 binlog→提交 redo log(commit) |
| 为什么两阶段提交？ | 保证 redo log 和 binlog 一致，避免主从数据不一致 |
| Binlog 三种格式？ | STATEMENT（记SQL）、ROW（记行变更）、MIXED（混合） |
| 主从复制原理？ | 主库写 binlog → 从库 IO 线程拉取 → 写 relay log → SQL 线程执行 |
| innodb_flush_log？ | 生产必须设 1（每次提交刷盘），保证不丢数据 |
