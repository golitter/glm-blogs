# 事务锁与MVCC

---

## 事务基础

- 事务（Transaction）：一组 SQL 操作，要么全部成功，要么全部回滚
- `START TRANSACTION` 开启事务，`COMMIT` 提交，`ROLLBACK` 回滚
- MySQL 默认 `autocommit=ON`，每条 SQL 自动提交；`START TRANSACTION` 会临时关闭自动提交

---

## ACID 特性

| 特性 | 英文 | 含义 | InnoDB 实现方式 |
|------|------|------|----------------|
| A | Atomicity | 原子性：全部成功或全部回滚 | undo log（回滚日志） |
| C | Consistency | 一致性：数据满足完整性约束 | 由 AID 共同保证 |
| I | Isolation | 隔离性：并发事务互不干扰 | 锁 + MVCC |
| D | Durability | 持久性：提交后数据永久保存 | redo log（重做日志） |

---

## 并发事务问题

| 问题 | 描述 | 核心区别 |
|------|------|----------|
| 脏读 | 读到其他事务**未提交**的数据，该事务可能回滚 | 数据根本不存在 |
| 不可重复读 | 同一事务内两次读**同一行**，结果不同（别人 UPDATE/DELETE 并提交了） | 已提交的修改 |
| 幻读 | 同一事务内两次**范围查询**，行数不同（别人 INSERT 并提交了） | 已提交的新增 |

- 关键：问题不是"看到了新数据"，而是"自己事务还没结束，中途数据变了，导致基于此数据的业务逻辑不可靠"

---

## 四种隔离级别

| 隔离级别 | 脏读 | 不可重复读 | 幻读 | 性能 |
|----------|------|-----------|------|------|
| READ UNCOMMITTED | 可能 | 可能 | 可能 | 最高 |
| READ COMMITTED | 避免 | 可能 | 可能 | ↑ |
| REPEATABLE READ（默认） | 避免 | 避免 | 大部分避免* | ↓ |
| SERIALIZABLE | 避免 | 避免 | 避免 | 最低 |

- \* InnoDB 的 REPEATABLE READ 通过 MVCC + 间隙锁在很大程度避免幻读，特殊场景仍可能出现
- 查看隔离级别：`SELECT @@transaction_isolation`
- 设置：`SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ`

---

## MVCC 原理（多版本并发控制）

### 核心思想

让读操作不加锁，通过给每行数据维护多个历史版本，让不同事务看到不同时刻的数据快照。快照读（普通 SELECT）走 MVCC，当前读（FOR UPDATE / UPDATE / DELETE）走锁。

### 隐藏列

每行数据自动包含：
- **DB_TRX_ID**：最后修改该行的事务 ID
- **DB_ROLL_PTR**：回滚指针，指向 undo log 中的上一个版本

### undo log 版本链

每次修改一行数据时，旧版本通过 DB_ROLL_PTR 串成链表：当前版本 → 旧版本2 → 旧版本1 → NULL

### ReadView（读视图）

事务做快照读时创建的可见性判断工具，包含四个字段：

| 字段 | 含义 |
|------|------|
| creator_trx_id | 创建该 ReadView 的事务 ID |
| m_ids | 创建时所有活跃（未提交）事务的 ID 列表 |
| min_trx_id | m_ids 中最小的事务 ID |
| max_trx_id | 下一个将分配的事务 ID（当前最大 + 1） |

### 可见性判断规则

对版本链中每个版本的 trx_id 依次判断：

| 条件 | 结果 |
|------|------|
| trx_id == creator_trx_id | 可见（自己的修改） |
| trx_id < min_trx_id | 可见（ReadView 创建前已提交） |
| trx_id >= max_trx_id | 不可见（ReadView 创建后才开始的事务） |
| min <= trx_id < max 且在 m_ids 中 | 不可见（事务还在跑，未提交） |
| min <= trx_id < max 且不在 m_ids 中 | 可见（事务已提交） |

- "还在跑" = 事务已经 START 但还没 COMMIT

### RR vs RC 的本质区别

唯一区别是 ReadView 的创建时机：

| 隔离级别 | ReadView 创建时机 | 效果 |
|----------|------------------|------|
| REPEATABLE READ | 第一次 SELECT 时创建，整个事务复用 | 事务内所有读看到同一快照，可重复读 |
| READ COMMITTED | 每次 SELECT 都创建新的 | 每次读都能看到最新已提交数据 |

### 快照读 vs 当前读

| 类型 | SQL | 走 MVCC？ | 加锁？ |
|------|-----|----------|-------|
| 快照读 | 普通 SELECT | 是 | 否 |
| 当前读 | SELECT FOR UPDATE | 否 | 是（X锁） |
| 当前读 | SELECT LOCK IN SHARE MODE | 否 | 是（S锁） |
| 当前读 | UPDATE / DELETE / INSERT | 否 | 是（X锁） |

- REPEATABLE READ 下幻读的特殊情况：快照读不会幻读，但如果事务中先快照读再当前读（如 UPDATE），当前读能看到最新数据并修改，修改后该行的 trx_id 变为自己的，后续快照读就能看到了

---

## 锁

### 按粒度分

| 锁类型 | 锁的是什么 | 作用 |
|--------|-----------|------|
| 行锁 | 单行数据 | 防止别人修改/删除该行 |
| 表锁 | 整张表 | 粒度大，并发低 |
| 间隙锁 | 行与行之间的空隙 | 防止别人往空隙中插入新行 |
| 临键锁 | 行锁 + 间隙锁 | 既防改删，又防插入 |

### 按模式分

| 锁模式 | 含义 | 兼容性 |
|--------|------|--------|
| 共享锁（S） | 读锁 | S 与 S 兼容，S 与 X 冲突 |
| 排他锁（X） | 写锁 | X 与所有锁冲突 |
| 意向锁（IS/IX） | 表级标识，表示打算对行加 S 或 X 锁 | 快速判断表里是否有行锁，避免逐行扫描 |

### 锁的兼容性矩阵

```
         │  S锁(读)  │  X锁(写)  │
─────────┼──────────┼──────────┤
 S锁(读)  │   ✅兼容   │  ❌冲突   │
 X锁(写)  │   ❌冲突   │  ❌冲突   │
```

### 关键规则

- UPDATE / DELETE / INSERT **自动加排他锁**，不需要手动写
- 普通 SELECT 不加锁（走 MVCC 快照读）
- InnoDB 的行锁加在**索引**上，不是数据行上
- **没有用到索引时，行锁升级为表锁**（最常见的事故原因之一）

### 间隙锁

- 只在 REPEATABLE READ 及以上隔离级别生效
- 锁的是索引记录之间的空隙，阻止插入
- 存在的意义：只有阻止插入才能真正防止幻读
- InnoDB 扫描索引时会顺手锁住匹配记录前面的间隙，宁可多锁不可漏锁

### 临键锁

- InnoDB 在 REPEATABLE READ 下的默认加锁方式
- = 行锁 + 间隙锁，锁定一个范围并包含记录本身
- `WHERE id >= 3 FOR UPDATE` 会锁定 (前一条记录, 3], (3, 5], (5, +∞)

### 查看锁信息

- MySQL 8.0+：`SELECT * FROM performance_schema.data_locks` 查看锁，`data_lock_waits` 查看锁等待
- `SELECT * FROM information_schema.innodb_trx` 查看正在运行的事务
- `SHOW ENGINE INNODB STATUS` 中的 TRANSACTIONS 部分包含锁信息

---

## 死锁

### 产生条件

两个或多个事务互相等待对方持有的锁，形成循环等待。

### 处理机制

InnoDB 自动检测死锁，回滚其中一个事务（通常是修改数据量少的），返回 ERROR 1213。

### 查看死锁日志

- `SHOW ENGINE INNODB STATUS` 中的 LATEST DETECTED DEADLOCK 部分
- `SET GLOBAL innodb_print_all_deadlocks = ON` 将死锁信息记录到错误日志

### 避免策略

| 策略 | 说明 |
|------|------|
| 固定加锁顺序 | 所有事务都按 id 升序操作，最有效 |
| 保持事务简短 | 持有锁的时间短，减少冲突 |
| 合理使用索引 | 避免行锁升级为表锁 |
| 设置锁等待超时 | `SET innodb_lock_wait_timeout = 10`（默认 50 秒） |
| 应用层重试 | 捕获死锁错误，自动重试整个事务 |

---

## 超卖问题

| 方案 | 做法 | 特点 |
|------|------|------|
| 悲观锁 | `SELECT ... FOR UPDATE` 先锁行再操作 | 简单直接，但并发度低 |
| 乐观锁 | `UPDATE ... WHERE stock > 0` 在 SQL 层面保证 | 不加锁，通过 WHERE 条件防止超卖 |

---

## 面试速查

| 问题 | 答案 |
|------|------|
| 事务四大特性？ | ACID — 原子性、一致性、隔离性、持久性 |
| MySQL 默认隔离级别？ | REPEATABLE READ |
| RR 和 RC 区别？ | RR 复用 ReadView，RC 每次 SELECT 创建新 ReadView |
| MVCC 原理？ | undo log 版本链 + ReadView 可见性判断 |
| 什么时候行锁变表锁？ | 没走索引时 |
| 间隙锁作用？ | 锁住间隙防止插入，防幻读 |
| 死锁怎么办？ | InnoDB 自动检测并回滚一个事务 |
| 怎么避免死锁？ | 固定加锁顺序 + 事务简短 + 合理索引 |
