# Agent 高危工具与长时间工具调用设计

> 调研时间：2026-07-28
>
> GPT 5.6 Sol


Agent 调用转账、交易、部署、删除等高危工具时，不能只靠模型"谨慎思考"。核心思路：**把安全边界、权限判断、任务状态和副作用放到模型之外，由确定性系统负责**。

长时间工具还需把"发起调用"与"真正执行"解耦：

```
Agent 提交意图 → 策略校验 → 持久化任务 → 受限执行器异步执行 → 查询/订阅结果 → Agent 继续
```

主流产品分两条路线：编码 Agent（Claude Code、Codex、TRAE）靠权限+沙箱+后台会话；Agent 框架（OpenAI Agents SDK、LangGraph、Google ADK）提供中断、恢复、持久化和异步原语。但**没有哪个框架能独立解决转账所需的幂等、事务、对账和补偿**。

## 一、核心概念

### 1. Agent 是提议者，不是执行者

Agent 不持有银行/邮箱通用凭证，只生成带 `idempotency_key` 的结构化提议；由模型外的程序校验白名单、审批、金额、额度、是否重复，通过后才由受限执行器真正调用。

本质是**权限分离**：Agent 有建议权，策略引擎有放行权，执行器有操作权。授权者可以是人（HITL），也可以是确定性策略程序——不强制依赖 HITL。

### 2. Dry Run 与两阶段提交

Dry Run 只是预测结果，状态可能变化，模拟成功不代表执行安全。

业务两阶段提交：**Prepare（冻结资金/预留额度/固定金额、返回不可篡改 `prepared_action_id`）→ Commit（只能提交既有 id，不能改金额或目标）**。

### 3. 借鉴事务与 Saga

Agent 工具调用可借鉴数据库事务（Begin→Validate→Lock→Commit→Rollback），但邮箱、银行、交易所不受同一事务管理器控制，无法 ACID。扣款成功但通知超时时，只能 **Saga 补偿，不能回滚**——反向交易有价差和手续费，错误邮件可能已被阅读。

### 4. 调用与执行解耦

高风险操作应提交 Command，立即返回 `operation_id` + `ACCEPTED`（仅表示已接收），Agent 再通过查询/事件获得终态。显式状态机：

```
PROPOSED → VALIDATED → QUEUED → PREPARING → PREPARED → EXECUTING
                                                          ├─ SUCCEEDED
                                                          ├─ FAILED
                                                          ├─ UNKNOWN
                                                          └─ COMPENSATING → COMPENSATED / FAILED
```

## 二、主流实现

| 框架 | 高危控制 | 长任务解耦 |
|---|---|---|
| **Claude Code** | PreToolUse Hook + allow/ask/deny 权限 + OS 级 Sandbox；无 HITL 路线 = 预授权白名单 + 沙箱 + 风险分类 | 本地后台 Bash（退出即清理）；云端 Remote Session 解耦"会话↔Agent Session"，非金融级事务 |
| **OpenAI Codex** | Approval Policy + Sandbox + Command Rules（allow/prompt/forbid）+ Managed Requirements + Auto-review（风险分类，非策略引擎） | 任务级解耦：Thread/Cloud Task 在隔离容器/worktree 执行，解耦"会话↔Task" |
| **TRAE** | Sandbox + Shell Interception（拦 `rm`/`rmdir`）+ allowlist + Manual/Auto Run | SOLO 可跑长链路开发任务，但公开资料未见 `operation_id` 类通用异步协议 |
| **OpenAI Agents SDK** | `needs_approval` + 动态审批 + input/output guardrails；审批可程序化 callback，不强制 HITL | 序列化 `RunState` 暂停/恢复；非可靠任务引擎，需自行实现 `start_xxx`/`get_xxx_status` |
| **LangGraph** | — | checkpointer 逐节点持久化、`interrupt()`、故障从最后节点恢复；但节点可能重执行，**副作用须幂等** |
| **Google ADK** | `require_confirmation`（可由监督程序确认） | `LongRunningFunctionTool`：工具启动外部任务返回 `operation_id`，暂停 Run，按 `invocation_id` 恢复——**与"调用/执行解耦"最吻合** |

三者共同点：**自主执行依赖事先划定的运行边界，而不是 Prompt 里要求 Agent 小心**。

## 三、共同分层

```
Agent（产生结构化意图）
    ↓
Agent Runtime（审批、Guardrail、暂停/恢复）  ← SDK/LangGraph/ADK
    ↓
Job / Workflow Service（队列、状态机、重试、幂等）  ← Temporal/Step Functions/Outbox
    ↓
Capability Executor（受限凭证、额度、白名单）  ← 需自建
    ↓
外部系统（邮箱、银行、交易所）
```

Agent 框架只覆盖前两层；后两层（可靠任务系统 + 业务安全执行器：白名单、限额、exactly-once、对账、Saga、capability token、审计）**必须由框架外的确定性业务层实现**。

## 四、推荐架构与实践

**架构**：`Agent 框架 → 持久化 Job Service → 受限业务执行器 → 邮箱/银行/交易所 API`

**工具接口**应拆成 `propose → validate → prepare → commit → get_status → cancel`，而不是只暴露一个 `transfer_money()`。

**关键实践**：

1. **执行时重新校验**——入队时合法不代表执行时仍合法（授权过期、收款人掉出白名单、余额变化）。需两次检查：提交时决定是否入队，执行前重新确认。
2. **正确处理 UNKNOWN**——网络超时但银行可能已成功，**不能直接重试**，按 `operation_id` 查询：已成功→SUCCEEDED，明确失败→FAILED，仍未知→保持冻结继续核对。
3. **Transactional Outbox**——在同一数据库事务中写业务状态 + Outbox 事件，后台发布器投递队列，执行器用 `idempotency_key` 去重，避免"记录了但消息没发"。

## 五、结论

- **编码 Agent**（Claude Code/Codex/TRAE）：解决 Agent 如何在环境里安全、长时间工作。
- **Agent 框架**（LangGraph/Agents SDK/ADK）：解决工作流如何中断、持久化、恢复。
- **金融/通信等高危业务**真正需要的是**幂等、事务、对账、限额、补偿和不可逆副作用管理**——这必须由框架外的确定性业务执行层实现。

最终原则：

```
Agent 无通用秘密、无无限权限、无最终裁决权
单次错误有硬上限
每个副作用都可唯一识别、查询和审计
```

安全目标不是保证 Agent 永不犯错，而是让它**即使犯错，也无法把错误放大成灾难**。
