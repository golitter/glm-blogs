# context：协作式取消与超时

跨 goroutine 传取消信号、超时、请求级元数据。配套阅读：[select](go并发-select详解.md)、[通道](go通道.md)、[sync、atomic](go并发-sync、atomic.md)。

> **核心**：context 不强杀 goroutine，而是通知它「该自己退出了」。

```go
ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
defer cancel()   // 拿到 cancel 就必须调用
```

## 为什么是协作式

Go 无法安全地外部强杀 goroutine——它可能正持锁、写文件、改共享变量。所以采用**协作式取消**：外部发信号，goroutine 自己感知 `ctx.Done()` 并清理退出。

```text
外部 cancel() → ctx.Done() 关闭 → 下游 select 触发 → 自己 return
```

## context 是一棵树

派生形成树形结构，根节点通常是 `context.Background()`：

```text
Background
   ├─ WithCancel   父取消 → 所有子取消（级联）
   ├─ WithTimeout
   └─ WithValue
```

> **父被取消，所有子孙自动取消**。后端里 HTTP 请求 ctx 一断，下游查库 / 调 RPC / 子 goroutine 都该跟着退。

## 两个根

| 根 | 含义 | 场景 |
| ---- | ---- | ---- |
| `context.Background()` | 明确是根，永不取消 | main、服务启动、初始化 |
| `context.TODO()` | 暂不知该传哪个 | 过渡代码 |

> **永远别传 `nil`**，不知道传什么就用 `context.TODO()`。

## 四个派生函数

```go
ctx, cancel := context.WithCancel(parent)                       // 手动取消
ctx, cancel := context.WithTimeout(parent, 3*time.Second)       // 多久后取消
ctx, cancel := context.WithDeadline(parent, time.Now().Add(3*time.Second)) // 到某点取消
ctx       = context.WithValue(ctx, key, val)                    // 存请求级元数据
```

`WithTimeout` = `WithDeadline(现在+时长)`，二者都会自动取消但**仍要 `defer cancel()`**——派生出的 ctx 持有 timer、父子引用，不调会延迟到超时才释放。

## 下游怎么感知取消

`ctx.Done()` 是个 channel，取消或超时后关闭；`ctx.Err()` 返回原因：

```go
func worker(ctx context.Context) {
    ticker := time.NewTicker(time.Second); defer ticker.Stop()
    for {
        select {
        case <-ctx.Done():
            return ctx.Err()      // context.Canceled 或 context.DeadlineExceeded
        case <-ticker.C:
            doWork()
        }
    }
}
```

> 未取消时 `ctx.Err()` 是 `nil`。

## WithValue：只放请求级元数据

放 `traceID` / `requestID` / `userID` 这种贯穿链路的辅助信息，**别塞业务参数**。

```go
// ❌ 业务数据走函数参数，别塞 ctx
ctx = context.WithValue(ctx, "user", user)
ctx = context.WithValue(ctx, "pageSize", 20)

// ✅ 元数据才放 ctx
ctx = context.WithValue(ctx, reqIDKey, "x1")
```

**key 别用 string**（不同包易冲突），用自定义类型：

```go
type ctxKey struct{}                                  // 未导出，最安全
var reqIDKey ctxKey
ctx = context.WithValue(ctx, reqIDKey, "x1")

reqID, ok := ctx.Value(reqIDKey).(string)             // 取出 + 类型断言
```

## ctx 透传

约定 `context.Context` 放第一个参数，每一层都往下传，取消才能级联：

```go
func Handler(w http.ResponseWriter, r *http.Request) {
    ctx := r.Context()
    if err := service.Do(ctx); err != nil { /* ... */ }
}

func (s *Service) Do(ctx context.Context) error { return s.repo.Query(ctx) }

func (r *Repo) Query(ctx context.Context) error {
    return r.db.QueryContext(ctx, "select ...")   // 标准库尊重 ctx
}
```

> 下游 API 必须**接收并尊重** ctx，超时控制才有效。

## 常见坑

- **忘了 `defer cancel()`**：哪怕 WithTimeout 会自动超时，也要手动调，否则资源延迟释放。
- **传 `nil`**：用 `context.TODO()` 代替。
- **ctx 存进 struct**：ctx 是一次请求的生命周期，该跟调用链走，不是结构体字段。
- **WithValue 塞业务数据**：业务参数走函数签名。
- **key 用 string**：改用未导出结构体类型。
- **ctx.Err() 当成「已取消」判空**：未取消时它是 `nil`。

## 三个最小模板

掌握这三个就够覆盖绝大多数场景：

```go
// 标准签名
func DoSomething(ctx context.Context, id string) error { /* ... */ }

// 创建超时 ctx
ctx, cancel := context.WithTimeout(ctx, 3*time.Second); defer cancel()

// select 监听取消
select {
case <-ctx.Done():
    return ctx.Err()
case v := <-ch:
    use(v)
}
```

## 速查

| 需求 | 用法 |
| ---- | ---- |
| 根节点 | `context.Background()` |
| 不知道传啥 | `context.TODO()` |
| 手动取消 | `WithCancel` |
| 超时控制 | `WithTimeout` |
| 到点截止 | `WithDeadline` |
| 请求级元数据 | `WithValue`（key 用自定义类型） |
| 感知取消 | `<-ctx.Done()` |
| 取消原因 | `ctx.Err()` |

> **核心理念**：ctx 放第一个参数、层层透传、下游监听 `ctx.Done()`、拿到 cancel 就 `defer cancel()`——这不是强杀，是协作退出。
