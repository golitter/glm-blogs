# select：多路复用

围绕 channel 的多路等待器。像 `switch`，但每个 `case` 是一个通道操作，**随机**挑一个已就绪的执行。配套阅读：[通道](go通道.md)、[sync、atomic](go并发-sync、atomic.md)。

```go
select {
case v := <-ch1:    // 接收取值
    fmt.Println(v)
case ch2 <- 100:    // 发送
    fmt.Println("sent")
default:            // 都没就绪
    fmt.Println("nothing")
}
```

执行规则：

| 情况 | 行为 |
| ---- | ---- |
| 有 case 就绪 | 执行其中一个 |
| 多个同时就绪 | **随机**挑一个（不按代码顺序，不会饿死某通道） |
| 没有任何就绪 | 阻塞 |
| 没有就绪但有 `default` | 走 `default` |

> `select` 不是 `if-else`，不从上到下判断。

## case 写法

```go
case v := <-ch:        // 接收取值
case v, ok := <-ch:    // ok=false 表示通道已关闭且空
case <-ch:             // 只收信号，不关心值
case ch <- v:          // 发送
```

关键是 `v, ok := <-ch`：关闭后的通道返回类型零值 + `ok=false`。

```go
ch := make(chan int); close(ch)
v, ok := <-ch   // 0 false
```

## default：非阻塞收发

`default` 让 select 不阻塞。队列满了就丢弃的典型写法：

```go
select {
case ch <- msg:    // 入队
default:           // 满了直接丢，不等
}
```

> 无限循环里慎用 `default`——通道一直没数据会疯狂空转打满 CPU。没有明确非阻塞需求别随手加。

## 超时控制

```go
select {
case v := <-ch:
    fmt.Println(v)
case <-time.After(2 * time.Second):
    fmt.Println("timeout")
}
```

`time.After` 返回一个通道，到点收到信号。和 `time.Sleep` 的区别：`Sleep` 强制睡满；`After` 是「最多等」，`ch` 提前来数据就立刻继续。

## 配合 context 取消

真实后端最常见：长期 goroutine 配 `ctx.Done()` 退出。

```go
func waitResult(ctx context.Context, ch <-chan string) error {
    select {
    case v := <-ch:
        fmt.Println(v); return nil
    case <-ctx.Done():
        return ctx.Err()   // Canceled 或 DeadlineExceeded
    }
}
```

发送也可能阻塞，同样要防：

```go
select {
case out <- v:
case <-ctx.Done():
    return ctx.Err()
}
```

## nil channel：动态禁用分支

`nil` 通道在 select 里**永不就绪**，可用来动态移除某个 case：

```text
关闭的通道：一直就绪（返回零值，ok=false）
nil 通道：  永不就绪
```

所以合并多个通道时，某通道关闭后置 `nil`，相当于把它踢出 select：

```go
for a != nil || b != nil {
    select {
    case v, ok := <-a:
        if !ok { a = nil; continue }
        fmt.Println("a:", v)
    case v, ok := <-b:
        if !ok { b = nil; continue }
        fmt.Println("b:", v)
    }
}
```

## 空 select

`select {}` 无 case 无 default，**永久阻塞**。可让 `main` 启动 goroutine 后不退出，但实际项目更推荐 `WaitGroup` / `os.Signal` / `http.Server.Shutdown` 明确控制生命周期。

## for-select 循环

`select` 只执行一次，持续监听要套 `for`：

```go
for {
    select {
    case v, ok := <-in:
        if !ok { return }   // 通道关闭
        handle(v)
    case <-ctx.Done():
        return
    }
}
```

**经典坑**：`break` 只跳出 select，跳不出外层 for。

```go
for {
    select {
    case <-done:
        break        // ❌ 只跳出 select，for 继续
    }
}
// 正确：用 return，或带 label —— break loop
```

## 常见并发模式

**取消传播**（stage：转发数据 + 随时取消）：

```go
func stage(ctx context.Context, in <-chan int) <-chan int {
    out := make(chan int)
    go func() {
        defer close(out)
        for {
            select {
            case v, ok := <-in:
                if !ok { return }
                select {
                case out <- v * 2:
                case <-ctx.Done(): return
                }
            case <-ctx.Done(): return
            }
        }
    }()
    return out
}
```

**fan-in**（多通道合一）：合并多个输入到一个 out，就是上面 nil channel 的写法。

**fan-out**（多 worker 抢同一队列）：channel 当任务队列，一个任务只被一个 worker 拿走。

```go
for i := 0; i < 3; i++ {
    go worker(i, jobs, results)   // 3 个 worker 共享 jobs
}
```

## 完整示例

worker 池，串起 for-select、`ok` 判断、`ctx.Done()`、发送防阻塞、超时、`close` 通知结束：

```go
func worker(ctx context.Context, jobs <-chan int, results chan<- int) {
    for {
        select {
        case job, ok := <-jobs:
            if !ok { return }
            select {
            case results <- job * 2:
            case <-ctx.Done(): return
            }
        case <-ctx.Done(): return
        }
    }
}

func main() {
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()

    jobs := make(chan int)
    results := make(chan int)

    go func() { worker(ctx, jobs, results); close(results) }()
    go func() {
        defer close(jobs)
        for i := 1; i <= 3; i++ { jobs <- i }
    }()

    for {
        select {
        case r, ok := <-results:
            if !ok { fmt.Println("all done"); return }
            fmt.Println("result:", r)
        case <-time.After(2 * time.Second):
            fmt.Println("timeout"); return
        }
    }
}
// result: 2 / 4 / 6 / all done
```

## 常见坑

- **多个 case 同时就绪**：随机选，不按顺序。
- **关闭的通道一直就绪**：接收要检查 `ok`。
- **default 导致 CPU 空转**：无限循环里慎用。
- **break 只跳出 select**：要退出函数用 `return`，或带 label。
- **发送也会阻塞**：发送也要配 `ctx.Done()`。
- **nil channel 永不就绪**：可用来动态禁用 case。
- **select {} 永久阻塞**：让 goroutine 永远卡住。

## 三个最小模板

非阻塞、超时、取消——掌握这三个就够覆盖绝大多数场景：

```go
// 非阻塞
select { case v := <-ch: use(v); default: }

// 超时
select {
case v := <-ch: use(v)
case <-time.After(time.Second): /* timeout */
}

// 取消（持续监听）
for {
    select {
    case v, ok := <-ch: if !ok { return }; use(v)
    case <-ctx.Done(): return
    }
}
```

> **核心理念**：每个 case 是一个可能阻塞的通信动作，`select` 负责等其中一个能走；`context`、`timeout`、`nil/closed channel` 都是在控制这些分支何时能走、何时该退。
