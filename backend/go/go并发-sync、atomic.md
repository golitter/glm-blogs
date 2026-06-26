# sync 包与 sync/atomic

channel 用「通信」做同步（见 [通道](go通道.md)）；`sync` 包走另一条路——用「锁」保护共享内存。

## 先备好工具

```bash
go run -race main.go     # 检测数据竞争
go test -race ./...
go vet ./...             # copylocks：复制了含锁结构体会报警
```

含锁类型按值传递 = 锁被复制、保护失效，一律用指针：

```go
func bad(c Counter)  { ... }  // ❌
func ok(c *Counter)  { ... }  // ✅
```

## sync.Mutex

保护临界区，同一时刻只进一个 goroutine。`n++` 是「读-加-写」三步，并发会互相覆盖：

```go
mu.Lock(); n++; mu.Unlock()
mu.Lock(); defer mu.Unlock()   // 逻辑复杂时用 defer，防漏解锁
```

匿名嵌入后可直接 `c.Lock()`；`TryLock()`（1.18+）非阻塞尝试加锁，极少用。**两种模式**：普通模式（新来者和队列一起抢，吞吐高）↔ 饥饿模式（等太久就优先给队列，保公平），自动切换。

**不可重入**：同一 goroutine 二次 `Lock` 必自死锁。最常见的踩法是「持锁方法又调了另一个加同一把锁的方法」：

```go
func (c *Counter) Inc() {
    c.Lock(); defer c.Unlock()
    c.Add(1)                 // Add 内部又 c.Lock() → 死锁
}
func (c *Counter) Add(n int) {
    c.Lock(); defer c.Unlock()
    c.n += n
}
```

Go 刻意不做可重入锁（对比 Java `ReentrantLock`、Python `RLock`）：可重入会模糊「调用方到底持没持锁」，掩盖不变量破坏，催生隐蔽 bug。标准解法是拆成「公开方法加锁 + 私有方法假定已持锁」：

```go
func (c *Counter) Inc() {
    c.Lock(); defer c.Unlock()
    c.add(1)                 // 小写不加锁，调用方负责持锁
}
func (c *Counter) add(n int) { c.n += n }   // 调用前必须已持锁
```

## sync.RWMutex

读锁可共享、写锁独占，适合**读多写少**：

```go
func (c *Cache) Get(k string) (string, bool) {
    c.mu.RLock(); defer c.mu.RUnlock()   // 读
    v, ok := c.m[k]; return v, ok
}
func (c *Cache) Set(k, v string) {
    c.mu.Lock(); defer c.mu.Unlock()     // 写
    c.m[k] = v
}
```

有 writer 在等时，新 reader 会被挡住（写优先），避免写者被饿死。

> 写多 / 临界区小时可能比 `Mutex` 还慢，别无脑换；不能升级（持读锁再 `Lock` → 死锁），也不能降级。

## sync.WaitGroup

```go
wg.Add(1); go func() { defer wg.Done(); /* ... */ }(); wg.Wait()
```

`Done()` ≡ `Add(-1)`。坑：`Add` 必须在启动 goroutine 之前（否则 `Wait` 先看到 0 直接返回）；`Done` 不能多调，counter 变负会 panic。复用要等上一轮 `Wait` 返回后再 `Add`。

## sync.Once

懒加载、单例。并发调用 `Do`，函数体只跑一次，其余阻塞等它完成：

```go
once.Do(func() { config = loadConfig() })
```

实现 = `atomic`（判断是否跑过）+ `Mutex`（保证只一个执行）。

> `Do` 里的函数 panic 后，`Once` 仍认为已执行，不会重跑。

1.21+ 带返回值封装：`sync.OnceValue(func() string { return "hi" })`。

## sync.Cond

等「某条件成立」再继续，用于生产者-消费者、任务队列：

```go
cond := sync.NewCond(mu)
// 等待方
mu.Lock()
for !ready { cond.Wait() }   // 必须持锁调用
mu.Unlock()
// 唤醒方
mu.Lock(); ready = true; cond.Signal(); mu.Unlock()
```

`Wait()` = 释放锁 → 阻塞 → 唤醒后重新加锁。`Signal` 唤醒一个，`Broadcast` 唤醒全部。

> 必须用 `for` 检查条件：被唤醒不代表条件还成立，抢到锁要重新检查。

## sync.Map

并发安全 map，**不是普通 map 的万能替代**，只适合读多写少 / key 稳定：

```go
m.Store("k", "v")
v, ok := m.Load("k")
m.LoadOrStore("k", "default")
m.Delete("k")
m.Range(func(k, v any) bool { return true })
```

写多 / key 高频增删时性能未必好，多数业务里 `map + RWMutex` 反而更清晰。

## sync.Pool

复用临时对象（`bytes.Buffer`、Context），减少分配、压 GC：

```go
buf := pool.Get().(*bytes.Buffer)
buf.Reset(); defer pool.Put(buf)
```

> Pool 不保证存活，GC 可能清空——不能当缓存用。它是「有就复用，没就新建」，不是 LRU。

## sync/atomic

保护单变量，比锁轻。新版 typed atomic：

```go
var n atomic.Int64
n.Add(1)                       // 加
n.Load()                       // 读
n.Store(10)                    // 写
n.Swap(20)                     // 交换，返回旧值
n.CompareAndSwap(20, 30)       // CAS
```

类型：`Int32/Int64/Uint32/Uint64/Uintptr/Bool/Pointer[T]`。`Pointer[T]`（1.19+）比旧的 `atomic.Value` 更类型安全。

**CAS** 是无锁核心：值还是预期就改，否则重试：

```go
for {
    old := n.Load()
    if n.CompareAndSwap(old, old+1) { break }
}
```

> atomic 只管单变量，多变量一致性 / 改 map slice 要用锁。

## 速查

| 场景 | 推荐 |
| ---- | ---- |
| 保护临界区 | `sync.Mutex` |
| 读多写少 | `sync.RWMutex` |
| 等一组 goroutine 完成 | `sync.WaitGroup` |
| 只初始化一次 | `sync.Once` |
| 等条件成立 | `sync.Cond` |
| 并发 map（读多 / key 稳定） | `sync.Map` |
| 临时对象复用、降 GC | `sync.Pool` |
| 单变量原子读写 | `sync/atomic` |

> **核心**：通信共享数据用 channel，保护共享内存用 sync，写完都跑 `-race`。
