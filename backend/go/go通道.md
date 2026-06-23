# 通道（channel）

goroutine 之间传递数据的管道，自带同步能力。必须用 `make` 创建。

```go
make(chan int)     // 非缓冲，容量 0
make(chan int, 10) // 缓冲，容量 10
```

## 基本操作

```go
c <- 1       // 发送
x := <-c     // 接收
v, ok := <-c // 接收，ok 为 false 表示通道已关闭且空
close(c)     // 关闭，之后不能再发送
```

操作四象限：

| 操作 | 正常 | 已关闭 |
| ---- | ---- | ------ |
| 发送 | 可能阻塞 | **panic** |
| 接收 | 可能阻塞 | 返回零值，ok=false |
| 关闭 | 正常 | **panic** |

> **谁发送谁关闭**。向已关闭通道发数据会 panic，最常见的崩溃点。

## 非缓冲 vs 缓冲

非缓冲没有存放空间，发送和接收**当面交接**：

```go
c := make(chan int)
go func() { c <- 1 }() // 没人接收就阻塞
x := <-c               // 接收后发送方才继续
```

缓冲有临时空间，没满就能发、不空就能收：

```go
c := make(chan int, 2)
c <- 3 // 不阻塞
c <- 5 // 不阻塞，刚好满
// c <- 7 // 再发就阻塞
```

```text
非缓冲：必须有人接，才能发。  —— 同步交接
缓冲：  有空位就能发，有数据就能收。 —— 临时排队
```

## 遍历

`for range` 一直取数据，直到通道关闭才退出，所以发送方结束后**必须 `close`**。

```go
go func() {
    for i := 0; i < 3; i++ { c <- i }
    close(c) // 不关，range 永久阻塞 → 死锁
}()
for v := range c { fmt.Println(v) }
```

## 单向通道

限定方向，主要用于函数参数防呆：

```go
chan<- T  // 只发送
<-chan T  // 只接收
```

```go
func producer(out chan<- int) { out <- 1; close(out) }
func consumer(in <-chan int)  { for v := range in { fmt.Println(v) } }
```

> 双向可隐式转单向，反向不行。

## select 多路复用

像 switch，但每个 case 是通道操作，**随机**挑一个已就绪的执行。

```go
select {
case v := <-c1:
    fmt.Println("收到", v)
case <-time.After(2 * time.Second):
    fmt.Println("超时")
default:
    fmt.Println("都没就绪") // 非阻塞探测
}
```

要点：多个就绪随机选（不会饿死某个通道）；都没就绪有 default 走 default，无 default 则阻塞。常用于**超时控制**和**非阻塞读写**。

## 常见死锁坑

```go
c := make(chan int)
c <- 1   // 死锁！主协程无人接收
```

- 无缓冲通道在主协程直接发 → 死锁。
- 向已关闭通道发 → panic。
- range 前没 close → 永久阻塞。
- `nil` 通道的收发永久阻塞（select 里可用来临时禁用某个 case）。

## 典型用法

**同步信号**（`chan struct{}` 只传事件不传数据，零开销）：

```go
done := make(chan struct{})
go func() { /* 干活 */ close(done) }()
<-done // 阻塞到完成
```

**Worker Pool**：

```go
jobs := make(chan int, 10)
for w := 0; w < 3; w++ {
    go func() { for j := range jobs { /* 处理 j */ } }()
}
for i := 1; i <= 5; i++ { jobs <- i }
close(jobs)
```

**优雅退出**：用 `select` 监听一个 `quit` 通道，`close(quit)` 通知退出。

## 速查

| 场景 | 选择 |
| ---- | ---- |
| 同步交接 | 非缓冲 `chan T` |
| 削峰排队 | 缓冲 `chan T, N` |
| 多通道监听 | `select` |
| 超时 | `select` + `time.After` |
| 只传信号 | `chan struct{}` |

> **核心理念**：不要通过共享内存来通信，而要通过通信来共享内存。
