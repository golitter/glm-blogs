# error：错误是值

Go 没有异常和 `try/catch`——错误就是一个普通的返回值，由调用者显式判断、显式处理。

> **核心**：错误是值（errors are values）。要么处理，要么 `return`，不要既处理又返回。

## error 接口

内置接口，只要实现 `Error() string` 就是 error：

```go
type error interface { Error() string }
```

## 创建错误

```go
err := errors.New("user not found")           // 固定信息
err := fmt.Errorf("user %d not found", id)    // 带变量
```

## 标准模式

```go
data, err := os.ReadFile(path)
if err != nil {
    return nil, err        // 处理不了就往上抛
}
```

> **不要 `_` 吞错误**。真要忽略就写清原因：`_ = file.Close() // 只读，关失败无所谓`。

## 错误包装：%w（Go 1.13）

`%w` 给错误加上下文，**保留错误链**：

```go
return fmt.Errorf("query user by id %d failed: %w", id, err)
// query user by id 10 failed: sql: no rows in result set
```

### %v vs %w

```go
fmt.Errorf("failed: %v", err)   // ❌ 错误链断了，errors.Is 失效
fmt.Errorf("failed: %w", err)   // ✅ 保留错误链
```

> 包装永远用 `%w`，除非你确定要切断链。

## errors.Is：判断是不是某个错误值

配**哨兵错误**（提前定义的固定错误值）使用：

```go
var ErrNotFound = errors.New("not found")

// 包装后返回
return fmt.Errorf("get user failed: %w", ErrNotFound)

// 即使被包了一层也能认出来
if errors.Is(err, ErrNotFound) { /* ... */ }
```

标准库哨兵：`io.EOF`、`sql.ErrNoRows`、`os.ErrNotExist`。

> **别用字符串比错误**（`err.Error() == "..."` / `strings.Contains`）——错误文案一改就失效。

## errors.As：提取某种错误类型

错误需要带信息（错误码、字段名）时用**自定义错误类型**，再用 `errors.As` 取出：

```go
type BizError struct{ Code int; Msg string }
func (e *BizError) Error() string { return e.Msg }

return &BizError{Code: 10001, Msg: "name empty"}

var biz *BizError
if errors.As(err, &biz) {
    fmt.Println(biz.Code)      // 即使被 %w 包过也能取出
}
```

| 方法 | 用途 | 适合 |
| ---- | ---- | ---- |
| `errors.Is` | 判断是不是某错误**值** | 哨兵错误 |
| `errors.As` | 提取某错误**类型** | 自定义错误类型 |

> `errors.Unwrap(err)` 能一层层拆 `%w`，但日常基本不用——Is/As 已经够。

## panic / recover

`panic` 是「程序没法继续了」，**不要拿来处理业务错误**。适用于：配置缺失启动失败、不变量被破坏、初始化失败。

```go
func MustLoadConfig() Config {
    cfg, err := LoadConfig()
    if err != nil { panic(err) }   // 启动期失败，该崩
    return cfg
}
```

`recover` 只在 `defer` 里生效，捕获 panic 防崩溃：

```go
defer func() {
    if r := recover(); r != nil { log.Println("panic:", r) }
}()
```

> **goroutine 里的 panic 不 recover 会拖垮整个进程**。后台 goroutine、HTTP 中间件都要包 recover。

## nil 接口陷阱

经典坑：返回一个「值为 nil 的具体错误指针」，**接口不是 nil**。

```go
func DoSomething() error {
    var err *MyError = nil   // *MyError 类型的 nil
    return err
}
err := DoSomething()
fmt.Println(err == nil)      // false！接口(类型=*MyError, 值=nil) ≠ nil
```

> 接口 = 动态类型 + 动态值，**两者都 nil 才等于 nil**。别返回 nil 的具体错误指针，要 `return nil`。

## 后端错误分层

```text
DAO      → 返回原始错误（不知道 HTTP）
Service  → %w 加业务上下文，或转成业务错误
Handler  → errors.Is/As 判断，转成 HTTP 响应
```

```go
// Service：底层错误 → 业务语义
if errors.Is(err, gorm.ErrRecordNotFound) {
    return nil, ErrUserNotFound
}

// Handler：业务错误 → HTTP
var appErr *AppError
if errors.As(err, &appErr) {
    c.JSON(appErr.StatusCode, gin.H{"code": appErr.Code, "msg": appErr.Message})
    return
}
```

> 日志**只在最外层打一次**——内层每层都 `log+return` 会让一条错误被刷屏。

## 速查

| 需求 | 用法 |
| ---- | ---- |
| 创建固定错误 | `errors.New("msg")` |
| 创建带变量错误 | `fmt.Errorf("id %d invalid", id)` |
| 包装错误 | `fmt.Errorf("x: %w", err)` |
| 判断某错误值 | `errors.Is(err, ErrX)` |
| 提取错误类型 | `errors.As(err, &target)` |
| 业务错误 | 返回 `error` |
| 严重错误 | `panic` |
| 捕获 panic | `defer + recover` |

> **核心公式**：底层返回原始错误 → 中层 `%w` 加上下文 → 最外层 `errors.Is`/`errors.As` 判断 → 统一日志与响应。
