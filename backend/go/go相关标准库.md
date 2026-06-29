# Go 常用标准库

后端日常 90% 的活用标准库就够，不用引第三方。这篇集中记最常用的：`strings/strconv/time/os/filepath/io/bufio/regexp`，外加 `json`、`http`。

## strings

最常用的字符串工具：

```go
strings.Contains(s, "go")            // 是否包含
strings.HasPrefix(s, "http")         // 前缀
strings.HasSuffix(s, ".txt")         // 后缀
strings.Index(s, "go")               // 首次位置，没有返回 -1
strings.Count(s, "a")                // 出现次数
strings.TrimSpace(s)                 // 去首尾空白
strings.Trim(s, " ")                 // 去首尾指定字符集
strings.Split("a,b,c", ",")          // 分割 → []string
strings.Join([]string{"a","b"}, "-") // 拼接
strings.Replace(s, "a", "b", n)      // 替换 n 次，-1 = 全部
strings.ReplaceAll(s, "a", "b")      // 替换全部
strings.Repeat("ab", 3)              // 重复
strings.ToLower / ToUpper
```

**循环拼接别用 `+`**（每次生成新串，整体 O(n²)），用 `strings.Builder`：

```go
var b strings.Builder
for i := 0; i < 1000; i++ { b.WriteString(strconv.Itoa(i)) }
s := b.String()
```

> `Builder` 非并发安全；预知长度用 `b.Grow(n)` 预分配减少扩容。

## strconv

字符串与基础类型互转：

```go
n, err := strconv.Atoi("123")    // string → int
s := strconv.Itoa(123)           // int → string

strconv.ParseInt("123", 10, 64)  // 进制 10、bitSize 64 → int64
strconv.FormatInt(255, 16)       // → "ff"
strconv.ParseFloat("3.14", 64)   // bitSize: 32/64
strconv.ParseBool("true")        // 接受 1/0/t/f/true/false...
```

> `ParseInt(s, base, bitSize)` 的 `bitSize` 是「限制取值范围」，返回值类型固定 `int64`；如 `ParseInt("255",10,8)` 超出 int8 范围会报错。

## regexp

```go
ok, _ := regexp.MatchString(`^\d+$`, "12345") // 临时用一次

re := regexp.MustCompile(`^\d+$`)             // 复用：包级变量先编译
re.MatchString("123")
re.FindStringSubmatch("name=tom")             // [name=tom tom]，下标 1 是捕获组
re.FindAllString("a1 b2 c3", -1)              // 所有匹配
re.ReplaceAllString(s, "x")                   // 替换
```

> Go 用 RE2 语法，**不支持反向引用 / 前瞻后瞻**；`MustCompile` 正则写错会 panic，正则来自用户输入用 `Compile`。

## time

```go
now := time.Now()
now.Format("2006-01-02 15:04:05")             // 格式化
t, _ := time.Parse("2006-01-02 15:04:05", "2026-06-27 10:30:00") // 解析

now.Unix()                                    // 秒级时间戳
now.UnixMilli()                               // 毫秒
now.UnixNano()                                // 纳秒
time.Unix(1730000000, 0)                      // 时间戳 → Time
```

格式串不是 `yyyy-MM-dd`，而是固定参考时间 `2006-01-02 15:04:05`（= 1/2 3:04:05 06，正好 1~6）。

间隔与计时：

```go
time.Sleep(2 * time.Second)
cost := time.Since(start)   // 自 start 至今，等价 time.Now().Sub(start)
now.Add(24 * time.Hour)     // 加减时间
```

定时器 / 超时控制：

```go
<-time.After(2 * time.Second)                  // 等 2 秒一次
ticker := time.NewTicker(time.Second); defer ticker.Stop()  // 周期
for range ticker.C { fmt.Println("tick") }

select {                                       // 超时控制
case v := <-ch: fmt.Println(v)
case <-time.After(3 * time.Second): fmt.Println("timeout")
}
```

> `time.Parse` 默认按 **UTC**，本地时间用 `time.ParseInLocation`，否则差 8 小时；`time.Since` 靠单调时钟，系统时间被改也不影响计时。

## os

文件、目录、环境变量、进程：

```go
data, _ := os.ReadFile("config.json")          // 读整个文件（1.16+，替代 ioutil）
os.WriteFile("out.txt", []byte("hi"), 0644)    // 写，存在则覆盖

f, _ := os.Open("app.log"); defer f.Close()    // 只读打开
f, _ := os.Create("out.txt"); defer f.Close()  // 创建（存在则清空）
f, _ := os.OpenFile("a.log", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644) // 追加写

os.MkdirAll("data/logs/app", 0755)             // 递归建目录
os.Remove("a.txt")                             // 删文件 / 空目录

os.Getenv("APP_ENV")                           // 不存在返回 ""
v, ok := os.LookupEnv("APP_ENV")               // 区分「不存在」和「空值」
os.Args                                        // 命令行参数，[0] 是程序路径
os.Exit(1)                                     // 直接退出，defer 不执行
```

判断文件是否存在：

```go
_, err := os.Stat("config.json")
if errors.Is(err, os.ErrNotExist) { /* 不存在 */ }
```

> 别只看 `err != nil` 就认定不存在，也可能是权限不足；`os.Exit` 前别依赖 `defer` 做清理。

## path/filepath

跨平台路径，**别手拼 `dir + "/" + name`**（Windows 分隔符是 `\`）：

```go
filepath.Join("data", "logs", "app.log")  // 自动选分隔符 + 清理 . ..
filepath.Dir(p)   // 目录 data/logs
filepath.Base(p)  // 文件名 app.log
filepath.Ext(p)   // 扩展名 .log
filepath.Abs(p)   // 绝对路径
filepath.Glob("*.go")  // 通配匹配
```

遍历目录（`import "io/fs"`，比老的 `Walk` 更快）：

```go
filepath.WalkDir(".", func(path string, d fs.DirEntry, err error) error {
    if err != nil || d.IsDir() { return err }
    if filepath.Ext(path) == ".go" { fmt.Println(path) }
    return nil
})
```

## io

流式读写抽象，核心两接口：

```go
type Reader interface { Read(p []byte) (n int, err error) }
type Writer interface { Write(p []byte) (n int, err error) }
```

文件、网络连接、HTTP Body、`bytes.Buffer` 都是 Reader/Writer；组合接口如 `io.ReadCloser`（`http.Response.Body` 就是它）。

```go
data, _ := io.ReadAll(r)          // 一次读完（小数据用）
io.Copy(dst, src)                 // 流式复制，到 EOF 为止
io.CopyN(dst, src, n)             // 只复制 n 字节
io.LimitReader(r, 1024)           // 限制最多读 1KB，防撑爆内存
```

> `io.EOF` 是「读到末尾」的正常信号，不是错误；大文件别 `ReadAll`。

## bufio

带缓冲的 I/O，减少系统调用：

```go
scanner := bufio.NewScanner(file)
for scanner.Scan() {              // 逐行读
    line := scanner.Text()
}
if err := scanner.Err(); err != nil { /* 别漏 */ }

w := bufio.NewWriter(file)
w.WriteString("hello\n")
w.Flush()                         // 必须 Flush 才真正落盘
```

> ⚠️ Scanner 默认单行上限 **64KB**，超长行报 `ErrTooLong`；手动扩：`scanner.Buffer(make([]byte, 1024), 1024*1024)`。

## encoding/json

序列化 / 反序列化。**字段必须大写（导出）**才能被读写，小写字段对外不可见。

```go
b, _  := json.Marshal(user)                 // 序列化
b, _  := json.MarshalIndent(user, "", "  ") // 带缩进，方便阅读
err  := json.Unmarshal(b, &user)            // 反序列化，第二参必须传指针
```

struct tag：

```go
type User struct {
    ID    int    `json:"id"`               // 重命名
    Name  string `json:"name,omitempty"`   // omitempty：零值时省略
    Pass  string `json:"-"`                // 忽略该字段
}
```

> `omitempty` 的零值 = 空串 / 0 / false / nil / 空切片；`Unmarshal` 传值类型只是副本，外部拿不到结果，必须传 `&`。

## net/http

客户端：

```go
resp, err := http.Get("https://example.com") // 简单 GET
defer resp.Body.Close()
body, _ := io.ReadAll(resp.Body)

// POST JSON
b, _ := json.Marshal(p)
resp, _ := http.Post(url, "application/json", bytes.NewReader(b))
```

更细的控制（超时、Header、方法）用 `http.Client` + `http.NewRequest`：

```go
client := &http.Client{Timeout: 10 * time.Second}
req, _ := http.NewRequest("GET", url, nil)
req.Header.Set("Authorization", "Bearer xxx")
resp, _ := client.Do(req); defer resp.Body.Close()
```

服务端（标准库自带，生产多用 Gin 等框架）：

```go
http.HandleFunc("/index", func(w http.ResponseWriter, r *http.Request) {
    fmt.Fprintln(w, "hello")     // 往响应里写
})
http.ListenAndServe(":8080", nil)
```

## 速查

| 想做 | 用 |
| ---- | ---- |
| 判前缀/包含/分割 | `strings` |
| string ↔ int | `strconv.Atoi / Itoa` |
| 正则匹配/提取 | `regexp` |
| 时间格式化/时间戳 | `time` |
| 读写文件 | `os.ReadFile / WriteFile` |
| 跨平台拼路径 | `filepath.Join` |
| 流式复制 | `io.Copy` |
| 逐行读文件 | `bufio.Scanner` |
| JSON 编解码 | `encoding/json` |
| HTTP 请求 | `net/http` |

> **核心**：把 `strings/strconv/time/os/filepath/io/bufio` 加上 `json/http` 用熟，后端日常基本不用引第三方库。
