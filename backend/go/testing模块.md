# Go testing 模块

Go 内置了 `testing` 测试框架，并通过 `go test` 完成测试代码的编译和执行，不需要额外安装测试运行器。

## 命名约定

只有符合约定的文件和函数才会被 `go test` 识别：

- 测试文件必须以 `_test.go` 结尾，例如 `calc_test.go`。
- 单元测试函数：`func TestXxx(t *testing.T)`。
- 基准测试函数：`func BenchmarkXxx(b *testing.B)`。
- 模糊测试函数：`func FuzzXxx(f *testing.F)`。
- 示例测试函数：`func ExampleXxx()`。

测试文件可以使用原包名，访问包内未导出的内容：

~~~go
package calc
~~~

也可以使用 `calc_test` 这样的外部包名，只通过导出的 API 测试，更接近真实调用方。

## 单元测试

被测代码：

~~~go
package calc

func Add(a, b int) int {
	return a + b
}
~~~

测试代码：

~~~go
package calc

import "testing"

func TestAdd(t *testing.T) {
	got := Add(1, 2)
	want := 3

	if got != want {
		t.Fatalf("Add(1, 2) = %d, want %d", got, want)
	}
}
~~~

Go 标准库没有内置断言函数，通常直接用 `if` 比较结果。

常用方法：

- `t.Logf`：记录日志，测试失败或使用 `-v` 时显示。
- `t.Errorf`：标记测试失败，但继续执行当前测试。
- `t.Fatalf`：标记测试失败，并立即结束当前测试。
- `t.Skip`：跳过当前测试。
- `t.Helper`：将函数标记为测试辅助函数。
- `t.Cleanup`：注册测试结束后的清理操作。
- `t.TempDir`：创建测试专用临时目录并自动清理。

## go test 常用命令

`go test` 主要按“包”运行，通常在 `go.mod` 所在目录执行。

~~~bash
go test                         # 测试当前包
go test ./calc                  # 测试指定包
go test ./...                   # 递归测试当前模块的所有包
go test -v ./...                # 显示每个测试的详细结果
go test -run '^TestAdd$' .      # 只运行指定测试
go test -count=1 ./...          # 忽略成功结果缓存，强制重跑
go test -count=10 ./...         # 连续运行 10 次
go test -shuffle=on ./...       # 随机打乱测试顺序
go test -failfast ./...         # 失败后不再启动新测试
go test -timeout=30s ./...      # 设置单个包的测试超时
~~~

`-run` 接受正则表达式。精确匹配时应使用 `^` 和 `$`，避免运行名称相似的测试。

子测试使用 `/` 分隔：

~~~bash
go test -run '^TestAdd$/negative_numbers$' .
~~~

### 为什么不推荐只测试一个文件

`go test calc.go calc_test.go` 只编译列出的文件。如果遗漏同包的其他源码，可能出现未定义符号。项目中应优先按包执行，再用 `-run` 筛选：

~~~bash
go test -run '^TestAdd$' ./calc
~~~

完整参数可以通过 `go help testflag` 查看。

## 表格测试

多组输入可以放进同一张测试表，并用 `t.Run` 创建子测试：

~~~go
func TestAdd(t *testing.T) {
	tests := []struct {
		name string
		a    int
		b    int
		want int
	}{
		{name: "positive", a: 1, b: 2, want: 3},
		{name: "negative", a: -1, b: -2, want: -3},
		{name: "zero", a: 5, b: 0, want: 5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Add(tt.a, tt.b)
			if got != tt.want {
				t.Fatalf("got %d, want %d", got, tt.want)
			}
		})
	}
}
~~~

这样新增用例只需增加一行数据，并且可以单独运行某个子测试。

## 隔离数据库等外部依赖

单元测试不应直接依赖 MySQL、网络或第三方服务，否则会变慢，并受本地环境和测试数据影响。

常见做法是让业务代码依赖一个小接口：

~~~go
type UserRepository interface {
	Create(user *User) error
	GetByName(name string) (*User, error)
}

type UserService struct {
	repo UserRepository
}
~~~

生产环境注入真实 repository，测试中注入 fake：

~~~go
type fakeUserRepo struct {
	created   *User
	createErr error
}

func (f *fakeUserRepo) Create(user *User) error {
	f.created = user
	return f.createErr
}

func (f *fakeUserRepo) GetByName(name string) (*User, error) {
	return nil, ErrNotFound
}
~~~

fake 可以记录调用参数并返回预设结果，用于验证业务逻辑是否正确。接口应放在使用方一侧，只声明实际需要的方法。

涉及真实 SQL 的 repository 测试更适合作为集成测试，使用独立测试数据库并负责准备、隔离和清理数据。

## 使用 httptest 测试 HTTP

`net/http/httptest` 可以在内存中构造请求和响应，不需要监听真实端口：

~~~go
func TestHelloHandler(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/hello?name=Go", nil)
	recorder := httptest.NewRecorder()

	helloHandler(recorder, req)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusOK)
	}
	if recorder.Body.String() != "hello, Go" {
		t.Fatalf("body = %q, want %q", recorder.Body.String(), "hello, Go")
	}
}
~~~

Handler 测试主要检查：

- JSON、path 和 query 参数是否正确解析。
- 非法输入是否返回正确状态码。
- 响应状态码和数据结构是否正确。
- middleware 写入的上下文值能否传递下去。

业务规则应主要由业务层单元测试覆盖，不必在 Handler 测试中全部重复。

## 覆盖率与竞态检测

~~~bash
go test -cover ./...                         # 显示覆盖率
go test -coverprofile=coverage.out ./...     # 生成覆盖率文件
go tool cover -func=coverage.out             # 按函数查看
go tool cover -html=coverage.out             # 浏览器可视化
go test -race ./...                          # 检测数据竞争
~~~

覆盖率用于发现遗漏，不等于测试质量。应优先覆盖关键行为、错误分支和边界条件。

## 常见问题

- 显示 `[no test files]`：当前包没有符合规则的测试文件或测试函数。
- 显示 `(cached)`：Go 复用了成功结果，使用 `-count=1` 强制重跑。
- fake 无法实现接口：检查方法名、参数和返回值是否完全一致。
- 测试一直不结束：用 `-timeout` 获取 goroutine 堆栈，再检查 channel、锁和未退出的 goroutine。

