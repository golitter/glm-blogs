Go 语言的 `testing` 模块是官方内置的测试框架，它简单、高效且与 Go 工具链（如 `go test` 命令）无缝集成。

Go 的测试有严格的命名约定，不符合约定的文件和函数不会被 `go test` 识别：

1. **文件命名**：必须以 `_test.go` 结尾。例如 `calc.go` 的测试文件必须是 `calc_test.go`。

2. 函数命名

   ：

   - 单元测试函数：必须以 `Test` 开头，参数为 `t *testing.T`。例如 `func TestAdd(t *testing.T)`。
   - 基准测试函数：必须以 `Benchmark` 开头，参数为 `b *testing.B`。例如 `func BenchmarkAdd(b *testing.B)`。
   - 模糊测试函数：必须以 `Fuzz` 开头，参数为 `f *testing.F`（Go 1.18 引入）。
   - 示例测试函数：必须以 `Example` 开头，无参数。例如 `func ExampleAdd()`。

3. **包规则**：测试文件通常与被测试文件在同一个包下，这样就可以访问包内的未导出（小写字母开头）的变量和函数。



## 单元测试

单元测试是最常用的功能，主要通过 `testing.T` 提供的方法来控制测试流程和报告测试结果。

- `t.Log() / t.Logf()`：打印日志，默认只有在测试失败或使用 `-v` 参数时才会显示。
- `t.Error() / t.Errorf()`：打印错误日志，**并标记当前测试为失败（Fail），但会继续执行后续代码**。
- `t.Fatal() / t.Fatalf()`：打印错误日志，**标记当前测试为失败，并立即终止当前测试函数的执行**（调用 `runtime.Goexit`）。
- `t.Skip() / t.Skipf()`：跳过当前测试。



> `-v`是`go test`命令的**可选参数**，全称是`verbose`（详细模式）。它**不是代码里的设置**，而是运行测试时在终端添加的参数，用于控制测试输出的详细程度。



e.g. `lab_test.go`：

```go
package main

import "testing"

func Add(a, b int) int {
    return a + b
}

func TestAdd(t *testing.T) {
    result := Add(1, 2)
    expected := 3
    
    // Go 推荐的断言方式
    if result != expected {
        // 使用 Errorf 不会中断当前测试，可以收集更多错误信息
        t.Errorf("Add(1, 2) = %d; expected %d", result, expected)
    }
}

```



测试：

- ```go
  go test -v lab_test.go
  ```

- ```go
  go test -v -run TestAdd hello/ 
  ```

  测试hello模块下的TestAdd测试函数

