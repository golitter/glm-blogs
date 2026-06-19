[Go 类型系统概述 | Go 101](https://gfw.go101.org/article/type-system-overview.html)

> 由 Claude Code（glm-5.2）整理，经人工审查。

Go 是静态强类型、编译型语言。本文梳理 Go 类型系统里的各种概念术语，不熟这些概念很难精通 Go。

## 基本类型（basic type）

内置 17 种基本类型，都属于**预声明类型**（predeclared type）：

- 字符串 `string`、布尔 `bool`。
- 数值：`int8/uint8/uint16/int16/int32/uint32/int64/uint64/int/uint/uintptr`、`float32/float64`、`complex64/complex128`。

注意 `byte` 是 `uint8` 的内置别名，`rune` 是 `int32` 的内置别名。

## 组合类型（composite type）

由基本类型组合而来，包括：

- **指针** `*T`
- **结构体** `struct{...}`
- **函数** `func(int) (bool, string)` —— Go 中函数是一等公民
- **容器**：数组 `[5]T`、切片 `[]T`、映射 `map[Tkey]T`
- **通道** `chan T`、`chan<- T`、`<-chan T`
- **接口** `interface{...}`

其中数组、切片、映射是正式意义上的**容器类型**，它们的元素类型由字面形式中的 `T` 决定；字符串的元素类型是 `byte`。

## 类型定义与类型别名

`type` 有两种用法，差别在是否多一个 `=`：

```go
// 类型定义：创造全新类型，与源类型是两个不同类型，赋值需显式转换
type Age int
var a int = 10
var b Age = Age(a) // 必须转换

// 类型别名：编译期完全替换为原类型，两者等价
type MyInt = int
var d MyInt = 10 // 无需转换
```

类型定义可同时声明多个（用 `()` 包裹），也可出现在函数体内：

```go
type (
    MyInt int
    Age   int
    Text  string
)
type IntPtr *int
type StringSlice []string
```

## 具名类型与无名类型

- **具名类型**：预声明类型（非别名）、定义类型、泛型实例化类型、类型参数类型。
- **无名类型**：用字面形式表示的组合类型，如 `[]int`、`map[string]int`。无名类型一定是组合类型（反之未必）。

> 注意：类型别名虽有个名字，但它可能表示一个无名类型，例如 `type table = map[string]int` 中的 `table` 表示的是无名类型。

## 底层类型（underlying type）

每个类型都有底层类型，溯源规则：**遇到内置类型或无名类型时结束**。

```go
type MyInt int          // 底层类型 int
type Age MyInt          // MyInt → int
type IntSlice []int     // 底层类型 []int（无名类型，溯源直接结束）
type MyIntSlice []MyInt // []MyInt → []int
```

底层类型决定了很多分类：底层为 `bool` 的叫布尔类型、底层为内置整数的叫整数类型…… 统称为数字值类型。底层类型在**类型转换、赋值、比较**规则中起关键作用。

## 可比较类型与不可比较类型

以下类型的值**不能用 `==` / `!=` 比较**，称为不可比较类型：

- 切片、映射、函数
- 含不可比较字段的结构体、元素不可比较的数组

其余为可比较类型。**映射的键类型必须是可比较类型**。

## 几个重要概念速查

- **零值**：每个类型有零值。`nil` 是切片、映射、函数、通道、指针、接口零值的字面量。
- **方法集**：一个类型的所有方法组成它的方法集；某类型的方法集是接口方法集的超集，就说它**实现了**该接口（隐式实现）。
- **接口的动态类型/动态值**：接口值可包裹一个非接口值，被包裹值的类型与值分别叫动态类型、动态值。零值接口值两者均不存在。
- **通道方向**：双向 `chan T`、只写 `chan<- T`、只读 `<-chan T`。
- **值部**：运行时一个值有直接部分，可能还有间接部分（如切片/映射底层由指针引用）。谈"值尺寸"一般指直接部分的字节数。

## 错误处理

Go 没有 `try-catch`，采用显式的错误值返回。`error` 是内置接口类型，调用方需检查是否为 `nil`：

```go
content, err := ReadFile("data.txt")
if err != nil {
    log.Fatal(err)
}
// 正常逻辑
```