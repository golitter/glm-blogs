注释跟cpp一样。

标识符命名规则也是。

运算符：没有`~`取反，只有`^`，`3^3`表示异或，`^3`表示取反。没有++a，只有a++形式，同时a++是statement，a++本身没有值。a=b++是错误的。

字符串反引号不能转义字符。

类型声明说后置的，类似python：`const name string = "glm"`。

花括号不能省（cpp中单个语句此时可以省）。

**函数内的**所有的变量都要用上。

# 包（Package）

在 Go 中，程序是通过将包链接在一起来构建的。

## 基本概念
- **导入单位**：Go 中进行导入的最基本单位是一个**包**，而不是 `.go` 文件
- **物理结构**：包其实就是一个文件夹，英文名 package
- **共享范围**：包内共享所有变量、常量以及所有定义的类型
- **命名规范**：包的命名风格建议都是小写字母，并且要尽量简短

## 三个容易混淆的「名字」

| 名字 | 定义位置 | 作用 |
| ---- | -------- | ---- |
| **导入路径** | `import "xxx"` | 末段是**目录名**，告诉编译器去哪找源码 |
| **包名** | 源文件首行 `package xxx` | 代码里调用时的前缀（默认 = 目录名） |
| **模块路径** | `go.mod` 第一行 `module xxx` | 整个项目根身份，所有导入路径的前缀 |

> 导入路径 = 模块路径 + 目录相对路径。包名可以和目录名不一致，但**建议保持一致**。

## 目录结构示例

```
golearn/                  ← 项目根(go mod init)
├── go.mod                ← module golearn
├── hello/hello.go        ← package main, import "golearn/tea"
└── tea/tea.go            ← package tea, func Make(...)
```

## go.mod

一个项目只需**一个根 `go.mod`**，子目录不要再各自 `go mod init`。

- `go mod init <模块名>`：初始化
- `go mod tidy`：整理依赖
- `go get <包路径>`：添加第三方包

## import 写法

```go
import (
    "fmt"               // 批量导入(推荐)
    "golearn/tea"
)
// import _ "pkg"  // 匿名引入：仅执行目标包 init()，常用于驱动注册
// import f "fmt"  // 别名 / import . "fmt" 句点引入(少用)
```

> **铁律**：导入必须使用（否则 `imported and not used`）；导入路径末段 = 目录名。

## 依赖规则

- **禁止循环依赖** —— a→b→c，则 c 不能再引 a/b。
- **`internal` 包** —— 只能被其直接父目录（及子目录）引入。
- **程序包 vs 库包** —— `main` 包是程序包（唯一且不可引入），其余是库包。

## init 函数

- 可声明多个，**无参无返回**，不能声明名为 `init` 的变量/常量/类型。
- `main` 前串行执行一次。**初始化顺序**：被依赖的包先加载 → 包级变量先于 `init` → `init` 先于 `main`。

## 可见性规则

Go 没有 `public`/`private`，用首字母大小写控制：

| 命名规则             | 可见性                           | 示例                  |
| -------------------- | -------------------------------- | --------------------- |
| **大写字母开头**     | 公有类型/变量/常量（包外可见）   | `MyName`              |
| **小写或下划线开头** | 私有类型/变量/常量（仅包内可见） | `mySalary` 或 `_temp` |

### 示例说明
```go
const MyName = "John"     // 公开常量（大写开头）
const mySalary = 5000     // 私有常量（小写开头）
```



# 数据类型

在go中，0和false等表示的是不同的。

整数中有：

- uintX，X可以为8、16、32、64，表示无符号
- intX，同上，有符号
- uint、int至少32位，**跟着系统走**
- rune是int32别名，**强调是一个unicode字符**
- uintptr：等效于无符号64位整数，但是专门存放指针运算。

> int、uint和int32、uint32是完全不一样的，不能直接相加。
浮点数：float32、64，IEEE754格式的

## 类型转换
go中不可以**隐式转换**。
- 数值之间的相互转换：小转大可以，大转小可能会被截断
- 浮点数、整数之间的转换：浮点数转整数，丢掉小数部分；整数转浮点数可能精度丢失
- 数值、字符串转换：需要使用`strconv`包来进行转换，因为正常的`string(123)`是转为对应ascii码
- 字符串转byte/rune：字符串本身就是byte，转byte简单。
```go
s1 := "go"
byteSlice := []byte(s1)
s11 := string(byteSlice)

s2 := "go语"
runeSlice := []rune(s1)
runeSlice[2] = "言"
s22 := string(runeSlice) // go言
```

# 常量

只能是基本数据类型。常量在声明时就必须初始化一个值。批量声明可以使用（）

`iota`是一个内置的常量标识符，通常用于表示一个常量声明中的无类型整数序数，一般都是在括号中使用。



# 变量

```go
var name string = "glm"
var age int = 32

name, age = "kkglm", 22
// 具体是什么类型交给编译器自行推断
// 短变量初始化
name1 := "kkglm"
// 只能用在声明截断，重新赋值不行

// 交换
num1, num2 = num2, num1

// 左侧不需要全是新变量，但是只要要有一个
f, err = os.Open("a.txt")
g, err = os.Open("b.txt")
```

**没有强制类型转换**，需要手动进行转换。



# 输入/输出

输出的`Printf`跟cpp相似。



# 条件语句

## if-else

```go
func main() { // 函数
	var score int = 82
	var answer string
	if score >= 0 && score < 60 {
		answer = "不及格"
	} else if score < 80 {
		answer = "及格"
	} else if score < 90 {
		answer = "良好"
	} else {
		answer = "优秀"
	}
	fmt.Printf("%s", answer)
}
```

## switch

```go
func main() { // 函数
	var score int = 44
	var answer string
	switch { // 等价于 switch true {}
		case score >= 0 && score < 60:
			answer = "不及格"
		case score < 80:
			answer = "及格"
		case score < 90:
			answer = "良好"
		case score <= 100:
			answer = "优秀"
		default:
			answer = "成绩不合法"
	}
	fmt.Printf("%s", answer)
}
```



# 循环

```go
for init statement; expression; post statement {
  execute statement
}
```

```go
func main() {
	for i := 0; i < 10; i = i + 1 {
		fmt.Printf("i = %d\n", i)
	}
}
```



for range 

```go
for i := range 10 {
    fmt.Println(i)
}

n := 10
for i := range n {
    fmt.Println(i)
}

const n = 10
for i := range n {
  fmt.Println(i)
}
```

Continue break

# 切片与数组

## 数组

```go
	const n = 3
	var nums [n]int
	for i := 0; i < n; i += 1 {
		nums[i] = i
	}
	fmt.Println(nums)

	var nums2 = [n]int{1,2}
	fmt.Println(nums2)

	var nums3 = [...]int{1,2,3,4}
	fmt.Println(nums3)
```

指针

```go
	nums := new([5]int)
	fmt.Printf("%T\n", nums) // *[5]int
```

长度：`len() / cap()`

> 对于数组 len/cap是相同的。

切割数组的格式为`arr[startIndex:endIndex]`，切割的区间为**左闭右开**。且数组在切割后，就会变为切片类型。

若要将数组转换为切片类型，不带参数进行切片即可，转换后的切片与原数组指向的是同一片内存，修改切片会导致原数组内容的变化。

## 切片

```go
nums := []int{1,2}
nums := make([]int, 0, 0) // 类型 长度 容量
// len(nums) cap(nums)

// 追加
nums = append(nums, 1,2)
// 头部添加
nums = append([]int{1,2}, nums...)

// 从中间下标 i 插入元素
nums = append(nums[:i+1], append([]int{999, 999}, nums[i+1:]...)...)


// 删除 前面n个
nums = nums[n:]
// 尾部nge
nums = nums[: len(nums) - n]

// 删除所有
nums = nums[:0]
```

1.2版本 **扩展表达式**

```shell
slice[low:high:max]
```

使用扩展表达式的切片容量为`max-low`。

```go
	nums := []int{1, 2, 3, 4, 5}
	nums = append([]int{0}, nums...)
	fmt.Println(nums, cap(nums)) // 6
	nums2 := nums[1:4]
	fmt.Println(nums2, cap(nums2)) // 5
```



# 字符串

不可变、只读的字节序列。

```go
s := "abc"
// s[0] 是byte，打印出来是字节序列（数字，不是'a')
bytes := []byte(s) // 字符串转字节切片
bytes = append(bytes, 96, 97, 98)
s2 := string(bytes) // 序列切片转字符串
```

字符串的长度，其实并不是字符的个数，而是字节序列的长度。

ascii码是一个字符一个字节。如果是中文则不同，文字占三个字节。

```go
	s := "abc"
	s2 := "你好啊"
	fmt.Printf("s: %s, s2: %s\n", s, s2)
	fmt.Printf("s: %d, s2: %d\n", len(s), len(s2)) // 3 9
```

默认for遍历是字节，for range遍历是utf8（可以遍历正确的中文字符）

使用`[]rune()`可以转为utf8格式，此时支持中文字符。

```go
func main() {
   str := "hello 世界!"
   runes := []rune(str)
   for i := 0; i < len(runes); i++ {
      fmt.Println(string(runes[i]))
   }
}
```



# 映射表

map的key必须是可比较的。

```go
	mp := map[int]string {
		1: "one",
		2: "two",
		3: "three",
	}
	mp2 := make(map[int]string, 8) // 类型，初始容量
```

maps是引用类型，零值或者未初始化可以访问，但是无法存放元素。



`mp[val]` 返回两个值，`value, status`，其中`status`是布尔值，代表键是否存在。

```go
val, exist := mp[3333]
	if exist {
		fmt.Println(val)
	} else {
		fmt.Println("不存在")
	}
```



# 指针

跟c语言类似，`&`取地址，`*`解地址。

```go
	s := 3
	s_ptr := &s
	fmt.Println(s_ptr)
	fmt.Println(*s_ptr)

// 声明
var s *int // s是int型指针
// 或者
s := new(int) // 其中，new(TYPE)

```



# 函数

```shell
func 函数名([参数列表]) [返回值列表] {
  函数体
}
```

```go
func main() { // main函数，程序入口
	a := 3
	b := 4
	fmt.Printf("Sum of %d and %d is %d\n", a, b, Sum(a, b))
}
func Sum(a int, b int) int {
	return a + b
}
```

**明确禁止函数重载**。

相同类型时，声明类型可以只写一个。
```go
// 冗余写法
func Sum(a int, b int) int {}

// 简洁写法
func Sum(a, b int) int {}
```

使用变长参数`...`接收不定长参数必须**放到参数列表最后**。在函数内部，变长参数实际是切片。如果外部有切片要传给变长参数，用`slice...`

**都是值传递**。不过不会消耗大量内存，底层结构本身就包含指针。

多返回值。

```go
func main() { // main函数，程序入口
	a := 3
	b := 4
	result, err := Sum(a, b)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	fmt.Printf("The sum of %d and %d is %d\n", a, b, result)
}
func Sum(a int, b int) (int, error) {
	return a + b, nil
}
```

具名返回值，return可以不写哪个变量。**return后面的优先级更高**。

```go
func main() { // main函数，程序入口
	a := 3
	b := 4
	result, err := Sum(a, b)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	fmt.Printf("The sum of %d and %d is %d\n", a, b, result)
}
func Sum(a int, b int) (answer int, err error) {
	answer, err = a + b, nil
	return
}
```

匿名函数，没有函数声明！！

```go
func main() {
    // 1. 立即执行 (你的示例)
    s := func(a, b int) int { return a + b }(3, 4)

    // 2. 赋值给变量 (更常见，常用于回调、闭包)
    add := func(a, b int) int { return a + b }
    fmt.Println(add(3, 4))
}

```

# 协程

## 协程（goroutine）

- **并发（concurrent）**：多个任务在**某些时间片段**交替推进（同时存在，未必同时执行）。
- **并行（parallel）**：多个任务在**任一时刻**真正同时执行（依赖多核），是并发的特殊情形。

- 协程又称**绿色线程**，由 Go runtime 维护，开销远小于系统线程，一个程序可轻松跑上万个协程。
- Go 不支持创建系统线程，**协程是唯一的并发实现方式**；程序启动时只有**主协程**。
- 函数调用前加 `go` 即在新协程中运行，**返回值必须全部舍弃**。

```go
go SayGreetings("hi!", 10)
```

> ⚠️ **主协程退出，整个程序就退出**，即使其它协程还在跑。并发打印用 `log`（已同步）而非 `fmt`，否则输出会交织。

## 并发同步

多个计算同时读写同一内存会产生**数据竞争**，需用并发同步避免。常用 `sync.WaitGroup`：

- `Add(n)` 注册 n 个任务；`Done()` 通知完成；`Wait()` 阻塞到全部完成。

## 协程状态

只有 **运行态** 和 **阻塞态**。`time.Sleep`/等系统调用属**运行态**；协程只能从运行态退出，阻塞态须被其它协程被动解除；全部阻塞 → **死锁崩溃**。

## 协程调度（M-P-G）

**M**=系统线程、**P**=逻辑处理器、**G**=协程，调度由 P 完成；同时执行数 ≤ 逻辑CPU数。`runtime.GOMAXPROCS` 设置 P 数量。

## 延迟函数调用（defer）

函数调用前加 `defer`，**不立即执行**，压栈后在外层函数退出阶段**按 LIFO 逆序**执行。返回值须全部舍弃，且可修改命名返回值：

```go
func Triple(n int) (r int) {
    defer func() { r += n }()
    return n + n   // r = n+n 再 += n → Triple(5)=15
}
```

### 实参估值时刻（重要）

- **defer 实参**：入栈时即估值（拍照快照）。
- **函数体内变量**：执行时才取值（监控探头，引用捕获）。

```go
for i := 0; i < 3; i++ {
    defer fmt.Println("a:", i)              // 实参 → 入栈即定 → a:2 a:1 a:0（所有版本）
    defer func() { fmt.Println("b:", i) }() // 函数体内 → 执行时取 i
}
// b 行：Go 1.22+ 每轮新建 i → b:2 b:1 b:0；Go 1.21 复用同一 i（=3）→ b:3 b:3 b:3
```

> 受不受 Go 1.22 影响，只看 `i` 在哪：**实参位置**入栈即定（与版本无关）；**函数体内**才受 for 语义影响。强制捕获每轮 i：传参 `func(i int){...}(i)` 或循环内 `i := i`。

## 恐慌（panic）和恢复（recover）

Go 不支持异常，推荐用返回值报错；提供类似的 panic/recover 机制。

- `panic(v)` 产生恐慌，函数立即进入退出阶段（触发 defer）。
- `recover()` **只能在延迟函数内调用**，消除当前协程恐慌并返回传给 panic 的值。
- ⚠️ 协程在恐慌状态下退出 → **整个程序崩溃**；新协程里 panic 若未 recover 也会崩整个程序。

```go
defer func() {
    if v := recover(); v != nil {
        fmt.Println("恢复：", v)
    }
}()
panic("拜拜！")
```

- runtime 也会 panic，如**整数除 0**：`runtime error: integer divide by zero`。
- 恐慌用于**不该发生的逻辑错误**（bug），非逻辑错误应正常处理而非 panic。
- ⚠️ **致命错误**（栈溢出、内存不足）不属于恐慌，**无法 recover**，程序直接崩溃。

# 结构体

类似 C，Go 用 `struct` 把若干**字段（field / 成员变量）**组合成复合类型。

## 声明

```go
// 无名结构体类型
struct {
    title, author string // 同类型字段可合并
    pages         int
}

// 具名（实践中更常用）
type Book struct {
    title, author string
    pages         int
}
```

- 尺寸 ≈ 各字段尺寸之和 + **padding**（内存对齐）；零字段结构体尺寸为 0。
- **不支持 union**；**不能自包含**（直接或间接含自身类型字段）。
- 字段可带**标签（tag）**，反引号写法，供反射/`encoding/json`（只看导出字段）使用，别当注释用。

```go
type Book struct {
    Title string `json:"title,omitempty"`
}
```

## 字面量与字段访问

`T{...}` 是**组合字面量**（类型确定值）。两种写法：

```go
book := Book{"Go语言101", "老貘", 256} // 不带字段名：顺序须一致，全部指定
book = Book{author: "老貘"}             // 带字段名：顺序无所谓，缺省取零值
book = Book{}                           // 全零值（最常用）

book.pages = 300                        // 选择器 v.x 访问字段
```

> 引用其它包的结构体**用带字段名**写法，避免对方日后加字段导致编译失败。

## 可寻址性 & 取地址

- 字段可寻址性跟随属主：可寻址结构体的字段可寻址；**所有组合字面量都不可寻址**。
- 语法糖：组合字面量虽不可寻址，却**可被取地址**：`p := &Book{100}`。
- 选择器属主是指针时**自动解引用**：`bookN.pages` 等价于 `(*bookN).pages`。

## 赋值 / 比较 / 转换

- **赋值**：逐字段拷贝。
- **比较**：所有字段都可比较才行（`_` 字段忽略），按声明顺序逐个比。
- **转换**：两个结构体值**能隐式互转**（底层类型相同且至少一方为无名类型）时才能互相赋值/比较。

```go
type S1 = struct{ x int "foo" } // 无名类型
type S2 = struct{ x int "bar" } // 无名类型
type S3 S2                      // 具名

v1, v3 := S1{}, S3{}
v1 = S1(v3); v3 = S3(v1)        // 显式转换
v2 := S2{}; v3 = v2             // S2↔S3 可隐式（S2 无名）
```

# 方法

方法拥有接受者，函数无。同时方法只有自定义类型能够拥有方法。



```go
package main // 当前go文件是哪个包的，入口文件必须是main包

import ( // 导入包
	"fmt"
)
type ADT []int
func (i ADT) Len() int {
	return len(i)
}
func (i ADT) Get(index int) (int, error) {
	lgh := i.Len()
	if index < 0 || index >= lgh {
		return 0, fmt.Errorf("index out of bounds")
	}
	return i[index], nil
}

func main() {
	s := []int{1, 2, 3} // 定义一个切片，包含三个整数
	adt := ADT(s) // 将切片转换为ADT类型
	fmt.Println("Length of ADT:", adt.Len()) // 输出ADT的长度
	value, err := adt.Get(133) // 获取索引为1的元素
	if err != nil {
		fmt.Println("Error:", err) // 如果发生错误，输出错误信息
	} else {
		fmt.Println("Value at index 1:", value) // 输出索引为1的元素值
	}
}
```



接受者，分为值接受者和指针接受者，其中**值接受者类似于形参**。

 如果是指针接受者，那么按照值的操作执行，go回将其转为指针

```go
ype MyInt int

func (i MyInt) Set(val int) {
   i = MyInt(val)
}

func main() {
   myInt := MyInt(1)
   myInt.Set(2) // 指针接受者，但是go将值转为指针了！
   fmt.Println(myInt)
}
```



# 接口

接口是一种抽象类型，用于定义一组方法签名而不提供方法的实现。

