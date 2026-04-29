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

## 可见性规则

前面提到包内共享所有变量、常量及类型，但对于包外并非如此。Go 语言中没有 `public`、`private` 等关键字，控制可见性的方式非常简单：

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
- uintptr：等效于无符号64位整数，但是专门存放指针运算。

浮点数：float32、64，IEEE754格式的



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



## 字符串

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



## 映射表

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



## 指针

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



## 函数

```shell
func 函数名([参数列表]) [返回值] {
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

想通类型时，声明类型可以只写一个。

使用变长参数`...`接收不定长参数必须**放到参数列表最后**。

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

匿名函数，没有签名

```go
func main() {
	s := func(a, b int) int {
		return a + b
	}(3, 4)
	fmt.Println(s)
}
```



## 结构体



## 方法

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



## 接口

接口是一种抽象类型，用于定义一组方法签名而不提供方法的实现。

