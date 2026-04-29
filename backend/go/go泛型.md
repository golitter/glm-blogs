[泛型 | Golang 中文学习文档](https://golang.halfiisland.com/essential/senior/90.generic.html#注意点)

两种成熟但是极端的方案：

1. stenciling，cpp、rust选择的：编译器将不同的类型原封不动复制一份全新的函数代码。优点是极致的性能，缺点是编译时间长。
2. dictionarier，java选择的：编译器只生成一份代码，同时生成了一张**字典**，里面记录了T的详细信息。调用函数时，把数据和字典一起传过去。

对于go，是基于上面两种的一个融合。当值类型可以完美复杂（内存形状完美复制）编译器只会生成一份代码**公用**。但是遇到解引用、赋值等具体操作，编译器会根据一个字典进行具体操作。

```go
func sum[T int | float64](a T, b T) T {
	return a + b
}
```

泛型如上：

- T是类型形参
- `int | float64`是类型约束
- `sum[int](1,2)`是类型实参

```go
sum[int](a,b) // ok
sum(a,b) // ok
```

对于泛型切片，不能省略：

```go
type GenericSlice[T int | int32 | int64] []T

GenericSlice[int]{1,2,3}
```

```go
// 泛型哈希表
type GenericMap[K comparable, V int | string | byte] map[K]V

// 泛型结构体
type GenericStruct[T int | string] struct {
   Name string
   Id   T
}
```



## 类型集

只能用于泛型中的类型约束，不能用作类型声明、转换和断言。

并集：

```go
type SignedInt interface {
   int8 | int16 | int | int32 | int64
}
```

交集：

```go
type SignedInt interface {
   int8 | int16 | int | int32 | int64
}

type Integer interface {
   int8 | int16 | int | int32 | int64 | uint8 | uint16 | uint | uint32 | uint64
}

type Number interface {
  SignedInt
  Integer
}
```

交集即为`SignedInt`

使用 `~` 符号，来表示底层类型，如果一个类型的底层类型属于该类型集，那么该类型就属于该类型集，如下所示

```go
type Int interface {
   ~int8 | ~int16 | ~int | ~int32 | ~int64 | ~uint8 | ~uint16 | ~uint | ~uint32 | ~uint64
}
```

**泛型不能作为一个类型的基本类型，无法使用类型断言**。







