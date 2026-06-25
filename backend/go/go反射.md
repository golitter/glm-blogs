运行时检查类型和值的机制。go 中反射与 `interface{}` 密切相关（见 `interface{}` 多半有反射）。

> 在程序运行期间，动态获取变量的类型信息和内存结构，并能动态修改它的值。

由 `reflect` 包提供，核心两个对象：

- `reflect.Type`：类型信息，`reflect.TypeOf(x)` 获取
- `reflect.Value`：值信息，`reflect.ValueOf(x)` 获取

## 三大法则

1. 接口值 → 反射对象（`TypeOf` / `ValueOf`）
2. 反射对象 → 接口值（`Interface()`）
3. 要修改反射对象，值必须可设置（`CanSet() == true`）

```go
x := 6
t := reflect.TypeOf(x)   // int
v := reflect.ValueOf(x)  // 6

y := v.Interface().(int) // 法则二：还原回接口值
```

## 修改值：必须传指针

`ValueOf` 拿到的是**副本**，改副本不影响原变量。想改原值：传指针 + `Elem()` 解引用。

```go
a := 5
v := reflect.ValueOf(&a).Elem() // 传 &a，再 Elem()
v.SetInt(10)
fmt.Println(a) // 10
```

```go
a := 5
reflect.ValueOf(a).SetInt(10) // panic：副本不可设置
```

## Type 与 Kind

- `Name()`：精确类型名，如 `MyInt`
- `Kind()`：底层大类，如 `int`

```go
type MyInt int
var x MyInt = 10
t := reflect.TypeOf(x)
t.Name() // MyInt
t.Kind() // int
```

常见 Kind：`Int Float String Bool Struct Ptr Slice Map Array Func Chan Interface`。

## %T vs reflect.TypeOf vs Kind

三个都能「看类型」，但角度不同，容易混：

```go
type MyInt int
x := MyInt(123)
v := reflect.ValueOf(x)

fmt.Printf("%T\n", x)                 // main.MyInt  ← 静态类型
fmt.Println(reflect.TypeOf(x))        // main.MyInt  ← 同上
fmt.Println(reflect.TypeOf(x).Kind()) // int         ← 底层种类
fmt.Println(v.Kind())                 // int         ← 底层种类
```

| 方式 | 看到的是 | 回答的问题 |
| ---- | ---- | ---- |
| `%T` | 静态类型 | 这个变量是什么类型？ |
| `reflect.TypeOf(x)` | 静态类型 | 同上 |
| `Kind()` | 底层种类 | 它底层用什么内存结构存？ |

> ⚠️ 把 `%T` 用在反射对象 `v` 上要小心：`%T` 打印的是变量本身的类型 `reflect.Value`，不是盒子里装的 `int`：

```go
v := reflect.ValueOf(123)
fmt.Printf("%T\n", v)   // reflect.Value  ← 盒子本身
fmt.Printf("%v\n", v)   // 123            ← 盒子里的值
fmt.Println(v.Type())   // int            ← 盒子里东西的真实类型
```

一句话：`%T` 看「是什么类型」，`%v` 看「是什么值」；`%T` 告诉你 `v` 是「反射盒子」，`v.Type()` 才告诉你盒子里东西的真实类型。

## 遍历结构体字段与 tag

```go
type User struct {
	Name string `json:"name" db:"user_name"`
	Age  int    `json:"age" db:"user_age"`
}

u := User{"Alice", 25}
t := reflect.TypeOf(u)
v := reflect.ValueOf(u)

for i := 0; i < t.NumField(); i++ {
	f := t.Field(i)
	fmt.Printf("%s %s %v tag=%s\n", f.Name, f.Type, v.Field(i), f.Tag.Get("json"))
}
// Name string Alice tag=name
// Age int 25 tag=age
```

- `t.Field(i)` / `v.Field(i)`：字段的元信息 / 值
- `f.Tag.Get("json")`：读 tag
- `t.FieldByName("Age")`：按名取

改结构体字段同样要 `reflect.ValueOf(&u).Elem()`。**未导出字段 `CanSet()` 为 false，Set 会 panic**。

## 调用方法

```go
v := reflect.ValueOf(dog)
method := v.MethodByName("Say")
out := method.Call([]reflect.Value{reflect.ValueOf("hi")}) // 参数也得是 Value
```

## 用在哪

`fmt` 打印、`encoding/json` 编解码、ORM 映射字段、配置解析、依赖注入、RPC、validator——凡是处理「未知类型」的框架都靠它。

## 代价

- 性能比普通代码慢
- 编译期类型检查变少，错误变成运行时 panic
- 可读性差：`u.Name = "Bob"` vs `reflect.ValueOf(&u).Elem().FieldByName("Name").SetString("Bob")`

常见 panic：对副本 Set、改未导出字段、类型不匹配的 Set、对 nil 取字段。防御：先判 `CanSet()` 和 `Kind()`。

## 速查

| 想做 | 代码 |
| ---- | ---- |
| 看类型 | `reflect.TypeOf(x)` |
| 看值 | `reflect.ValueOf(x)` |
| 还原接口 | `v.Interface()` |
| 改原变量 | `reflect.ValueOf(&x).Elem()` |
| 底层大类 | `t.Kind()` |
| 精确类型名 | `t.Name()` |
| 字段数 / 第 i 字段 | `t.NumField()` / `t.Field(i)` |
| 读 tag | `f.Tag.Get("json")` |
| 调方法 | `v.MethodByName("X").Call(args)` |
| 是否可设置 | `v.CanSet()` |

> **核心公式**：`reflect.TypeOf(x)` 看类型，`reflect.ValueOf(x)` 看值，想改就 `reflect.ValueOf(&x).Elem()`。
