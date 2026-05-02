[反射 | Golang 中文学习文档](https://golang.halfiisland.com/essential/senior/105.reflect.html)

反射是一种在运行时检查语言自身结构的机制，它可以很灵活的去应对一些问题，但同时带来的弊端也很明显，例如性能问题等等。在go中，反射与`interface{}`密切相关（只要出现这个大概率有反射）

> 反射就是在程序运行期间，能够动态地获取一个变量的类型信息和内存结构，并且能够动态地修改它的值的能力。



反射是由`reflect`包提供。



在go中，接口本质上是结构体。go在运行时将接口分为了两大类：

- 没有方法集的接口
- 有方法集的接口

对于有方法集的接口，运行时采用`iface`结构体表示；而无方法集的接口则用`eface`接口。

这两个结构体在`reflect`包下对应的结构体：

```go
type nonEmptyInterface struct {
  itab *struct {
    ityp *rtype // 静态接口类型
    typ  *rtype // 动态具体类型
    hash uint32 // 类型哈希
    _    [4]byte
    fun  [100000]unsafe.Pointer // 方法集
  }
  word unsafe.Pointer // 指向值的指针
}
type emptyInterface struct {
   typ  *rtype // 动态具体类型
   word unsafe.Pointer // 指向指针的值
}
```

- `iface` -> `nonEmptyInterface`
- `eface` -> `emptyInterface`

在`reflect`包下，有`reflect.Type`接口类型表示go中的类型，`reflect.Value`结构体类型表示go中的值。

go中的所有反射相关的操作都是基于这两个类型。使用`TypeOf`和`ValueOf`来进行上述类型的转换。



反射的核心：

1. 反射可以将`interface{}`类型变量转换成反射对象
2. 反射可以将反射对象还原成`interface{}`类型变量
3. 要修改反射对象，其值必须是可以设置的



```go
 func main() {
  str := "hello world!"
  reflectType := reflect.TypeOf(str)
  fmt.Println(reflectType)
}
```

```go
  // 定理一
  x := 6
  x_type := reflect.TypeOf(x)
  x_value := reflect.ValueOf(x)
  fmt.Printf("Type of x: %s\n", x_type)
  fmt.Printf("Value of x: %d\n", x_value.Int())

  // 定理二
  x_reflect := x_value.Interface()
  fmt.Printf("Value of x using Interface: %d\n", x_reflect.(int))
  fmt.Println(reflect.TypeOf(x_reflect)) // int
```

`reflect.ValueOf`拿到的是**变量副本**，修改副本不会影响原变量。想要通过反射修改原变量，**必须传入指针**，并使用`Elem()`解引用拿到指向的值，同时需要保证值是可设置的`Canset()==true`。

```go
  a := 5
  a_ptr_reflect := reflect.ValueOf(&a)
  a_elem := a_ptr_reflect.Elem()
  fmt.Println(reflect.TypeOf(a_elem))

  if a_elem.CanSet() {
    a_elem.SetInt(10)
  }
  fmt.Println(a)
```

reflect中，`Kind`代表底层的大类，比如`int string`等。`Name`表示精确的类型名称，比如`MyInt`。

```go
type MyInt int
var x MyInt = 10
t := reflect.TypeOf(x)

fmt.Println(t.Name()) // 输出: MyInt (精确类型名)
fmt.Println(t.Kind()) // 输出: int   (底层种类)
```



## 遍历结构体的字段和标签

```go
type AName struct {
    Name string `json:"name" db:"user_name"`
}
type User struct {
    Name AName `json:"na33me" db:"user_name"`
    Age  int   `json:"age" db:"user_age"`
}

u := User{AName{"Alice"}, 25}
t := reflect.TypeOf(u)

// 遍历结构体字段
for i := 0; i < t.NumField(); i++ {
    field := t.Field(i)
    fmt.Printf("字段名: %s, 类型: %s, json标签: %s\n", 
        field.Name, field.Type, field.Tag.Get("json"))
}
// 输出：
// 字段名: Name, 类型: string, json标签: na33me
// 字段名: Age, 类型: int, json标签: age
```

